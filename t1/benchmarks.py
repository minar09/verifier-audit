"""Pinned DARE/MM-UPD conversions. Published-answer task and T1 validity stay separate."""
from __future__ import annotations
import argparse
import base64
import collections
import hashlib
import io
import json
import random
from pathlib import Path

from PIL import Image
from .itemgen import DATA
from .records import atomic_text, atomic_json, sha256, check_coverage, fingerprint, load_rows

RAW = DATA/'external/20261021'
DEST = DATA/'external_eval'
LETTERS = 'ABCDE'


def image_file(blob, directory):
    digest = hashlib.sha256(blob).hexdigest()
    p = directory/(digest+'.png')
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(io.BytesIO(blob)).convert('RGB')
        im.thumbnail((448,448))
        im.save(p)
    return str(p.relative_to(DATA))


def instance(iid, group, q, options, image, answers, task):
    numbers = [i+1 for i, _ in enumerate(options)]
    correct = [numbers[i] for i,(k,_) in enumerate(options) if k in answers]
    native_q = q+'\n'+'\n'.join(f'{k}: {v}' for k,v in options)
    q = q+'\n'+'\n'.join(f'Option {i+1}: {v}' for i,(_,v) in enumerate(options))
    return {'id': iid, 'base_id': group, 'family': 'A', 'slot_id': task, 'task': task,
        'op': 'published', 'question': q, 'native_question': native_q, 'native_answers': answers,
        'option_letters': [k for k,_ in options], 'key': str(correct[0] if correct else 1),
        'images': [{'role': 'panel','path': image,'display_label': 'Panel 1'}], 'keep_aspect': True,
        'label': {'valid': len(correct)==1, 'defect': 'none' if len(correct)==1 else 'multi_correct' if correct else 'no_correct',
                  'correct_set': correct}}


def validity_rows(r):
    correct = r['label']['correct_set']
    rng = random.Random('20261022-'+r['id'])
    if len(correct)==1:
        choices = [('gold',correct[0]), ('wrong',rng.choice([i+1 for i in range(len(r['option_letters'])) if i+1 not in correct]))]
    else:
        choices = [('multi' if correct else 'unanswerable', rng.choice(correct or list(range(1,len(r['option_letters'])+1))))]
    result = []
    for name,key in choices:
        valid = len(correct)==1 and key==correct[0]
        defect = 'none' if valid else 'no_correct' if not correct else 'multi_correct' if len(correct)>1 else 'key_mismatch'
        result.append(dict(r,id=r['id']+'__'+name,key=str(key),op=r['task']+'__'+name,
                           label={'valid':valid,'defect':defect,'correct_set':correct}))
    return result


def save_suite(name, native, source_files):
    check_coverage(native, {r['id']: {} for r in native})
    validity = [x for r in native for x in validity_rows(r)]
    check_coverage(validity,{r['id']: {} for r in validity})
    groups = sorted({r['base_id'] for r in native})
    # One immutable exploration sample across all configurations/conditions of a source problem.
    chosen = set(random.Random(20261022).sample(groups,min(400,len(groups))))
    root = DEST/name
    manifest = {'version':1,'source_files':{str(p.relative_to(DATA)):sha256(p) for p in source_files},
        'native_n':len(native),'validity_n':len(validity),'groups':len(groups),
        'first_pass_groups':sorted(chosen),'selection_seed':20261022,
        'native_rows_sha256':fingerprint(native),'validity_rows_sha256':fingerprint(validity),
        'native_prompt': 'All correct option letters for DARE; one letter or F for unsolvable MM-UPD.',
        'comparison_scope':'Pinned public release with fixed project prompts, not an official leaderboard reproduction.'}
    m = root/'protocol_v1.json'
    if m.exists() and json.loads(m.read_text()) != manifest:
        raise ValueError('Refusing to change frozen external protocol '+name)
    for kind,rows in [('native',native),('validity',validity)]:
        for split,subset in [('full',rows),('first_pass',[r for r in rows if r['base_id'] in chosen])]:
            p=root/kind/split/'instances.jsonl'
            text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset)
            if p.exists() and p.read_text()!=text: raise ValueError('Frozen conversion changed '+str(p))
            atomic_text(p,text)
    atomic_json(m,manifest)
    print(name, len(native),'native,',len(validity),'validity,',len(groups),'source groups',flush=True)


def build_dare():
    import pyarrow.parquet as pq
    native,files=[],[]
    for config in ('1_correct','1_correct_var','n_correct'):
        p=RAW/'dare'/config/'validation/0000.parquet';files.append(p)
        for ordinal,row in enumerate(pq.read_table(p).to_pylist()):
            opts=[(k,str(row[k])) for k in 'ABCD' if row.get(k) is not None and str(row[k]).strip()]
            answers=list(row['answer'])
            if not set(answers)<=set(k for k,_ in opts): raise ValueError('Invalid published DARE answer')
            image=image_file(row['img']['bytes'],DEST/'dare/images')
            r=instance(f'dare_{config}_{ordinal:05d}_'+row['id'],f"dare_{row['category']}_{row['instance_id']}",
                row['question'],opts,image,answers,config)
            r['category']=row['category'];native.append(r)
    save_suite('dare',native,files)


def build_upd():
    import pandas as pd
    native,files=[],[]
    for task in ('aad','iasd','ivqd'):
        p=RAW/'mm_upd/data'/f'mm{task}_20240303_base.tsv';files.append(p)
        d=pd.read_csv(p,sep='\t',keep_default_na=False)
        for row in d.to_dict('records'):
            opts=[(k,str(row[k])) for k in LETTERS if str(row.get(k,'')).strip()]
            answers=[] if row['answer']=='F' else [row['answer']]
            if bool(answers)!=(row['type']=='standard'): raise ValueError('UPD type/answer contract mismatch')
            if not set(answers)<=set(k for k,_ in opts): raise ValueError('Missing published standard option')
            blob=base64.b64decode(row['image'],validate=True)
            image=image_file(blob,DEST/'mm_upd/images')
            q=(('Hint: '+row['hint']+'\n') if row['hint'] else '')+row['question']
            # Group same original index across all three benchmark conditions.
            r=instance(f"mmupd_{task}_{row['index']}_{row['type']}",f"mmupd_{row['index']}",q,opts,image,answers,task)
            r['source_type']=row['type'];r['category']=row['category'];native.append(r)
    save_suite('mm_upd',native,files)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('suite',choices=['dare','mm_upd','all']);a=ap.parse_args()
    if a.suite in ('dare','all'):build_dare()
    if a.suite in ('mm_upd','all'):build_upd()


if __name__=='__main__':main()
