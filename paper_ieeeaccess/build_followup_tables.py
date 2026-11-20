"""Generate prompt-sensitivity and pixel-duplicate audit tables from frozen reports."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;T=P.parent
files={'neutral':T/'frozen/results/NEUTRAL_PROMPT_RESULTS_20261101.json','pixel':T/'frozen/results/PIXEL_DUPLICATE_BASELINE_20261030.json'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
n=json.loads(files['neutral'].read_text());p=json.loads(files['pixel'].read_text())
if not n['complete'] or n['missing']:raise ValueError('Neutral evaluation incomplete')
labels=[('Qwen3-VL-2B','qwen3vl2b_base'),('OmniVerifier-7B, common','omniverifier7b_base')]+[(f'Structured, seed {i}',f'v3_structured_s{i}') for i in range(3)]+[(f'Describe, seed {i}',f'v3_describe_s{i}') for i in range(3)]
rows=[]
for label,tag in labels:
 cells=[]
 for ds in ('muir','dare','mm_upd'):
  v=n['rows'][ds][tag]['paired_neutral_minus_original'];lo,hi=v['ci95'];cells.append(f"{100*v['delta_bacc']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
 rows.append(label+' & '+' & '.join(cells)+r' \\')
(P/'tables/neutral_prompt.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for key,label in [('test_id','Assets and wording'),('test_ood','Two templates')]:
 d=p['splits'][key];v=d['all_items'];q=d['by_defect_choice'];rows.append(f"{label} & {v['n']:,} & {100*v['invalid_recall']:.1f} & {100*v['invalid_precision']:.1f} & {100*v['valid_retention']:.1f} & {100*q['multi_correct']['invalid_recall']:.1f} & {100*q['no_correct']['invalid_recall']:.1f} & {100*q['key_mismatch']['invalid_recall']:.1f} "+r'\\')
(P/'tables/pixel_duplicate.tex').write_text('\n'.join(rows)+'\n')
manifest={'inputs':{k:{'path':str(v.relative_to(T)),'sha256':sha(v)} for k,v in files.items()},'outputs':{f:sha(P/'tables'/f) for f in ('neutral_prompt.tex','pixel_duplicate.tex')},'generator_sha256':sha(Path(__file__))}
(P/'tables/FOLLOWUP_MANIFEST_20261102.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Wrote two tables and hash manifest')
