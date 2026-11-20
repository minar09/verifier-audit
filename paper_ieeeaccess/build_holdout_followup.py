import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
BASE={'qwen3vl2b':'Qwen3-VL-2B','qwen3vl8b':'Qwen3-VL-8B','omniverifier7b':'OmniVerifier-7B'}
SPLITS={'test_id':'Assets / wording','test_ood':'Two templates'}


def tables(o):
    """Write the holdout comparison and contrast tables, with a source/output hash manifest."""
    label={'R1_spec_rules':'R1 rules (spec.)',**BASE,'structured':'Tuned, structured','describe':'Tuned, describe-then-verify'}
    f=lambda x:'---' if x is None else f"{x:.3f}"
    lines=[]
    for split,name in SPLITS.items():
        ms=o['splits'][split]['methods'];image=[k for k in ('qwen3vl2b','qwen3vl8b','omniverifier7b','structured','describe')]
        best=max(ms[k]['balanced_accuracy']['mean'] for k in image)
        for k in ['R1_spec_rules']+image:
            m=ms[k];ba=m['balanced_accuracy']['mean'];cell=f(ba)
            if k in image and abs(ba-best)<5e-4:cell=r'\textbf{'+cell+'}'
            if m['n_checkpoints']>1:cell+=rf" $\pm$ {m['balanced_accuracy']['sd']:.3f}"
            row=[name if k=='R1_spec_rules' else '',label[k],str(m['n_checkpoints']) if k in ('structured','describe') else '---',cell,
                 f(m['invalid_recall']['mean']),f(m['valid_retention']['mean']),f(m['ppv']['0.1']),f(m['ppv']['0.05'])]
            lines.append(' & '.join(row)+r' \\')
        if split!=list(SPLITS)[-1]:lines.append(r'\midrule')
    (HERE/'tables/holdout_baselines.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    names={'structured-qwen3vl2b':'Structured $-$ Qwen3-VL-2B','describe-qwen3vl2b':'Describe $-$ Qwen3-VL-2B',
           'structured-qwen3vl8b':'Structured $-$ Qwen3-VL-8B','describe-qwen3vl8b':'Describe $-$ Qwen3-VL-8B','structured-omniverifier7b':'Structured $-$ OmniVerifier-7B',
           'describe-omniverifier7b':'Describe $-$ OmniVerifier-7B','describe-structured':'Describe $-$ structured'}
    for key,name in names.items():
        cells=[]
        for split in SPLITS:
            c=o['splits'][split]['contrasts'][key];cells.append(f"{100*c['delta_bacc']:+.1f} [{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}]")
        lines.append(name+' & '+' & '.join(cells)+r' \\')
    (HERE/'tables/holdout_contrasts.tex').write_text('\n'.join(lines)+'\n')
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    manifest={'inputs':{'frozen/results/HOLDOUT_FOLLOWUP_20261103.json':sha(TRACK/'frozen/results/HOLDOUT_FOLLOWUP_20261103.json')},'generator_sha256':sha(Path(__file__)),
              'outputs':{n:sha(HERE/'tables'/n) for n in ('holdout_baselines.tex','holdout_contrasts.tex')}}
    (HERE/'tables/HOLDOUT_FOLLOWUP_MANIFEST_20261103.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    tables(json.loads((TRACK / 'frozen/results/HOLDOUT_FOLLOWUP_20261103.json').read_text(encoding='utf-8')))
