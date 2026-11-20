import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
SPLITS={'test_id':'Assets / wording','test_ood':'Two templates'}
LABEL={'qwen3vl2b':'Qwen3-VL-2B','omniverifier7b':'OmniVerifier-7B','structured':'Tuned, structured','describe':'Tuned, describe-then-verify'}


def table(o):
    lines=[]
    for split,name in SPLITS.items():
        res=o['splits'].get(split,{})
        for i,(g,r) in enumerate(res.items()):
            c=r['english_minus_korean'];d=f"{100*c['delta_bacc']:+.1f} [{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}]"
            lines.append(' & '.join([name if i==0 else '',LABEL[g],str(len(r['checkpoints'])),f"{r['korean']['balanced_accuracy']:.3f}",
                f"{r['english']['balanced_accuracy']:.3f}",d,f"{r['verdict_agreement']:.3f}"])+r' \\')
        if split!=list(SPLITS)[-1] and res:lines.append(r'\midrule')
    (HERE/'tables/english_holdout.tex').write_text('\n'.join(lines)+'\n')
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    (HERE/'tables/ENGLISH_HOLDOUT_MANIFEST_20261104.json').write_text(json.dumps({'inputs':{'frozen/results/ENGLISH_HOLDOUT_RESULT_20261104.json':sha(TRACK/'frozen/results/ENGLISH_HOLDOUT_RESULT_20261104.json')},
        'generator_sha256':sha(Path(__file__)),'outputs':{'english_holdout.tex':sha(HERE/'tables/english_holdout.tex')}},indent=2)+'\n')


if __name__ == '__main__':
    table(json.loads((TRACK / 'frozen/results/ENGLISH_HOLDOUT_RESULT_20261104.json').read_text(encoding='utf-8')))
