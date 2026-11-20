"""Generate two held-out-template rows from the verified five-seed summary."""
import hashlib,json
from collections import Counter
from pathlib import Path
P=Path(__file__).resolve().parent;T=P.parent
summary=T/'frozen/results/HOLDOUT_FOLLOWUP_20261103.json';population=P/'revision_evidence.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads(summary.read_text())['splits']['test_ood']['by_template'];counts={k:v['n'] for k,v in json.loads(population.read_text())['internal']['procgen_v3_test_ood']['v3_structured_s0']['by_template'].items()};assert counts=={'A4_rotonly':1095,'B_add':700}
names={'A4_rotonly':'Rotation-only choice','B_add':'Addition detection'}
lines=[]
for template in ('A4_rotonly','B_add'):
 for method,label in [('structured','Structured'),('describe','Describe')]:
  m=d[template][method];assert m['n_checkpoints']==5
  b=m['balanced_accuracy'];a=m['accuracy']
  ncell=f"{counts[template]:,}".replace(',',r'{,}')
  lines.append(names[template]+' & '+label+' & '+ncell+' & '+f"{b['mean']:.3f} "+r'$\pm$ '+f"{b['sd']:.3f} & {a['mean']:.3f} "+r'$\pm$ '+f"{a['sd']:.3f} "+r'\\')
out=P/'tables/templates_five.tex';out.write_text('\n'.join(lines)+'\n');manifest={'summary_sha256':sha(summary),'population_sha256':sha(population),'generator_sha256':sha(Path(__file__)),'table_sha256':sha(out)};(P/'tables/TEMPLATES_FIVE_MANIFEST_20261103.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Five-seed template table verified:',dict(counts))
