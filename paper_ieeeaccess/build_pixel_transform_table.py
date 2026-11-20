"""Generate same-subset comparison directly from the exploratory result file."""
import hashlib
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent
source = P.parent / 'frozen/results/PIXEL_TRANSFORM_RESULT_20261105.json'
data = json.loads(source.read_text())['splits']
rows = []
for split, name in [('test_id', 'Assets/wording'), ('test_ood', 'New templates')]:
    v = data[split]
    c = v['english_model_comparators']
    mean = lambda fmt: statistics.mean(c[f'v3_{fmt}_s{i}']['supported']['balanced_accuracy'] for i in range(5))
    vals = [v['supported']['balanced_accuracy'], c['qwen3vl2b']['supported']['balanced_accuracy'],
            c['omniverifier7b']['supported']['balanced_accuracy'], mean('structured'), mean('describe')]
    rows.append(name + ' & ' + f"{v['supported']['n']} ({100*v['coverage']:.1f}\\%)" +
                ' & ' + ' & '.join(f'{100*x:.1f}' for x in vals) + r' \\')
target = P / 'tables/pixel_transform.tex'
target.write_text('\n'.join(rows) + '\n')
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {'result_sha256': digest(source), 'generator_sha256': digest(Path(__file__)),
            'table_sha256': digest(target)}
(P / 'tables/PIXEL_TRANSFORM_MANIFEST_20261105.json').write_text(json.dumps(manifest, indent=2) + '\n')
