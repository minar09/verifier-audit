"""Tables for the locked 901-pair Muir-V2 confirmation, generated directly from its report."""
import hashlib
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent
TRACK = P.parent
source = TRACK / 'data/runs/muirv2_confirmation_v1/confirmation_report.json'
protocol = TRACK / 'frozen/MUIRV2_CONFIRMATION_20261105.md'
r = json.loads(source.read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if r['protocol_sha256'] != digest(protocol) or r['missing'] or r['pairs'] != 901 or r['items'] != 6307:
    raise ValueError('Confirmation report incomplete or protocol changed')
M = r['methods']
COLS = ('balanced_accuracy', 'accuracy', 'invalid_recall', 'invalid_precision', 'valid_retention')


def single(tag, name):
    m = M[tag]['metrics']
    return name + ' & ' + ' & '.join(f'{m[c]:.3f}' for c in COLS) + f" & {M[tag]['parse_fail']}" + r' \\'


def seeds(fmt, name):
    tags = [f'v3_{fmt}_s{i}' for i in range(3)]
    cells = []
    for c in COLS:
        v = [M[t]['metrics'][c] for t in tags]
        cells.append(f'{statistics.mean(v):.3f} $\\pm$ {statistics.stdev(v):.3f}')
    return name + ' & ' + ' & '.join(cells) + ' & ' + '/'.join(str(M[t]['parse_fail']) for t in tags) + r' \\'


heur = r['text_baselines']
lines = [name + ' & ' + ' & '.join(f'{heur[k][c]:.3f}' for c in COLS) + r' & -- \\'
         for k, name in (('always_invalid', 'Always invalid'), ('none_present', 'None option present'),
                         ('key_is_none', 'Key is None'))]
lines += [r'\midrule', single('qwen3vl2b_base', 'Qwen3-VL-2B'), single('qwen3vl8b_base', 'Qwen3-VL-8B'),
          single('omniverifier7b_base', 'OmniVerifier-7B, common'), single('omniverifier7b_native', 'OmniVerifier-7B, native'),
          r'\midrule', seeds('structured', 'Tuned, structured'), seeds('describe', 'Tuned, describe-then-verify')]
(P / 'tables/muir_confirmation.tex').write_text('\n'.join(lines) + '\n')

# Per-type flag rates: every Muir-V2 variant type is single-class, so report the share flagged invalid.
TYPES = [('ans_gold', 'Answerable, None removed, gold key'), ('ans_wrong', 'Answerable, None removed, wrong key'),
         ('ans_gold_none', 'Answerable, None shown, gold key'), ('ans_none_key', 'Answerable, None shown, key None'),
         ('unans_native', 'Unanswerable, None shown, key None'), ('unans_wrong', 'Unanswerable, None shown, other key'),
         ('unans_drop', 'Unanswerable, None removed')]
def flag_rate(tag, kind):
    m = M[tag]['by_type'][kind]
    return m['invalid_recall'] if m.get('invalid_recall') is not None else 1 - m['valid_retention']
def valid_type(kind):
    return M['qwen3vl2b_base']['by_type'][kind].get('invalid_recall') is None
rows = []
for kind, name in TYPES:
    vals = [flag_rate('qwen3vl2b_base', kind), flag_rate('omniverifier7b_base', kind),
            statistics.mean(flag_rate(f'v3_structured_s{i}', kind) for i in range(3)),
            statistics.mean(flag_rate(f'v3_describe_s{i}', kind) for i in range(3))]
    rows.append(f"{name} & {'valid' if valid_type(kind) else 'invalid'} & " + ' & '.join(f'{v:.3f}' for v in vals) + r' \\')
(P / 'tables/muir_confirmation_types.tex').write_text('\n'.join(rows) + '\n')

c = r['primary_contrasts']
facts = {k: {'difference_points': round(100 * v['difference'], 2), 'ci97_5_points': [round(100 * x, 2) for x in v['ci97_5']]}
         for k, v in c.items()}
manifest = {'report_sha256': digest(source), 'protocol_sha256': digest(protocol), 'generator_sha256': digest(Path(__file__)),
            'tables_sha256': {n: digest(P / f'tables/{n}') for n in ('muir_confirmation.tex', 'muir_confirmation_types.tex')},
            'primary_contrasts': facts}
(P / 'tables/MUIR_CONFIRMATION_MANIFEST_20261106.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(facts))
