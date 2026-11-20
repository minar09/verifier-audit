"""Generate every results table of the IEEE Access draft from the frozen evidence files.

Run from this directory:  python build_tables.py
Writes tables/*.tex and tables/MANIFEST.json (source paths and SHA-256), so each number
in the paper can be traced to one record. Never edit tables/*.tex by hand.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
DATA = TRACK / 'data'
OUT = HERE / 'tables'
SEEDS = (0, 1, 2)
SOURCES = {}


def load(path):
    path = Path(path)
    SOURCES[str(path.relative_to(TRACK))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def rows(path):
    path = Path(path)
    SOURCES[str(path.relative_to(TRACK))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def mark(values, fmt, higher=True):
    """Bold the best and underline the second distinct displayed value."""
    shown = [None if v is None else fmt.format(v) for v in values]
    distinct = sorted({float(s) for s in shown if s is not None}, reverse=higher)
    out = []
    for s in shown:
        if s is None:
            out.append('--')
        elif distinct and float(s) == distinct[0]:
            out.append(r'\textbf{' + s + '}')
        elif len(distinct) > 1 and float(s) == distinct[1]:
            out.append(r'\underline{' + s + '}')
        else:
            out.append(s)
    return out


def table(rows_, cols, groups, fmt='{:.3f}', higher=None, rank=None, highlight=None, spreads=None):
    """rows_: list of (label, [values]); groups: indices after which a midrule goes.
    rank: indices of rows that compete for best/second (a matched population); others are
    shown unhighlighted, e.g. rules with specification access or text-only baselines."""
    higher = higher or [True] * len(cols)
    rank = set(range(len(rows_))) if rank is None else set(rank)
    # One-sided rates (recall, valid kept) are maximized trivially by flagging everything or
    # nothing, so they are shown but never highlighted.
    highlight = highlight or [True] * len(cols)
    marked = []
    for c in range(len(cols)):
        if not highlight[c]:
            marked.append(['--' if r[1][c] is None else fmt.format(r[1][c]) for r in rows_])
            continue
        ranked = mark([r[1][c] if i in rank else None for i, r in enumerate(rows_)], fmt, higher[c])
        plain = ['--' if r[1][c] is None else fmt.format(r[1][c]) for r in rows_]
        marked.append([ranked[i] if i in rank else plain[i] for i in range(len(rows_))])
    marked = list(zip(*marked))
    lines = []
    for i, ((label, _), cells) in enumerate(zip(rows_, marked)):
        if spreads and spreads[i] is not None:
            cells=[cell+rf' $\pm$ {fmt.format(spread)}' for cell,spread in zip(cells,spreads[i])]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
        if i in groups:
            lines.append(r'\midrule')
    return '\n'.join(lines) + '\n'


def mean(xs):
    return statistics.mean(xs)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def pilot():
    s = load(DATA / 'runs/natural_pilot/_scores.json')
    pick = [('Pipeline validator', 'R0_validator'), ('Strengthened rules', 'R1_spec_rules'),
            ('Qwen3-VL-2B, holistic', 'qwen3vl2b__V1'), ('Qwen3-VL-8B, per option', 'qwen3vl8b__V3'),
            ('OmniVerifier-7B, binary', 'omniverifier7b__V4'), ('OmniVerifier-7B, native', 'omniverifier7b__OV')]
    body = [(lab, [s[k]['invalid_recall'], s[k]['invalid_precision'], s[k]['valid_retention'], s[k]['accuracy']])
            for lab, k in pick]
    return table(body, range(4), groups={1}, fmt='{:.2f}', rank=range(2, len(body)),
                 highlight=[False, False, False, True])


def internal():
    d = load(DATA / 'reports/continuation_v3/internal_three_seeds.json')['by_run']
    sets = [('natural_pilot', 'Pilot'), ('procgen_v3_test_id', 'Held-out assets'), ('procgen_v3_test_ood', 'Held-out templates')]
    out, raw = [], {}
    for sup, lab in (('structured', 'Structured'), ('describe', 'Describe-then-verify')):
        vals = []
        for key, _ in sets:
            accs = [100 * d[f'v3_{sup}_s{s}:{key}']['accuracy'] for s in SEEDS]
            vals.append((mean(accs), sd(accs)))
        raw[lab] = vals
    for c in range(len(sets)):
        best = max(raw, key=lambda k: round(raw[k][c][0], 1))
        for lab in raw:
            m, s_ = raw[lab][c]
            cell = f'{m:.1f}'
            raw[lab][c] = (r'\textbf{' + cell + '}' if lab == best else cell) + rf' $\pm$ {s_:.1f}'
    for lab, cells in raw.items():
        out.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    return '\n'.join(out) + '\n'


def muir():
    s = load(TRACK / 'frozen/results/PRIMARY_EXTERNAL_SUMMARY_20261025.json')['methods']
    text = load(TRACK / 'frozen/results/MUIRV2_TEXT_AUDIT_20261023.json')
    def avg(sup, key):
        return mean([s[f'v3_{sup}_s{k}'][key] for k in SEEDS])
    body = [('Text-only heuristics', [0.5, 4 / 7, None, None])];spreads=[None]
    for lab, k in (('Qwen3-VL-2B', 'qwen3vl2b_base'), ('Qwen3-VL-8B', 'qwen3vl8b_base'),
                   ('OmniVerifier-7B, common', 'omniverifier7b_base'), ('OmniVerifier-7B, native', 'omniverifier7b_native')):
        body.append((lab, [s[k]['balanced_accuracy'], s[k]['accuracy'], s[k]['valid_retention'], s[k]['native_accuracy']]))
        spreads.append(None)
    for lab, sup in (('Tuned, structured', 'structured'), ('Tuned, describe-then-verify', 'describe')):
        body.append((lab, [avg(sup, 'balanced_accuracy'), avg(sup, 'accuracy'), avg(sup, 'valid_retention'),
                           avg(sup, 'native_accuracy')]))
        spreads.append([sd([s[f'v3_{sup}_s{k}'][metric] for k in SEEDS]) for metric in
                        ('balanced_accuracy','accuracy','valid_retention','native_accuracy')])
    return table(body, range(4), groups={0, 4}, rank=range(1, len(body)), highlight=[True, True, False, True],spreads=spreads)


def public():
    out = {}
    for suite in ('dare', 'mm_upd'):
        d = load(DATA / f'reports/continuation_v3/{suite}_first_pass.json')
        v, nat = d['validity'], d['native']
        def avg(sup, part, key):
            return mean([part[f'v3_{sup}_s{k}'][key] for k in SEEDS])
        def pf(part, key):
            return part[key].get('parse_fail', 0) / part[key]['n']
        body = [];spreads=[]
        for lab, k in (('Qwen3-VL-2B', 'qwen3vl2b_base'), ('Qwen3-VL-8B', 'qwen3vl8b_base'),
                       ('OmniVerifier-7B, common', 'omniverifier7b_base'), ('OmniVerifier-7B, native', 'omniverifier7b_native')):
            body.append((lab, [v[k]['balanced_accuracy'], v[k]['valid_retention'], nat[k]['accuracy'], pf(nat, k)]))
            spreads.append(None)
        for lab, sup in (('Tuned, structured', 'structured'), ('Tuned, describe-then-verify', 'describe')):
            body.append((lab, [avg(sup, v, 'balanced_accuracy'), avg(sup, v, 'valid_retention'), avg(sup, nat, 'accuracy'),
                               mean([pf(nat, f'v3_{sup}_s{k}') for k in SEEDS])]))
            spreads.append([sd([v[f'v3_{sup}_s{k}'][metric] for k in SEEDS]) for metric in
                            ('balanced_accuracy','valid_retention')]+[
                            sd([nat[f'v3_{sup}_s{k}']['accuracy'] for k in SEEDS]),
                            sd([pf(nat,f'v3_{sup}_s{k}') for k in SEEDS])])
        out[suite] = (table(body, range(4), groups={3}, higher=[True, True, True, False],
                            highlight=[True, False, True, True],spreads=spreads),
                      v['qwen3vl2b_base']['n'], nat['qwen3vl2b_base']['n'])
    return out


def data_tables():
    s = load(DATA / 'procgen_v3/summary.json')['by_split']
    names = (('train', 'Training'), ('dev', 'Development'), ('test_id', 'Held-out assets and wordings'),
             ('test_ood', 'Held-out templates'))
    lines = [f"{lab} & {s[k]['instances']:,} & {100 * s[k]['valid_rate']:.1f} \\\\".replace(',', '{,}')
             for k, lab in names]
    nat = rows(DATA / 'natural_pilot/table_counts.jsonl')
    counts = collections.Counter(r['label']['defect'] for r in nat)
    return '\n'.join(lines) + '\n', len(nat), counts


def operators():
    nat = rows(DATA / 'natural_pilot/table_counts.jsonl')
    train = rows(DATA / 'procgen_v3/train/instances.jsonl')
    def fam(op):
        for k in ('qswap_angle_rekeyed', 'qswap_angle', 'qswap_color', 'qswap_', 'key_swap', 'dup_key', 'perm'):
            if op.startswith(k):
                return {'qswap_': 'qswap_type'}.get(k, k)
        return op
    names = {'orig': 'Original item', 'perm': 'Permute options (key remapped)', 'restyle': 'Recolor an untouched object',
             'dup_key': 'Copy the keyed option into a distractor', 'key_swap': 'Move the key to another option',
             'break_key': 'Break the keyed option', 'dup_distractor': 'Copy one distractor onto another',
             'invisible_literal': 'Distractor with a same-value change', 'invisible_symmetry': 'Distractor with a symmetric turn or flip',
             'sym_reference': 'Replace the stimulus by a symmetric shape', 'qswap_angle': 'Change the asked angle (stale key)',
             'qswap_angle_rekeyed': 'Change the asked angle (key updated)', 'qswap_color': 'Change the asked color',
             'qswap_type': 'Ask about another change type', 'noop_literal': 'Make the change a same-value change',
             'wrong_op': 'Replace the change by another type', 'extra_change': 'Add one more change of the asked type',
             'drop_change': 'Remove one of several changes'}
    order = list(names)
    short = {'none': 'V', 'no_correct': 'N', 'multi_correct': 'M', 'key_mismatch': 'K'}
    def dist(data):
        c = collections.defaultdict(collections.Counter)
        for r in data:
            c[fam(r['op'])][short[r['label']['defect']]] += 1
        return c
    dn, dt = dist(nat), dist(train)
    lines = []
    for op in order:
        if op not in dn and op not in dt:
            continue
        cell = lambda c: ' & '.join(str(c[k]) for k in 'VNMK')
        lines.append(f'{names[op]} & {cell(dn[op])} & {cell(dt[op])} \\\\')
    return '\n'.join(lines) + '\n'


def main():
    OUT.mkdir(exist_ok=True)
    calibration=load(DATA/'calibration.json');display=load(DATA/'display_audit_448.json');small=load(DATA/'calibration_448.json')
    def scientific(value):
        if not value:return '0'
        exponent=math.floor(math.log10(value))
        return rf'${value/10**exponent:.1f}\times10^{{{exponent}}}$'
    values=[('Visually null edits',str(calibration['n_same'])),('Minimal real edits',str(calibration['n_different'])),
        ('Largest null-edit score',scientific(calibration['max_same'])),
        ('Smallest real-edit score',scientific(calibration['min_different'])),('Threshold',scientific(calibration['same_frac']))]
    (OUT/'calibration.tex').write_text('\n'.join(name+' & '+value+r' \\' for name,value in values)+'\n')
    macros={'DisplayAuditMinReal':scientific(small['min_different'])}
    for label,key in [('Pilot','natural_pilot'),('Assets','procgen_v3:test_id'),('Templates','procgen_v3:test_ood')]:
        for field,suffix in [('n','N'),('label_changes','Changes')]:
            macros['DisplayAudit'+label+suffix]=f"{display['sets'][key][field]:,}".replace(',',r'{,}')
    (OUT/'facts.tex').write_text('\n'.join(r'\newcommand{'+chr(92)+name+'}{'+value+'}' for name,value in macros.items())+'\n')
    from build_revision_tables import build
    revision=load(HERE/'revision_evidence.json')
    build(revision, OUT)
    costs=load(HERE/'training_cost_evidence.json')['runs']
    lines=[]
    for sup in ('structured','describe'):
        rr=[costs[f'v3_{sup}_s{k}'] for k in SEEDS];tt=[revision['training'][f'v3_{sup}_s{k}'] for k in SEEDS]
        lines.append(sup.capitalize()+' & '+f"{mean([r['mean_supervised_tokens'] for r in rr]):.2f}"+' & '+f"{mean([r['held_asset_decoded_retokenized_mean'] for r in rr]):.2f}"+' & '+f"{min(r['elapsed_hours'] for r in tt):.2f}--{max(r['elapsed_hours'] for r in tt):.2f}"+r' \\')
    (OUT/'training_cost.tex').write_text('\n'.join(lines)+'\n')
    (OUT / 'pilot.tex').write_text(pilot())
    (OUT / 'internal.tex').write_text(internal())
    (OUT / 'muir.tex').write_text(muir())
    pub = public()
    for suite, (tex, n_val, n_nat) in pub.items():
        (OUT / f'{suite}.tex').write_text(tex)
    data_tex, n_pilot, counts = data_tables()
    (OUT / 'data.tex').write_text(data_tex)
    (OUT / 'operators.tex').write_text(operators())
    facts = {'pilot_instances': n_pilot, 'pilot_defects': dict(counts),
             'dare_validity_n': pub['dare'][1], 'dare_native_n': pub['dare'][2],
             'mm_upd_validity_n': pub['mm_upd'][1], 'mm_upd_native_n': pub['mm_upd'][2]}
    (OUT / 'MANIFEST.json').write_text(json.dumps({'sources_sha256': SOURCES, 'facts': facts}, indent=1))
    print(json.dumps(facts))


if __name__ == '__main__':
    main()
