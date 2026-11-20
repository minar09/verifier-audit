"""Score prediction files against oracle labels (PLAN.md Section 9).

A prediction file maps instance id -> {"valid": bool, "defect"?: str, "p_invalid"?: float,
"parse_ok"?: bool}. Unparseable or missing outputs count as "valid" (the verifier did not
flag the item), which is how a pipeline would treat them.

    python -m t1.score [run_dir]
"""

from __future__ import annotations

import json
import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from .itemgen import DATA
from .records import load_rows as canonical_load, atomic_json, sha256, fingerprint

PILOT = DATA / "natural_pilot"
DEFECTS = ["none", "no_correct", "multi_correct", "key_mismatch"]
# Prespecified consequential families for the gate (PLAN.md Section 9).
GATE = {"symmetry_multi": {"invisible_symmetry", "sym_reference"},
        "key_mismatch": {"key_swap", "key_swap_0", "key_swap_1", "qswap_angle"}}


def load_rows(path=None):
    return canonical_load(path or PILOT / 'instances.jsonl')


def _bin(rows, preds):
    tp = fp = fn = tn = 0
    for r in rows:
        inval = not r["label"]["valid"]
        flag = not preds.get(r["id"], {}).get("valid", True)
        tp += inval and flag; fp += (not inval) and flag
        fn += inval and not flag; tn += (not inval) and not flag
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"invalid_recall": rec, "invalid_precision": prec, "valid_retention": spec,
            "accuracy": (tp + tn) / max(len(rows), 1), "n": len(rows)}


def _macro_f1(rows, preds):
    f1s = []
    for d in DEFECTS:
        tp = sum(1 for r in rows if r["label"]["defect"] == d and preds.get(r["id"], {}).get("defect") == d)
        p = sum(1 for r in rows if preds.get(r["id"], {}).get("defect") == d)
        g = sum(1 for r in rows if r["label"]["defect"] == d)
        f1s.append(2 * tp / (p + g) if p + g else 0.0)
    return sum(f1s) / len(f1s)


SCORE_VERSION = 3   # explicit evaluated population, coverage, and frozen dev calibration


def _scored(rows, preds):
    """(p_invalid or None, is_invalid) for every row; missing scores are kept, never dropped."""
    def finite(v):
        return v if isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 else None
    return [(finite(preds.get(r['id'], {}).get('p_invalid')), not r['label']['valid']) for r in rows]


def _at_threshold(pairs, thr):
    """Flag items with score >= thr; unscored items are never flagged."""
    tp = sum(1 for s, y in pairs if s is not None and s >= thr and y)
    fp = sum(1 for s, y in pairs if s is not None and s >= thr and not y)
    pos = sum(1 for _, y in pairs if y)
    return tp, fp, pos


def _threshold_points(pairs):
    blocks = defaultdict(lambda: [0, 0])
    pos = sum(y for _, y in pairs)
    for s, y in pairs:
        if s is not None: blocks[s][0 if y else 1] += 1
    tp = fp = 0
    for s in sorted(blocks, reverse=True):
        tp += blocks[s][0]; fp += blocks[s][1]
        yield s, tp, fp, pos


def recall_at_precision(rows, preds, target=0.9):
    """Descriptive PR statistic on the evaluated set: best recall over realizable score
    thresholds whose precision >= target. Not a deployable operating point (see
    `calibrated_operating_point`)."""
    pairs = _scored(rows, preds)
    if not any(s is not None for s, _ in pairs):
        return None
    best = 0.0
    for thr, tp, fp, pos in _threshold_points(pairs):
        if tp + fp and tp / (tp + fp) >= target:
            best = max(best, tp / pos if pos else 0.0)
    return best


def calibrate_threshold(rows, preds, target=0.9):
    """Lowest realizable threshold reaching `target` precision on a calibration set."""
    pairs = _scored(rows, preds)
    ok = [thr for thr, tp, fp, pos in _threshold_points(pairs) if tp+fp and tp/(tp+fp) >= target]
    return min(ok) if ok else None


def calibrated_operating_point(cal_rows, cal_preds, rows, preds, target=0.9):
    """Threshold frozen on calibration data, then applied unchanged to the evaluated set."""
    thr = calibrate_threshold(cal_rows, cal_preds, target)
    if thr is None:
        return {"threshold": None}
    tp, fp, pos = _at_threshold(_scored(rows, preds), thr)
    return {"threshold": thr, "recall": tp / pos if pos else None,
            "precision": tp / (tp + fp) if tp + fp else None}


def bootstrap(rows, preds, metric="invalid_recall", n=2000, seed=0):
    import numpy as np
    groups = defaultdict(list)
    for r in rows:
        groups[r["base_id"]].append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    if not keys: return (None, None)
    counts = []
    for k in keys:
        tp=fp=fn=tn=0
        for r in groups[k]:
            inv=not r['label']['valid'];flag=not preds.get(r['id'],{}).get('valid',True)
            tp+=inv and flag;fp+=not inv and flag;fn+=inv and not flag;tn+=not inv and not flag
        counts.append([tp,fp,fn,tn])
    counts=np.asarray(counts);vals=[]
    for start in range(0,n,128):
        idx=np.asarray([rng.choices(range(len(keys)),k=len(keys)) for _ in range(min(128,n-start))])
        totals=counts[idx].sum(axis=1);tp,fp,fn,tn=totals.T
        num,den={'invalid_recall':(tp,tp+fn),'invalid_precision':(tp,tp+fp),
            'valid_retention':(tn,tn+fp),'accuracy':(tp+tn,tp+tn+fp+fn)}[metric]
        vals.extend((num[den>0]/den[den>0]).tolist())
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (None, None)


def score(rows, preds, calibration=None) -> dict:
    out = _bin(rows, preds)
    out["invalid_recall_ci95"] = bootstrap(rows, preds)
    out['invalid_precision_ci95'] = bootstrap(rows,preds,'invalid_precision')
    out['valid_retention_ci95'] = bootstrap(rows,preds,'valid_retention')
    out["macro_f1_defect"] = _macro_f1(rows, preds) if any("defect" in p for p in preds.values()) else None
    out["recall_at_p90"] = recall_at_precision(rows, preds)
    out["parse_fail"] = sum(1 for r in rows if preds.get(r["id"], {}).get("parse_ok") is False)
    out['missing_predictions'] = sum(r['id'] not in preds for r in rows)
    out['scored_items'] = sum(s is not None for s, _ in _scored(rows, preds))
    out['balanced_accuracy'] = (out['invalid_recall'] + out['valid_retention']) / 2
    groups = defaultdict(list)
    for r in rows:
        groups[r['base_id']].append(r)
    out['all_instances_correct_per_group'] = sum(all(r['id'] in preds and preds[r['id']].get('valid') == r['label']['valid']
        for r in g) for g in groups.values()) / max(len(groups), 1)
    out['accuracy_ci95'] = bootstrap(rows, preds, 'accuracy')
    out['by_op'] = {op: _bin([r for r in rows if r['op'] == op], preds) for op in sorted({r['op'] for r in rows})}
    if all('task' in r for r in rows):
        out['by_task']={task:_bin([r for r in rows if r['task']==task],preds) for task in sorted({r['task'] for r in rows})}
    out['dev_calibrated'] = {'status': 'calibration_not_supplied'}
    if calibration is not None:
        thr = calibration.get('threshold')
        if thr is None:
            out['dev_calibrated'] = {'status': 'no_qualifying_dev_threshold', **calibration}
        else:
            pairs = _scored(rows, preds)
            tp, fp, pos = _at_threshold(pairs, thr)
            flags = {r['id']: {'valid': not (s is not None and s >= thr)} for r, (s, _) in zip(rows, pairs)}
            out['dev_calibrated'] = {**calibration, 'status': 'applied_unchanged',
                **_bin(rows, flags), 'invalid_recall_ci95': bootstrap(rows, flags),
                'invalid_precision_ci95': bootstrap(rows,flags,'invalid_precision'),
                'test_precision_target_met': bool(tp+fp and tp/(tp+fp) >= calibration.get('target_precision', .9))}
    out["by_family"] = {f: _bin([r for r in rows if r["family"] == f], preds) for f in ("A", "B")
                        if any(r["family"] == f for r in rows)}
    out["gate"] = {g: (_bin(sel, preds)["invalid_recall"] if sel else None)
                   for g, ops in GATE.items()
                   for sel in [[r for r in rows if r["op"] in ops and not r["label"]["valid"]]]}
    ops = sorted({r["op"] for r in rows})
    out["by_op_flag_rate"] = {op: round(sum(1 for r in rows if r["op"] == op and
                                            not preds.get(r["id"], {}).get("valid", True)) /
                                        sum(1 for r in rows if r["op"] == op), 3) for op in ops}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('run', nargs='?', type=Path, default=DATA/'runs/natural_pilot')
    ap.add_argument('instances', nargs='?', type=Path)
    ap.add_argument('--first-pass', action='store_true')
    ap.add_argument('--calibration-rows', type=Path)
    ap.add_argument('--calibration-dir', type=Path)
    ap.add_argument('--thresholds', type=Path)
    ap.add_argument('--freeze-only',action='store_true')
    ap.add_argument('--methods',nargs='+',help='Explicit method population, e.g. primary versus supplementary seeds')
    a = ap.parse_args()
    run, rows = a.run, load_rows(a.instances)
    if a.first_pass:
        from .external import first_pass_pairs
        keep = first_pass_pairs()
        rows = [r for r in rows if r['base_id'] in keep]
    if bool(a.calibration_rows) != bool(a.calibration_dir):
        ap.error('Supply both --calibration-rows and --calibration-dir')
    frozen = json.loads(a.thresholds.read_text()) if a.thresholds and a.thresholds.exists() else {}
    if a.calibration_rows:
        cal_rows = load_rows(a.calibration_rows)
        if {r['base_id'] for r in cal_rows} & {r['base_id'] for r in rows}:
            raise ValueError('Calibration and evaluation groups overlap')
        for f in sorted(a.calibration_dir.glob('*.json')):
            if f.name.startswith('_'): continue
            if a.methods and f.stem not in a.methods:continue
            ps = json.loads(f.read_text())
            from .records import check_coverage
            check_coverage(cal_rows, ps)
            entry = {'threshold': calibrate_threshold(cal_rows, ps), 'target_precision': .9,
                'calibration_rows_sha256': fingerprint(cal_rows), 'calibration_predictions_sha256': sha256(f)}
            if entry['threshold'] is not None:
                tp,fp,pos=_at_threshold(_scored(cal_rows,ps),entry['threshold'])
                entry['calibration_flagged']=tp+fp
                entry['calibration_precision']=tp/(tp+fp)
                entry['calibration_recall']=tp/pos if pos else None
            if f.stem in frozen and frozen[f.stem] != entry:
                raise ValueError('Refusing to change frozen threshold: '+f.stem)
            frozen[f.stem] = entry
        if a.methods and not set(a.methods)<=set(frozen):
            raise ValueError('Missing calibration methods; no threshold file written')
        if a.thresholds: atomic_json(a.thresholds, frozen)
    if a.freeze_only:
        if not a.thresholds or not a.calibration_rows:ap.error('--freeze-only requires calibration data and --thresholds')
        if a.methods and not set(a.methods)<=set(frozen):raise ValueError('Missing calibration methods')
        return
    table = {}
    for f in sorted(run.glob("*.json")):
        if f.name.startswith("_"):
            continue
        if a.methods and f.stem not in a.methods:continue
        table[f.stem] = score(rows, json.loads(f.read_text()), frozen.get(f.stem))
    if a.methods and set(a.methods)!=set(table):raise ValueError('Requested methods are missing from the score table')
    table = {"_score_version": SCORE_VERSION, **table}
    previous=run/'_scores.json'
    if previous.exists():
        old=json.loads(previous.read_text());version=old.get('_score_version',1)
        archive=run/f'_scores_v{version}.json'
        if version!=SCORE_VERSION and not archive.exists():atomic_json(archive,old)
    atomic_json(run/'_scores.json', table)
    table.pop("_score_version")
    hdr = f'{"method":34s} {"inv_rec":>7s} {"CI95":>13s} {"inv_prec":>8s} {"val_ret":>7s} {"acc":>5s} {"mF1":>5s} {"R@P90":>5s} {"sym":>5s} {"key":>5s} {"A_rec":>5s} {"B_rec":>5s} {"fail":>4s}'
    print(hdr)
    fmt = lambda v: "  -  " if v is None or v != v else f"{v:5.2f}"
    for name, s in table.items():
        ci = s["invalid_recall_ci95"]
        print(f'{name:34s} {s["invalid_recall"]:7.2f} {f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci[0] is not None else "-":>13s} '
              f'{s["invalid_precision"]:8.2f} {s["valid_retention"]:7.2f} {s["accuracy"]:5.2f} {fmt(s["macro_f1_defect"])} '
              f'{fmt(s["recall_at_p90"])} {fmt(s["gate"]["symmetry_multi"])} {fmt(s["gate"]["key_mismatch"])} '
              f'{fmt(s["by_family"].get("A", {}).get("invalid_recall"))} {fmt(s["by_family"].get("B", {}).get("invalid_recall"))} {s["parse_fail"]:4d}')


if __name__ == "__main__":
    main()
