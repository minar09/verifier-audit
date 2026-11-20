"""Compute untouched non-first-pass and full-release metrics for DARE and MM-UPD.

Generates:
  - ../FULL_EXTERNAL_SUMMARY_20261118.json
  - tables/public_full.tex
  - tables/public_remainder_contrasts.tex
  - tables/FULL_EXTERNAL_MANIFEST_20261118.json

Runs in pure stdlib Python so table verification requires no external wheels.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
DATA = TRACK / "data"
TABLES = HERE / "tables"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from build_tables import mark, mean, sd  # noqa: E402

TAGS = (
    "qwen3vl2b_base",
    "qwen3vl8b_base",
    "omniverifier7b_base",
    "omniverifier7b_native",
    "v3_structured_s0",
    "v3_structured_s1",
    "v3_structured_s2",
    "v3_describe_s0",
    "v3_describe_s1",
    "v3_describe_s2",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path, sources: dict[str, str]) -> list[dict[str, Any]]:
    sources[str(path.relative_to(TRACK))] = sha256_file(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path, sources: dict[str, str]) -> Any:
    sources[str(path.relative_to(TRACK))] = sha256_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def check_coverage(rows: list[dict[str, Any]], preds: dict[str, Any], label: str) -> None:
    expected = {r["id"] for r in rows}
    missing = expected - set(preds)
    if missing:
        raise ValueError(f"{label}: missing {len(missing)} predictions (e.g. {sorted(missing)[:3]})")


def validity_metrics(rows: list[dict[str, Any]], preds: dict[str, Any]) -> dict[str, Any]:
    tp = fp = fn = tn = pf = 0
    for r in rows:
        is_inv = not r["label"]["valid"]
        pred = preds[r["id"]]
        if pred.get("parse_ok") is False:
            pf += 1
        flagged = not pred.get("valid", True)
        if is_inv and flagged:
            tp += 1
        elif is_inv and not flagged:
            fn += 1
        elif not is_inv and flagged:
            fp += 1
        else:
            tn += 1
    rec = tp / (tp + fn) if (tp + fn) else None
    ret = tn / (tn + fp) if (tn + fp) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    bacc = 0.5 * (rec + ret) if (rec is not None and ret is not None) else None
    return {
        "n": len(rows),
        "groups": len({r["base_id"] for r in rows}),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": bacc,
        "invalid_recall": rec,
        "invalid_precision": prec,
        "valid_retention": ret,
        "parse_fail": pf,
    }


def native_metrics(rows: list[dict[str, Any]], preds: dict[str, Any]) -> dict[str, Any]:
    correct = 0
    format_fail = 0
    for r in rows:
        pred = preds[r["id"]]
        if pred.get("correct"):
            correct += 1
        if pred.get("parse_ok") is False or pred.get("answer") is None:
            format_fail += 1
    n = len(rows)
    return {
        "n": n,
        "groups": len({r["base_id"] for r in rows}),
        "accuracy": correct / n,
        "format_fail_count": format_fail,
        "format_fail": format_fail / n,
    }


def paired_contrasts_bootstrap(
    rows: list[dict[str, Any]],
    preds_by_method: dict[str, list[dict[str, Any]]],
    contrasts: list[tuple[str, str, str]],
    resamples: int = 10000,
    seed: int = 20261118,
) -> dict[str, dict[str, Any]]:
    """Compute paired source-group bootstrap 95% CIs on pre-aggregated group counts."""
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        grouped[r["base_id"]].append(r)
    group_keys = sorted(grouped)
    g = len(group_keys)

    # Pre-aggregate per-group denominators and per-method (tp_mean, tn_mean)
    n_inv_g = [sum(1 for r in grouped[k] if not r["label"]["valid"]) for k in group_keys]
    n_val_g = [sum(1 for r in grouped[k] if r["label"]["valid"]) for k in group_keys]

    method_tp: dict[str, list[float]] = {}
    method_tn: dict[str, list[float]] = {}
    for m_key, pred_list in preds_by_method.items():
        tp_list: list[float] = []
        tn_list: list[float] = []
        n_ckpts = len(pred_list)
        for k in group_keys:
            rs = grouped[k]
            tp_sum = 0.0
            tn_sum = 0.0
            for p in pred_list:
                for r in rs:
                    is_inv = not r["label"]["valid"]
                    flagged = not p[r["id"]].get("valid", True)
                    if is_inv and flagged:
                        tp_sum += 1.0
                    elif not is_inv and not flagged:
                        tn_sum += 1.0
            tp_list.append(tp_sum / n_ckpts)
            tn_list.append(tn_sum / n_ckpts)
        method_tp[m_key] = tp_list
        method_tn[m_key] = tn_list

    def bacc_from_totals(tp: float, den_i: int, tn: float, den_v: int) -> float:
        return 0.5 * (tp / den_i + tn / den_v)

    total_inv = sum(n_inv_g)
    total_val = sum(n_val_g)
    observed = {}
    boot_vals: dict[str, list[float]] = {name: [] for name, _, _ in contrasts}
    for name, a_key, b_key in contrasts:
        ba_a = bacc_from_totals(sum(method_tp[a_key]), total_inv, sum(method_tn[a_key]), total_val)
        ba_b = bacc_from_totals(sum(method_tp[b_key]), total_inv, sum(method_tn[b_key]), total_val)
        observed[name] = ba_a - ba_b

    rng = random.Random(seed)
    randrange = rng.randrange
    tp_base = method_tp["2b_base"]
    tn_base = method_tn["2b_base"]
    tp_omni = method_tp["omni_common"]
    tn_omni = method_tn["omni_common"]
    tp_struct = method_tp["structured"]
    tn_struct = method_tn["structured"]
    tp_desc = method_tp["describe"]
    tn_desc = method_tn["describe"]
    packed = list(zip(n_inv_g, n_val_g, tp_base, tn_base, tp_omni, tn_omni, tp_struct, tn_struct, tp_desc, tn_desc))
    bv_sb = boot_vals["structured-base"]
    bv_so = boot_vals["structured-omni"]
    bv_ds = boot_vals["describe-structured"]
    for _ in range(resamples):
        di = 0
        dv = 0
        s_tp_b = s_tn_b = s_tp_o = s_tn_o = s_tp_s = s_tn_s = s_tp_d = s_tn_d = 0.0
        for _ in range(g):
            ni, nv, tb, nb, to, no, ts, ns, td, nd = packed[randrange(g)]
            di += ni
            dv += nv
            s_tp_b += tb
            s_tn_b += nb
            s_tp_o += to
            s_tn_o += no
            s_tp_s += ts
            s_tn_s += ns
            s_tp_d += td
            s_tn_d += nd
        if di == 0 or dv == 0:
            continue
        ba_b = 0.5 * (s_tp_b / di + s_tn_b / dv)
        ba_o = 0.5 * (s_tp_o / di + s_tn_o / dv)
        ba_s = 0.5 * (s_tp_s / di + s_tn_s / dv)
        ba_d = 0.5 * (s_tp_d / di + s_tn_d / dv)
        bv_sb.append(ba_s - ba_b)
        bv_so.append(ba_s - ba_o)
        bv_ds.append(ba_d - ba_s)

    out: dict[str, dict[str, Any]] = {}
    for name, _, _ in contrasts:
        vals = sorted(boot_vals[name])
        lo = vals[int(0.025 * len(vals))]
        hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
        out[name] = {
            "delta_bacc": observed[name],
            "ci95": [lo, hi],
            "groups": g,
            "resamples": resamples,
            "accepted_resamples": len(vals),
            "scope": "paired source-group percentile interval on untouched post-first-pass groups; fixed checkpoints; exploratory",
        }
    return out


def aggregate_family(rows_by_tag: dict[str, dict[str, Any]], fmt: str, metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    tags = [f"v3_{fmt}_s{i}" for i in range(3)]
    for m in metrics:
        vals = [rows_by_tag[t][m] for t in tags]
        out[m] = {"mean": mean(vals), "sd": sd(vals)}
    return out


def build_summary(resamples: int = 10000) -> tuple[dict[str, Any], dict[str, str]]:
    sources: dict[str, str] = {}
    summary: dict[str, Any] = {
        "date": "2026-11-18",
        "note": (
            "Evaluates the completed full public-source releases of DARE and MM-UPD "
            "(EXTERNAL_PROTOCOL.md lines 57-69) and isolates untouched non-first-pass source groups "
            "disjoint from the 400-group exploratory first passes."
        ),
        "suites": {},
    }

    for suite in ("dare", "mm_upd"):
        fp_val = load_jsonl(DATA / f"external_eval/{suite}/validity/first_pass/instances.jsonl", sources)
        full_val = load_jsonl(DATA / f"external_eval/{suite}/validity/full/instances.jsonl", sources)
        fp_nat = load_jsonl(DATA / f"external_eval/{suite}/native/first_pass/instances.jsonl", sources)
        full_nat = load_jsonl(DATA / f"external_eval/{suite}/native/full/instances.jsonl", sources)

        fp_groups = {r["base_id"] for r in fp_val}
        if fp_groups != {r["base_id"] for r in fp_nat}:
            raise ValueError(f"{suite}: validity vs native first-pass base_ids differ")
        full_groups = {r["base_id"] for r in full_val}
        if not fp_groups.issubset(full_groups):
            raise ValueError(f"{suite}: first-pass base_ids not subset of full")

        rem_val = [r for r in full_val if r["base_id"] not in fp_groups]
        rem_nat = [r for r in full_nat if r["base_id"] not in fp_groups]
        rem_groups = {r["base_id"] for r in rem_val}
        if rem_groups & fp_groups:
            raise ValueError(f"{suite}: overlap between first-pass and remainder base_ids")

        val_preds_full: dict[str, Any] = {}
        nat_preds_full: dict[str, Any] = {}
        for tag in TAGS:
            pv_fp = load_json(DATA / f"runs/{suite}_validity_first_pass/{tag}.json", sources)
            pv_full = load_json(DATA / f"runs/{suite}_validity_full/{tag}.json", sources)
            check_coverage(fp_val, pv_fp, f"{suite}/validity/fp/{tag}")
            check_coverage(full_val, pv_full, f"{suite}/validity/full/{tag}")
            for r in fp_val:
                if pv_fp[r["id"]] != pv_full[r["id"]]:
                    raise ValueError(f"{suite}/validity/{tag}: first_pass != full on {r['id']}")
            val_preds_full[tag] = pv_full

            pn_fp = load_json(DATA / f"runs/{suite}_native_first_pass/{tag}.json", sources)
            pn_full = load_json(DATA / f"runs/{suite}_native_full/{tag}.json", sources)
            check_coverage(fp_nat, pn_fp, f"{suite}/native/fp/{tag}")
            check_coverage(full_nat, pn_full, f"{suite}/native/full/{tag}")
            for r in fp_nat:
                if pn_fp[r["id"]] != pn_full[r["id"]]:
                    raise ValueError(f"{suite}/native/{tag}: first_pass != full on {r['id']}")
            nat_preds_full[tag] = pn_full

        val_rem_res = {tag: validity_metrics(rem_val, val_preds_full[tag]) for tag in TAGS}
        val_full_res = {tag: validity_metrics(full_val, val_preds_full[tag]) for tag in TAGS}
        nat_rem_res = {tag: native_metrics(rem_nat, nat_preds_full[tag]) for tag in TAGS}
        nat_full_res = {tag: native_metrics(full_nat, nat_preds_full[tag]) for tag in TAGS}

        preds_grouped = {
            "2b_base": [val_preds_full["qwen3vl2b_base"]],
            "omni_common": [val_preds_full["omniverifier7b_base"]],
            "structured": [val_preds_full[f"v3_structured_s{i}"] for i in range(3)],
            "describe": [val_preds_full[f"v3_describe_s{i}"] for i in range(3)],
        }
        rem_contrasts = paired_contrasts_bootstrap(
            rem_val,
            preds_grouped,
            contrasts=[
                ("structured-base", "structured", "2b_base"),
                ("structured-omni", "structured", "omni_common"),
                ("describe-structured", "describe", "structured"),
            ],
            resamples=resamples,
        )

        summary["suites"][suite] = {
            "counts": {
                "first_pass": {"groups": len(fp_groups), "validity": len(fp_val), "native": len(fp_nat)},
                "untouched_remainder": {"groups": len(rem_groups), "validity": len(rem_val), "native": len(rem_nat)},
                "full": {"groups": len(full_groups), "validity": len(full_val), "native": len(full_nat)},
            },
            "validity_remainder": {
                "by_tag": val_rem_res,
                "three_seed_means": {
                    fmt: aggregate_family(val_rem_res, fmt, ("balanced_accuracy", "accuracy", "invalid_recall", "invalid_precision", "valid_retention"))
                    for fmt in ("structured", "describe")
                },
            },
            "validity_full": {
                "by_tag": val_full_res,
                "three_seed_means": {
                    fmt: aggregate_family(val_full_res, fmt, ("balanced_accuracy", "accuracy", "invalid_recall", "invalid_precision", "valid_retention"))
                    for fmt in ("structured", "describe")
                },
            },
            "native_remainder": {
                "by_tag": nat_rem_res,
                "three_seed_means": {
                    fmt: aggregate_family(nat_rem_res, fmt, ("accuracy", "format_fail"))
                    for fmt in ("structured", "describe")
                },
            },
            "native_full": {
                "by_tag": nat_full_res,
                "three_seed_means": {
                    fmt: aggregate_family(nat_full_res, fmt, ("accuracy", "format_fail"))
                    for fmt in ("structured", "describe")
                },
            },
            "remainder_paired_contrasts": rem_contrasts,
        }

    summary["sources_sha256"] = sources
    return summary, sources


def render_public_full_tex(summary: dict[str, Any]) -> str:
    singles = (
        ("Qwen3-VL-2B", "qwen3vl2b_base"),
        ("Qwen3-VL-8B", "qwen3vl8b_base"),
        ("OmniVerifier-7B, common", "omniverifier7b_base"),
        ("OmniVerifier-7B, native", "omniverifier7b_native"),
    )
    tuned = (
        ("Tuned, structured", "structured"),
        ("Tuned, describe-then-verify", "describe"),
    )
    sections: list[str] = []
    for suite, title in (("dare", "DARE"), ("mm_upd", "MM-UPD")):
        s = summary["suites"][suite]
        c_rem = s["counts"]["untouched_remainder"]
        c_full = s["counts"]["full"]
        sections.append(
            rf"\multicolumn{{7}}{{l}}{{\textit{{{title}}} "
            rf"(untouched: {c_rem['validity']:,} val., {c_rem['groups']:,} groups; "
            rf"full: {c_full['validity']:,} val., {c_full['native']:,} nat.)}} \\".replace(",", "{,}")
        )
        rows: list[tuple[str, list[float], list[float] | None]] = []
        vr = s["validity_remainder"]
        vf = s["validity_full"]
        nf = s["native_full"]
        for label, tag in singles:
            rows.append(
                (
                    label,
                    [
                        vr["by_tag"][tag]["balanced_accuracy"],
                        vr["by_tag"][tag]["valid_retention"],
                        vf["by_tag"][tag]["balanced_accuracy"],
                        vf["by_tag"][tag]["valid_retention"],
                        nf["by_tag"][tag]["accuracy"],
                        nf["by_tag"][tag]["format_fail"],
                    ],
                    None,
                )
            )
        for label, fmt in tuned:
            m_vr = vr["three_seed_means"][fmt]
            m_vf = vf["three_seed_means"][fmt]
            m_nf = nf["three_seed_means"][fmt]
            rows.append(
                (
                    label,
                    [
                        m_vr["balanced_accuracy"]["mean"],
                        m_vr["valid_retention"]["mean"],
                        m_vf["balanced_accuracy"]["mean"],
                        m_vf["valid_retention"]["mean"],
                        m_nf["accuracy"]["mean"],
                        m_nf["format_fail"]["mean"],
                    ],
                    [
                        m_vr["balanced_accuracy"]["sd"],
                        m_vr["valid_retention"]["sd"],
                        m_vf["balanced_accuracy"]["sd"],
                        m_vf["valid_retention"]["sd"],
                        m_nf["accuracy"]["sd"],
                        m_nf["format_fail"]["sd"],
                    ],
                )
            )

        # Highlight matched columns (BA, Full BA, Full Nat Acc high; Full Format fail low).
        # One-sided valid retention columns are unhighlighted.
        highlight_cols = (True, False, True, False, True, True)
        higher_cols = (True, True, True, True, True, False)
        cols_marked = []
        for c_idx in range(6):
            vals = [r[1][c_idx] for r in rows]
            if highlight_cols[c_idx]:
                cols_marked.append(mark(vals, "{:.3f}", higher=higher_cols[c_idx]))
            else:
                cols_marked.append([f"{v:.3f}" for v in vals])
        for r_idx, (label, _, sds) in enumerate(rows):
            cells = []
            for c_idx in range(6):
                cell = cols_marked[c_idx][r_idx]
                if sds is not None:
                    cell += rf" $\pm$ {sds[c_idx]:.3f}"
                cells.append(cell)
            sections.append(label + " & " + " & ".join(cells) + r" \\")
            if r_idx == 3:
                sections.append(r"\midrule")
        if suite == "dare":
            sections.append(r"\midrule")
    return "\n".join(sections) + "\n"


def render_remainder_contrasts_tex(summary: dict[str, Any]) -> str:
    names = (
        ("structured-base", "Structured $-$ 2B base"),
        ("structured-omni", "Structured $-$ Omni common"),
        ("describe-structured", "Describe $-$ Structured"),
    )
    lines: list[str] = []
    for suite, title in (("dare", "DARE (603)"), ("mm_upd", "MM-UPD (3{,}225)")):
        rc = summary["suites"][suite]["remainder_paired_contrasts"]
        for key, label in names:
            m = rc[key]
            lines.append(
                f"{title} & {label} & {100 * m['delta_bacc']:+.2f} & "
                f"[{100 * m['ci95'][0]:+.2f}, {100 * m['ci95'][1]:+.2f}] \\\\"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-predictions", action="store_true",
                        help="Recompute metrics from locally regenerated inputs and predictions")
    args = parser.parse_args()
    TABLES.mkdir(exist_ok=True)
    summary_path = TRACK / "frozen/results/FULL_EXTERNAL_SUMMARY_20261118.json"
    if args.from_predictions:
        summary, _ = build_summary(resamples=10000)
        summary_path = TABLES / "recomputed_external_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    full_tex_path = TABLES / "public_full.tex"
    rem_tex_path = TABLES / "public_remainder_contrasts.tex"
    full_tex_path.write_text(render_public_full_tex(summary), encoding="utf-8")
    rem_tex_path.write_text(render_remainder_contrasts_tex(summary), encoding="utf-8")

    manifest = {
        "summary_sha256": sha256_file(summary_path),
        "generator_sha256": sha256_file(Path(__file__)),
        "outputs_sha256": {
            "public_full.tex": sha256_file(full_tex_path),
            "public_remainder_contrasts.tex": sha256_file(rem_tex_path),
        },
        "counts": {suite: summary["suites"][suite]["counts"] for suite in ("dare", "mm_upd")},
    }
    manifest_path = TABLES / "FULL_EXTERNAL_MANIFEST_20261118.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
