"""Build tables from the published reviewer aggregate and rule baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
T = P.parent

src = {
    "reviewer_a": T / "english_revision/REVIEWER_A_AGGREGATE_20261102.json",
    "r1": T / "frozen/results/HOLDOUT_R1_BASELINE_20261102.json",
    "protocol": T / "frozen/HOLDOUT_BASELINES_V1_20261102.json",
    "disjoint": T / "frozen/results/DISJOINT_CALIBRATION_SENSITIVITY_20261102.json",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()



def main() -> int:
    a = json.loads(src["reviewer_a"].read_text(encoding="utf-8"))
    r = json.loads(src["r1"].read_text(encoding="utf-8"))
    if a["results"]["n"] != 240 or r["protocol_sha256"] != sha(src["protocol"]):
        raise ValueError("Input verification failed")

    lines = []
    for label, field, pos, neg in [
        ("Item valid", "validity", "valid", "invalid"),
        ("Wording clear", "wording", "clear", "unclear"),
        ("Needed differences visible", "visual", "yes", "no"),
    ]:
        c = a["results"]["counts"][field]
        values = [c.get(pos, 0), c.get(neg, 0), c.get("uncertain", 0)]
        assert sum(values) == 240
        lines.append(label + " & " + " & ".join(map(str, values)) + r" \\")
    (P / "tables/reviewer_a.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = []
    for split, label, n in [("test_id", "Assets / wording", 2505), ("test_ood", "Two templates", 1795)]:
        m = r["splits"][split]["metrics"]
        assert m["n"] == n
        lines.append(
            label
            + " & R1 (spec.) & "
            + " & ".join(f"{m[k]:.3f}" for k in ("balanced_accuracy", "invalid_recall", "invalid_precision", "valid_retention", "accuracy"))
            + r" \\"
        )
    (P / "tables/r1_holdout.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    d = json.loads(src["disjoint"].read_text(encoding="utf-8"))
    assert d["calibration_n"] == 51 and d["calibration_groups"] == 23
    lines = []
    for suite, label in [("muir", "Muir-V2"), ("dare", "DARE"), ("mm_upd", "MM-UPD")]:
        for method, tags in [
            ("Omni common", ["omniverifier7b_base"]),
            ("Structured", [f"v3_structured_s{i}" for i in range(3)]),
            ("Describe", [f"v3_describe_s{i}" for i in range(3)]),
        ]:
            vals = [d["external"][suite][tag] for tag in tags]
            cells = []
            for metric in ("recall", "precision", "valid_retention"):
                v = [x[metric] for x in vals]
                cells.append("---" if any(x is None for x in v) else f"{sum(v)/len(v):.3f}")
            lines.append(label + " & " + method + " & " + " & ".join(cells) + r" \\")
    (P / "tables/disjoint_calibration.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "inputs": {k: {"path": str(p.relative_to(T)), "sha256": sha(p)} for k, p in src.items()},
        "outputs": {
            n: sha(P / "tables" / n)
            for n in ("reviewer_a.tex", "r1_holdout.tex", "disjoint_calibration.tex")
        },
        "generator_sha256": sha(Path(__file__)),
    }
    (P / "tables/REVIEW_UPDATES_MANIFEST_20261102.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Review-update tables generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
