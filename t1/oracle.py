"""Render-level ground truth for the executable families (PLAN.md Sections 2-3).

Two panels are the same to a viewer when their deterministic compose renders are
the same, so correctness is decided on pixels, not on the specification. This is
what makes a 180-degree turn of a square, or a recolor to the current color, a
non-change.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from .itemgen import DATA, render

# Render size for labels. 768 is native; T1_ORACLE_SIZE=448 audits the size models are shown.
SIZE = int(os.environ.get("T1_ORACLE_SIZE", 768))
DIFF_LEVEL = 40          # a pixel counts as changed when any channel moves more than this
CALIBRATION = DATA / ("calibration.json" if SIZE == 768 else f"calibration_{SIZE}.json")


def _same_frac() -> float:
    if CALIBRATION.exists():
        return json.loads(CALIBRATION.read_text())["same_frac"]
    return 0.002         # pre-calibration placeholder; calibrate.py overwrites it


SAME_FRAC = _same_frac()


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_cache: dict[tuple[str, int], np.ndarray] = {}


def render_scene(scene: dict, size: int = SIZE) -> np.ndarray:
    key = (canon(scene), size)
    if key not in _cache:
        img = render.render_compose(scene, size).convert("RGB")
        _cache[key] = np.asarray(img, dtype=np.int16)
    return _cache[key]


SHIFT = 2                # sub-pixel edge shifts up to this many pixels are not changes


def _one_way(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pixels of `a` whose colour lies outside the local colour range of `b`.

    The range is the per-channel min..max of `b` over a (2*SHIFT+1)^2 window. An
    anti-aliased edge value sits between the ink and paper values around it, so it
    is explained; a recoloured, moved or resized region is not.
    """
    h, w, _ = a.shape
    pad = np.pad(b, ((SHIFT, SHIFT), (SHIFT, SHIFT), (0, 0)), mode="edge")
    lo = np.full(a.shape, 10_000, dtype=np.int16)
    hi = np.full(a.shape, -10_000, dtype=np.int16)
    for dy in range(2 * SHIFT + 1):
        for dx in range(2 * SHIFT + 1):
            win = pad[dy:dy + h, dx:dx + w]
            np.minimum(lo, win, out=lo)
            np.maximum(hi, win, out=hi)
    return ((a < lo - DIFF_LEVEL) | (a > hi + DIFF_LEVEL)).any(axis=2)


def diff_frac(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels that changed beyond a small spatial tolerance.

    Rotating a shape onto itself moves its anti-aliased edges by a pixel or so, and
    those must not count. A moved or recoloured object, even a thin line drawing,
    has pixels with no match nearby and does count. Exact equality is checked first.
    """
    raw = np.abs(a - b).max(axis=2) > DIFF_LEVEL
    if not raw.any():
        return 0.0
    # Only pixels that differ outright can be changes, so work on their bounding box.
    ys, xs = np.nonzero(raw)
    m = 2 * SHIFT
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, a.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, a.shape[1])
    ca, cb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    changed = _one_way(ca, cb) | _one_way(cb, ca)
    return float(changed.sum() / raw.size)


def same_look(scene_a: dict, scene_b: dict) -> bool:
    return diff_frac(render_scene(scene_a), render_scene(scene_b)) <= SAME_FRAC


def defect_of(n_correct: int, key_ok: bool) -> str:
    if n_correct == 0:
        return "no_correct"
    if n_correct >= 2:
        return "multi_correct"
    return "none" if key_ok else "key_mismatch"


# ------------------------------------------------------------------ family A

def options(resolved: list[dict]) -> list[dict]:
    return sorted((p for p in resolved if p.get("kind") == "option"),
                  key=lambda p: p["option_index"])


def reference_stage(item: dict) -> str:
    stages = {p["edit_of"] for p in item["panels"] if p.get("kind") == "option"}
    if len(stages) != 1:
        raise ValueError(f"options derive from {sorted(stages)}; family A needs one reference")
    return stages.pop()


def intended_scene(item: dict, intended_changes: list[dict]) -> dict:
    """The answer the question asks for, built from the item's current reference."""
    stage = reference_stage(item)
    ref = next(p for p in render.resolve_panels(item) if p.get("stage") == stage
               and p.get("kind") != "option")
    return render.apply_changes(ref["scene"], intended_changes)


def alternatives(intended: list) -> list[list[dict]]:
    """`intended` is one change list, or a list of acceptable change lists (any one counts)."""
    if intended and isinstance(intended[0], list):
        return intended
    return [intended or []]


def label_choice(item: dict, intended_changes: list) -> dict:
    resolved = render.resolve_panels(item)
    opts = options(resolved)
    alts = alternatives(intended_changes)
    targets = [render_scene(intended_scene(item, alt)) for alt in alts]
    stage = reference_stage(item)
    ref = next(p for p in resolved if p.get("stage") == stage and p.get("kind") != "option")
    ref_img = render_scene(ref["scene"])
    looks = {p["option_index"]: render_scene(p["scene"]) for p in opts}
    correct = [i for i, im in looks.items() if any(diff_frac(im, t) <= SAME_FRAC for t in targets)]
    wrong = [i for i in looks if i not in correct]
    dup_distractors = sorted({(a, b) for a in wrong for b in wrong if a < b
                              and diff_frac(looks[a], looks[b]) <= SAME_FRAC})
    equals_reference = [i for i, im in looks.items() if diff_frac(im, ref_img) <= SAME_FRAC]
    key = int(str(item["answer"]).strip())
    key_ok = key in correct
    return {
        "family": "A",
        "n_correct": len(correct),
        "correct_set": correct,
        "key": key,
        "key_ok": key_ok,
        "valid": len(correct) == 1 and key_ok,
        "defect": defect_of(len(correct), key_ok),
        "flags": {"dup_distractor": bool(dup_distractors),
                  "options_equal_to_reference": equals_reference,
                  # The question asks for a transformation that leaves the picture unchanged
                  # (e.g. mirroring a symmetric triangle): answerable by matching alone.
                  "transformation_invisible": all(alts) and any(
                      diff_frac(t, ref_img) <= SAME_FRAC for t in targets)},
    }


# ------------------------------------------------------------------ family B

ATTR_OP = {"cell": "move", "pos": "move", "color": "recolor", "rotation": "orient",
           "mirrored": "orient", "facing": "orient", "scale": "rescale",
           "name": "replace", "appearance": "replace"}


def _placed(scene: dict) -> dict[str, dict]:
    """Objects by ref, with an explicit grid cell so each can be drawn alone in place."""
    out = {}
    n = len(scene["objects"])
    for k, obj in enumerate(scene["objects"]):
        solo = {key: v for key, v in scene.items() if key != "objects"}
        o = copy.deepcopy(obj)
        if scene.get("layout") == "row":
            solo.update(layout="grid", rows=1, cols=n)
            o["cell"] = [0, k]
        solo["objects"] = [o]
        out[obj["ref"]] = {"obj": obj, "solo": solo}
    return out


def visible_changes(before: dict, after: dict) -> list[dict]:
    a, b = _placed(before), _placed(after)
    changes = []
    for ref in a.keys() - b.keys():
        changes.append({"ref": ref, "name": a[ref]["obj"]["name"], "ops": {"remove"}})
    for ref in b.keys() - a.keys():
        changes.append({"ref": ref, "name": b[ref]["obj"]["name"], "ops": {"add"}})
    for ref in a.keys() & b.keys():
        if same_look(a[ref]["solo"], b[ref]["solo"]):
            continue
        oa, ob = a[ref]["obj"], b[ref]["obj"]
        ops = {op for attr, op in ATTR_OP.items() if oa.get(attr) != ob.get(attr)}
        changes.append({"ref": ref, "name": oa["name"], "ops": ops or {"move"}})
    return sorted(changes, key=lambda c: c["ref"])


def change_panels(item: dict) -> tuple[dict, dict]:
    resolved = render.resolve_panels(item)
    edit = next(p for p in resolved if p.get("kind") == "edit")
    base = next(p for p in resolved if p.get("stage") == edit["edit_of"] and p.get("kind") != "edit")
    return base, edit


def names_in(answer: str, names: set[str]) -> set[str]:
    # Longest names first so "빨간 공" is not also counted as "공".
    found, text = set(), answer
    for name in sorted(names, key=len, reverse=True):
        if name and name in text:
            found.add(name)
            text = text.replace(name, " ")
    return found


def label_change(item: dict, asked_ops: set[str] | None, required: int) -> dict:
    base, edit = change_panels(item)
    seen = visible_changes(base["scene"], edit["scene"])
    relevant = [c for c in seen if asked_ops is None or c["ops"] & asked_ops]
    all_names = {o["name"] for o in base["scene"]["objects"]} | \
                {o["name"] for o in edit["scene"]["objects"]}
    mentioned = names_in(str(item["answer"]), all_names)
    relevant_names = {c["name"] for c in relevant}
    key_ok = bool(relevant) and mentioned == relevant_names
    if len(relevant) < required:
        n_correct = 0
    elif len(relevant) > required:
        n_correct = 2
    else:
        n_correct = 1
    name_counts = {}
    for o in base["scene"]["objects"] + edit["scene"]["objects"]:
        name_counts.setdefault(o["name"], set()).add(o["ref"])
    return {
        "family": "B",
        "n_correct": n_correct,
        "correct_set": sorted(relevant_names),
        "visible_changes": [{**c, "ops": sorted(c["ops"])} for c in seen],
        "key": str(item["answer"]),
        "key_ok": key_ok,
        "valid": n_correct == 1 and key_ok,
        "defect": defect_of(n_correct, key_ok),
        "flags": {"ambiguous_referent": any(len(name_counts[c["name"]]) > 1 for c in relevant)},
    }


def save_panel_images(item: dict, out_dir: Path, size: int = 448) -> list[dict]:
    """Render every panel a viewer sees and return their roles in display order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shown = []
    for k, panel in enumerate(render.resolve_panels(item)):
        if panel.get("kind") == "option":
            label, role = f"보기 {panel['option_index']}", "option"
        else:
            label, role = panel["stage"], "panel"
        path = out_dir / f"p{k:02d}.png"
        render.render_compose(panel["scene"], size).convert("RGB").save(path, optimize=True)
        shown.append({"role": role, "label": label, "option_index": panel.get("option_index"),
                      "path": str(path)})
    return shown
