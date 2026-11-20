"""Load the frozen ItemGen renderer, scene specification, and assets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1]
PACKAGE = TRACK / "itemgen_reference"
DATA = TRACK / "data"

os.environ["ITEMGEN_ASSETS_DIR"] = str(PACKAGE / "assets")
sys.path.insert(0, str(PACKAGE))

from itemgen_bench import render, shapes, spec  # noqa: E402

_reference_tint = render._tint


def _fast_tint(art, obj: dict):
    """Vectorised copy of render._tint; `verify_tint` checks it is pixel-identical.

    The original loops over pixels in Python and dominates render time.
    """
    import numpy as np
    from PIL import Image
    if not (obj.get("color") or "").strip():
        return art
    rgb = shapes.pick_color(obj)
    if rgb == shapes.DEFAULT_FILL:
        return art
    a = np.asarray(art.convert("RGBA")).copy()
    r, g, b, alpha = (a[..., k].astype(np.int64) for k in range(4))
    lum = (r * 299 + g * 587 + b * 114) / 1000
    paint = (alpha >= 16) & (lum >= 110)
    f = np.minimum(1.0, lum / 255 + 0.15)
    for k in range(3):
        a[..., k] = np.where(paint, (rgb[k] * f).astype(np.int64), a[..., k])
    return Image.fromarray(a, "RGBA")


render._tint = _fast_tint

SLOTS = {s["id"]: s for s in json.loads((PACKAGE / "spec/slots.json").read_text(encoding="utf-8"))["slots"]}

# Executable families (PLAN.md Section 2). EXE_09 is excluded: its key is a free-text
# move description whose destination wording the oracle cannot check.
FAMILY_A = ["MEM_02", "MEM_04", "MEM_05", "MEM_06", "MEM_10", "VIS_01", "VIS_02",
            "VIS_03", "VIS_06", "VIS_09", "EXE_03", "EXE_05"]
FAMILY_B = ["MEM_03", "MEM_07", "MEM_08", "MEM_09", "ATT_05", "ATT_06"]

# Change types each family-B question asks about. None means any visible change.
ASKED_OPS = {"MEM_03": {"move"}, "MEM_07": {"remove"}, "MEM_08": {"add"},
             "MEM_09": {"recolor"}, "ATT_05": None, "ATT_06": None}


def family(slot_id: str) -> str | None:
    if slot_id in FAMILY_A:
        return "A"
    if slot_id in FAMILY_B:
        return "B"
    return None


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
