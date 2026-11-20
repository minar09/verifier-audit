"""Fix the render-equivalence threshold before any item is labelled (PLAN.md Section 6).

For every object in every base scene, apply visually null edits ("same") and the
smallest real edits ("different"), render the whole scene, and record the changed-
pixel fraction. Asset rotations and mirrors are kept apart as "uncertain", because
some drawings (a ball, an apple) are close to symmetric; they are reported, not used.

"""

from __future__ import annotations

import copy
import math

from . import oracle
from .itemgen import render, shapes

# Rotation angles that map each code shape onto itself, and whether a left-right flip does.
SYMMETRY = {
    "circle": ([45, 90, 137, 180], True), "dot": ([45, 90, 180], True),
    "square": ([90, 180, 270], True), "diamond": ([90, 180], True),
    "cross": ([180], True), "triangle": ([120, 240], True),
    "pentagon": ([72, 144], True), "hexagon": ([60, 120, 180], True),
    "star": ([72, 144], True), "rect": ([180], True), "ellipse": ([180], True),
    "line": ([180], True), "heart": ([], True), "semicircle": ([], True),
}
ASYMMETRIC_SHAPES = {"asym", "elbow", "arrow", "quarter_circle"}
# A cross turned 90 degrees is NOT the same picture: its bars are drawn in a fixed order,
# so the inner outline swaps. Measured as a visible difference in the first calibration.
PALETTE = {k: v for k, v in shapes.COLORS.items() if not k.isascii()}


def nearest_other_color(name: str | None) -> str:
    rgb = PALETTE.get(name or "", shapes.DEFAULT_FILL)
    best = min((c for c, v in PALETTE.items() if v != rgb),
               key=lambda c: math.dist(PALETTE[c], rgb))
    return best


def _with(scene: dict, ref: str, **attrs) -> dict:
    out = copy.deepcopy(scene)
    for o in out["objects"]:
        if o["ref"] == ref:
            o.update(attrs)
    return out


def _free_neighbor(scene: dict, obj: dict):
    if "cell" not in obj or scene.get("layout") != "grid":
        return None
    taken = {tuple(o["cell"]) for o in scene["objects"] if "cell" in o}
    r, c = obj["cell"]
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        cell = (r + dr, c + dc)
        if 0 <= cell[0] < scene["rows"] and 0 <= cell[1] < scene["cols"] and cell not in taken:
            return list(cell)
    return None


def pairs_for(scene: dict):
    for obj in scene["objects"]:
        ref, key = obj["ref"], shapes.shape_key(obj.get("name", ""))
        rot = int(obj.get("rotation") or 0)
        if obj.get("color"):
            yield "same", "recolor_same", _with(scene, ref, color=obj["color"])
        yield "different", "recolor_near", _with(scene, ref, color=nearest_other_color(obj.get("color")))
        yield "different", "rescale_1.25", _with(scene, ref, scale=float(obj.get("scale", 1.0)) * 1.25)
        cell = _free_neighbor(scene, obj)
        if cell:
            yield "different", "move_cell", _with(scene, ref, cell=cell)
        if key in SYMMETRY:
            angles, mirror_sym = SYMMETRY[key]
            for a in angles:
                yield "same", f"rotate_sym_{key}", _with(scene, ref, rotation=(rot + a) % 360)
            if mirror_sym:
                yield "same", f"mirror_sym_{key}", _with(scene, ref, mirrored=not obj.get("mirrored", False))
        elif key in ASYMMETRIC_SHAPES:
            yield "different", f"rotate90_{key}", _with(scene, ref, rotation=(rot + 90) % 360)
            yield "different", f"mirror_{key}", _with(scene, ref, mirrored=not obj.get("mirrored", False))
        elif key is None:
            yield "uncertain", "asset_rotate90", _with(scene, ref, rotation=(rot + 90) % 360)
            yield "uncertain", "asset_mirror", _with(scene, ref, mirrored=not obj.get("mirrored", False))


def probe_scenes():
    """Shape-bank scenes so every symmetric shape is tested, alone and small in a 4x4 grid."""
    for key in list(SYMMETRY) + sorted(ASYMMETRIC_SHAPES):
        name = {"circle": "원", "dot": "점", "square": "정사각형", "diamond": "마름모",
                "cross": "십자", "triangle": "삼각형", "pentagon": "오각형", "hexagon": "육각형",
                "star": "별", "rect": "직사각형", "ellipse": "타원", "line": "선분",
                "heart": "하트", "semicircle": "반원", "asym": "비대칭 도형", "elbow": "L자형 블록",
                "arrow": "화살표", "quarter_circle": "사분원"}[key]
        yield {"layout": "free", "objects": [{"ref": "s", "name": name, "color": "파랑", "pos": [0.5, 0.5]}]}
        yield {"layout": "grid", "rows": 4, "cols": 4, "objects": [
            {"ref": "s", "name": name, "color": "빨강", "cell": [1, 1]},
            {"ref": "t", "name": "원", "color": "초록", "cell": [2, 3]}]}

