"""Specification-level corruptions of base items (PLAN.md Section 4).

Each operator edits the item JSON only; the oracle then renders and labels the
result. An operator's name is a hypothesis about its effect, never a label.
Every operator returns zero or more `Variant`s; ones that cannot apply to a base
(no symmetric shape, no free cell, ...) return nothing.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field

from .calibrate import ASYMMETRIC_SHAPES, SYMMETRY, nearest_other_color
from .itemgen import render, shapes, spec

# Korean colour adjectives used in rule questions, keyed by the scene colour noun.
COLOR_ADJ = {"빨강": "빨간색", "파랑": "파란색", "초록": "초록색", "노랑": "노란색", "보라": "보라색",
             "주황": "주황색", "분홍": "분홍색", "회색": "회색", "갈색": "갈색", "검정": "검은색"}


@dataclass
class Variant:
    op: str
    item: dict
    intended: list | None = None      # family A: intended changes relative to the reference
    asked: set | None = None          # family B: change types the question asks about
    required: int = 1                 # family B: number of changes the question asks for
    meta: dict = field(default_factory=dict)


def _opts(item):
    return sorted((p for p in item["panels"] if p.get("kind") == "option"),
                  key=lambda p: p["option_index"])


def _key(item) -> int:
    return int(str(item["answer"]).strip())


def _ref_panel(item):
    stage = next(p["edit_of"] for p in item["panels"] if p.get("kind") == "option")
    return next(p for p in item["panels"] if p.get("stage") == stage and p.get("kind") != "option")


def _swap_key(item, new: int) -> dict:
    out = copy.deepcopy(item)
    old = _key(out)
    out["answer"] = str(new)
    out["answer_rationale"] = spec._renumber_rationale(out.get("answer_rationale", ""), {old: new, new: old})
    return out


def _targets(item) -> set[str]:
    return {c.get("target") for p in item["panels"] for c in (p.get("changes") or [])}


def _free_cell(scene, rng, exclude=()):
    if scene.get("layout") != "grid":
        return None
    taken = {tuple(o["cell"]) for o in scene["objects"] if "cell" in o} | set(exclude)
    free = [[r, c] for r in range(scene["rows"]) for c in range(scene["cols"]) if (r, c) not in taken]
    return rng.choice(free) if free else None


# ------------------------------------------------------------------ family A

def family_a(base_item: dict, seed: int, intended: list | None = None, qcp: bool = False) -> list[Variant]:
    """`intended` overrides the keyed option's changes (procedural items may accept several)."""
    rng = random.Random(seed)
    item = copy.deepcopy(base_item)
    key = _key(item)
    keyed = next(p for p in _opts(item) if p["option_index"] == key)
    keyed_changes = copy.deepcopy(keyed.get("changes") or [])
    if intended is None:
        intended = keyed_changes
    distractors = [p["option_index"] for p in _opts(item) if p["option_index"] != key]
    out = [Variant("orig", item, intended)]

    # Option permutations with the key remapped: task-preserving controls.
    for k in range(2):
        idx = [p["option_index"] for p in _opts(item)]
        while True:
            perm = idx[:]
            rng.shuffle(perm)
            if perm[idx.index(key)] != key:
                break
        mapping = dict(zip(idx, perm))
        v = copy.deepcopy(item)
        for p in v["panels"]:
            if p.get("kind") == "option":
                p["option_index"] = mapping[p["option_index"]]
        pos = [i for i, p in enumerate(v["panels"]) if p.get("kind") == "option"]
        for i, p in zip(pos, sorted((v["panels"][i] for i in pos), key=lambda p: p["option_index"])):
            v["panels"][i] = p
        v["answer"] = str(mapping[key])
        out.append(Variant(f"perm_{k}", v, intended))

    # Recolor an object no option touches, in the reference only; options inherit it.
    ref = _ref_panel(item)
    free_objs = [o for o in ref["scene"]["objects"] if o["ref"] not in _targets(item)]
    if free_objs:
        v = copy.deepcopy(item)
        pick = rng.choice(free_objs)["ref"]
        o = next(x for x in _ref_panel(v)["scene"]["objects"] if x["ref"] == pick)
        o["color"] = nearest_other_color(o.get("color"))
        out.append(Variant("restyle", v, intended, meta={"object": o["ref"]}))

    for k, j in enumerate(rng.sample(distractors, min(2, len(distractors)))):
        v = copy.deepcopy(item)
        p = next(x for x in _opts(v) if x["option_index"] == j)
        p["changes"] = copy.deepcopy(keyed.get("changes") or [])
        out.append(Variant(f"dup_key_{k}", v, intended, meta={"copied_to": j}))
        out.append(Variant(f"key_swap_{k}", _swap_key(item, j), intended, meta={"key_to": j}))

    # Break the keyed option: recolor one of its targets away from the intended colour.
    v = copy.deepcopy(item)
    want = render.apply_changes(ref["scene"], keyed_changes)
    target = keyed_changes[0]["target"] if keyed_changes else rng.choice(want["objects"])["ref"]
    want_obj = next(o for o in want["objects"] if o["ref"] == target)
    p = next(x for x in _opts(v) if x["option_index"] == key)
    p["changes"] = copy.deepcopy(keyed_changes) + [
        {"op": "recolor", "target": target, "to_color": nearest_other_color(want_obj.get("color"))}]
    out.append(Variant("break_key", v, intended, meta={"target": target}))

    if len(distractors) >= 2:
        j, k2 = rng.sample(distractors, 2)
        v = copy.deepcopy(item)
        src = next(x for x in _opts(v) if x["option_index"] == j)
        dst = next(x for x in _opts(v) if x["option_index"] == k2)
        if (src.get("changes") or []) != (dst.get("changes") or []):
            dst["changes"] = copy.deepcopy(src.get("changes") or [])
            out.append(Variant("dup_distractor", v, intended, meta={"from": j, "to": k2}))

    # A distractor whose change a viewer cannot see. Literal: same value. Symmetry:
    # a turn or flip that maps the shape onto itself.
    j = rng.choice(distractors)
    colored = [o for o in ref["scene"]["objects"] if o.get("color")]
    if colored:
        o = rng.choice(colored)
        v = copy.deepcopy(item)
        next(x for x in _opts(v) if x["option_index"] == j)["changes"] = [
            {"op": "recolor", "target": o["ref"], "to_color": o["color"]}]
        out.append(Variant("invisible_literal", v, intended, meta={"option": j, "object": o["ref"]}))
    sym = [o for o in ref["scene"]["objects"] if shapes.shape_key(o.get("name", "")) in SYMMETRY]
    if sym:
        o = rng.choice(sym)
        angles, mirror_ok = SYMMETRY[shapes.shape_key(o["name"])]
        if angles:
            change = {"op": "rotate", "target": o["ref"],
                      "to_rotation": (int(o.get("rotation") or 0) + rng.choice(angles)) % 360}
        else:
            change = {"op": "mirror", "target": o["ref"]}
        v = copy.deepcopy(item)
        next(x for x in _opts(v) if x["option_index"] == j)["changes"] = [change]
        out.append(Variant("invisible_symmetry", v, intended, meta={"option": j, "change": change}))

    # Make the transformed stimulus symmetric so its turns and flips collapse.
    turned = {c["target"] for p in _opts(item) for c in (p.get("changes") or [])
              if c.get("op") in ("rotate", "mirror")}
    for o in ref["scene"]["objects"]:
        if o["ref"] in turned and (shapes.shape_key(o.get("name", "")) in ASYMMETRIC_SHAPES
                                   or shapes.shape_key(o.get("name", "")) is None):
            v = copy.deepcopy(item)
            t = next(x for x in _ref_panel(v)["scene"]["objects"] if x["ref"] == o["ref"])
            t["name"], t["appearance"] = "원", "단색 원 낱개"
            out.append(Variant("sym_reference", v, intended, meta={"object": o["ref"]}))
            break

    # Same images, different question: change the angle the question asks for. Question
    # angles are relative to the reference; option rotations are absolute, so convert.
    # With `qcp` (question-counterfactual pairs) every other angle is used, the colour rule
    # of two-rule items is swapped too, and both the stale-key and re-keyed versions are kept.
    m = re.search(r"(\d+)\s*도", item["question"])
    flat = bool(intended) and isinstance(intended[0], dict)
    rot_changes = [c for c in intended if c.get("op") == "rotate"] if flat else []
    if m and rot_changes and (qcp or all(c.get("op") == "rotate" for c in intended)):
        asked = int(m.group(1))
        tgt = rot_changes[0]["target"]
        r0 = int(next(o for o in ref["scene"]["objects"] if o["ref"] == tgt).get("rotation") or 0)
        shown = [(p["option_index"], (c["to_rotation"] - r0) % 360) for p in _opts(item)
                 if p["option_index"] != key for c in (p.get("changes") or [])
                 if c.get("op") == "rotate" and c.get("to_rotation") is not None]
        shown = [(j, a) for j, a in shown if a != asked and (not qcp or a != 0)]
        picks = shown if qcp else ([rng.choice(shown)] if shown else [])
        for j, angle in picks:
            q = item["question"][:m.start(1)] + str(angle) + item["question"][m.end(1):]
            new_intended = [dict(c, to_rotation=(r0 + angle) % 360) if c.get("op") == "rotate" else c
                            for c in intended]
            v = copy.deepcopy(item)
            v["question"] = q
            out.append(Variant(f"qswap_angle{'_' + str(angle) if qcp else ''}", v, new_intended,
                               meta={"angle": angle}))
            out.append(Variant(f"qswap_angle_rekeyed{'_' + str(angle) if qcp else ''}", _swap_key(v, j),
                               new_intended, meta={"angle": angle}))
    recolors = [c for c in intended if c.get("op") == "recolor"] if flat else []
    if qcp and recolors:
        adj = {v_: k for k, v_ in COLOR_ADJ.items()}
        said = next((a for a in adj if a in item["question"]), None)
        if said:
            for other in rng.sample([a for a in adj if a != said], 2):
                v = copy.deepcopy(item)
                v["question"] = item["question"].replace(said, other)
                new_intended = [dict(c, to_color=adj[other]) if c.get("op") == "recolor" else c
                                for c in intended]
                out.append(Variant(f"qswap_color_{adj[other]}", v, new_intended, meta={"color": other}))
    return out


# ------------------------------------------------------------------ family B

QUESTION_OPS = {"MEM_03": "move", "MEM_07": "remove", "MEM_08": "add", "MEM_09": "recolor"}


def _edit(item):
    return next(p for p in item["panels"] if p.get("kind") == "edit")


def _base_scene(item):
    e = _edit(item)
    return next(p for p in item["panels"] if p.get("stage") == e["edit_of"] and p.get("kind") != "edit")["scene"]


def _visible_change(op: str, scene: dict, obj: dict, rng, taken=()):
    """A visible change of the given type to `obj`, or None if impossible here."""
    if op == "move":
        cell = _free_cell(scene, rng, taken)
        return {"op": "move", "target": obj["ref"], "to_cell": cell} if cell else None
    if op == "recolor":
        return {"op": "recolor", "target": obj["ref"], "to_color": nearest_other_color(obj.get("color"))}
    if op == "remove":
        return {"op": "remove", "target": obj["ref"]}
    return None


def family_b(base_item: dict, slot_id: str, asked: set | None, required: int,
             questions: dict[str, str], seed: int, qcp: bool = False) -> list[Variant]:
    rng = random.Random(seed)
    item = copy.deepcopy(base_item)
    out = [Variant("orig", item, asked=asked, required=required)]
    scene = _base_scene(item)
    changes = _edit(item)["changes"]
    targets = {c["target"] for c in changes}
    others = [o for o in scene["objects"] if o["ref"] not in targets]

    if others:
        v = copy.deepcopy(item)
        pick = rng.choice(others)["ref"]
        o = next(x for x in _base_scene(v)["objects"] if x["ref"] == pick)
        o["color"] = nearest_other_color(o.get("color"))
        out.append(Variant("restyle", v, asked=asked, required=required, meta={"object": o["ref"]}))

        # Wrong key: name an unchanged object instead of one changed object.
        named = [c["target"] for c in changes if c["op"] != "add"] or [changes[0]["target"]]
        changed_name = next((o["name"] for o in scene["objects"] if o["ref"] == named[0]),
                            changes[0].get("object", {}).get("name"))
        decoy = rng.choice(others)["name"]
        if changed_name and changed_name in item["answer"] and decoy not in item["answer"]:
            v = copy.deepcopy(item)
            v["answer"] = item["answer"].replace(changed_name, decoy)
            out.append(Variant("key_swap", v, asked=asked, required=required, meta={"to": decoy}))

    # Literal no-op of the same type (same cell, same colour).
    first = changes[0]
    obj = next((o for o in scene["objects"] if o["ref"] == first["target"]), None)
    noop = None
    if obj is not None and first["op"] == "move" and "cell" in obj:
        noop = dict(first, to_cell=list(obj["cell"]))
    elif obj is not None and first["op"] == "recolor" and obj.get("color"):
        noop = dict(first, to_color=obj["color"])
    if noop:
        v = copy.deepcopy(item)
        _edit(v)["changes"][0] = noop
        out.append(Variant("noop_literal", v, asked=asked, required=required))

    # Replace the change by a visible change of another type on the same object.
    if obj is not None and slot_id != "ATT_06":
        for op in ("move", "recolor", "remove"):
            if op == first["op"]:
                continue
            ch = _visible_change(op, scene, obj, rng)
            if ch:
                v = copy.deepcopy(item)
                _edit(v)["changes"][0] = ch
                out.append(Variant("wrong_op", v, asked=asked, required=required, meta={"op": op}))
                break

    # One more change of an asked type on another object.
    if others:
        o = rng.choice(others)
        ops = sorted(asked) if asked else ["recolor", "move"]
        taken = [tuple(c["to_cell"]) for c in changes if c.get("to_cell")] + \
                [tuple(c["object"]["cell"]) for c in changes if c.get("op") == "add"]
        for op in ops:
            if op == "add":
                cell = _free_cell(scene, rng, taken)
                ch = cell and {"op": "add", "target": "extra", "object": {
                    "ref": "extra", "name": "책", "appearance": "책 한 권 낱개", "cell": cell}}
            else:
                ch = _visible_change(op, scene, o, rng, taken)
            if ch:
                v = copy.deepcopy(item)
                _edit(v)["changes"].append(ch)
                out.append(Variant("extra_change", v, asked=asked, required=required, meta={"op": op}))
                break

    if required > 1:
        v = copy.deepcopy(item)
        _edit(v)["changes"].pop(rng.randrange(len(changes)))
        out.append(Variant("drop_change", v, asked=asked, required=required))

    # Same images, a question about a different change type (natural wording from another slot).
    if required == 1:
        swaps = []
        for qslot, qop in QUESTION_OPS.items():
            if qslot == slot_id or qslot not in questions:
                continue
            same_type = first["op"] == qop
            if asked is None and not same_type:
                continue                      # "any change" items only get matching-type questions
            if asked is not None and same_type:
                continue
            swaps.append((qslot, qop))
        for qslot, qop in (swaps if qcp else rng.sample(swaps, min(2, len(swaps)))):
            v = copy.deepcopy(item)
            v["question"] = questions[qslot]
            out.append(Variant(f"qswap_{qop}", v, asked={qop}, required=1, meta={"question_from": qslot}))
    return out
