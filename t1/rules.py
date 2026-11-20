"""Deterministic baselines in the specification regime (PLAN.md Section 5).

R1  a strengthened specification checker that compares resolved scene JSON but
    never renders, so it cannot know that a turn or flip may be invisible:
    - A: the keyed option must differ from every other option; for "same as" questions
      the keyed option must equal the reference and no other option may.
    - B: attribute-level change set between the panels, filtered by the change type the
      question names, compared with the count it asks for and with the key's object names.

"""

from __future__ import annotations

import re

from . import oracle
from .itemgen import render

IDENTITY = re.compile(r"같은|똑같|그대로")
OP_WORDS = [("move", r"자리가 바뀐|위치가 바뀐|옮겨|이동"), ("remove", r"사라진|없어진"),
            ("add", r"없던|새로 생긴|추가된|나타난"), ("recolor", r"색이 바뀐|색깔이 바뀐")]
COUNT_WORDS = {"두 가지": 2, "세 가지": 3, "네 가지": 4}


def r1_choice(item: dict) -> dict:
    panels = _resolved(item)
    opts = oracle.options(panels)
    key = int(str(item["answer"]).strip())
    sig = {p["option_index"]: oracle.canon(p["scene"]) for p in opts}
    stage = oracle.reference_stage(item)
    ref = next(p for p in panels if p.get("stage") == stage and p.get("kind") != "option")
    ref_sig = oracle.canon(ref["scene"])
    if any(s == sig[key] for i, s in sig.items() if i != key):
        return {"valid": False, "defect": "multi_correct"}
    if IDENTITY.search(item["question"]):
        same = [i for i, s in sig.items() if s == ref_sig]
        if len(same) > 1:
            return {"valid": False, "defect": "multi_correct"}
        if not same:
            return {"valid": False, "defect": "no_correct"}
        if same != [key]:
            return {"valid": False, "defect": "key_mismatch"}
    return {"valid": True, "defect": "none"}


def r1_change(item: dict) -> dict:
    base, edit = oracle.change_panels(item)
    a = {o["ref"]: o for o in base["scene"]["objects"]}
    b = {o["ref"]: o for o in edit["scene"]["objects"]}
    changed = [(r, a[r]["name"], {"remove"}) for r in a.keys() - b.keys()]
    changed += [(r, b[r]["name"], {"add"}) for r in b.keys() - a.keys()]
    for r in a.keys() & b.keys():
        ops = {op for attr, op in oracle.ATTR_OP.items() if a[r].get(attr) != b[r].get(attr)}
        if ops:
            changed.append((r, a[r]["name"], ops))
    asked = {op for op, rx in OP_WORDS if re.search(rx, item["question"])} or None
    required = next((n for w, n in COUNT_WORDS.items() if w in item["question"]), 1)
    relevant = [c for c in changed if asked is None or c[2] & asked]
    names = {o["name"] for o in list(a.values()) + list(b.values())}
    key_ok = bool(relevant) and oracle.names_in(str(item["answer"]), names) == {c[1] for c in relevant}
    n = 0 if len(relevant) < required else (2 if len(relevant) > required else 1)
    return {"valid": n == 1 and key_ok, "defect": oracle.defect_of(n, key_ok)}

