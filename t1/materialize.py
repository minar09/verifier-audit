"""Label and render procedural variants, and write their instance rows."""

from . import oracle
from .itemgen import DATA, spec, write_json


def materialize(base: dict, variants, out_root, domain: str, answer_kind: str, slot=None) -> list[dict]:
    """Label each variant with the oracle, save its item and images, return instance rows."""
    rows, seen, used_ids = [], set(), set()
    for v in variants:
        key = oracle.canon(v.item)
        if key in seen:
            continue
        seen.add(key)
        iid = f'{base["base_id"]}__{v.op}'
        if iid in used_ids:
            import hashlib
            iid += '__input_' + hashlib.sha256(key.encode()).hexdigest()[:12]
        if iid in used_ids: raise ValueError('Variant identity collision: '+iid)
        used_ids.add(iid)
        try:
            if base["family"] == "A":
                label = oracle.label_choice(v.item, v.intended)
            else:
                label = oracle.label_change(v.item, v.asked, v.required)
        except Exception as exc:          # an operator produced an unrenderable item
            rows.append({"id": iid, "error": f"{type(exc).__name__}: {exc}"})
            continue
        write_json(out_root / "items" / f"{iid}.json", v.item)
        shown = oracle.save_panel_images(v.item, out_root / "images" / iid)
        for s in shown:
            s["path"] = str((out_root / "images" / iid / s["path"].rsplit("/", 1)[1]).relative_to(DATA))
        rows.append({
            "id": iid, "base_id": base["base_id"], "family": base["family"], "slot_id": base["slot_id"],
            "domain": domain, "op": v.op, "meta": v.meta,
            "question": v.item["question"], "key": str(v.item["answer"]),
            "answer_kind": answer_kind, "images": shown,
            "intended": v.intended, "asked": sorted(v.asked) if v.asked else None,
            "required": v.required, "label": label,
            "spec_errors": spec.validate_item(v.item, slot) if slot else None,
        })
    return rows

