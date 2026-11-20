"""Convert the pinned MuirBench release into native and T1 validity rows."""

from __future__ import annotations

import argparse
import glob
import io
import json
import random
import re
from pathlib import Path

from PIL import Image

from .itemgen import DATA, TRACK
from .records import load_rows, atomic_json, sha256

SRC = DATA / "external/20261021/muirbench/data"
OUT = DATA / "external_eval/muir"
SEED = 20261021
NONE_TEXT = "None of the choices provided"
METHODS = ['qwen3vl2b_base', 'omniverifier7b_base', 'omniverifier7b_native', 'qwen3vl8b_base'] + [
    f'v3_{sup}_s{seed}' for seed in range(3) for sup in ('describe', 'structured')]


def _fit(img: Image.Image, n_images: int) -> Image.Image:
    side = 448 if n_images <= 6 else 336
    img = img.convert("RGB")
    img.thumbnail((side, side))
    return img


def _load_muir():
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SRC / "*.parquet")))])
    rows = df.to_dict("records")
    for r in rows:
        n_q = r["question"].count("<image>")
        n_o = sum(str(o).count("<image>") for o in r["options"])
        assert n_q + n_o == len(r["image_list"]), r["idx"]
    return {str(r["idx"]): r for r in rows}


def _save_images(r, out_dir):
    """Save images in placeholder order: question images first, then option images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    n = len(r["image_list"])
    for k, im in enumerate(r["image_list"]):
        p = out_dir / f"{k:02d}.png"
        if not p.exists():
            _fit(Image.open(io.BytesIO(im["bytes"])), n).save(p)
        paths.append(str(p.relative_to(DATA)))
    return paths


def _instance(r, paths, options, key_pos, label, iid, kind, letters=False):
    """T1 row. `options` is a list of (original_index, text); images keep original order."""
    n_q = r["question"].count("<image>")
    images, img_iter = [], iter(paths)
    for k in range(n_q):
        images.append({"role": "panel", "path": next(img_iter), "display_label": f"Panel {k + 1}"})
    # Option images in original order; skip images of removed options.
    shown = {orig for orig, _ in options}
    text_lines = []
    for orig, text in enumerate(r["options"]):
        n_img = str(text).count("<image>")
        imgs = [next(img_iter) for _ in range(n_img)]
        if orig not in shown:
            continue
        pos = [o for o, _ in options].index(orig) + 1
        name = "ABCDEFGH"[pos - 1] if letters else str(pos)
        for p in imgs:
            images.append({"role": "option", "option_index": pos, "path": p, "display_label": f"Option {name}"})
        if n_img == 0:
            text_lines.append(f"Option {name}: {text}")
        elif str(text).strip() != "<image>":
            text_lines.append(f"Option {name}: {str(text).replace('<image>', '(image)')}")
    q = r["question"].replace("<image>", "(image)")
    if text_lines:
        q += "\n" + "\n".join(text_lines)
    return {"id": iid, "base_id": f"muir_pair_{min(int(r['idx']), int(r['counterpart_idx']))}",
            "family": "A", "slot_id": f"MUIR_{r['task']}", "op": kind, "task": r["task"],
            "question": q, "key": "ABCDEFGH"[key_pos - 1] if letters else str(key_pos),
            "images": images, "keep_aspect": True, "label": label, "intended": None}


def build_muir():
    rows = _load_muir()
    native, muirv, done = [], [], set()
    for idx, r in rows.items():
        img_paths = _save_images(r, OUT / "images" / idx)
        opts = [(i, str(t)) for i, t in enumerate(r["options"])]
        gold = ord(r["answer"]) - 65
        native.append(_instance(r, img_paths, opts, gold + 1,
                                {"valid": True, "defect": "none", "correct_set": [gold + 1]},
                                f"muir_{idx}", "native", letters=True))
        pair = tuple(sorted((idx, str(r["counterpart_idx"]))))
        if pair in done:
            continue
        done.add(pair)
        a, b = rows[pair[0]], rows[pair[1]]
        ans, una = (a, b) if NONE_TEXT not in str(a["options"][ord(a["answer"]) - 65]) else (b, a)
        assert NONE_TEXT in str(una["options"][ord(una["answer"]) - 65]), pair
        rng = random.Random(f"{SEED}-{pair[0]}-{pair[1]}")
        # Answerable question with the None option removed: one correct option remains.
        a_opts = [(i, t) for i, t in enumerate(ans["options"]) if NONE_TEXT not in str(t)]
        a_gold = [o for o, _ in a_opts].index(ord(ans["answer"]) - 65) + 1
        a_paths = _save_images(ans, OUT / "images" / str(ans["idx"]))
        wrong = rng.choice([k for k in range(1, len(a_opts) + 1) if k != a_gold])
        muirv.append(_instance(ans, a_paths, a_opts, a_gold, {"valid": True, "defect": "none",
                               "correct_set": [a_gold]}, f"muirv_{ans['idx']}_ans_gold", "ans_gold"))
        muirv.append(_instance(ans, a_paths, a_opts, wrong, {"valid": False, "defect": "key_mismatch",
                               "correct_set": [a_gold]}, f"muirv_{ans['idx']}_ans_wrong", "ans_wrong"))
        # Unanswerable question with the None option removed: no correct option remains.
        u_opts = [(i, t) for i, t in enumerate(una["options"]) if NONE_TEXT not in str(t)]
        u_paths = _save_images(una, OUT / "images" / str(una["idx"]))
        muirv.append(_instance(una, u_paths, u_opts, rng.randrange(1, len(u_opts) + 1),
                               {"valid": False, "defect": "no_correct", "correct_set": []},
                               f"muirv_{una['idx']}_unans_drop", "unans_drop"))
        u_all = [(i, str(t)) for i, t in enumerate(una["options"])]
        u_gold = ord(una["answer"]) - 65 + 1
        muirv.append(_instance(una, u_paths, u_all, u_gold, {"valid": True, "defect": "none",
                               "correct_set": [u_gold]}, f"muirv_{una['idx']}_unans_native", "unans_native"))
    muirv2 = list(muirv)
    for pair in sorted(done):
        a, b = rows[pair[0]], rows[pair[1]]
        ans, una = (a, b) if NONE_TEXT not in str(a["options"][ord(a["answer"]) - 65]) else (b, a)
        rng = random.Random(f"v2-{SEED}-{pair[0]}-{pair[1]}")
        a_all = [(i, str(t)) for i, t in enumerate(ans["options"])]
        a_paths = _save_images(ans, OUT / "images" / str(ans["idx"]))
        a_gold = ord(ans["answer"]) - 65 + 1
        a_none = next(i for i, t in a_all if NONE_TEXT in t) + 1
        muirv2.append(_instance(ans, a_paths, a_all, a_gold, {"valid": True, "defect": "none",
                                "correct_set": [a_gold]}, f"muirv2_{ans['idx']}_ans_gold_none", "ans_gold_none"))
        muirv2.append(_instance(ans, a_paths, a_all, a_none, {"valid": False, "defect": "key_mismatch",
                                "correct_set": [a_gold]}, f"muirv2_{ans['idx']}_ans_none_key", "ans_none_key"))
        u_all = [(i, str(t)) for i, t in enumerate(una["options"])]
        u_paths = _save_images(una, OUT / "images" / str(una["idx"]))
        u_gold = ord(una["answer"]) - 65 + 1
        u_wrong = rng.choice([k for k in range(1, len(u_all) + 1) if k != u_gold])
        muirv2.append(_instance(una, u_paths, u_all, u_wrong, {"valid": False, "defect": "key_mismatch",
                                "correct_set": [u_gold]}, f"muirv2_{una['idx']}_unans_wrong", "unans_wrong"))
    for name, data in (("native", native), ("muirv", muirv), ("muirv2", muirv2)):
        (OUT / name).mkdir(parents=True, exist_ok=True)
        with open(OUT / name / "instances.jsonl", "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("native", len(native), "muirv", len(muirv))


def first_pass_pairs(n_pairs=400):
    """Seeded, task-stratified pair subset fixed in EXTERNAL_PROTOCOL.md (2026-10-21)."""
    rows = [json.loads(l) for l in open(OUT / "muirv" / "instances.jsonl", encoding="utf-8")]   # pairs fixed by v1
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], set()).add(r["base_id"])
    total = sum(len(v) for v in by_task.values())
    rng = random.Random(SEED)
    chosen = set()
    for task in sorted(by_task):
        pairs = sorted(by_task[task])
        k = min(len(pairs), max(10, round(n_pairs * len(pairs) / total)))
        chosen |= set(rng.sample(pairs, k))
    manifest = TRACK/'frozen/MUIR_SPLIT_20261022.json'
    if manifest.exists():
        frozen = json.loads(manifest.read_text())
        if chosen != set(frozen['first_pass_pairs']):
            raise ValueError('Frozen Muir first-pass assignment changed')
    return chosen


def freeze_splits():
    first = first_pass_pairs()
    all_pairs = {r['base_id'] for r in load_rows(OUT/'muirv/instances.jsonl')}
    value = {'version': 1, 'seed': SEED, 'first_pass_pairs': sorted(first),
        'sealed_confirmation_pairs': sorted(all_pairs-first),
        'policy': 'Automatic runs use first pass only. Remaining 901 pairs are sealed against inference and method selection.',
        'dataset_sha256': {name: sha256(OUT/name/'instances.jsonl') for name in ('native', 'muirv', 'muirv2')}}
    assert len(first) == 399 and len(all_pairs-first) == 901
    p = TRACK/'frozen/MUIR_SPLIT_20261022.json'
    if p.exists() and json.loads(p.read_text()) != value:
        raise ValueError('Refusing to change frozen Muir split or source datasets')
    atomic_json(p, value)
    return value



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build-muir", "freeze-splits"))
    args = parser.parse_args()
    if args.mode == "build-muir":
        build_muir()
    freeze_splits()
