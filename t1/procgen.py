"""Procedural items for training and development (PLAN.md Section 8).

Writes items in the same JSON schema as the LLM pipeline, so the same corruption
operators and the same render oracle label them. The natural pilot (LLM-written
items) is never used for training; it is the out-of-generator test.

Splits, fixed before any training:
  train      train templates, train assets, train paraphrases
  dev        same distribution as train, separate seeds (thresholds, early stopping)
  test_id    train templates, held-out assets and held-out paraphrases
  test_ood   held-out templates (A4 rotation-only, B add-detection), all assets

    python -m t1.procgen --out procgen_v1 --train 1500 --dev 150 --test-id 200 --test-ood 200
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from multiprocessing import Pool

from . import oracle
from .materialize import materialize
from .corrupt import family_a, family_b
from .itemgen import DATA, write_json

# ------------------------------------------------------------------ banks

ASSETS = ["사과", "열쇠", "머그컵", "안경", "연필", "볼펜", "책", "시계", "우산", "모자", "숟가락",
          "칼", "컵", "공", "바나나", "포도", "딸기", "레몬", "토마토", "당근", "감자", "식빵", "달걀",
          "자물쇠", "칫솔", "수건", "그릇", "쟁반", "운동화", "주사위", "물고기", "나뭇잎", "토끼", "새",
          "나무", "집", "구름", "태양", "상자", "벽돌", "해머", "깔때기", "거품기", "프라이팬", "유리컵",
          "오렌지", "배", "빵", "빗", "낫", "문", "돌"]
HELD_OUT_ASSETS = {"우산", "모자", "칫솔", "운동화", "주사위", "물고기", "토끼", "해머", "거품기", "프라이팬"}
SHAPES_ASYM = ["L자형 블록", "비대칭 도형", "화살표", "사분원"]
SHAPES_SYM = ["원", "정사각형", "삼각형", "별", "육각형", "마름모", "하트", "오각형", "직사각형", "타원"]
COLORS = ["빨강", "파랑", "초록", "노랑", "보라", "주황", "분홍", "회색", "갈색", "검정"]
COLOR_ADJ = {"빨강": "빨간색", "파랑": "파란색", "초록": "초록색", "노랑": "노란색", "보라": "보라색",
             "주황": "주황색", "분홍": "분홍색", "회색": "회색", "갈색": "갈색", "검정": "검은색"}

# The last paraphrase of each template is held out for test_id and test_ood.
Q = {
    "A1_identity": ["학습한 장면과 완전히 같은 장면을 고르세요.", "앞에서 본 그림과 똑같은 그림은 몇 번입니까?",
                    "처음 그림과 모든 것이 같은 보기를 찾으세요.", "기억한 장면과 하나도 다르지 않은 보기를 고르세요."],
    "A2_rotate": ["주어진 모양을 시계 방향으로 {t}도 돌린 모양은 어느 것입니까?",
                  "기준 도형을 시계 방향으로 {t}도 회전하면 어떤 모양이 됩니까?",
                  "왼쪽 그림을 시계 방향으로 {t}도 돌렸을 때의 모습을 고르세요."],
    "A3_mirror": ["세로로 세워진 거울에 비추면 어떤 모습이 될까요?", "기준 도형을 좌우로 뒤집은 모양을 고르세요.",
                  "거울에 비친 것처럼 좌우가 바뀐 그림은 어느 것입니까?"],
    "A4_rotonly": ["첫 번째 그림의 도형을 회전만 했을 때와 같은 모양은 어느 보기입니까?",
                   "뒤집지 않고 돌리기만 해서 만들 수 있는 모양을 고르세요."],
    "A5_tworule": ["첫 번째 그림의 도형에 두 가지 규칙을 모두 적용한 그림을 고르세요. 규칙 1: 색을 {c}으로 바꾼다. 규칙 2: 시계 방향으로 {t}도 회전한다.",
                   "다음 두 규칙을 함께 적용한 결과를 고르세요. 첫째, {c}으로 칠한다. 둘째, 시계 방향으로 {t}도 돌린다."],
    "A6_order": ["학습 장면에서 사물이 왼쪽에서 오른쪽으로 놓인 순서를 기억하세요. 검사 보기 중 학습 장면과 순서가 같은 것을 고르세요.",
                 "처음 본 순서와 똑같이 놓인 보기를 고르세요.", "사물이 놓인 순서가 학습 장면과 같은 보기는 몇 번입니까?"],
    "A7_sequence": ["왼쪽에서 오른쪽으로 이어지는 규칙을 계속하면, 다음에 올 그림은 어느 것입니까?",
                    "그림이 놓인 규칙을 이어서 다음 차례에 올 줄을 고르세요.",
                    "같은 규칙으로 계속 놓으면 다음에 올 그림들은 무엇입니까?"],
    "A8_symcomplete": ["세로 대칭으로 완성된 오른쪽 그림을 고르시오.",
                       "가운데 세로선을 기준으로 거울처럼 뒤집어 오른쪽에 그린 그림을 고르세요.",
                       "왼쪽 그림과 좌우 대칭이 되도록 오른쪽을 완성한 보기는 어느 것입니까?"],
    "A9_rotscale": ["첫 번째 그림의 모양은 그대로 유지하면서, 시계 방향으로 {t}도 회전하고 크기를 {s}배로 확대한 그림은 어느 것입니까?",
                    "기준 도형을 시계 방향으로 {t}도 돌리고 {s}배 크게 만든 모양을 고르세요.",
                    "도형을 {s}배로 키운 뒤 시계 방향으로 {t}도 돌린 그림은 어느 것입니까?"],
    "B_move": ["첫 번째 그림과 두 번째 그림을 비교했을 때, 자리가 바뀐 사물의 이름을 말해 주세요.",
               "두 그림에서 위치가 바뀐 사물은 무엇입니까?", "두 번째 그림에서 자리를 옮긴 사물의 이름을 말하세요."],
    "B_remove": ["첫 번째 그림과 두 번째 그림을 비교했을 때, 사라진 사물의 이름을 말해 주세요.",
                 "두 번째 그림에서 없어진 사물은 무엇입니까?", "처음 그림에는 있었는데 사라진 사물을 말하세요."],
    "B_add": ["첫 번째 그림에 없던 사물은 무엇입니까?", "두 번째 그림에 새로 생긴 사물의 이름을 말하세요."],
    "B_recolor": ["첫 번째 그림과 두 번째 그림을 비교했을 때, 색이 바뀐 사물의 이름을 말해 주세요.",
                  "두 그림에서 색깔이 바뀐 사물은 무엇입니까?", "두 번째 그림에서 색이 바뀐 사물의 이름을 말하세요."],
    "B_any": ["두 그림을 비교해 보세요. 달라진 사물의 이름을 말하세요.", "두 그림에서 다른 곳이 있는 사물은 무엇입니까?",
              "두 그림 사이에 달라진 사물 하나를 말해 주세요."],
    "B_multi": ["첫 번째 그림과 두 번째 그림을 비교하여 달라진 사물을 {n} 가지 모두 말해 주세요.",
                "두 그림에서 달라진 사물 {n} 개의 이름을 모두 말하세요."],
}
NUM = {2: "두", 3: "세"}
TRAIN_TEMPLATES = ["A1_identity", "A2_rotate", "A3_mirror", "A5_tworule", "A6_order",
                   "A7_sequence", "A8_symcomplete", "A9_rotscale",
                   "B_move", "B_remove", "B_recolor", "B_any", "B_multi"]
OOD_TEMPLATES = ["A4_rotonly", "B_add"]
# Pseudo slots tell the family-B operators which change type a template asks about.
B_SLOT = {"B_move": "MEM_03", "B_remove": "MEM_07", "B_add": "MEM_08", "B_recolor": "MEM_09",
          "B_any": "ATT_05", "B_multi": "ATT_06"}
B_ASKED = {"B_move": {"move"}, "B_remove": {"remove"}, "B_add": {"add"}, "B_recolor": {"recolor"},
           "B_any": None, "B_multi": None}
DOMAIN = {"A1": "기억력", "A2": "시공간능력", "A3": "시공간능력", "A4": "시공간능력", "A5": "집행기능",
          "A6": "기억력", "A7": "집행기능", "A8": "시공간능력", "A9": "시공간능력", "B_": "주의력"}


# ------------------------------------------------------------------ scenes

class Gen:
    def __init__(self, rng: random.Random, split: str):
        self.rng = rng
        held = split in ("test_id",)
        self.assets = [a for a in ASSETS if (a in HELD_OUT_ASSETS) == held] if split != "test_ood" else ASSETS
        self.held_q = split in ("test_id", "test_ood")

    def question(self, template: str, **kw) -> str:
        pool = Q[template]
        pool = pool[-1:] if self.held_q and len(pool) > 1 else (pool[:-1] if len(pool) > 1 else pool)
        return self.rng.choice(pool).format(**kw)

    def color(self, exclude=()):
        return self.rng.choice([c for c in COLORS if c not in exclude])

    def objects(self, n, shapes_too=True):
        names = self.rng.sample(self.assets + (SHAPES_SYM if shapes_too else []), n)
        out = []
        for k, name in enumerate(names):
            o = {"ref": f"o{k + 1}", "name": name}
            if name in SHAPES_SYM or self.rng.random() < 0.5:
                o["color"] = self.color()
            out.append(o)
        return out

    def grid(self, n_objects, rows=None, cols=None, free_cells=1):
        while True:
            r, c = rows or self.rng.choice([2, 3]), cols or self.rng.choice([2, 3])
            if r * c >= n_objects + free_cells:
                break
        cells = self.rng.sample([[i, j] for i in range(r) for j in range(c)], n_objects)
        objs = self.objects(n_objects)
        for o, cell in zip(objs, cells):
            o["cell"] = cell
        return {"layout": "grid", "rows": r, "cols": c, "background": "흰 바닥", "objects": objs}

    def single(self, name=None, rotation=0):
        name = name or self.rng.choice(SHAPES_ASYM)
        return {"layout": "free", "background": "흰 바닥", "objects": [
            {"ref": "shape", "name": name, "color": self.color(), "pos": [0.5, 0.5], "rotation": rotation}]}


def _free(scene, exclude=()):
    taken = {tuple(o["cell"]) for o in scene["objects"]} | set(exclude)
    return [[i, j] for i in range(scene["rows"]) for j in range(scene["cols"]) if (i, j) not in taken]


def choice_item(sid, stage, scene, question, key_changes, distractors, rng, opt_stage="보기"):
    opts = [key_changes] + distractors
    order = list(range(4))
    rng.shuffle(order)
    panels = [{"stage": stage, "kind": "scene", "scene": scene}]
    key = None
    for idx, k in enumerate(order, 1):
        panels.append({"stage": opt_stage, "kind": "option", "option_index": idx, "edit_of": stage,
                       "changes": opts[k]})
        if k == 0:
            key = idx
    return {"slot_id": sid, "question": question, "answer": str(key),
            "answer_rationale": f"보기 {key}만 조건을 만족한다.", "panels": panels}


def rot(t):
    return {"op": "rotate", "target": "shape", "to_rotation": t % 360}


def make_a(template, g: Gen):
    rng = g.rng
    if template == "A1_identity":
        sc = g.grid(rng.randint(3, 5))
        objs = rng.sample(sc["objects"], 3)
        ds = []
        for o in objs:
            kind = rng.choice(["move", "recolor", "rescale"])
            free = _free(sc)
            if kind == "move" and free:
                ds.append([{"op": "move", "target": o["ref"], "to_cell": rng.choice(free)}])
            elif kind == "recolor":
                ds.append([{"op": "recolor", "target": o["ref"], "to_color": g.color((o.get("color"),))}])
            else:
                ds.append([{"op": "rescale", "target": o["ref"], "to_scale": rng.choice([0.6, 1.4])}])
        return choice_item("PG_A1", "학습", sc, g.question(template), [], ds, rng, "검사 보기"), None
    if template == "A2_rotate":
        r0, t = rng.choice([0, 90, 180, 270]), rng.choice([90, 180, 270])
        others = [a for a in (0, 90, 180, 270) if a != t]
        ds = [[rot(r0 + a)] for a in others]
        if rng.random() < 0.5:
            ds[rng.randrange(3)] = [{"op": "mirror", "target": "shape"}, rot(r0 + t)]
        return choice_item("PG_A2", "기준", g.single(rotation=r0), g.question(template, t=t), [rot(r0 + t)], ds, rng), None
    if template == "A3_mirror":
        r0 = rng.choice([0, 90, 180, 270])
        true_mirror = [{"op": "mirror", "target": "shape"}, rot(-r0)]
        ds = [[rot(r0 + a)] for a in rng.sample([90, 180, 270], 2)]
        # Toggling `mirrored` alone is the pipeline's mirror; for r0 = 90/270 it is not the
        # left-right mirror of what is shown, so it makes a hard distractor.
        ds.append([{"op": "mirror", "target": "shape"}] if r0 in (90, 270) else [rot(r0 + rng.choice([90, 270]))])
        return choice_item("PG_A3", "기준", g.single(rotation=r0), g.question(template), true_mirror, ds, rng), None
    if template == "A4_rotonly":
        r0, t = rng.choice([0, 90, 180, 270]), rng.choice([90, 180, 270])
        ds = [[{"op": "mirror", "target": "shape"}, rot(r0 + a)] for a in rng.sample([0, 90, 180, 270], 3)]
        alts = [[rot(r0 + a)] for a in (90, 180, 270)]      # a 0-degree turn is not a rotation
        return choice_item("PG_A4", "기준", g.single(rotation=r0), g.question(template), [rot(r0 + t)], ds, rng), alts
    if template == "A5_tworule":
        r0, t = rng.choice([0, 90, 180, 270]), rng.choice([90, 180, 270])
        sc = g.single(rotation=r0)
        c0 = sc["objects"][0]["color"]
        c1 = g.color((c0,))
        rc = {"op": "recolor", "target": "shape", "to_color": c1}
        wrong_t = rng.choice([a for a in (90, 180, 270) if a != t])
        ds = [[rc], [rot(r0 + t)], [rc, rot(r0 + wrong_t)]]
        q = g.question(template, c=COLOR_ADJ[c1], t=t)
        return choice_item("PG_A5", "기준", sc, q, [rc, rot(r0 + t)], ds, rng), None
    if template == "A6_order":
        n = rng.randint(3, 5)
        sc = g.grid(n, rows=1, cols=n, free_cells=0)
        sc["objects"].sort(key=lambda o: o["cell"][1])
        pairs = rng.sample([(i, j) for i in range(n) for j in range(i + 1, n)], 3)
        ds = [[{"op": "move", "target": sc["objects"][i]["ref"], "to_cell": [0, j]},
               {"op": "move", "target": sc["objects"][j]["ref"], "to_cell": [0, i]}] for i, j in pairs]
        return choice_item("PG_A6", "학습", sc, g.question(template), [], ds, rng, "검사 보기"), None
    if template == "A7_sequence":
        # Two item types share one shape and differ in colour and/or turn; continue A B A ...
        n = rng.choice([3, 5])
        shape = rng.choice(SHAPES_ASYM + ["삼각형", "별", "하트"])
        ca, cb = rng.sample(COLORS, 2)
        ra, rb = rng.choice([0, 90]), rng.choice([0, 90, 180, 270])
        look = {"A": (ca, ra), "B": (cb, rb)}
        pattern = "".join("AB"[k % 2] for k in range(n))
        sc = {"layout": "grid", "rows": 1, "cols": n, "background": "흰 바닥",
              "objects": [{"ref": f"o{k + 1}", "name": shape, "color": look[t][0], "rotation": look[t][1],
                           "cell": [0, k]} for k, t in enumerate(pattern)]}
        nxt = "".join("AB"[(n + k) % 2] for k in range(n))

        def to(pat):
            ch = []
            for k, (now, want) in enumerate(zip(pattern, pat)):
                if now != want:
                    ch += [{"op": "recolor", "target": f"o{k + 1}", "to_color": look[want][0]},
                           {"op": "rotate", "target": f"o{k + 1}", "to_rotation": look[want][1]}]
            return ch
        others = ["".join(rng.choice("AB") for _ in range(n)) for _ in range(12)]
        others = [o for o in dict.fromkeys(others) if o != nxt][:3]
        if len(others) < 3:
            raise ValueError("not enough distinct patterns")
        return choice_item("PG_A7", "기준", sc, g.question(template), to(nxt), [to(o) for o in others], rng), None
    if template == "A8_symcomplete":
        rows = rng.choice([2, 3])
        names = [rng.choice(SHAPES_ASYM)] + rng.sample(SHAPES_ASYM + SHAPES_SYM, rows - 1)
        rng.shuffle(names)
        objs = []
        for k, name in enumerate(names):
            objs.append({"ref": f"o{k + 1}", "name": name, "color": g.color(),
                         "rotation": rng.choice([0, 90, 180, 270]), "cell": [k, 0]})
        sc = {"layout": "grid", "rows": rows, "cols": 2, "background": "흰 바닥", "objects": objs}
        # The true left-right mirror of a turned object: toggle the flip and negate the turn.
        flip = lambda o: [{"op": "mirror", "target": o["ref"]},
                          {"op": "rotate", "target": o["ref"], "to_rotation": (-o["rotation"]) % 360}]
        move = lambda o, r: [{"op": "move", "target": o["ref"], "to_cell": [r, 1]}]
        key = [c for o in objs for c in move(o, o["cell"][0]) + flip(o)]
        d_move = [c for o in objs for c in move(o, o["cell"][0])]
        d_flip = [c for o in objs for c in flip(o)]
        shift = [(o["cell"][0] + 1) % rows for o in objs]
        d_row = [c for o, r in zip(objs, shift) for c in move(o, r) + flip(o)]
        return choice_item("PG_A8", "기준", sc, g.question(template), key, [d_move, d_flip, d_row], rng), None
    if template == "A9_rotscale":
        r0, t = rng.choice([0, 90, 180, 270]), rng.choice([90, 180, 270])
        sc_ = rng.choice([1.3, 1.5])
        size = {"op": "rescale", "target": "shape", "to_scale": sc_}
        wrong_t = rng.choice([a for a in (90, 180, 270) if a != t])
        ds = [[rot(r0 + t)], [size], [rot(r0 + wrong_t), size]]
        if rng.random() < 0.5:
            ds[rng.randrange(3)] = [{"op": "mirror", "target": "shape"}, rot(-(r0 + t)), size]
        q = g.question(template, t=t, s=sc_)
        return choice_item("PG_A9", "기준", g.single(rotation=r0), q, [rot(r0 + t), size], ds, rng), None
    raise KeyError(template)


def make_b(template, g: Gen):
    rng = g.rng
    n_req = rng.choice([2, 3]) if template == "B_multi" else 1
    sc = g.grid(rng.randint(3, 5), free_cells=2)
    for o in sc["objects"]:
        o.setdefault("color", g.color()) if template == "B_recolor" else None
    targets = rng.sample(sc["objects"], n_req)
    changes, names = [], []
    taken = []
    for o in targets:
        op = {"B_move": "move", "B_remove": "remove", "B_add": "add", "B_recolor": "recolor"}.get(
            template, rng.choice(["move", "recolor", "remove", "add"]))
        if op == "move":
            cell = rng.choice(_free(sc, taken))
            taken.append(tuple(cell))
            changes.append({"op": "move", "target": o["ref"], "to_cell": cell})
            names.append(o["name"])
        elif op == "recolor":
            changes.append({"op": "recolor", "target": o["ref"], "to_color": g.color((o.get("color"),))})
            names.append(o["name"])
        elif op == "remove":
            changes.append({"op": "remove", "target": o["ref"]})
            names.append(o["name"])
        else:
            cell = rng.choice(_free(sc, taken))
            taken.append(tuple(cell))
            new = rng.choice([a for a in g.assets if a not in {x["name"] for x in sc["objects"]} | set(names)])
            ref = f"n{len(changes) + 1}"
            changes.append({"op": "add", "target": ref, "object": {"ref": ref, "name": new, "cell": cell}})
            names.append(new)
    q = g.question(template, n=NUM.get(n_req, n_req))
    item = {"slot_id": f"PG_{template}", "question": q, "answer": ", ".join(names),
            "answer_rationale": "바뀐 사물: " + ", ".join(names),
            "panels": [{"stage": "기준", "kind": "scene", "scene": sc},
                       {"stage": "검사", "kind": "edit", "edit_of": "기준", "changes": changes}]}
    return item, n_req


# ------------------------------------------------------------------ build

def build_one(args):
    split, template, idx, seed, out_name, qcp = args
    rng = random.Random(seed)
    g = Gen(rng, split)
    out_root = DATA / out_name / split
    for attempt in range(20):
        base_id = f"pg_{split}_{template}_{idx:05d}"
        if template.startswith("A"):
            try:
                item, alts = make_a(template, g)
                label = oracle.label_choice(item, alts if alts else next(
                    p["changes"] for p in item["panels"] if p.get("kind") == "option"
                    and str(p["option_index"]) == item["answer"]))
            except Exception:
                continue
            if not label["valid"] or label["flags"]["dup_distractor"] or label["flags"]["transformation_invisible"]:
                continue                       # bases must be clean; corruptions add the defects
            base = {"base_id": base_id, "slot_id": item["slot_id"], "family": "A", "item": item}
            try:
                variants = family_a(item, seed, intended=alts, qcp=qcp)
            except Exception:
                continue
        else:
            asked = B_ASKED[template]
            try:
                item, n_req = make_b(template, g)       # may run out of free cells: retry
                label = oracle.label_change(item, asked, n_req)
            except Exception:
                continue
            if not label["valid"] or label["flags"]["ambiguous_referent"]:
                continue
            base = {"base_id": base_id, "slot_id": item["slot_id"], "family": "B", "item": item}
            questions = {B_SLOT[t]: g.question(t, n="") for t in ("B_move", "B_remove", "B_add", "B_recolor")}
            try:
                variants = family_b(item, B_SLOT[template], asked, n_req, questions, seed, qcp=qcp)
            except Exception:
                continue
        rows = materialize(base, variants, out_root, DOMAIN[template[:2]], "보기번호" if template[0] == "A" else "사물명")
        for r in rows:
            r["split"], r["template"] = split, template
        return rows
    return [{"id": f"pg_{split}_{template}_{idx:05d}", "error": "no valid base in 20 attempts"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="procgen_v1")
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--dev", type=int, default=150)
    ap.add_argument("--test-id", type=int, default=200)
    ap.add_argument("--test-ood", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--qcp", action="store_true", help="add question-counterfactual pairs")
    a = ap.parse_args()
    jobs, rng = [], random.Random(a.seed)
    for split, n, templates in (("train", a.train, TRAIN_TEMPLATES), ("dev", a.dev, TRAIN_TEMPLATES),
                                ("test_id", a.test_id, TRAIN_TEMPLATES), ("test_ood", a.test_ood, OOD_TEMPLATES)):
        for i in range(n):
            jobs.append((split, templates[i % len(templates)], i, rng.randrange(1 << 30), a.out, a.qcp))
    with Pool(a.workers) as pool:
        rows = [r for chunk in pool.imap_unordered(build_one, jobs, chunksize=4) for r in chunk]
    ok = [r for r in rows if "error" not in r]
    summary = {"errors": collections.Counter(r["error"].split(":")[0] for r in rows if "error" in r),
               "by_split": {}}
    for split in ("train", "dev", "test_id", "test_ood"):
        sr = sorted((r for r in ok if r["split"] == split), key=lambda r: r["id"])
        (DATA / a.out / split).mkdir(parents=True, exist_ok=True)
        with open(DATA / a.out / split / "instances.jsonl", "w", encoding="utf-8") as f:
            for r in sr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["by_split"][split] = {
            "bases": len({r["base_id"] for r in sr}), "instances": len(sr),
            "valid_rate": round(sum(r["label"]["valid"] for r in sr) / max(len(sr), 1), 3),
            "by_defect": dict(collections.Counter(r["label"]["defect"] for r in sr)),
            "by_template": dict(collections.Counter(r["template"] for r in sr))}
    write_json(DATA / a.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
