# -*- coding: utf-8 -*-
"""make_register_sample.py — B 第二阶段：候选新语域盲标验证样本。

三个候选语域（由 _b2_mine.py 残差挖掘得出）:
  Quiet 幽居闲静   种子: 静禅苔钓隐亭栽莲炉畦眠
  Travel 羁旅行役  种子: 吏马灯店舫猿驿帆仆僮
  Home   思乡飘零  种子: 泪零萧堪飘蓬雁砧捣衣
抽样: 每语域取种子命中最高 ~40 首 (20-160字) + 30 首三语域零命中对照, 盲标 R 编号。
输出: register_sample.json / register_sample_blind.md / register_annot_template.csv
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import opencc

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260919)
MINL, MAXL = 20, 160
cc = opencc.OpenCC("t2s")

SEEDS = {
    "Quiet": list("静禅苔钓隐亭栽莲炉畦眠"),
    "Travel": list("吏马灯店舫猿驿帆仆僮"),
    "Home": list("泪零萧堪飘蓬雁砧捣"),
}
N_PER = 40
N_CTRL = 30


def load_poems(pattern, tag):
    out = []
    for fp in glob.glob(pattern):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        for p in d:
            a = cc.convert((p.get("author") or "").strip())
            t = cc.convert("".join(p.get("paragraphs") or []))
            if a and MINL <= len(t) <= MAXL:
                out.append({"author": a, "title": cc.convert((p.get("title") or "").strip()),
                            "text": t, "src": tag})
    return out


def main():
    print("loading corpora ...")
    pool = (load_poems(f"{BASE}/data/tang/poet.tang.*.json", "tang")
            + load_poems(f"{BASE}/data/songshi/poet.song.*.json", "song"))
    print(f"pool={len(pool)}")

    # 排除已金标诗
    for f in ("gold_sample.json", "gold_p0.json"):
        for p in json.load(open(f"{BASE}/{f}", encoding="utf-8")):
            pass
    used = set()
    for f in ("gold_sample.json", "gold_p0.json"):
        used |= {p["text"][:14] for p in json.load(open(f"{BASE}/{f}", encoding="utf-8"))}
    pool = [p for p in pool if p["text"][:14] not in used]
    print(f"after exclusion: {len(pool)}")

    # 语域得分
    scored = []
    for p in pool:
        hits = {}
        for reg, seeds in SEEDS.items():
            hits[reg] = sum(p["text"].count(s) for s in seeds)
        scored.append((p, hits))

    sample = []
    for reg in ("Quiet", "Travel", "Home"):
        cand = sorted([(p, h) for p, h in scored if h[reg] > 0],
                      key=lambda x: (-x[1][reg], RNG.random()))
        take, seen_txt = [], set()
        for p, h in cand:
            if p["text"][:14] in seen_txt:
                continue
            seen_txt.add(p["text"][:14])
            take.append((p, dict(h, _score=h[reg], _reg=reg)))
            if len(take) >= N_PER:
                break
        print(f"{reg}: top {len(take)} (score range {take[0][1]['_score']}-{take[-1][1]['_score']})")
        sample.extend(take)

    # 对照: 三语域全零
    ctrl = [(p, {"_score": 0, "_reg": "control"}) for p, h in scored
            if all(h[r] == 0 for r in SEEDS)]
    RNG.shuffle(ctrl)
    ctrl = ctrl[:N_CTRL]
    print(f"control: {len(ctrl)}")
    sample.extend(ctrl)

    # 去重与编号
    seen, uniq = set(), []
    for p, meta in sample:
        k = p["text"][:14]
        if k in seen:
            continue
        seen.add(k)
        uniq.append((p, meta))
    RNG.shuffle(uniq)
    for i, (p, meta) in enumerate(uniq):
        meta["pid"] = f"R{i+1:03d}"
        meta["dynasty"] = "Tang" if p["src"] == "tang" else "Song"
        meta["seed_hits"] = {r: sum(p["text"].count(s) for s in seeds)
                             for r, seeds in SEEDS.items()}
        meta["register"] = meta.pop("_reg")
        meta["score"] = meta.pop("_score")

    out = [{"pid": m["pid"], "author": p["author"], "title": p["title"], "text": p["text"],
            "dynasty": m["dynasty"], "register": m["register"], "score": m["score"],
            "seed_hits": m["seed_hits"]} for p, m in uniq]
    json.dump(out, open(f"{BASE}/register_sample.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    lines = ["# 候选情感—意象语域判读（盲标）", "",
             f"共 {len(out)} 首，已隐去作者与出处。请仅据原文与判定标准评分。", "",
             "三个候选语域，各按 0–3 整数评分（锚点同前：0=全篇无；1=偶一及之或程式套语；",
             "2=构成基调一部分；3=全篇主调）。意象≠情绪。", "",
             "`Quiet` 幽居闲静——园林/书斋/山居的静趣安顿（栽花莳苔、垂钓、禅榻、亭轩独坐、",
             "  焚香啜茶等幽居生活之自得；注意：写景本身不算，须落在幽居生活的心境上）",
             "`Travel` 羁旅行役——官旅行役漂泊之踪迹与劳顿（吏役、鞍马、津店、孤舟夜泊、",
             "  猿声驿路等行旅意象与奔波之感；注意：单纯的送别不算，须落在行役处境上）",
             "`Home` 思乡飘零——怀归之思与身世飘零（泪、断蓬飞蓬、旅雁、砧声、萧瑟、",
             "  不堪/那堪等怀归伤漂语；注意：一般羁旅描写不自动计入，须有怀归或飘零之叹）", ""]
    for p, m in zip([o for o in out], uniq):
        pass
    for item in out:
        lines.append(f"### {item['pid']}   [{item['dynasty']}]")
        lines.append(item["text"])
        lines.append("")
    open(f"{BASE}/register_sample_blind.md", "w", encoding="utf-8").write("\n".join(lines))

    with open(f"{BASE}/register_annot_template.csv", "w", encoding="utf-8") as f:
        f.write("pid,Quiet,Travel,Home\n")
        for item in out:
            f.write(f"{item['pid']},,,\n")

    cnt = Counter(o["register"] for o in out)
    print("strata:", dict(cnt), "total:", len(out))


if __name__ == "__main__":
    main()
