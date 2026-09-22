# -*- coding: utf-8 -*-
"""make_gold_p0.py — P0 作者级定向扩充金标样本。

设计（承接 make_gold_sample.py 的盲标协议，种子换新 20260918）：
  1. 12 位贬谪队列官员：每人抽最多 8 首（队列官员是作者级效度 CI 的主对象）。
  2. 队列外作者：唐/宋各 15 位（共 30 位），每人最多 7 首 —— 专攻作者级效度短板。
  3. 约束与原样本一致：20 <= 诗长 <= 160 字；排除原 160 首已标注诗（按文本前 14 字去重）。
  4. 盲标：文本不含作者/出处，只给朝代；作者与 NMI 存私有 key 文件。
输出:
  gold_p0.json            私有（author/title/nmi/carrier/strata）
  gold_p0_blind.md        交给两位标注员的盲标文本（格式与原 gold_sample_blind.md 一致）
  gold_p0_annot_template.csv  回填模板
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import opencc

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lexicon as L

cc = opencc.OpenCC("t2s")
RNG = np.random.default_rng(20260918)
MINL, MAXL = 20, 160
TARGET12 = ["柳宗元", "刘禹锡", "韩愈", "白居易",
            "苏轼", "欧阳修", "黄庭坚", "秦观", "王禹偁", "范仲淹", "辛弃疾", "陆游"]
N_COHORT_PER = 8      # 队列官员每人
N_NONCOHORT_AUTHORS = 30   # 队列外作者总数（唐 15 + 宋 15）
N_NONCOHORT_PER = 7   # 每人


def load_poems(pattern, root_tag):
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
            ti = cc.convert((p.get("title") or "").strip())
            if a and MINL <= len(t) <= MAXL:
                out.append({"author": a, "title": ti, "text": t, "carrier": root_tag})
    return out


def nmi_of(text):
    per, _n = L.cat_per1k([text])
    return L.net_index(per), per


def main():
    print("loading corpora ...")
    tang_qj = load_poems(f"{BASE}/data/tang/poet.tang.*.json", "quanji")
    tang_xb = load_poems(f"{BASE}/data/yuding/*.json", "xuanben")
    song_qj = load_poems(f"{BASE}/data/songshi/poet.song.*.json", "quanji")
    song_xb = load_poems(f"{BASE}/data/yusong/poet.song_selection.json", "xuanben")
    pool = tang_qj + tang_xb + song_qj + song_xb
    print(f"  pool={len(pool)} (tang_qj={len(tang_qj)} tang_xb={len(tang_xb)} "
          f"song_qj={len(song_qj)} song_xb={len(song_xb)})")

    # 排除原 160 首（文本前 14 字指纹）
    old = json.load(open(f"{BASE}/gold_sample.json", encoding="utf-8"))
    used = {p["text"][:14] for p in old}
    pool = [p for p in pool if p["text"][:14] not in used]
    print(f"  after excluding original 160: {len(pool)}")

    # 标朝代
    tang_authors = {p["author"] for p in tang_qj + tang_xb}
    song_authors = {p["author"] for p in song_qj + song_xb}
    for p in pool:
        p["dynasty"] = "Tang" if p["author"] in tang_authors else "Song"

    # NMI
    for p in pool:
        p["nmi"], p["per"] = nmi_of(p["text"])

    # 按作者聚合
    by_author = defaultdict(list)
    for p in pool:
        by_author[(p["author"], p["dynasty"])].append(p)

    sample = []

    # 1) 队列官员：每人最多 8 首（跨载体混抽）
    for a in TARGET12:
        cands = by_author.get((a, "Tang"), []) + by_author.get((a, "Song"), [])
        cands = sorted(cands, key=lambda p: RNG.random())
        take = cands[:N_COHORT_PER]
        for p in take:
            p["group"] = "cohort"
        sample.extend(take)
        print(f"  cohort {a}: {len(take)} poems")

    # 2) 队列外作者：唐 15 + 宋 15，要求可用 >= 7 首
    picked_authors = {"Tang": [], "Song": []}
    for dyn in ("Tang", "Song"):
        cands = [(a, ps) for (a, d), ps in by_author.items()
                 if d == dyn and a not in TARGET12 and len(ps) >= N_NONCOHORT_PER]
        cands.sort(key=lambda x: RNG.random())
        for a, ps in cands[:N_NONCOHORT_AUTHORS // 2]:
            ps2 = sorted(ps, key=lambda p: RNG.random())
            take = ps2[:N_NONCOHORT_PER]
            for p in take:
                p["group"] = "noncohort"
            sample.extend(take)
            picked_authors[dyn].append((a, len(take)))

    for dyn in ("Tang", "Song"):
        print(f"  noncohort {dyn}: {len(picked_authors[dyn])} authors, "
              f"{sum(n for _, n in picked_authors[dyn])} poems")

    # 去重 + 编号（与原协议一致）
    seen = set()
    uniq = []
    for p in sample:
        key = p["text"][:14]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    sample = uniq
    RNG.shuffle(sample)
    for i, p in enumerate(sample):
        p["pid"] = f"Q{i+1:03d}"   # Q 前缀区分于原 P 系列

    json.dump(sample, open(f"{BASE}/gold_p0.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    lines = ["# 逐首情感—意象判读（盲标·第二批）", "",
             f"共 {len(sample)} 首，已隐去作者与出处。请仅据原文与判定标准评分。",
             "判定标准与第一批完全相同（五维 0-3 整数，锚点与判读规则不变）。", ""]
    for p in sample:
        lines.append(f"### {p['pid']}   [{p['dynasty']}]")
        lines.append(p["text"])
        lines.append("")
    open(f"{BASE}/gold_p0_blind.md", "w", encoding="utf-8").write("\n".join(lines))

    with open(f"{BASE}/gold_p0_annot_template.csv", "w", encoding="utf-8") as f:
        f.write("pid,Bei,Quan,Man,Ziran,Konghuan\n")
        for p in sample:
            f.write(f"{p['pid']},,,,,\n")

    st = Counter((p["dynasty"], p["carrier"], p["group"]) for p in sample)
    print("strata:", dict(st))
    print("total:", len(sample))
    authors = sorted({p["author"] for p in sample})
    print(f"authors: {len(authors)}")
    nmis = np.array([p["nmi"] for p in sample])
    print(f"NMI range [{nmis.min():.2f}, {nmis.max():.2f}] mean {nmis.mean():.3f} "
          f"sd {nmis.std():.3f}")
    lens = np.array([len(p["text"]) for p in sample])
    print(f"len range [{lens.min()}, {lens.max()}] median {int(np.median(lens))}")


if __name__ == "__main__":
    main()
