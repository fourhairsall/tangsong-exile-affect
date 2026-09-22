# -*- coding: utf-8 -*-
"""make_gold_sample.py — 构建人工标注金标样本 (效标效度用)。

设计要点
  1. 逐首单位为**诗篇**(poem-level), 非作者级 —— 效标必须与词表读出同层级。
  2. 分层: 朝代(唐 80 / 宋 80) x 是否属十二人贬谪队列 x 词表 NMI 五分位 x
     载体(总集 60 / 选本 20), 固定随机种子 20260917, 使样本跨越 NMI 全域。
  3. 可标注性: 20 <= 诗长 <= 160 字 (过短无从判读, 过长超出逐首标注可行性)。
  4. **盲标**: 交给标注员的文本**不含作者名**, 只给朝代与原文, 以免光环效应;
     词表输出一律不给。作者与词表值只存于私有 key 文件。
输出:
  gold_sample.json          私有(含 author/title/nmi/strata)
  gold_sample_blind.md      交给标注员的盲标文本
  gold_annot_template.csv   标注回填模板
"""
import glob
import json
import os
import sys

import numpy as np
import opencc

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lexicon as L

cc = opencc.OpenCC("t2s")
RNG = np.random.default_rng(20260917)
MINL, MAXL = 20, 160
TARGET12 = ["柳宗元", "刘禹锡", "韩愈", "白居易",
            "苏轼", "欧阳修", "黄庭坚", "秦观", "王禹偁", "范仲淹", "辛弃疾", "陆游"]


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


def stratified(pool, n_main, n_anth, tag):
    """总集侧抽 n_main 首, 选本侧抽 n_anth 首; 各自跨 NMI 五分位 x 队列分层."""
    picks = []
    for carrier, n in (("quanji", n_main), ("xuanben", n_anth)):
        sub = [p for p in pool if p["carrier"] == carrier]
        if not sub:
            continue
        vals = np.array([p["nmi"] for p in sub])
        qs = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
        for p in sub:
            p["quintile"] = int(np.searchsorted(qs, p["nmi"]))
            p["queued"] = p["author"] in TARGET12
        # 8 层: 队列 x NMI 四分位组(为凑够样本量, 用中位数二分而非五分位)
        med = float(np.median(vals))
        for p in sub:
            p["nmi_hi"] = p["nmi"] >= med
        strata = {}
        for p in sub:
            strata.setdefault((p["queued"], p["nmi_hi"]), []).append(p)
        keys = sorted(strata, key=lambda k: (-len(strata[k]), k))
        per = max(1, n // len(keys))
        chosen = []
        for k in keys:
            grp = sorted(strata[k], key=lambda p: RNG.random())
            chosen.extend(grp[:per])
        # 补齐到 n
        rest = [p for k in keys for p in strata[k] if p not in chosen]
        RNG.shuffle(rest)
        chosen.extend(rest[:max(0, n - len(chosen))])
        chosen = chosen[:n]
        for p in chosen:
            p["dynasty"] = tag
        picks.extend(chosen)
    return picks


def main():
    print("loading corpora ...")
    tang_qj = load_poems(f"{BASE}/data/tang/poet.tang.*.json", "quanji")
    tang_xb = load_poems(f"{BASE}/data/yuding/*.json", "xuanben")
    song_qj = load_poems(f"{BASE}/data/songshi/poet.song.*.json", "quanji")
    song_xb = load_poems(f"{BASE}/data/yusong/poet.song_selection.json", "xuanben")
    for p in tang_qj + tang_xb + song_qj + song_xb:
        p["nmi"], p["per"] = nmi_of(p["text"])
    print(f"  Tang quanji={len(tang_qj)} xuanben={len(tang_xb)} | "
          f"Song quanji={len(song_qj)} xuanben={len(song_xb)}")

    tang = stratified(tang_qj + tang_xb, 60, 20, "Tang")
    song = stratified(song_qj + song_xb, 60, 20, "Song")
    sample = tang + song
    # 去重(同首异版可能重复)
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
        p["pid"] = f"P{i+1:03d}"

    json.dump(sample, open(f"{BASE}/gold_sample.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 盲标文本
    lines = ["# 逐首情感—意象判读（盲标）", "",
             f"共 {len(sample)} 首，已隐去作者与出处。请仅据原文与判定标准评分。", ""]
    for p in sample:
        lines.append(f"### {p['pid']}   [{p['dynasty']}]")
        lines.append(p["text"])
        lines.append("")
    open(f"{BASE}/gold_sample_blind.md", "w", encoding="utf-8").write("\n".join(lines))

    with open(f"{BASE}/gold_annot_template.csv", "w", encoding="utf-8") as f:
        f.write("pid,Bei,Quan,Man,Ziran,Konghuan\n")
        for p in sample:
            f.write(f"{p['pid']},,,,,\n")

    # 描述统计
    import collections
    st = collections.Counter((p["dynasty"], p["carrier"]) for p in sample)
    print("strata:", dict(st))
    print("queue authors in sample:", sorted({p["author"] for p in sample
                                              if p["author"] in TARGET12}))
    nmis = np.array([p["nmi"] for p in sample])
    print(f"NMI range [{nmis.min():.2f}, {nmis.max():.2f}] mean {nmis.mean():.3f} "
          f"sd {nmis.std():.3f}")
    lens = np.array([len(p["text"]) for p in sample])
    print(f"len range [{lens.min()}, {lens.max()}] median {int(np.median(lens))}")


if __name__ == "__main__":
    main()
