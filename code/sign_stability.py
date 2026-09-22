# -*- coding: utf-8 -*-
"""sign_stability.py — per-author sign-flip probability under lexicon perturbation.

Converts the leave-k-out sign-flip *narrative* into a per-author quantitative
metric for tab:nmi.

Method (as prescribed by the devil's-advocate review, A3):
  For each of the 12 cohort authors, draw B=2,000 perturbed lexicons; in each
  draw, independently remove a random 10% of the entries from each of the two
  affective poles (悲苦孤寂 Bei, 旷达闲适 Kuang). Recompute the author's NMI
  from the surviving word counts over the author's complete collection and
  record whether its sign reverses relative to the full-lexicon sign.
  flip rate = #(sign reversal) / B.

Counting and aggregation replicate methods_compare_v5.py exactly:
  author NMI = (sum Kuang counts - sum Bei counts) / total chars * 1000,
  counts via str.count on OpenCC-t2s-normalized joined paragraphs.
The baseline is asserted against results_v5_methods.json M1_lexicon.

Output: results_sign_stability.json
"""
import sys, os, json, glob, random
from collections import defaultdict
import numpy as np
import opencc

BASE = "D:/databuddy/2026-09-14-10-59-40/tangsong_research"
sys.path.insert(0, BASE)
import lexicon as L

cc = opencc.OpenCC("t2s")
DATA = f"{BASE}/data"
AUTH = ["柳宗元", "刘禹锡", "韩愈", "白居易", "苏轼", "欧阳修",
        "黄庭坚", "秦观", "王禹偁", "范仲淹", "辛弃疾", "陆游"]
SEED = 20260918
B = 2000
DROP = 0.10

POLES = ("旷达闲适", "悲苦孤寂")  # NMI = Kuang - Bei


def load_poems(folder, authors):
    out = {a: [] for a in authors}
    for fp in glob.glob(folder):
        d = json.load(open(fp, encoding="utf-8"))
        for p in d:
            a = cc.convert(p.get("author", ""))
            if a in out:
                txt = "".join(p.get("paragraphs", []))
                if txt:
                    out[a].append(cc.convert(txt))
    return out


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    tang = load_poems(f"{DATA}/tang/poet.tang.*.json",
                      ["柳宗元", "刘禹锡", "韩愈", "白居易"])
    song = load_poems(f"{DATA}/songshi/poet.song.*.json",
                      ["苏轼", "欧阳修", "黄庭坚", "秦观", "王禹偁", "范仲淹", "辛弃疾", "陆游"])
    shi = {**tang, **song}
    n_total = sum(len(v) for v in shi.values())
    print(f"[ss] unified shi: {n_total} poems across {len(AUTH)} authors")

    ku_words = list(L.LEX[POLES[0]])
    be_words = list(L.LEX[POLES[1]])
    print(f"[ss] pole sizes: Kuang={len(ku_words)}  Bei={len(be_words)}")

    # per-author per-word counts over the complete collection + total chars
    ku_cnt, be_cnt, chars, npoems = {}, {}, {}, {}
    for a in AUTH:
        ck = np.zeros(len(ku_words))
        cb = np.zeros(len(be_words))
        nch = 0
        for t in shi[a]:
            for j, w in enumerate(ku_words):
                c = t.count(w)
                if c:
                    ck[j] += c
            for j, w in enumerate(be_words):
                c = t.count(w)
                if c:
                    cb[j] += c
            nch += len(L.strip_punct(t))
        ku_cnt[a], be_cnt[a], chars[a], npoems[a] = ck, cb, nch, len(shi[a])
        print(f"[ss]   {a}: poems={npoems[a]:5d} chars={nch:8d}")

    # baseline NMI and assertion against the published table
    ref = json.load(open(f"{BASE}/results_v5_methods.json", encoding="utf-8"))
    ref = ref["net_mind_methods"]["M1_lexicon"]
    base, bad = {}, []
    for a in AUTH:
        nmi = 1000.0 * (ku_cnt[a].sum() - be_cnt[a].sum()) / chars[a]
        base[a] = nmi
        if abs(round(nmi, 3) - ref[a]) > 1.1e-3:
            bad.append((a, round(nmi, 3), ref[a]))
    if bad:
        print("[ss] BASELINE MISMATCH:", bad)
        sys.exit(1)
    print("[ss] baseline NMI matches results_v5_methods.json M1_lexicon for all 12 authors")

    # perturbation: drop 10% of entries within each pole, B draws
    n_drop_ku = max(1, int(round(DROP * len(ku_words))))
    n_drop_be = max(1, int(round(DROP * len(be_words))))
    print(f"[ss] dropout per draw: Kuang {n_drop_ku}/{len(ku_words)}, Bei {n_drop_be}/{len(be_words)}, B={B}")

    rng = np.random.default_rng(SEED)
    out = {}
    for a in AUTH:
        s0 = np.sign(base[a])
        flips = 0
        vals = np.empty(B)
        for b in range(B):
            mk = np.ones(len(ku_words), dtype=bool)
            mb = np.ones(len(be_words), dtype=bool)
            mk[rng.choice(len(ku_words), n_drop_ku, replace=False)] = False
            mb[rng.choice(len(be_words), n_drop_be, replace=False)] = False
            v = 1000.0 * (mk @ ku_cnt[a] - mb @ be_cnt[a]) / chars[a]
            vals[b] = v
            if np.sign(v) != s0:
                flips += 1
        out[a] = {
            "nmi_baseline": round(base[a], 3),
            "flip_rate": round(flips / B, 4),
            "flip_pct": round(100.0 * flips / B, 1),
            "perturbed_mean": round(float(vals.mean()), 3),
            "perturbed_p05": round(float(np.percentile(vals, 5)), 3),
            "perturbed_p95": round(float(np.percentile(vals, 95)), 3),
        }
        print(f"[ss]   {a}: base={base[a]:+.3f}  flip={out[a]['flip_pct']:5.1f}%  "
              f"[p05,p95]=[{out[a]['perturbed_p05']:+.2f},{out[a]['perturbed_p95']:+.2f}]")

    res = {
        "method": ("per-author sign-flip probability under lexicon perturbation: "
                   "each of B draws independently removes a random 10% of entries from "
                   "each affective pole (Kuang, Bei), NMI recomputed from surviving word "
                   "counts over the author's complete collection; flip = sign reversal "
                   "relative to the full-lexicon NMI"),
        "seed": SEED, "B": B, "drop_fraction": DROP,
        "n_drop_kuang": n_drop_ku, "n_drop_bei": n_drop_be,
        "pole_sizes": {"旷达闲适": len(ku_words), "悲苦孤寂": len(be_words)},
        "baseline_source": "results_v5_methods.json:net_mind_methods:M1_lexicon (asserted equal)",
        "aggregation": "identical to methods_compare_v5.py:agg_net (str.count, per-1000-chars)",
        "flip_rates": out,
    }
    with open(f"{BASE}/results_sign_stability.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[ss] wrote results_sign_stability.json")


if __name__ == "__main__":
    main()
