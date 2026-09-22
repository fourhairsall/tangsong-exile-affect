# -*- coding: utf-8 -*-
"""Phase 1b -- invariance recheck with the induced lexicon, vs M1 baseline.

Same cross-edition pipeline, two lexicons (M1 hand lexicon vs Phase-1 induced
lexicon).  We report:
  * Tang same-source invariance: QTS (tang) vs 御选 (yuding), author-level NMI r.
  * Song cross-source divergence: 全宋诗 (songshi) vs 御选 (yuding), author NMI r.
  * Per-dimension preservation count (|r|>=0.5 across matched authors).
This verifies the core scientific claims survive the lexicon swap.
"""
import json, glob, os
import numpy as np
import opencc
import lexicon as L
import lexicon_learned as LL

cc = opencc.OpenCC('t2s')
BASE = "D:/databuddy/2026-09-14-10-59-40/tangsong_research"
DATA = f"{BASE}/data"
TANG_AUTH = ["柳宗元", "刘禹锡", "韩愈", "白居易"]
SONG_AUTH = ["苏轼", "欧阳修", "黄庭坚", "秦观", "王禹偁", "范仲淹", "辛弃疾", "陆游"]

def author_nmi(folder, authors, lexmod):
    out = {a: [] for a in authors}
    for fp in glob.glob(folder):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for p in d:
            a = cc.convert(p.get("author", ""))
            if a in out:
                txt = "".join(p.get("paragraphs", []))
                if txt:
                    out[a].append(cc.convert(txt))
    res = {}
    for a, texts in out.items():
        if not texts:
            continue
        per, nch = lexmod.cat_per1k(texts)
        res[a] = {"nmi": lexmod.net_index(per), "per": per, "n": len(texts), "chars": nch}
    return res

def invariance(lexmod, srcA, srcB, authors, label):
    A = author_nmi(srcA, authors, lexmod)
    B = author_nmi(srcB, authors, lexmod)
    common = [a for a in authors if a in A and a in B]
    r_nmi = float(np.corrcoef([A[a]["nmi"] for a in common],
                              [B[a]["nmi"] for a in common])[0, 1]) if len(common) > 1 else None
    dim = []
    for cat in lexmod.CATS:
        xs = [A[a]["per"][cat] for a in common]
        ys = [B[a]["per"][cat] for a in common]
        if len(common) >= 3:
            r = float(np.corrcoef(xs, ys)[0, 1])
        else:
            r = None
        dim.append((cat, r))
    n_preserved = sum(1 for _, r in dim if r is not None and abs(r) >= 0.5)
    return {"label": label, "common": common, "r_nmi": r_nmi,
            "dim_r": dim, "n_preserved": n_preserved}

def run(lexmod, name):
    tang = invariance(lexmod, f"{DATA}/tang/poet.tang.*.json", f"{DATA}/yuding/*.json",
                      TANG_AUTH, f"{name} | Tang QTS vs 御选")
    song = invariance(lexmod, f"{DATA}/songshi/poet.song.*.json", f"{DATA}/yuding/*.json",
                      SONG_AUTH, f"{name} | Song 全宋诗 vs 御选")
    return tang, song

print("=== M1 (hand lexicon) ===")
t1, s1 = run(L, "M1")
print("Tang r_nmi =", t1["r_nmi"], "| common:", t1["common"])
print("  dim:", [(c, round(r, 3) if r is not None else None) for c, r in t1["dim_r"]],
      "| preserved:", t1["n_preserved"], "/5")
print("Song r_nmi =", s1["r_nmi"], "| common:", s1["common"])
print("  dim:", [(c, round(r, 3) if r is not None else None) for c, r in s1["dim_r"]],
      "| preserved:", s1["n_preserved"], "/5")

print("\n=== Learned (Phase-1 induced) ===")
t2, s2 = run(LL, "Learned")
print("Tang r_nmi =", t2["r_nmi"], "| common:", t2["common"])
print("  dim:", [(c, round(r, 3) if r is not None else None) for c, r in t2["dim_r"]],
      "| preserved:", t2["n_preserved"], "/5")
print("Song r_nmi =", s2["r_nmi"], "| common:", s2["common"])
print("  dim:", [(c, round(r, 3) if r is not None else None) for c, r in s2["dim_r"]],
      "| preserved:", s2["n_preserved"], "/5")

out = {
    "M1": {"tang": t1, "song": s1},
    "Learned": {"tang": t2, "song": s2},
}
# round for json
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return round(o, 4)
    return o
with open(f"{BASE}/results_phase1b_invariance.json", "w", encoding="utf-8") as f:
    json.dump(clean(out), f, ensure_ascii=False, indent=2)
print("\nwrote results_phase1b_invariance.json")
