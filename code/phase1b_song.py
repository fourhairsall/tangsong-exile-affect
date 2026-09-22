# -*- coding: utf-8 -*-
"""Phase 1b (Song side) -- previously BLOCKED because 御選宋詩 was missing.
Now that data/song2 (御選宋金元明四朝詩) is parsed into data/yusong, we run the
cross-edition invariance check for the SONG dynasty:
    全宋诗 (songshi)  vs  御選宋詩 (yusong)
at the author level, using a DYNAMIC author intersection (all authors present in
both corpora), for both the M1 hand lexicon and the Phase-1 induced lexicon.
We report Pearson r on net NMI and per-dimension preservation (|r|>=0.5).
For context we also re-run the TANG check (QTS vs 御選唐詩 = yuding).
"""
import json, glob
import numpy as np
import opencc
import lexicon as L
import lexicon_learned as LL

cc = opencc.OpenCC('t2s')
BASE = "D:/databuddy/2026-09-14-10-59-40/tangsong_research"
DATA = f"{BASE}/data"

def aggregate(folder):
    agg = {}
    for fp in glob.glob(folder):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, list):
            for p in d:
                a = cc.convert(p.get("author", "") or "")
                t = "".join(p.get("paragraphs", []))
                if a and t:
                    agg.setdefault(a, []).append(cc.convert(t))
    return agg

def to_nmi(agg, lexmod):
    res = {}
    for a, texts in agg.items():
        per, nch = lexmod.cat_per1k(texts)
        res[a] = {"nmi": lexmod.net_index(per), "per": per,
                  "n": len(texts), "chars": nch}
    return res

def invariance(lexmod, aggA, aggB, label, min_chars=300):
    A = to_nmi(aggA, lexmod)
    B = to_nmi(aggB, lexmod)
    common = [a for a in A if a in B and A[a]["chars"] >= min_chars
              and B[a]["chars"] >= min_chars]
    if len(common) > 1:
        r_nmi = float(np.corrcoef([A[a]["nmi"] for a in common],
                                  [B[a]["nmi"] for a in common])[0, 1])
    else:
        r_nmi = None
    dim = []
    for cat in lexmod.CATS:
        xs = [A[a]["per"][cat] for a in common]
        ys = [B[a]["per"][cat] for a in common]
        r = float(np.corrcoef(xs, ys)[0, 1]) if len(common) >= 3 else None
        dim.append((cat, r))
    n_pres = sum(1 for _, r in dim if r is not None and abs(r) >= 0.5)
    return {"label": label, "n_common": len(common), "r_nmi": r_nmi,
            "dim_r": dim, "n_preserved": n_pres}

print("aggregating corpora ...")
songshi_agg = aggregate(f"{DATA}/songshi/poet.song.*.json")
yusong_agg = aggregate(f"{DATA}/yusong/poet.song_selection.json")
tang_agg = aggregate(f"{DATA}/tang/poet.tang.*.json")
yuding_agg = aggregate(f"{DATA}/yuding/*.json")
print(f"  songshi authors={len(songshi_agg)}  yusong authors={len(yusong_agg)}")
print(f"  tang authors={len(tang_agg)}  yuding(唐選) authors={len(yuding_agg)}")

def report(lexmod, name):
    print(f"\n=== {name} ===")
    s = invariance(lexmod, songshi_agg, yusong_agg, "Song 全宋诗 vs 御選宋詩")
    print(f"  Song  r_nmi={s['r_nmi']:.3f}  n_common={s['n_common']}  preserved={s['n_preserved']}/5")
    print("    dim:", [(c, (round(r,3) if r is not None else None)) for c, r in s["dim_r"]])
    t = invariance(lexmod, tang_agg, yuding_agg, "Tang QTS vs 御選唐詩")
    print(f"  Tang  r_nmi={t['r_nmi']:.3f}  n_common={t['n_common']}  preserved={t['n_preserved']}/5")
    print("    dim:", [(c, (round(r,3) if r is not None else None)) for c, r in t["dim_r"]])
    return s, t

s1, t1 = report(L, "M1 (hand lexicon)")
s2, t2 = report(LL, "Learned (Phase-1 induced)")

out = {
    "M1": {"song": s1, "tang": t1},
    "Learned": {"song": s2, "tang": t2},
    "n_authors": {"songshi": len(songshi_agg), "yusong": len(yusong_agg),
                  "tang": len(tang_agg), "yuding": len(yuding_agg)},
}
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, float): return round(o, 4)
    return o
with open(f"{BASE}/results_phase1b_song.json", "w", encoding="utf-8") as f:
    json.dump(clean(out), f, ensure_ascii=False, indent=2)
print("\nwrote results_phase1b_song.json")
