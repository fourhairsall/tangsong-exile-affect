# -*- coding: utf-8 -*-
"""xval_yuding.py — cross-edition consistency check (异源偏差消解).

Compare the net-mind index (净心指数) of the 4 Tang authors as measured from
TWO independent Tang-poetry editions:
  (A) 全唐诗  — chinese-poetry (data/tang), already in results_v4 shi_stat
  (B) 御定全唐詩 — kanripo/chinese-poetry (data/yuding), 2nd independent edition
Both normalized to simplified (OpenCC t2s) before lexicon counting.
If the two editions yield close 净心指数 despite different poem subsets, the
cross-source (异源) bias is demonstrably controlled.
"""
import json, glob, sys, os
from collections import defaultdict
sys.path.insert(0, "D:/databuddy/2026-09-14-10-59-40/tangsong_research")
import opencc
import lexicon as L
cc = opencc.OpenCC('t2s')

BASE = "D:/databuddy/2026-09-14-10-59-40/tangsong_research"
TANG = ["柳宗元","刘禹锡","韩愈","白居易"]

# ---- (B) 御定全唐詩 ----
texts = defaultdict(list)
nvol = 0
for fp in glob.glob(f"{BASE}/data/yuding/*.json"):
    nvol += 1
    d = json.load(open(fp, encoding="utf-8"))
    if not isinstance(d, list): 
        continue
    for p in d:
        a = cc.convert(p.get("author", "") or "")
        if a in TANG:
            txt = cc.convert("".join(p.get("paragraphs", [])))
            if txt.strip():
                texts[a].append(txt)

yuding_stat = {}
for a in TANG:
    if texts[a]:
        per, nch = L.cat_per1k(texts[a])
        yuding_stat[a] = dict(per)
        yuding_stat[a]["__n"] = len(texts[a])
        yuding_stat[a]["__chars"] = nch
    else:
        yuding_stat[a] = {k: 0.0 for k in L.LEX}
        yuding_stat[a]["__n"] = 0
        yuding_stat[a]["__chars"] = 0
    yuding_stat[a]["净心指数"] = L.net_index(yuding_stat[a])

# ---- (A) 全唐诗 (from results_v4 shi_stat) ----
R = json.load(open(f"{BASE}/results_v4.json", encoding="utf-8"))
shi_stat = R["shi_stat"]

# ---- compare ----
print(f"御定全唐詩 volumes scanned: {nvol}")
print(f"{'作者':6} | {'全唐诗 n/chars':>14} | {'御定 n/chars':>14} | {'全唐诗净心':>10} | {'御定净心':>10} | {'Δ净心':>8}")
rows = []
for a in TANG:
    qa = shi_stat[a]; yb = yuding_stat[a]
    d = yb["净心指数"] - qa["净心指数"]
    rows.append({"author":a,
                 "qt_n":qa.get("n"),"qt_chars":qa.get("chars"),
                 "yd_n":yb["__n"],"yd_chars":yb["__chars"],
                 "qt_net":round(qa["净心指数"],3),"yd_net":round(yb["净心指数"],3),
                 "delta":round(d,3)})
    print(f"{a:6} | {str(qa.get('n'))+'/'+str(qa.get('chars')):>14} | {str(yb['__n'])+'/'+str(yb['__chars']):>14} | {qa['净心指数']:>10.2f} | {yb['净心指数']:>10.2f} | {d:>8.2f}")

import numpy as np
deltas = [r["delta"] for r in rows]
print(f"\nMean|Δ净心| = {np.mean(np.abs(deltas)):.3f}   max|Δ| = {np.max(np.abs(deltas)):.3f}")
# category-level correlation across the 4 authors (5 dims)
cats = list(L.LEX.keys())
qmat = np.array([[shi_stat[a]["cat_per1k"][c] for c in cats] for a in TANG], float)
ymat = np.array([[yuding_stat[a][c] for c in cats] for a in TANG], float)
from numpy import corrcoef
cr = corrcoef(qmat.flatten(), ymat.flatten())[0,1]
print(f"Category-frequency correlation (全唐诗 vs 御定, 4×5=20 cells) = {cr:.3f}")

out = {"n_volumes": nvol, "rows": rows,
       "mean_abs_delta_net": round(float(np.mean(np.abs(deltas))),4),
       "max_abs_delta_net": round(float(np.max(np.abs(deltas))),4),
       "cat_corr": round(float(cr),4)}
json.dump(out, open(f"{BASE}/results_v4_xval.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote results_v4_xval.json")
