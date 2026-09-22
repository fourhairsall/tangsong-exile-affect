# -*- coding: utf-8 -*-
r"""cross_edition_points.py -- emit the PER-AUTHOR Net-Mind Index pairs for the
two population-scale cross-edition comparisons, so the paper can plot the
invariance continuum directly (scatter, not bars):

  Tang : QTS (data/tang)     vs  YD   (data/yuding)   -- same work
  Song : QSS (data/songshi)  vs  YXS  (data/yusong)   -- corpus vs selection

Same inclusion rule as phase1b_song.py: authors present in BOTH editions with
>= MINC characters in each.  Output: cross_edition_points.json
"""
import json, glob, os
import numpy as np
import opencc
import lexicon as L

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MINC = 300
cc = opencc.OpenCC("t2s")


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


def rates(texts):
    return L.cat_per1k(texts)


def compare(A, B, label):
    pts = []
    for a in A:
        if a not in B:
            continue
        pa, na = rates(A[a]); pb, nb = rates(B[a])
        if na >= MINC and nb >= MINC:
            pts.append({"author": a, "x": L.net_index(pa), "y": L.net_index(pb),
                        "nx": na, "ny": nb})
    xs = np.array([p["x"] for p in pts]); ys = np.array([p["y"] for p in pts])
    r = float(np.corrcoef(xs, ys)[0, 1]) if len(pts) > 2 else float("nan")
    dimr = []
    for c in L.CATS:
        xs2 = [rates(A[p["author"]])[0][c] for p in pts]
        ys2 = [rates(B[p["author"]])[0][c] for p in pts]
        dimr.append([c, round(float(np.corrcoef(xs2, ys2)[0, 1]), 4)])
    S = sum(1 for _, rr in dimr if abs(rr) >= 0.5)
    print(f"{label}: n={len(pts)}  r={r:.4f}  S={S}/5")
    for c, rr in dimr:
        print(f"    {c}: r={rr:+.3f}")
    return {"label": label, "n": len(pts), "r_nmi": round(r, 4), "S": S,
            "dim_r": dimr, "points": pts}


if __name__ == "__main__":
    out = {}
    print("aggregating Tang (QTS, YD) ...")
    A = aggregate(f"{DATA}/tang/poet.tang.*.json"); B = aggregate(f"{DATA}/yuding/*.json")
    out["Tang_same_work"] = compare(A, B, "Tang QTS vs YD (same work)")

    print("aggregating Song (QSS, YXS) ...")
    A = aggregate(f"{DATA}/songshi/poet.song.*.json"); B = aggregate(f"{DATA}/yusong/poet.song_selection.json")
    out["Song_corpus_vs_selection"] = compare(A, B, "Song QSS vs YuXuan (selection)")

    json.dump(out, open(os.path.join(BASE, "cross_edition_points.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote cross_edition_points.json")
