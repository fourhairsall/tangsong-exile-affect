# -*- coding: utf-8 -*-
"""verify_punct_fix.py -- PRE-PIPELINE gate for the P0 punctuation fix.

Computes, for every source, the per-author Net-Mind Index under BOTH
denominators:
  OLD : rate = count / len(text_incl_punctuation) * 1000
  NEW : rate = count / len(content_chars_only)   * 1000   (lexicon.py P0 fix)
and reports the four cross-edition consistency coefficients A (Pearson of the
NMI vectors) and the 12-author NMI table under each convention, so we can see
exactly how the correction moves the paper's headline numbers before touching
the pipeline.
"""
import glob, json, os
import numpy as np
import opencc
import lexicon as L

cc = opencc.OpenCC("t2s")
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MINC = 300

SOURCES = {
    "QTS":  ("%s/tang/poet.tang.*.json" % DATA, "glob"),
    "YD":   ("%s/yuding/*.json" % DATA, "glob"),
    "QSS":  ("%s/songshi/poet.song.*.json" % DATA, "glob"),
    "YXSS": ("%s/yusong/poet.song_selection.json" % DATA, "file"),
    "SSC":  ("%s/songchao/poet.song_songchao.json" % DATA, "file"),
}

def load(pattern, kind):
    acc = {}
    if kind == "file":
        d = json.load(open(pattern, encoding="utf-8"))
        if isinstance(d, list):
            for p in d:
                a = cc.convert(p.get("author", "") or "")
                t = "".join(p.get("paragraphs", []))
                if a and t:
                    acc.setdefault(a, []).append(cc.convert(t))
    else:
        for fp in sorted(glob.glob(pattern)):
            try:
                d = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(d, list):
                for p in d:
                    a = cc.convert(p.get("author", "") or "")
                    t = "".join(p.get("paragraphs", []))
                    if a and t:
                        acc.setdefault(a, []).append(cc.convert(t))
    return acc

def nmi_old(texts):
    """OLD denominator (punctuation included)."""
    nch = sum(len(t) for t in texts)
    if nch == 0:
        return None
    per = L.norm(L.count_cat("".join(texts)), nch)
    return L.net_index(per)

def nmi_new(texts):
    """NEW denominator (punctuation excluded) via lexicon.cat_per1k."""
    per, n = L.cat_per1k(texts)
    return L.net_index(per) if n else None

AGG_OLD, AGG_NEW = {}, {}
for name, (pat, kind) in SOURCES.items():
    agg = load(pat, kind)
    AGG_OLD[name] = {a: nmi_old(v) for a, v in agg.items() if nmi_old(v) is not None}
    AGG_NEW[name] = {a: nmi_new(v) for a, v in agg.items() if nmi_new(v) is not None}
    # also report mean punctuation fraction per source
    tot = pn = 0
    for v in agg.values():
        for t in v:
            tot += len(t); pn += len(L.strip_punct(t))
    print(f"  {name:5s} authors={len(agg):4d}  punct_fraction={1-pn/max(tot,1):.4f}")

def A_of(pair):
    sa, sb = pair
    a_old = {k: v for k, v in AGG_OLD[sa].items() if k in AGG_OLD[sb]}
    b_old = {k: v for k, v in AGG_OLD[sb].items() if k in AGG_OLD[sa]}
    a_new = {k: v for k, v in AGG_NEW[sa].items() if k in AGG_NEW[sb]}
    b_new = {k: v for k, v in AGG_NEW[sb].items() if k in AGG_NEW[sa]}
    common_old = sorted(set(a_old) & set(b_old))
    common_new = sorted(set(a_new) & set(b_new))
    xo = np.array([a_old[k] for k in common_old]); yo = np.array([b_old[k] for k in common_old])
    xn = np.array([a_new[k] for k in common_new]); yn = np.array([b_new[k] for k in common_new])
    ro = float(np.corrcoef(xo, yo)[0, 1]) if len(xo) > 2 else float("nan")
    rn = float(np.corrcoef(xn, yn)[0, 1]) if len(xn) > 2 else float("nan")
    return ro, len(common_old), rn, len(common_new)

print("\n=== Cross-edition consistency A (NMI Pearson) : OLD vs NEW denom ===")
pairs = [("QTS", "YD"), ("QSS", "YXSS"), ("QSS", "SSC"), ("YXSS", "SSC")]
for sa, sb in pairs:
    ro, no, rn, nn = A_of((sa, sb))
    print(f"  {sa:4s} vs {sb:4s} : OLD A={ro:+.4f} (n={no})   NEW A={rn:+.4f} (n={nn})")

print("\n=== 12-author Net-Mind Index : OLD vs NEW (Tang=QTS, Song=QSS) ===")
TANG4 = ["柳宗元", "刘禹锡", "韩愈", "白居易"]
SONG8 = ["苏轼", "欧阳修", "黄庭坚", "秦观", "王禹偁", "范仲淹", "辛弃疾", "陆游"]
order = TANG4 + SONG8
print(f"  {'author':6s}  OLD     NEW    Δ")
for a in order:
    o = AGG_OLD["QTS" if a in TANG4 else "QSS"].get(a)
    n_ = AGG_NEW["QTS" if a in TANG4 else "QSS"].get(a)
    if o is None or n_ is None:
        print(f"  {a:6s}  missing"); continue
    print(f"  {a:6s}  {o:+.3f}  {n_:+.3f}  {n_-o:+.3f}")
