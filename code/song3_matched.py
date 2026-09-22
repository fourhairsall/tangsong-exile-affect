# -*- coding: utf-8 -*-
"""song3_matched.py — 两部宋人选本的**同作者集匹配对照**。

动机: 《御選宋詩》对照用了 256 位共有作者, 《宋詩鈔》对照用了 75 位。两组的作者
构成不同(宋詩鈔所收多为大诗人, 在《全宋诗》中篇幅大), 因此 A/S 不可直接比较。
本脚本把两个对照**限制在三向共有的同一批作者**上重算, 使"哪一部选本与总集更一致"
这一问题成为可比问题。同时给出两部选本在同一作者集上的 δ 向量与逐维 CI。

输出 results_song3_matched.json
"""
import glob
import json
import os
import sys

import numpy as np
import opencc
from scipy.stats import pearsonr, binomtest

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lexicon as L

cc = opencc.OpenCC("t2s")
EASE, BITTER = L.NET
EPS = 1e-9
MINC = 300
RNG = np.random.default_rng(20260917)


def _lp(rate, n):
    return (rate / 1000.0 * n + 0.5) / (n + 1.0)


def sig_nmi(per, n):
    pe, pb = _lp(per[EASE], n), _lp(per[BITTER], n)
    return 1000.0 / np.sqrt(n + 1.0) * np.sqrt(pe * (1 - pe) + pb * (1 - pb))


def sig_dim(rate, n):
    p = _lp(rate, n)
    return 1000.0 / np.sqrt(n + 1.0) * np.sqrt(p * (1 - p))


def wdelta(Y1, S1, Y2, S2):
    w = 1.0 / (S1 ** 2 + S2 ** 2 + EPS)
    d = Y2 - Y1
    return float(np.sum(w * d) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w)))


def aggregate(pattern):
    agg = {}
    for fp in glob.glob(pattern):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, list):
            for p in d:
                a = cc.convert(p.get("author", "") or "")
                t = cc.convert("".join(p.get("paragraphs", [])))
                if a and t:
                    agg.setdefault(a, []).append(t)
    return agg


def stats(agg):
    out = {}
    for a, texts in agg.items():
        txt = "".join(texts)
        if not txt:
            continue
        per, n = L.cat_per1k([txt])
        out[a] = {"nmi": L.net_index(per), "per": per, "n": n}
    return out


def report(A, B, authors, label, k):
    Y1 = np.array([A[a]["nmi"] for a in authors])
    Y2 = np.array([B[a]["nmi"] for a in authors])
    S1 = np.array([sig_nmi(A[a]["per"], A[a]["n"]) for a in authors]) * k
    S2 = np.array([sig_nmi(B[a]["per"], B[a]["n"]) for a in authors]) * k
    r_nmi = float(pearsonr(Y1, Y2)[0])
    dims = {}
    for d in L.CATS:
        x = np.array([A[a]["per"][d] for a in authors])
        y = np.array([B[a]["per"][d] for a in authors])
        gx = np.array([sig_dim(A[a]["per"][d], A[a]["n"]) for a in authors]) * k
        gy = np.array([sig_dim(B[a]["per"][d], B[a]["n"]) for a in authors]) * k
        dd, sed = wdelta(x, gx, y, gy)
        boot = [float(np.mean(y[i] - x[i]))
                for i in (RNG.choice(len(authors), len(authors), True) for _ in range(4000))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        dims[d] = {"r": round(float(pearsonr(x, y)[0]), 4),
                   "preserved": bool(abs(float(pearsonr(x, y)[0])) >= 0.5),
                   "delta": round(dd, 3), "ci": [round(dd - 1.96 * sed, 3),
                                                 round(dd + 1.96 * sed, 3)],
                   "boot_ci": [round(float(lo), 3), round(float(hi), 3)]}
    dn, sen = wdelta(Y1, S1, Y2, S2)
    bootn = [float(np.mean(Y2[i] - Y1[i]))
             for i in (RNG.choice(len(authors), len(authors), True) for _ in range(4000))]
    return {"label": label, "n_authors": len(authors),
            "A_nmi": round(r_nmi, 4),
            "S_dims": sum(1 for d in L.CATS if dims[d]["preserved"]),
            "delta_nmi": {"delta": round(dn, 3),
                          "ci": [round(dn - 1.96 * sen, 3), round(dn + 1.96 * sen, 3)],
                          "boot_ci": [round(float(np.percentile(bootn, 2.5)), 3),
                                      round(float(np.percentile(bootn, 97.5)), 3)],
                          "contains_zero": bool(dn - 1.96 * sen < 0 < dn + 1.96 * sen)},
            "per_dimension": dims}


def main():
    qss = stats(aggregate(f"{BASE}/data/songshi/poet.song.*.json"))
    yxs = stats(aggregate(f"{BASE}/data/yusong/poet.song_selection.json"))
    ssc = stats(aggregate(f"{BASE}/data/songchao/poet.song_songchao.json"))
    k = 1.0108
    try:
        k = json.load(open(f"{BASE}/results_song3_rep.json",
                           encoding="utf-8"))["design"]["clustering_inflation_k"]
    except Exception:
        pass

    tri = sorted(a for a in qss if a in yxs and a in ssc
                 and min(qss[a]["n"], yxs[a]["n"], ssc[a]["n"]) >= MINC)
    print(f"three-way shared authors (>= {MINC} chars each side): {len(tri)}")

    rep_yx = report(qss, yxs, tri, "YXSS_on_tri", k)
    rep_sc = report(qss, ssc, tri, "SSC_on_tri", k)
    print(f"  YXSS(on tri): A={rep_yx['A_nmi']} S={rep_yx['S_dims']}/5 "
          f"delta_nmi={rep_yx['delta_nmi']['delta']} {rep_yx['delta_nmi']['ci']}")
    print(f"  SSC (on tri): A={rep_sc['A_nmi']} S={rep_sc['S_dims']}/5 "
          f"delta_nmi={rep_sc['delta_nmi']['delta']} {rep_sc['delta_nmi']['ci']}")
    print("  per-dimension delta (matched):")
    dv1, dv2 = {}, {}
    for d in L.CATS:
        a = rep_yx["per_dimension"][d]
        b = rep_sc["per_dimension"][d]
        dv1[d] = a["delta"]
        dv2[d] = b["delta"]
        print(f"    {d:6} YXSS {a['delta']:+7.3f} {a['ci']} r={a['r']:.3f} | "
              f"SSC {b['delta']:+7.3f} {b['ci']} r={b['r']:.3f}")
    signs1 = [int(np.sign(dv1[d])) for d in L.CATS]
    signs2 = [int(np.sign(dv2[d])) for d in L.CATS]
    agree = sum(1 for a, b in zip(signs1, signs2) if a == b and a != 0)
    print(f"  sign agreement (matched): {agree}/5 ; "
          f"spearman={float(pearsonr([dv1[d] for d in L.CATS], [dv2[d] for d in L.CATS])[0]):.3f}")
    print(f"  sign test p={float(binomtest(agree, 5, 0.5, 'greater').pvalue):.5f}")

    # 作者级位移在两选本间的相关 (匹配集上)
    cross = {}
    for d in L.CATS:
        x = np.array([yxs[a]["per"][d] - qss[a]["per"][d] for a in tri])
        y = np.array([ssc[a]["per"][d] - qss[a]["per"][d] for a in tri])
        cross[d] = round(float(pearsonr(x, y)[0]), 4)
    xn = np.array([yxs[a]["nmi"] - qss[a]["nmi"] for a in tri])
    yn = np.array([ssc[a]["nmi"] - qss[a]["nmi"] for a in tri])
    cross["nmi"] = round(float(pearsonr(xn, yn)[0]), 4)
    print("  author-level shift correlation across anthologies (matched):", cross)

    # 挖改/阙字符审计: 选本文本中的阙字标记数量
    marks = 0
    total = 0
    for fp in glob.glob(f"{BASE}/data/songchao/KR4h0157_*.txt"):
        raw = open(fp, encoding="utf-8").read()
        total += len(raw)
        marks += raw.count("□") + raw.count("�")
    print(f"  missing-char markers in 宋詩鈔 text: {marks} over {total} bytes "
          f"({marks/total*1e6:.1f} per million)")

    json.dump({
        "note": "同作者集匹配对照: 两选本限制在三向共有的同一批作者上, 使 A/S/delta "
                "成为可比量。A 未匹配时不可直接比较(作者构成与 n 不同)。",
        "n_three_way_authors": len(tri),
        "authors": tri,
        "YXSS_matched": rep_yx,
        "SSC_matched": rep_sc,
        "sign_agreement_matched": {
            "n_sign_agree": agree, "n_dims": 5,
            "pearson_of_delta_vectors": round(float(pearsonr(
                [dv1[d] for d in L.CATS], [dv2[d] for d in L.CATS])[0]), 4),
            "sign_test_p": round(float(binomtest(agree, 5, 0.5, "greater").pvalue), 5),
        },
        "author_level_shift_correlation_matched": cross,
        "missing_char_markers": {"count": marks, "per_million": round(marks / total * 1e6, 2)},
    }, open(f"{BASE}/results_song3_matched.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)
    print("wrote results_song3_matched.json")


if __name__ == "__main__":
    main()
