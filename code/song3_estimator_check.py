# -*- coding: utf-8 -*-
"""song3_estimator_check.py — 估计量口径核查 (estimator check)。

question: song3_rep.py 的 `robustness_size_matched` 报出的量是
``per_QSS_ds - per_anth`` (即 QSS 减选本), 符号与论文主用的 δ = 选本 − 总集
相反, 直接引用极易误读。本脚本用**统一符号** (一律 选本 − 总集) 重算三种
口径, 用来回答一个实质问题: δ 的符号是否依赖每位作者的样本量?

  weighted_delta_full      逆方差加权 δ (论文主用量, 全量文本)
  unweighted_shift_full    未加权作者均值 (选本 − 总集, 全量文本)
  unweighted_shift_sizematch
                           未加权作者均值, 但把总集每位作者的文本随机截取到
                           与该选本等长 (200 次重复的均值与 95% 区间)

输出 results_song3_estimator.json
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lexicon as L
import phase1b_song as P
import song3_rep as R

REPS = 200
OUT = {}


def common(st, qss):
    return sorted(a for a in st
                  if a in qss and st[a]["n"] >= R.MINC and qss[a]["n"] >= R.MINC)


def run(tag, st, qss, k):
    comm = common(st, qss)
    rec = {"n_authors": len(comm), "per_dimension": {}, "nmi": {}}
    for d in L.CATS:
        x = np.array([qss[a]["per"][d] for a in comm])          # 总集
        y = np.array([st[a]["per"][d] for a in comm])           # 选本
        gx = np.array([R.sigma_dim(qss[a]["per"][d], qss[a]["n"]) for a in comm]) * k
        gy = np.array([R.sigma_dim(st[a]["per"][d], st[a]["n"]) for a in comm]) * k
        wv, _ = R.wdelta(x, gx, y, gy)
        rec["per_dimension"][d] = {
            "weighted_delta_full": round(float(wv), 3),
            "unweighted_shift_full": round(float(np.mean(y - x)), 3),
        }
    x = np.array([qss[a]["nmi"] for a in comm])
    y = np.array([st[a]["nmi"] for a in comm])
    gx = np.array([R.sigma_nmi(qss[a]["per"], qss[a]["n"]) for a in comm]) * k
    gy = np.array([R.sigma_nmi(st[a]["per"], st[a]["n"]) for a in comm]) * k
    wv, _ = R.wdelta(x, gx, y, gy)
    rec["nmi"] = {"weighted_delta_full": round(float(wv), 3),
                  "unweighted_shift_full": round(float(np.mean(y - x)), 3)}

    # size-matched: 把总集侧截到与选本等长, 未加权均值, 符号统一为 (选本 − 总集)
    rng = np.random.default_rng(20260917)
    acc = {d: [] for d in L.CATS}
    acc_nmi = []
    for _ in range(REPS):
        dd = {d: [] for d in L.CATS}
        dn = []
        for a in comm:
            target = st[a]["n"]
            txt = qss[a]["text"]
            if len(txt) > target:
                i0 = int(rng.integers(0, len(txt) - target + 1))
                txt = txt[i0:i0 + target]
            per, _ = L.cat_per1k([txt])
            for d in L.CATS:
                dd[d].append(st[a]["per"][d] - per[d])
            dn.append(st[a]["nmi"] - L.net_index(per))
        for d in L.CATS:
            acc[d].append(float(np.mean(dd[d])))
        acc_nmi.append(float(np.mean(dn)))
    for d in L.CATS:
        rec["per_dimension"][d]["unweighted_shift_sizematch"] = {
            "mean": round(float(np.mean(acc[d])), 3),
            "ci": [round(float(np.percentile(acc[d], 2.5)), 3),
                   round(float(np.percentile(acc[d], 97.5)), 3)],
        }
        f = rec["per_dimension"][d]["unweighted_shift_full"]
        m = rec["per_dimension"][d]["unweighted_shift_sizematch"]["mean"]
        rec["per_dimension"][d]["sizematch_minus_full"] = round(m - f, 3)
    rec["nmi"]["unweighted_shift_sizematch"] = {
        "mean": round(float(np.mean(acc_nmi)), 3),
        "ci": [round(float(np.percentile(acc_nmi, 2.5)), 3),
               round(float(np.percentile(acc_nmi, 97.5)), 3)],
    }
    rec["nmi"]["sizematch_minus_full"] = round(
        rec["nmi"]["unweighted_shift_sizematch"]["mean"]
        - rec["nmi"]["unweighted_shift_full"], 3)
    return rec


def main():
    print("aggregating ...")
    qss = R.stats_of(P.aggregate(f"{BASE}/data/songshi/poet.song.*.json"))
    yxs = R.stats_of(P.aggregate(f"{BASE}/data/yusong/poet.song_selection.json"))
    ssc = R.stats_of(P.aggregate(f"{BASE}/data/songchao/poet.song_songchao.json"))
    k, _ = R.load_k()
    if k is None:
        k = 1.0108
    OUT["note"] = ("符号统一为 (选本 − 总集)。song3_rep.py 的 "
                   "robustness_size_matched 存的是 (总集 − 选本), 即本表之相反数。")
    OUT["YXSS"] = run("YXSS", yxs, qss, k)
    OUT["SSC"] = run("SSC", ssc, qss, k)
    json.dump(OUT, open(f"{BASE}/results_song3_estimator.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    for tag in ("YXSS", "SSC"):
        print(f"\n[{tag}] n={OUT[tag]['n_authors']}")
        for d in L.CATS:
            r = OUT[tag]["per_dimension"][d]
            print(f"  {d:6} weighted {r['weighted_delta_full']:+.3f} | "
                  f"unweighted {r['unweighted_shift_full']:+.3f} | "
                  f"size-matched {r['unweighted_shift_sizematch']['mean']:+.3f} "
                  f"(Δ {r['sizematch_minus_full']:+.3f})")
        r = OUT[tag]["nmi"]
        print(f"  NMI    weighted {r['weighted_delta_full']:+.3f} | "
              f"unweighted {r['unweighted_shift_full']:+.3f} | "
              f"size-matched {r['unweighted_shift_sizematch']['mean']:+.3f} "
              f"(Δ {r['sizematch_minus_full']:+.3f})")
    print("\nwrote results_song3_estimator.json")


if __name__ == "__main__":
    main()
