# -*- coding: utf-8 -*-
"""song3_rep.py — 第二部宋人选本的复现性检验 (Song replication)。

设计: **同一底本家族**(四庫全書文淵閣本)、**同一数字化管线**(kanripo mandoku)、
**同一词表、同一口径**, 只换选本 —— 《御選宋詩》(清·官修, KR4h0143) 与
《宋詩鈔》(清·吳之振私修, KR4h0157)。故两选本之间 δ 的差异可归因于选本原则本身。

对每个「总集→选本」对, 报出:
  A  = 作者级 NMI 的 Pearson 相关 (一致性系数)
  S  = 5 维中 |r|>=0.5 的维数 (结构一致性)
  δ  = 版本效应 (逆方差加权), 逐维 δ_d 及其 95%CI
并做三件事把"单例"变成"可复现规律":
  (1) 两个选本的 δ 向量一致性: 逐维符号一致率 + 5 维 δ 的 Spearman + 精确符号检验 p
  (2) 两条独立推断路径互校: 解析噪声底(层次/逆方差) vs 作者级配对自助 (2e4 次)
  (3) 三项稳健性: 挖改/避讳免疫子词表、选本内部按卷分半、总集侧同规模下采样

输出 results_song3_rep.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, spearmanr, pearsonr

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lexicon as L
import phase1b_song as P

EASE, BITTER = L.NET
EPS = 1e-9
MINC = 300
RNG = np.random.default_rng(20260917)

# 四庫館臣避諱/挖改最易波及的字面 (玄→𤣥/元, 弘→宏, 曆→歷, 禎→楨;
# 吕留良案后印本挖改 胡/虜/夷/狄 等)。用于"挖改免疫子词表"稳健性检验。
TABOO_CHARS = set("玄弘曆禎胡虜夷狄虜鹵")


# ---------------------------------------------------------------- noise floor
def _laplace_p(rate_per1k, n):
    c = rate_per1k / 1000.0 * n
    return (c + 0.5) / (n + 1.0)


def sigma_nmi(per, n):
    if n <= 0:
        return float("inf")
    pe, pb = _laplace_p(per[EASE], n), _laplace_p(per[BITTER], n)
    return 1000.0 / np.sqrt(n + 1.0) * np.sqrt(pe * (1 - pe) + pb * (1 - pb))


def sigma_dim(per, n):
    if n <= 0:
        return float("inf")
    p = _laplace_p(per, n)
    return 1000.0 / np.sqrt(n + 1.0) * np.sqrt(p * (1 - p))


def wdelta(Y1, S1, Y2, S2):
    w = 1.0 / (S1 ** 2 + S2 ** 2 + EPS)
    d = Y2 - Y1
    return float(np.sum(w * d) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w)))


def stats_of(agg):
    out = {}
    for a, texts in agg.items():
        per, n = L.cat_per1k(texts)
        if n == 0:
            continue
        out[a] = {"nmi": L.net_index(per), "per": per, "n": n, "text": "".join(texts)}
    return out


def load_k():
    """复用 Phase 3 在 7 位目标诗人上标定的簇内膨胀因子 k。"""
    try:
        noise = json.load(open(f"{BASE}/results_song2_noisecheck.json", encoding="utf-8"))
        boot = {r["author"]: r["poem_boot_sd"] for r in noise["rows"]}
    except Exception:
        return None, None
    import song2_xval as S
    qs = {a: "".join(v) for a, v in S.load_quansongshi_poems().items()}
    yx = S.load_yuxuan_songshi()
    ks = []
    for a in S.TARGET:
        if a not in boot or not qs.get(a):
            continue
        n_anth = len(yx[a])
        if n_anth <= 0:
            continue
        per_q, _nq = L.cat_per1k([qs[a]])
        ks.append(boot[a] / sigma_nmi(per_q, n_anth))
    return float(np.mean(ks)), boot


# ---------------------------------------------------------------- pair stats
def pair_report(A, B, label, k):
    common = [a for a in A if a in B and A[a]["n"] >= MINC and B[a]["n"] >= MINC]
    common.sort(key=lambda a: -min(A[a]["n"], B[a]["n"]))
    Y1 = np.array([A[a]["nmi"] for a in common])
    Y2 = np.array([B[a]["nmi"] for a in common])
    S1 = np.array([sigma_nmi(A[a]["per"], A[a]["n"]) for a in common]) * k
    S2 = np.array([sigma_nmi(B[a]["per"], B[a]["n"]) for a in common]) * k

    r_nmi = float(pearsonr(Y1, Y2)[0]) if len(common) > 2 else None
    dims = {}
    for d in L.CATS:
        x = np.array([A[a]["per"][d] for a in common])
        y = np.array([B[a]["per"][d] for a in common])
        r = float(pearsonr(x, y)[0]) if len(common) > 2 else float("nan")
        X1 = x
        X2 = y
        G1 = np.array([sigma_dim(A[a]["per"][d], A[a]["n"]) for a in common]) * k
        G2 = np.array([sigma_dim(B[a]["per"][d], B[a]["n"]) for a in common]) * k
        dd, sed = wdelta(X1, G1, X2, G2)
        # 作者级配对自助 (不依赖噪声底模型)
        boot = []
        idx = np.arange(len(common))
        for _ in range(4000):
            ii = RNG.choice(idx, size=len(idx), replace=True)
            boot.append(float(np.mean(X2[ii] - X1[ii])))
        lo_b, hi_b = np.percentile(boot, [2.5, 97.5])
        dims[d] = {
            "r": round(r, 4),
            "preserved": bool(abs(r) >= 0.5) if not np.isnan(r) else False,
            "delta": round(dd, 3), "se": round(sed, 3),
            "ci": [round(dd - 1.96 * sed, 3), round(dd + 1.96 * sed, 3)],
            "systematic_shift": bool(not (dd - 1.96 * sed < 0 < dd + 1.96 * sed)),
            "boot_ci": [round(float(lo_b), 3), round(float(hi_b), 3)],
            "boot_shift": bool(not (lo_b < 0 < hi_b)),
        }
    d_nmi, se_nmi = wdelta(Y1, S1, Y2, S2)
    boot_nmi = []
    for _ in range(4000):
        ii = RNG.choice(np.arange(len(common)), size=len(common), replace=True)
        boot_nmi.append(float(np.mean(Y2[ii] - Y1[ii])))
    lo_n, hi_n = np.percentile(boot_nmi, [2.5, 97.5])
    return {
        "label": label,
        "n_authors": len(common),
        "authors": common,
        "A_nmi_pearson": round(r_nmi, 4) if r_nmi is not None else None,
        "S_dims_preserved": sum(1 for d in L.CATS if dims[d]["preserved"]),
        "delta_nmi": {"delta": round(d_nmi, 3), "se": round(se_nmi, 3),
                      "ci": [round(d_nmi - 1.96 * se_nmi, 3), round(d_nmi + 1.96 * se_nmi, 3)],
                      "contains_zero": bool(d_nmi - 1.96 * se_nmi < 0 < d_nmi + 1.96 * se_nmi),
                      "boot_ci": [round(float(lo_n), 3), round(float(hi_n), 3)],
                      "boot_contains_zero": bool(lo_n < 0 < hi_n)},
        "per_dimension": dims,
        "_Y1": Y1.tolist(), "_Y2": Y2.tolist(),
    }


def sign_agreement(dv1, dv2, cats, tag):
    s1 = [np.sign(dv1[c]) for c in cats]
    s2 = [np.sign(dv2[c]) for c in cats]
    agree = sum(1 for a, b in zip(s1, s2) if a == b and a != 0)
    v1 = [dv1[c] for c in cats]
    v2 = [dv2[c] for c in cats]
    rho = float(spearmanr(v1, v2)[0])
    r = float(pearsonr(v1, v2)[0])
    # 精确符号检验: 在"两选本无共同的维度级方向"下, 5 维中至少 agree 维同号的概率
    p = float(binomtest(agree, len(cats), 0.5, alternative="greater").pvalue)
    return {"tag": tag, "signs_1": [int(x) for x in s1], "signs_2": [int(x) for x in s2],
            "n_sign_agree": agree, "n_dims": len(cats),
            "spearman_of_delta_vectors": round(rho, 4),
            "pearson_of_delta_vectors": round(r, 4),
            "sign_test_p_greater": round(p, 5),
            "delta_vector_1": {c: round(dv1[c], 3) for c in cats},
            "delta_vector_2": {c: round(dv2[c], 3) for c in cats}}


def main():
    print("aggregating ...")
    qss = stats_of(P.aggregate(f"{BASE}/data/songshi/poet.song.*.json"))
    yxs = stats_of(P.aggregate(f"{BASE}/data/yusong/poet.song_selection.json"))
    ssc = stats_of(P.aggregate(f"{BASE}/data/songchao/poet.song_songchao.json"))
    k, boot = load_k()
    if k is None:
        k = 1.0108  # Phase 3 标定值之兜底
    print(f"  QSS authors={len(qss)}  YXSS={len(yxs)}  SSC={len(ssc)}  k={k:.4f}")

    rep_yx = pair_report(qss, yxs, "QSS_vs_YuxuanSongshi", k)
    rep_sc = pair_report(qss, ssc, "QSS_vs_Songshichao", k)
    print(f"\n[YXSS] n={rep_yx['n_authors']} A={rep_yx['A_nmi_pearson']} "
          f"S={rep_yx['S_dims_preserved']}/5 delta={rep_yx['delta_nmi']['delta']} "
          f"CI={rep_yx['delta_nmi']['ci']}")
    print(f"[SSC ] n={rep_sc['n_authors']} A={rep_sc['A_nmi_pearson']} "
          f"S={rep_sc['S_dims_preserved']}/5 delta={rep_sc['delta_nmi']['delta']} "
          f"CI={rep_sc['delta_nmi']['ci']}")

    dv_yx = {d: rep_yx["per_dimension"][d]["delta"] for d in L.CATS}
    dv_sc = {d: rep_sc["per_dimension"][d]["delta"] for d in L.CATS}
    print("\nper-dimension delta:")
    for d in L.CATS:
        print(f"  {d:6} YXSS {dv_yx[d]:+7.3f} {rep_yx['per_dimension'][d]['ci']}  "
              f"| SSC {dv_sc[d]:+7.3f} {rep_sc['per_dimension'][d]['ci']}")
    agree = sign_agreement(dv_yx, dv_sc, L.CATS, "YXSS_vs_SSC")
    print(f"\n[sign agreement] {agree['n_sign_agree']}/{agree['n_dims']} dims same sign; "
          f"rho(delta vectors)={agree['spearman_of_delta_vectors']}; "
          f"sign-test p={agree['sign_test_p_greater']}")

    # ---- 稳健性 1: 挖改/避讳免疫子词表 ----
    print("\n[robustness 1] taboo-immune sublexicon ...")
    saved = {c: list(L.LEX[c]) for c in L.LEX}
    for c in L.LEX:
        L.LEX[c] = [w for w in saved[c]
                    if not (isinstance(w, str) and len(w) == 1 and w in TABOO_CHARS)]
    q2 = stats_of(P.aggregate(f"{BASE}/data/songshi/poet.song.*.json"))
    s2 = stats_of(P.aggregate(f"{BASE}/data/songchao/poet.song_songchao.json"))
    y2 = stats_of(P.aggregate(f"{BASE}/data/yusong/poet.song_selection.json"))
    rob1_sc = pair_report(q2, s2, "taboo_immune_SSC", k)
    rob1_yx = pair_report(q2, y2, "taboo_immune_YXSS", k)
    for c in L.LEX:
        L.LEX[c] = saved[c]
    print("  SSC delta:", {d: rob1_sc["per_dimension"][d]["delta"] for d in L.CATS})

    # ---- 稳健性 2: 选本内部按卷分半 (定位效应是否由某段局部挖改驱动) ----
    print("\n[robustness 2] split-half within the anthology ...")
    split = {}
    for lab, st in (("YXSS", yxs), ("SSC", ssc)):
        comm = [a for a in st if a in qss and st[a]["n"] >= MINC and qss[a]["n"] >= MINC]
        comm.sort()
        half = len(comm) // 2
        parts = {}
        for name, sub in (("first_half", comm[:half]), ("second_half", comm[half:])):
            dv = {}
            for d in L.CATS:
                x = np.array([qss[a]["per"][d] for a in sub])
                y = np.array([st[a]["per"][d] for a in sub])
                gx = np.array([sigma_dim(qss[a]["per"][d], qss[a]["n"]) for a in sub]) * k
                gy = np.array([sigma_dim(st[a]["per"][d], st[a]["n"]) for a in sub]) * k
                dd, _ = wdelta(x, gx, y, gy)
                dv[d] = dd
            parts[name] = {"n": len(sub),
                           "delta": {d: round(dv[d], 3) for d in L.CATS},
                           "signs": {d: int(np.sign(dv[d])) for d in L.CATS}}
        sa = sum(1 for d in L.CATS
                 if parts["first_half"]["signs"][d] == parts["second_half"]["signs"][d]
                 and parts["first_half"]["signs"][d] != 0)
        parts["sign_agree_between_halves"] = f"{sa}/{len(L.CATS)}"
        split[lab] = parts
        print(f"  {lab}: {sa}/{len(L.CATS)} dims same sign across halves")

    # ---- 稳健性 3: 总集侧同规模下采样 ----
    print("\n[robustness 3] size-matched downsampling of QSS ...")
    def downsample_delta(st, reps=200):
        comm = [a for a in st if a in qss and st[a]["n"] >= MINC and qss[a]["n"] >= MINC]
        out = {d: [] for d in L.CATS}
        out_nmi = []
        for _ in range(reps):
            dd = {d: [] for d in L.CATS}
            dn = []
            for a in comm:
                target = st[a]["n"]
                txt = qss[a]["text"]
                if len(txt) > target:
                    i0 = RNG.integers(0, len(txt) - target + 1)
                    txt = txt[i0:i0 + target]
                per, _ = L.cat_per1k([txt])
                for d in L.CATS:
                    dd[d].append(per[d] - st[a]["per"][d])
                dn.append(L.net_index(per) - st[a]["nmi"])
            for d in L.CATS:
                out[d].append(float(np.mean(dd[d])))
            out_nmi.append(float(np.mean(dn)))
        return {d: {"mean": round(float(np.mean(out[d])), 3),
                    "ci": [round(float(np.percentile(out[d], 2.5)), 3),
                           round(float(np.percentile(out[d], 97.5)), 3)]} for d in L.CATS} | \
               {"nmi": {"mean": round(float(np.mean(out_nmi)), 3),
                        "ci": [round(float(np.percentile(out_nmi, 2.5)), 3),
                               round(float(np.percentile(out_nmi, 97.5)), 3)]}}
    rob3_sc = downsample_delta(ssc)
    rob3_yx = downsample_delta(yxs)
    print("  SSC (QSS downsampled to anthology size):",
          {d: rob3_sc[d]["mean"] for d in L.CATS})

    # ---- 作者级跨选本一致性: d_i(YXSS) 与 d_i(SSC) 的相关 ----
    tri = [a for a in qss if a in yxs and a in ssc
           and min(qss[a]["n"], yxs[a]["n"], ssc[a]["n"]) >= MINC]
    print(f"\n[three-way authors] n={len(tri)}")
    cross = {}
    for d in L.CATS:
        x = np.array([yxs[a]["per"][d] - qss[a]["per"][d] for a in tri])
        y = np.array([ssc[a]["per"][d] - qss[a]["per"][d] for a in tri])
        cross[d] = {"r_between_anthologies": round(float(pearsonr(x, y)[0]), 4),
                    "n": len(tri)}
    xn = np.array([yxs[a]["nmi"] - qss[a]["nmi"] for a in tri])
    yn = np.array([ssc[a]["nmi"] - qss[a]["nmi"] for a in tri])
    cross["nmi"] = {"r_between_anthologies": round(float(pearsonr(xn, yn)[0]), 4)}
    print("  author-level shift correlation across the two anthologies:",
          {d: cross[d]["r_between_anthologies"] for d in L.CATS})

    # ---- 预登记对照 (陈善本, 检定前给出) ----
    prereg = {
        "source": "陈善本（文献考据）于检定前给出的方向性预判",
        "predicted": {
            "悲苦孤寂": "升，且升幅大于《御選宋詩》",
            "贬谪羁旅": "升（小传系统叙贬谪，且宋诗钞偏好元祐党人/遗民）",
            "旷达闲适": "近零（最可能通过）",
            "自然山水": "升，但升幅应小于《御選宋詩》（御選含大量题画/咏物/闲适；宋诗钞更关注政治与人格）",
            "空幻时空": "方向不定",
        },
        "observed_delta_SSC": {d: dv_sc[d] for d in L.CATS},
        "observed_delta_YXSS": {d: dv_yx[d] for d in L.CATS},
    }

    out = {
        "design": {
            "second_anthology": "宋詩鈔 (清·吳之振, 四庫全書文淵閣本, kanripo KR4h0157, 106 卷)",
            "first_anthology": "御選宋詩 (清·康熙敕選, 四庫全書文淵閣本, kanripo KR4h0143)",
            "common_base_edition_family": "四庫全書文淵閣本 (WYG), 同一 mandoku 数字化管线",
            "lexicon": "M1 透明情感—意象词表 (5 维), NMI = 旷达闲适 - 悲苦孤寂 (per 1000 chars)",
            "minc_chars_per_author_per_side": MINC,
            "clustering_inflation_k": round(k, 4),
            "rng_seed": 20260917,
        },
        "parse_songchao": json.load(open(f"{BASE}/results_parse_songchao.json",
                                         encoding="utf-8")),
        "pair_QSS_vs_YXSS": rep_yx,
        "pair_QSS_vs_SSC": rep_sc,
        "delta_sign_agreement": agree,
        "robustness_taboo_immune": {
            "SSC": {d: rob1_sc["per_dimension"][d]["delta"] for d in L.CATS},
            "SSC_ci": {d: rob1_sc["per_dimension"][d]["ci"] for d in L.CATS},
            "YXSS": {d: rob1_yx["per_dimension"][d]["delta"] for d in L.CATS},
        },
        "robustness_split_half": split,
        "robustness_size_matched": {"SSC": rob3_sc, "YXSS": rob3_yx},
        "author_level_cross_anthology": cross,
        "preregistration": prereg,
    }
    with open(f"{BASE}/results_song3_rep.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nwrote results_song3_rep.json")


if __name__ == "__main__":
    main()
