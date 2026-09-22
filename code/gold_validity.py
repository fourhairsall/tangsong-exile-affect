# -*- coding: utf-8 -*-
"""gold_validity.py — 人工标注金标的编码信度与效标效度。

输入: annotations/annotatorA_tang.txt, annotations/annotatorB_song.txt
      (两位标注员**独立**标注同一批 160 首; 盲标, 不见词表输出)
      gold_sample.json (私有 key: 作者/载体/词表读值)

三步:
  (1) 编码信度: 逐维二次加权 Cohen's kappa、序数 Krippendorff alpha、ICC(2,1);
      金标 NMI = mean(Quan)-mean(Bei) 的 ICC。
  (2) 效标效度: 金标(两员均值) 与 词表读出 的 Spearman rho, **按作者聚类自助** CI;
      逐维; 并给出**平凡基线**(仅用诗长)与**长度偏相关**作对照。
  (3) 对照检验: 只取情感两维(金标 Quan-Bei vs 词表 旷达-悲苦), 即 H2 的核心检验;
      另在十位贬谪队列作者上做作者级秩比较。

输出 results_gold_validity.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, pearsonr

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lexicon as L

DIMS = ["Bei", "Quan", "Man", "Ziran", "Konghuan"]
CATMAP = {"Bei": "悲苦孤寂", "Quan": "贬谪羁旅", "Man": "旷达闲适",
          "Ziran": "自然山水", "Konghuan": "空幻时空"}
RNG = np.random.default_rng(20260917)


def load_ann(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("pid"):
            continue
        p = line.split(",")
        if len(p) != 6:
            continue
        d[p[0]] = {k: int(v) for k, v in zip(DIMS, p[1:])}
    return d


def qwk(a, b, k=4):
    """二次加权 Cohen's kappa (有序 0..k-1)。"""
    a = np.asarray(a, int); b = np.asarray(b, int)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    n = O.sum()
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    ra = O.sum(1) / n; rb = O.sum(0) / n
    E = np.outer(ra, rb) * n
    num = (W * O).sum(); den = (W * E).sum()
    return float(1 - num / den) if den > 0 else float("nan")


def krippendorff_ordinal(M, k=4):
    """序数 Krippendorff alpha (标准巧合矩阵形式)。

    delta^2_ordinal(c,k) = ( sum_{g=c..k} n_g - (n_c + n_k)/2 )^2
    巧合矩阵 O_ck 按每个分析单元的编码者数 m_u 归一 (1/(m_u-1))。
    """
    M = np.asarray(M, float)
    n_units, m_max = M.shape
    flat = M[~np.isnan(M)]
    counts = {g: float((flat == g).sum()) for g in range(k)}
    tot = float(len(flat))

    def d2(c, kk):
        lo, hi = int(min(c, kk)), int(max(c, kk))
        s = sum(counts[g] for g in range(lo, hi + 1)) - (counts[int(c)] + counts[int(kk)]) / 2.0
        return s ** 2

    O = np.zeros((k, k))
    for row in M:
        obs = row[~np.isnan(row)].astype(int)
        m_u = len(obs)
        if m_u < 2:
            continue
        for a in obs:
            for b in obs:
                if a != b or True:
                    O[a, b] += 1.0 / (m_u - 1)
    n_c = O.sum(axis=1)
    E = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            E[i, j] = n_c[i] * n_c[j] / (tot - 1) if i != j else n_c[i] * (n_c[i] - 1) / (tot - 1)
    W = np.array([[d2(i, j) for j in range(k)] for i in range(k)])
    Do, De = (W * O).sum(), (W * E).sum()
    return float(1 - Do / De) if De > 0 else float("nan")


def lexicon_reliability(key, pids):
    """逐首诗的词表读出信度 (劈半法)。

    对每首诗按字符奇/偶位置劈成两个半篇, 各自计算 per-1000 频次与 NMI,
    跨诗求两半的 Spearman 相关 r_half, 再以 Spearman-Brown 公式
        r_full = 2 r_half / (1 + r_half)
    校正为整篇诗上的信度。这是逐首读出**测量误差**的直接估计, 用于衰减校正。
    """
    # 劈分方式: 用**连续前后半**(text[:n//2] vs text[n//2:]) 而非奇偶位置。
    # 理由: 奇偶切分会把双字词(如"凄凉""浮生")与上下句对仗切碎, 人为破坏
    # n-gram 信息, 使信度被低估为近零; 连续前后半在保留词完整性的同时仍
    # 构成两个独立半篇。
    half = {"nmi": ([], []), **{c: ([], []) for c in L.CATS}}
    ok = []
    for p in pids:
        t = key[p]["text"]
        m = len(t) // 2
        o, e = t[:m], t[m:]
        if len(o) < 6 or len(e) < 6:
            continue
        po, _no = L.cat_per1k([o])
        pe, _ne = L.cat_per1k([e])
        half["nmi"][0].append(L.net_index(po))
        half["nmi"][1].append(L.net_index(pe))
        for c in L.CATS:
            half[c][0].append(po[c])
            half[c][1].append(pe[c])
        ok.append(p)
    out = {}
    for kk, (a, b) in half.items():
        if len(a) < 20:
            out[kk] = None
            continue
        r = float(spearmanr(a, b)[0])
        sb = 2 * r / (1 + r) if r > -1 else float("nan")
        out[kk] = {"r_half": round(r, 4), "reliability_spearman_brown": round(sb, 4),
                   "n_poems": len(a)}
    return out


def disattenuate(rho, rel_x, rel_y):
    """衰减校正: rho_true ≈ rho_obs / sqrt(rel_x * rel_y)。"""
    if not rel_x or not rel_y or rel_x <= 0 or rel_y <= 0:
        return None
    return round(float(rho) / float(np.sqrt(rel_x * rel_y)), 4)




def icc21(X):
    """ICC(2,1): 两向随机效应、绝对一致性、单测量。X: 行=对象, 列=编码者。"""
    X = np.asarray(X, float)
    n, k = X.shape
    gm = X.mean()
    MSR = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    MSC = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    SST = ((X - gm) ** 2).sum()
    SSE = SST - (MSR * (n - 1)) - (MSC * (k - 1))
    MSE = SSE / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return float((MSR - MSE) / denom) if denom != 0 else float("nan")


def spearman_cluster(x, y, groups, reps=8000):
    """按作者聚类的自助 CI (诗篇在同一作者内不独立)。"""
    rho = float(spearmanr(x, y)[0])
    uniq = sorted(set(groups))
    idx_by = {g: np.where(np.array(groups) == g)[0] for g in uniq}
    boot = []
    for _ in range(reps):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        xx, yy = np.asarray(x)[ii], np.asarray(y)[ii]
        if len(set(xx)) < 3 or len(set(yy)) < 3:
            continue
        r = spearmanr(xx, yy)[0]
        if not np.isnan(r):
            boot.append(r)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return rho, float(lo), float(hi), len(boot)


def partial_spearman(x, y, z):
    """控制 z(诗长) 后 x,y 的偏相关 (Pearson 偏相关, 秩变换后等价于偏秩相关)."""
    def rank(v):
        order = np.argsort(np.asarray(v, float))
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v))
        return r
    rx, ry, rz = rank(x), rank(y), rank(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(pearsonr(ex, ey)[0])


def author_grouped_cv(key, pids, gold_nmi, k=5, lam=1e-6):
    """Author-grouped cross-validation (controls author leakage).

    Manual GroupKFold: split the UNIQUE authors into K=5 folds; poems inherit
    their author's fold. For each fold, fit a Ridge regression (numpy only,
    intercept unpenalised) predicting GOLD NMI from the 5 lexicon per-category
    rates (+ intercept) using ONLY training-fold authors; evaluate Pearson r and
    Spearman rho on the held-out (unseen) authors' poems. This shows the
    lexicon->gold relationship GENERALIZES out-of-author, not just within-author.
    """
    from collections import defaultdict
    authors = sorted(set(key[p]["author"] for p in pids))
    rng = np.random.default_rng(20260918)
    perm = rng.permutation(len(authors))
    folds = [set(authors[i::k]) for i in range(k)]  # author -> fold i (round-robin)
    order = [CATMAP[d] for d in DIMS]  # 5 lexicon per-category rates, aligned to DIMS

    def design(ps):
        X = np.array([[1.0] + [key[p]["per"][c] for c in order] for p in ps], float)
        y = np.array([gold_nmi[p] for p in ps], float)
        return X, y

    pear, sp = [], []
    n_test_total = 0
    for fi in range(k):
        test_auth = folds[fi]
        train_auth = set(authors) - test_auth
        train_ps = [p for p in pids if key[p]["author"] in train_auth]
        test_ps = [p for p in pids if key[p]["author"] in test_auth]
        if len(test_ps) < 2 or len(train_ps) < 2:
            continue
        Xtr, ytr = design(train_ps)
        Xte, yte = design(test_ps)
        # Ridge: (XᵀX + λP)⁻¹ Xᵀy, λ only on the 5 feature weights (not intercept)
        P = np.zeros((6, 6))
        P[1:, 1:] = lam * np.eye(5)
        beta = np.linalg.solve(Xtr.T @ Xtr + P, Xtr.T @ ytr)
        pred = Xte @ beta
        if np.std(pred) < 1e-9 or np.std(yte) < 1e-9:
            continue
        r = float(pearsonr(pred, yte)[0])
        rh = float(spearmanr(pred, yte)[0])
        if not np.isnan(r):
            pear.append(r)
        if not np.isnan(rh):
            sp.append(rh)
        n_test_total += len(test_ps)
    return {
        "k_folds": k,
        "pearson_mean": round(float(np.mean(pear)), 4) if pear else None,
        "pearson_std": round(float(np.std(pear)), 4) if pear else None,
        "spearman_mean": round(float(np.mean(sp)), 4) if sp else None,
        "spearman_std": round(float(np.std(sp)), 4) if sp else None,
        "n_test_poems_total": int(n_test_total),
    }


def dl_pool_authors():
    """DL training-pool author set: read build_dl_pool.py's AUTH literal.

    build_dl_pool.py defines AUTH (the 12 Tang+Song authors whose poems are
    labelled into dl_pool.json) and filters the source corpora to exactly that
    set, so AUTH is the authoritative DL training-corpus author set. Parsed from
    source (no opencc/torch import needed).
    """
    import ast, re
    src = open(f"{BASE}/build_dl_pool.py", encoding="utf-8").read()
    m = re.search(r"AUTH\s*=\s*(\[[^\]]*\])", src, re.S)
    return set(ast.literal_eval(m.group(1)))


def sub_criterion_validity(mask, lex_nmi, g_nmi, groups):
    """Existing criterion validity (cluster-bootstrapped Spearman) on a subset."""
    if int(mask.sum()) == 0:
        return {"n": 0, "rho": None, "ci_low": None, "ci_high": None,
                "underpowered": True}
    r = spearman_cluster(lex_nmi[mask], g_nmi[mask], list(np.array(groups)[mask]))
    return {"n": int(mask.sum()), "rho": round(r[0], 4),
            "ci_low": round(r[1], 3), "ci_high": round(r[2], 3),
            "underpowered": bool(mask.sum() < 20)}


def main():
    pA = f"{BASE}/annotations/annotatorA_tang.txt"
    pB = f"{BASE}/annotations/annotatorB_song.txt"
    A, B = load_ann(pA), load_ann(pB)
    sample = json.load(open(f"{BASE}/gold_sample.json", encoding="utf-8"))
    key = {p["pid"]: p for p in sample}
    pids = [p["pid"] for p in sample if p["pid"] in A and p["pid"] in B]
    print(f"annotated by both: {len(pids)}  (A={len(A)}, B={len(B)})")
    if len(pids) < 20:
        print("!! 需要两位标注员的结果才能计算信度与效度"); return

    # ---- (1) 编码信度 ----
    rel = {}
    for d in DIMS:
        a = [A[p][d] for p in pids]
        b = [B[p][d] for p in pids]
        M = np.array([[A[p][d], B[p][d]] for p in pids], float)
        rel[d] = {"kappa_quadratic_weighted": round(qwk(a, b), 4),
                  "krippendorff_alpha_ordinal": round(krippendorff_ordinal(M), 4),
                  "icc_2_1": round(icc21(M), 4),
                  "mean_A": round(float(np.mean(a)), 3),
                  "mean_B": round(float(np.mean(b)), 3)}
        print(f"  [{d:8}] qwk={rel[d]['kappa_quadratic_weighted']:.3f} "
              f"alpha={rel[d]['krippendorff_alpha_ordinal']:.3f} "
              f"icc={rel[d]['icc_2_1']:.3f}")
    gold_nmi = {}
    for p in pids:
        gold_nmi[p] = 0.5 * ((A[p]["Man"] - A[p]["Bei"]) + (B[p]["Man"] - B[p]["Bei"]))
    Mnmi = np.array([[(A[p]["Man"] - A[p]["Bei"]), (B[p]["Man"] - B[p]["Bei"])]
                     for p in pids], float)
    rel["NMI_gold"] = {"icc_2_1": round(icc21(Mnmi), 4),
                       "pearson_A_vs_B": round(float(pearsonr(Mnmi[:, 0], Mnmi[:, 1])[0]), 4),
                       "spearman_A_vs_B": round(float(spearmanr(Mnmi[:, 0], Mnmi[:, 1])[0]), 4)}
    print(f"  [NMI_gold] ICC={rel['NMI_gold']['icc_2_1']:.3f} "
          f"r(A,B)={rel['NMI_gold']['pearson_A_vs_B']:.3f}")

    # ---- (2) 效标效度 ----
    gold = {p: {d: 0.5 * (A[p][d] + B[p][d]) for d in DIMS} for p in pids}
    groups = [key[p]["author"] for p in pids]
    length = np.array([len(key[p]["text"]) for p in pids])
    lex_nmi = np.array([key[p]["nmi"] for p in pids])
    g_nmi = np.array([gold_nmi[p] for p in pids])

    rho_nmi = spearman_cluster(lex_nmi, g_nmi, groups)
    rho_len_nmi = float(spearmanr(length, g_nmi)[0])
    rho_lex_len = float(spearmanr(lex_nmi, length)[0])
    print(f"\n[criterion validity] lexicon NMI vs gold NMI: rho={rho_nmi[0]:.3f} "
          f"CI=[{rho_nmi[1]:.3f},{rho_nmi[2]:.3f}] n={len(pids)}")
    print(f"  baselines: rho(length, gold)={rho_len_nmi:.3f}; "
          f"rho(lexicon, length)={rho_lex_len:.3f}; "
          f"partial rho(lex,gold|length)={partial_spearman(lex_nmi, g_nmi, length):.3f}")

    per_dim = {}
    for d in DIMS:
        lex_rate = np.array([key[p]["per"][CATMAP[d]] for p in pids])
        gs = np.array([gold[p][d] for p in pids])
        r = spearman_cluster(lex_rate, gs, groups)
        per_dim[d] = {"rho": round(r[0], 4), "ci": [round(r[1], 3), round(r[2], 3)],
                      "rho_vs_length_baseline": round(float(spearmanr(length, gs)[0]), 3)}
        print(f"  {d:8} ({CATMAP[d]}): rho={r[0]:.3f} ")

    # 情感两维(H2 核心): 金标 Quan-Bei vs 词表 旷达-悲苦
    g_affect = np.array([gold[p]["Quan"] - gold[p]["Bei"] for p in pids])
    r_aff = spearman_cluster(lex_nmi, g_affect, groups)
    print(f"  affect-only (gold Quan-Bei vs lexicon NMI): rho={r_aff[0]:.3f} "
          f"CI=[{r_aff[1]:.3f},{r_aff[2]:.3f}]")

    # 只取情感两维读出(不用意象维)
    lex_affect = {p: None for p in pids}

    # ---- 敏感性: 剔除极端值与短诗 ----
    sens = {}
    for lab, mask in (("drop_len<28", length >= 28),
                      ("drop_extreme_nmi", np.abs(lex_nmi) < 60),
                      ("drop_long", length <= 120)):
        if mask.sum() >= 30:
            rr = spearman_cluster(lex_nmi[mask], g_nmi[mask],
                                  list(np.array(groups)[mask]))
            sens[lab] = {"n": int(mask.sum()), "rho": round(rr[0], 4),
                         "ci": [round(rr[1], 3), round(rr[2], 3)]}
            print(f"  [sens {lab}] n={mask.sum()} rho={rr[0]:.3f}")

    # ---- (4) 诊断: 分布、逐首读出信度、衰减校正 ----
    print("\n[diagnostics] distributions & attenuation:")
    dist = {}
    for d in DIMS:
        gs = np.array([gold[p][d] for p in pids])
        lexr = np.array([key[p]["per"][CATMAP[d]] for p in pids])
        dist[d] = {"gold_mean": round(float(gs.mean()), 3),
                   "gold_sd": round(float(gs.std()), 3),
                   "gold_share_zero": round(float((gs == 0).mean()), 3),
                   "gold_share_ge2": round(float((gs >= 2).mean()), 3),
                   "lexicon_rate_mean": round(float(lexr.mean()), 3),
                   "lexicon_rate_sd": round(float(lexr.std()), 3),
                   "lexicon_share_zero": round(float((lexr == 0).mean()), 3)}
    rei = lexicon_reliability(key, pids)
    print(f"  lexicon per-poem reliability (contiguous split-half, SB-corrected): "
          f"NMI={rei['nmi']['reliability_spearman_brown']} "
          f"(r_half={rei['nmi']['r_half']}, n={rei['nmi']['n_poems']})")
    for d in DIMS:
        print(f"    {d:8} rel={rei[CATMAP[d]]['reliability_spearman_brown']:.3f} "
              f"goldmean={dist[d]['gold_mean']} zero={dist[d]['gold_share_zero']}")
    # 金标(两员均值)的信度: ICC(2,1) -> 两员均值的信度 (Spearman-Brown)
    gold_rel_nmi = 2 * rel["NMI_gold"]["icc_2_1"] / (1 + rel["NMI_gold"]["icc_2_1"])
    dis = {"gold_reliability_2coder_mean_NMI": round(gold_rel_nmi, 4),
           "NMI_raw_rho": round(rho_nmi[0], 4),
           "NMI_disattenuated": disattenuate(
               rho_nmi[0], rei["nmi"]["reliability_spearman_brown"], gold_rel_nmi),
           "affect_raw_rho": round(r_aff[0], 4),
           "affect_disattenuated": disattenuate(
               r_aff[0], rei["nmi"]["reliability_spearman_brown"], gold_rel_nmi),
           "per_dimension": {}}
    for d in DIMS:
        g_rel = 2 * rel[d]["icc_2_1"] / (1 + rel[d]["icc_2_1"])
        dis["per_dimension"][d] = {
            "rho_raw": per_dim[d]["rho"],
            "lexicon_reliability": rei[CATMAP[d]]["reliability_spearman_brown"],
            "gold_reliability": round(g_rel, 4),
            "rho_disattenuated": disattenuate(
                per_dim[d]["rho"], rei[CATMAP[d]]["reliability_spearman_brown"], g_rel)}
    print(f"  disattenuated: NMI {dis['NMI_raw_rho']}->{dis['NMI_disattenuated']}; "
          f"affect {dis['affect_raw_rho']}->{dis['affect_disattenuated']}")
    for d in DIMS:
        print(f"    {d:8} raw={dis['per_dimension'][d]['rho_raw']} "
              f"-> dis={dis['per_dimension'][d]['rho_disattenuated']}")

    # ---- (5) 五分位"剂量—反应"检验 (对逐首量化噪声稳健的效度证据) ----
    # 逐首 per-1000 读出受词频量化噪声严重限制, 单首相关性会被衰减。故另按词表
    # NMI 把诗篇分为五分位, 看金标 NMI 是否随分位单调上升; 用作者级聚类自助
    # 给出"最高分位减最低分位"的差及其 CI。若词表完全无效, 该差应跨 0。
    order = np.argsort(lex_nmi)
    qidx = np.array_split(order, 5)
    quint = []
    uniq = sorted(set(groups))
    idx_by = {g: np.where(np.array(groups) == g)[0] for g in uniq}
    for qi, ii in enumerate(qidx):
        quint.append({"quintile": qi + 1, "n": len(ii),
                      "lexicon_nmi_range": [round(float(lex_nmi[ii].min()), 3),
                                            round(float(lex_nmi[ii].max()), 3)],
                      "gold_nmi_mean": round(float(g_nmi[ii].mean()), 3)})
    print("  [dose-response] gold NMI by lexicon-NMI quintile:",
          [q["gold_nmi_mean"] for q in quint])
    diff_boot = []
    for _ in range(6000):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        sel_lo = ii[np.isin(ii, qidx[0])]
        sel_hi = ii[np.isin(ii, qidx[4])]
        if len(sel_lo) < 3 or len(sel_hi) < 3:
            continue
        diff_boot.append(float(g_nmi[sel_hi].mean() - g_nmi[sel_lo].mean()))
    d_lo, d_hi = np.percentile(diff_boot, [2.5, 97.5])
    q5v1 = float(g_nmi[qidx[4]].mean() - g_nmi[qidx[0]].mean())
    print(f"    Q5-Q1 = {q5v1:.3f}  CI=[{d_lo:.3f},{d_hi:.3f}] "
          f"({'跨0' if d_lo < 0 < d_hi else '不含0'})")
    quintile = {"rows": quint, "Q5_minus_Q1": round(q5v1, 3),
                "Q5_minus_Q1_ci": [round(float(d_lo), 3), round(float(d_hi), 3)],
                "contains_zero": bool(d_lo < 0 < d_hi),
                "spearman_across_quintiles": round(float(spearmanr(
                    [q["quintile"] for q in quint],
                    [q["gold_nmi_mean"] for q in quint])[0]), 4)}

    # ---- (3) 作者级(十位贬谪队列作者) ----
    from collections import defaultdict
    by_a = defaultdict(list)
    for p in pids:
        by_a[key[p]["author"]].append(p)
    auth_rows = []
    for a, ps in by_a.items():
        if len(ps) < 3:
            continue
        auth_rows.append({"author": a, "n_poems": len(ps),
                          "lexicon_nmi": round(float(np.mean([key[p]["nmi"] for p in ps])), 3),
                          "gold_nmi": round(float(np.mean([gold_nmi[p] for p in ps])), 3)})
    auth_rows.sort(key=lambda r: -r["gold_nmi"])
    if len(auth_rows) >= 5:
        ra = spearman_cluster([r["lexicon_nmi"] for r in auth_rows],
                              [r["gold_nmi"] for r in auth_rows],
                              [r["author"] for r in auth_rows], reps=2000)
        print(f"\n[author-level] n_authors={len(auth_rows)} rho={ra[0]:.3f} "
              f"CI=[{ra[1]:.3f},{ra[2]:.3f}]")
    else:
        ra = (float("nan"), float("nan"), float("nan"), 0)

    # ---- (6) REVIEWER P0-11: author-leakage control via author-grouped CV ----
    agcv = author_grouped_cv(key, pids, gold_nmi)
    print(f"\n[author-grouped CV] K={agcv['k_folds']} "
          f"pearson={agcv['pearson_mean']}±{agcv['pearson_std']} "
          f"(n_test={agcv['n_test_poems_total']}) "
          f"spearman={agcv['spearman_mean']}±{agcv['spearman_std']}")

    # ---- (7) REVIEWER P0-21: in-pool vs out-of-pool external validity ----
    AUTH = dl_pool_authors()
    inpool_mask = np.array([key[p]["author"] in AUTH for p in pids])
    outpool_mask = ~inpool_mask
    ipv = sub_criterion_validity(inpool_mask, lex_nmi, g_nmi, groups)
    opv = sub_criterion_validity(outpool_mask, lex_nmi, g_nmi, groups)
    print(f"[in-pool vs out-of-pool] inpool n={ipv['n']} rho={ipv['rho']} "
          f"CI=[{ipv['ci_low']},{ipv['ci_high']}] underpowered={ipv['underpowered']}")
    print(f"  outpool n={opv['n']} rho={opv['rho']} "
          f"CI=[{opv['ci_low']},{opv['ci_high']}] underpowered={opv['underpowered']}")

    out = {
        "design": {
            "n_poems": len(pids), "n_authors": len(set(groups)),
            "coders": 2, "blind": True,
            "note": "两位专家独立编码同一批诗; 标注员不见词表输出与作者名",
            "gold_NMI_formula": "mean(Man)-mean(Bei) 两位编码者均值",
        },
        "reliability": rel,
        "criterion_validity": {
            "rho_lexiconNMI_vs_goldNMI": round(rho_nmi[0], 4),
            "ci_cluster_bootstrap": [round(rho_nmi[1], 3), round(rho_nmi[2], 3)],
            "n_bootstrap_reps_used": rho_nmi[3],
            "baseline_rho_length_vs_gold": round(rho_len_nmi, 4),
            "rho_lexicon_vs_length": round(rho_lex_len, 4),
            "partial_rho_controlling_length": round(partial_spearman(lex_nmi, g_nmi, length), 4),
            "per_dimension": per_dim,
            "affect_only_rho": round(r_aff[0], 4),
            "affect_only_ci": [round(r_aff[1], 3), round(r_aff[2], 3)],
        },
        "sensitivity": sens,
        "diagnostics": {"distributions": dist, "lexicon_reliability": rei,
                        "attenuation": dis},
        "quintile_dose_response": quintile,
        "author_level": {"n_authors": len(auth_rows),
                         "rho": round(ra[0], 4) if not np.isnan(ra[0]) else None,
                         "ci": [round(ra[1], 3), round(ra[2], 3)] if not np.isnan(ra[0]) else None,
                         "rows": auth_rows},
        "author_grouped_cv": agcv,
        "inpool_outpool_validity": {
            "inpool": ipv,
            "outpool": opv,
            "note": "out-of-pool is the stricter external-validity check",
        },
    }
    json.dump(out, open(f"{BASE}/results_gold_validity.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nwrote results_gold_validity.json")


if __name__ == "__main__":
    main()
