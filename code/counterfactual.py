# -*- coding: utf-8 -*-
r"""counterfactual.py — author-level counterfactual edition-channel projection.

科学问题: 一位诗人的测得情绪分 (NMI 及五维剖面) 有多少是「幸存版本」给的?
若其诗经由不同流传渠道 (官修选本 御選宋詩 / 私家选本 宋詩鈔 / 唐·同书重印
御定全唐诗) 到达我们手中, 测得的分数会变成多少?

方法 (与 hb_model.py / song3_rep.py 同一机件):
  y_i^QSS  = alpha_i            + eps,   eps ~ N(0, s_i1^2)
  y_i^C    = alpha_i + delta^C  + eps,   eps ~ N(0, s_i2^2)
  alpha_i ~ N(mu, tau^2);  delta^C = 精度加权均值; s = 解析噪声底 * k=1.0108

反事实: 后验 alpha_d (联合全集侧 + 官修选本侧 + 私家侧[在册时]观测),
投影到渠道 C:  rate_cf^C_d = alpha_d + delta^C_d;  NMI_cf = cf_旷达 - cf_悲苦.
反向反事实: 仅凭官修选本观测还原全集侧 alpha.
不确定性: 参数级 parametric bootstrap (mu/tau/delta 联合重抽, 每 replicate 重新
ML 拟合) x 作者级后验抽样, B=300; seed=20260918.

输出: results_counterfactual.json  (+ counterfactual_log.txt)
"""
import glob, json, os
import numpy as np
import opencc
import lexicon as L

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = f"{BASE}/data"
cc = opencc.OpenCC('t2s')
LOG = open(f"{BASE}/counterfactual_log.txt", "w", encoding="utf-8")
def log(*a):
    LOG.write(" ".join(str(x) for x in a) + "\n"); LOG.flush()

EPS = 1e-9
K_INFL = 1.0108      # 聚集膨胀因子 (hb_model.py 校准)
MINC = 300
SEED = 20260918
B = 300
EASE, BITTER = L.NET
DIMS = list(L.CATS)
QUEUE = ["苏轼", "欧阳修", "黄庭坚", "秦观", "王禹偁", "范仲淹", "辛弃疾", "陆游"]

# ---------- 语料聚合 (带缓存, 与 phase1b_song.aggregate 同一逻辑) ----------
def aggregate(folder, cache_key):
    cache_fp = f"{DATA}/_agg_cache_{cache_key}.json"
    if os.path.exists(cache_fp):
        log(f"[cache] hit {cache_key}")
        return json.load(open(cache_fp, encoding="utf-8"))
    agg, nfile = {}, 0
    for fp in glob.glob(folder):
        nfile += 1
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
    log(f"[agg] {cache_key}: {nfile} files -> {len(agg)} authors")
    with open(cache_fp, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False)
    return agg

def per_author_stats(agg):
    out = {}
    for a, texts in agg.items():
        per, nch = L.cat_per1k(texts)
        if nch == 0:
            continue
        out[a] = {"chars": nch, "per": per, "nmi": per[EASE] - per[BITTER]}
    return out

# ---------- 与 hb_model.py 相同的机件 ----------
def _laplace_p(rate, n):
    return ((rate / 1000.0 * n) + 0.5) / (n + 1.0)

def sig_dim(rate, n):
    if n <= 0: return float("inf")
    p = _laplace_p(rate, n)
    return 1000.0/np.sqrt(n+1.0)*np.sqrt(p*(1-p))

def sig_nmi(per, n):
    if n <= 0: return float("inf")
    pe, pb = _laplace_p(per[EASE], n), _laplace_p(per[BITTER], n)
    return 1000.0/np.sqrt(n+1.0)*np.sqrt(pe*(1-pe)+pb*(1-pb))

def weighted_delta(Y1, S1, Y2, S2):
    w = 1.0/(S1**2 + S2**2 + EPS)
    return float(np.sum(w*(Y2-Y1))/np.sum(w)), float(1.0/np.sqrt(np.sum(w)))

def fit_mu_tau(Y1, S1, mu0, tau0):
    from scipy.optimize import minimize
    def nll(th):
        mu, lt = th
        v = np.exp(2*lt) + S1**2 + EPS
        return 0.5*np.sum(np.log(2*np.pi*v) + (Y1-mu)**2/v)
    r = minimize(nll, [mu0, np.log(tau0)], method="Nelder-Mead",
                 options={"xatol":1e-9,"fatol":1e-9,"maxiter":20000})
    return float(r.x[0]), float(np.exp(r.x[1]))

def series(A, Bc, authors, key):
    Y1 = np.array([(A[a]["nmi"] if key == "nmi" else A[a]["per"][key]) for a in authors])
    Y2 = np.array([(Bc[a]["nmi"] if key == "nmi" else Bc[a]["per"][key]) for a in authors])
    if key == "nmi":
        S1 = np.array([sig_nmi(A[a]["per"], A[a]["chars"]) for a in authors])*K_INFL
        S2 = np.array([sig_nmi(Bc[a]["per"], Bc[a]["chars"]) for a in authors])*K_INFL
    else:
        S1 = np.array([sig_dim(A[a]["per"][key], A[a]["chars"]) for a in authors])*K_INFL
        S2 = np.array([sig_dim(Bc[a]["per"][key], Bc[a]["chars"]) for a in authors])*K_INFL
    return Y1, S1, Y2, S2

def rnd(x, nd=3):
    if isinstance(x, list): return [rnd(v, nd) for v in x]
    if isinstance(x, float): return round(x, nd)
    return x

# =====================================================================
log(f"=== counterfactual edition-channel projection | seed={SEED} B={B} k={K_INFL} MINC={MINC} ===")
log("loading corpora ...")
QSS = per_author_stats(aggregate(f"{DATA}/songshi/poet.song.*.json", "songshi"))
YX  = per_author_stats(aggregate(f"{DATA}/yusong/poet.song_selection.json", "yusong"))
SSC = per_author_stats(aggregate(f"{DATA}/songchao/poet.song_songchao.json", "songchao"))
TAN = per_author_stats(aggregate(f"{DATA}/tang/poet.tang.*.json", "tang"))
YUD = per_author_stats(aggregate(f"{DATA}/yuding/*.json", "yuding"))
log(f"authors: QSS={len(QSS)} YXSS={len(YX)} SSC={len(SSC)} QTS={len(TAN)} YUDING={len(YUD)}")

auth_off = sorted(a for a in QSS if a in YX and QSS[a]["chars"] >= MINC and YX[a]["chars"] >= MINC)
auth_prv = sorted(a for a in QSS if a in SSC and QSS[a]["chars"] >= MINC and SSC[a]["chars"] >= MINC)
auth_tri = sorted(a for a in auth_prv if a in YX and YX[a]["chars"] >= MINC)  # 三向共有 (作者级私家反事实限于此)
auth_tng = sorted(a for a in TAN if a in YUD and TAN[a]["chars"] >= MINC and YUD[a]["chars"] >= MINC)
sub_prv = np.array([auth_off.index(a) for a in auth_tri])
sub_prv_fit = np.array([auth_prv.index(a) for a in auth_tri])
log(f"common: official-channel={len(auth_off)} private-channel={len(auth_prv)} "
    f"tri-way={len(auth_tri)} tang={len(auth_tng)}")

rng = np.random.default_rng(SEED)
FIT = {"off": {}, "prv": {}, "tng": {}}
BOOT = {"off": {}, "prv": {}, "tng": {}}   # 每 series: delta/mu/tau 的 bootstrap 抽取

for key in DIMS + ["nmi"]:
    Y1, S1, Y2, S2 = series(QSS, YX, auth_off, key)
    d0, se0 = weighted_delta(Y1, S1, Y2, S2)
    mu0, tau0 = fit_mu_tau(Y1, S1, float(np.mean(Y1)), float(np.std(Y1)))
    FIT["off"][key] = {"delta": d0, "se": se0, "mu": mu0, "tau": tau0,
                       "ci": [d0-1.96*se0, d0+1.96*se0]}
    Y1p, S1p, Y2p, S2p = series(QSS, SSC, auth_prv, key)
    d1, se1 = weighted_delta(Y1p, S1p, Y2p, S2p)
    FIT["prv"][key] = {"delta": d1, "se": se1, "ci": [d1-1.96*se1, d1+1.96*se1]}
    Y1t, S1t, Y2t, S2t = series(TAN, YUD, auth_tng, key)
    d2, se2 = weighted_delta(Y1t, S1t, Y2t, S2t)
    mu_t, tau_t = fit_mu_tau(Y1t, S1t, float(np.mean(Y1t)), float(np.std(Y1t)))
    FIT["tng"][key] = {"delta": d2, "se": se2, "mu": mu_t, "tau": tau_t,
                       "ci": [d2-1.96*se2, d2+1.96*se2]}
    d_off_b, d_prv_b, d_tng_b, mu_b, tau_b, mu_tb, tau_tb = [], [], [], [], [], [], []
    for _ in range(B):
        a_star = rng.normal(mu0, tau0, len(auth_off))
        y1s = a_star + rng.normal(0, S1)
        y2s = a_star + d0 + rng.normal(0, S2)
        db, _ = weighted_delta(y1s, S1, y2s, S2)
        mb, tb = fit_mu_tau(y1s, S1, mu0, tau0)
        d_off_b.append(db); mu_b.append(mb); tau_b.append(tb)
        y2sp = a_star[sub_prv] + d1 + rng.normal(0, S2p[sub_prv_fit])
        dpb, _ = weighted_delta(y1s[sub_prv], S1[sub_prv], y2sp, S2p[sub_prv_fit])
        d_prv_b.append(dpb)
        a_t = rng.normal(mu_t, tau_t, len(auth_tng))
        y1t = a_t + rng.normal(0, S1t)
        y2t = a_t + d2 + rng.normal(0, S2t)
        dtb, _ = weighted_delta(y1t, S1t, y2t, S2t)
        mtb, ttb = fit_mu_tau(y1t, S1t, mu_t, tau_t)
        d_tng_b.append(dtb); mu_tb.append(mtb); tau_tb.append(ttb)
    BOOT["off"][key] = {"delta": np.array(d_off_b), "mu": np.array(mu_b), "tau": np.array(tau_b)}
    BOOT["prv"][key] = {"delta": np.array(d_prv_b)}
    BOOT["tng"][key] = {"delta": np.array(d_tng_b), "mu": np.array(mu_tb), "tau": np.array(tau_tb)}
    log(f"[fit] {key}: off_d={d0:.3f} prv_d={d1:.3f} tng_d={d2:.3f}  (mu={mu0:.3f} tau={tau0:.3f})")

# ---------- 复现性核对 ----------
hb = json.load(open(f"{BASE}/results_hb_model.json", encoding="utf-8"))
s3 = json.load(open(f"{BASE}/results_song3_rep.json", encoding="utf-8"))
repro = {
    "delta_nmi_official_recomputed": round(FIT["off"]["nmi"]["delta"], 3),
    "delta_nmi_official_published_hb": hb["edition_effect"]["delta"],
    "delta_nmi_private_recomputed": round(FIT["prv"]["nmi"]["delta"], 3),
    "delta_nmi_private_published_song3": s3["pair_QSS_vs_SSC"]["delta_nmi"]["delta"],
    "dim_deltas_official_recomputed": {d: round(FIT["off"][d]["delta"], 3) for d in DIMS},
    "dim_deltas_official_published_hb": {d: v["delta"] for d, v in hb["per_dimension_delta"].items()},
}
log(f"[repro] {json.dumps(repro, ensure_ascii=False)}")

# ---------- 作者级反事实 MC ----------
# 逐维收集 per replicate, 最后差分得 NMI_cf
obs_off = {a: {} for a in auth_off}      # 每维 cf (官修渠道)
obs_prv = {a: {} for a in auth_tri}      # 每维 cf (私家渠道, 三向共有作者)
obs_rev = {a: {} for a in auth_off}      # 每维反向还原
obs_tng = {a: {} for a in auth_tng}      # 每维 cf (唐重印渠道)
for d in DIMS:
    Y1, S1, Y2, S2 = series(QSS, YX, auth_off, d)
    _, _, Y2p, S2p = series(QSS, SSC, auth_prv, d)
    Y2p, S2p = Y2p[sub_prv_fit], S2p[sub_prv_fit]      # 三向共有作者
    Y1t, S1t, Y2t, S2t = series(TAN, YUD, auth_tng, d)
    do_arr = BOOT["off"][d]["delta"]; mu_arr = BOOT["off"][d]["mu"]; tau_arr = BOOT["off"][d]["tau"]
    dp_arr = BOOT["prv"][d]["delta"]
    dt_arr = BOOT["tng"][d]["delta"]; mu_tarr = BOOT["tng"][d]["mu"]; tau_tarr = BOOT["tng"][d]["tau"]
    for b in range(B):
        do, dp = do_arr[b], dp_arr[b]
        mu, tau = mu_arr[b], tau_arr[b]
        ptau = 1.0/tau**2
        p1 = 1.0/S1**2; p2 = 1.0/S2**2
        # 官修-only 后验 (所有 256 作者)
        v = 1.0/(ptau + p1 + p2)
        m = v*(mu*ptau + Y1*p1 + (Y2-do)*p2)
        alpha = rng.normal(m, np.sqrt(v))
        cf = alpha + do
        rev = rng.normal(1.0/(ptau+p2)*(mu*ptau + (Y2-do)*p2), np.sqrt(1.0/(ptau+p2)))
        for i, a in enumerate(auth_off):
            obs_off[a].setdefault(d, []).append(cf[i])
            obs_rev[a].setdefault(d, []).append(rev[i])
        # 私家在册作者: 联合三侧观测的后验
        idx = sub_prv
        p2pp = 1.0/S2p**2
        v_p = 1.0/(ptau + p1[idx] + p2[idx] + p2pp)
        m_p = v_p*(mu*ptau + Y1[idx]*p1[idx] + (Y2[idx]-do)*p2[idx] + (Y2p-dp)*p2pp)
        alpha_p = rng.normal(m_p, np.sqrt(v_p))
        cf_p = alpha_p + dp
        for j, a in enumerate(auth_tri):
            obs_prv[a].setdefault(d, []).append(cf_p[j])
        # 唐侧: 独立总体
        dt, mu_t, tau_t = dt_arr[b], mu_tarr[b], tau_tarr[b]
        ptau_t = 1.0/tau_t**2
        p1t = 1.0/S1t**2; p2t = 1.0/S2t**2
        v_t = 1.0/(ptau_t + p1t + p2t)
        m_t = v_t*(mu_t*ptau_t + Y1t*p1t + (Y2t-dt)*p2t)
        alpha_t = rng.normal(m_t, np.sqrt(v_t))
        cf_t = alpha_t + dt
        for i, a in enumerate(auth_tng):
            obs_tng[a].setdefault(d, []).append(cf_t[i])
    log(f"[mc] dim {d} done")

def nmi_from(obs, a):
    """NMI 抽样序列 = 旷达 - 悲苦 (逐 replicate 对齐)."""
    e = np.array(obs[a][EASE]); b = np.array(obs[a][BITTER])
    return e - b
    e = np.array(obs[a][EASE]); b = np.array(obs[a][BITTER])
    return e - b

def stat(samples):
    v = np.asarray(samples)
    return {"mean": round(float(v.mean()), 3),
            "ci95": [round(float(np.percentile(v, 2.5)), 3),
                     round(float(np.percentile(v, 97.5)), 3)]}

# ---------- 渠道 delta 汇总 ----------
def q95(v):
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

def chan_summary(tag, fit, boot, dims=DIMS):
    out = {"n_authors": {"off": len(auth_off), "prv": len(auth_prv), "tng": len(auth_tng)}[tag]}
    for key in dims + ["nmi"]:
        f = fit[key]; bb = boot[key]["delta"]
        out[key] = {"delta": round(f["delta"], 3), "se": round(f["se"], 3),
                    "ci95_normal": rnd(f["ci"]), "ci95_boot": q95(bb),
                    "ci_excludes_zero": not (min(f["ci"]) < 0 < max(f["ci"]))}
    return out

channel_deltas = {
    "official_yuxuan_songshi": chan_summary("off", FIT["off"], BOOT["off"]),
    "private_songshichao": chan_summary("prv", FIT["prv"], BOOT["prv"]),
    "tang_reprint_yuding_quantangshi": chan_summary("tng", FIT["tng"], BOOT["tng"]),
}
latent_pop = {d: {"mu": round(FIT["off"][d]["mu"], 3), "tau": round(FIT["off"][d]["tau"], 3),
                  "tau_ci95_boot": q95(BOOT["off"][d]["tau"])} for d in DIMS}

# ---------- 8 位宋队列官员 (含在册状态) ----------
queue_rows = []
for a in QUEUE:
    if a not in QSS:
        queue_rows.append({"author": a, "in_songshi_corpus": False})
        continue
    row = {"author": a, "in_songshi_corpus": True,
           "in_official_channel": a in YX, "in_private_channel": a in SSC,
           "chars_qss": QSS[a]["chars"]}
    if a in YX:
        row["chars_yuxuan"] = YX[a]["chars"]
        row["nmi_qss_measured"] = round(QSS[a]["nmi"], 3)
        row["nmi_yuxuan_measured"] = round(YX[a]["nmi"], 3)
        row["observed_shift_nmi"] = round(YX[a]["nmi"] - QSS[a]["nmi"], 3)
        row["nmi_cf_official"] = stat(nmi_from(obs_off, a))
        row["nmi_reverse_from_official_only"] = stat(nmi_from(obs_rev, a))
    if a in obs_prv:
        row["chars_songshichao"] = SSC[a]["chars"]
        row["nmi_songshichao_measured"] = round(SSC[a]["nmi"], 3)
        row["nmi_cf_private"] = stat(nmi_from(obs_prv, a))
    pd = {}
    for d in DIMS:
        e = {"qss_rate": round(QSS[a]["per"][d], 2)}
        if a in YX:
            e["yuxuan_rate"] = round(YX[a]["per"][d], 2)
            e["cf_official"] = stat(obs_off[a][d])
        if a in obs_prv:
            e["songshichao_rate"] = round(SSC[a]["per"][d], 2)
            e["cf_private"] = stat(obs_prv[a][d])
        pd[d] = e
    row["per_dim"] = pd
    queue_rows.append(row)
    log(f"[queue] {a}: qss={row.get('nmi_qss_measured')} cf_off={row.get('nmi_cf_official')} cf_prv={row.get('nmi_cf_private')}")

# ---------- 作者级测量脆弱性排序 (观测位移 d_i = 选本测得 - 全集测得) ----------
Y1n, S1n, Y2n, S2n = series(QSS, YX, auth_off, "nmi")
d_i = Y2n - Y1n
se_i = np.sqrt(S1n**2 + S2n**2)
rows = [{"author": auth_off[i], "d": round(float(d_i[i]), 2),
         "se": round(float(se_i[i]), 2),
         "ci": [round(float(d_i[i]-1.96*se_i[i]), 2), round(float(d_i[i]+1.96*se_i[i]), 2)],
         "ci_excludes_zero": bool(abs(d_i[i]) > 1.96*se_i[i])} for i in range(len(auth_off))]
frag = sorted([r for r in rows if r["ci_excludes_zero"]], key=lambda r: -abs(r["d"]))[:10]
stab = sorted([r for r in rows if not r["ci_excludes_zero"]], key=lambda r: abs(r["d"]))[:10]
n_frag = len([r for r in rows if r["ci_excludes_zero"]])

# ---------- 排序翻转 (8 队列作者内部 + 全体) ----------
qp = [r["author"] for r in queue_rows if r.get("nmi_qss_measured") is not None and "nmi_cf_official" in r]
order_qss = sorted(qp, key=lambda a: -QSS[a]["nmi"])
order_cf = sorted(qp, key=lambda a: -(stat(nmi_from(obs_off, a))["mean"]))
flips_q = []
for i in range(len(order_qss)):
    for j in range(i+1, len(order_qss)):
        a1, a2 = order_qss[i], order_qss[j]
        pos_cf1 = order_cf.index(a1); pos_cf2 = order_cf.index(a2)
        if pos_cf1 > pos_cf2:
            flips_q.append({"pair": [a1, a2],
                            "qss_order": [order_qss.index(a1)+1, order_qss.index(a2)+1],
                            "cf_official_order": [pos_cf1+1, pos_cf2+1]})
# 全体 256: 秩相关与翻转对数
rank_qss = np.argsort(np.argsort(-Y1n))
mean_cf_all = np.array([stat(nmi_from(obs_off, a))["mean"] for a in auth_off])
rank_cf = np.argsort(np.argsort(-mean_cf_all))
from scipy.stats import spearmanr
rho_cf, _ = spearmanr(Y1n, mean_cf_all)
flip_all = {"n_authors": len(auth_off),
            "spearman_rank_measured_vs_cf_official": round(float(rho_cf), 4),
            "n_discordant_pairs": int(np.sum((rank_qss[:, None] < rank_qss[None, :]) != (rank_cf[:, None] < rank_cf[None, :])))}

# ---------- 刘禹锡 (唐侧) ----------
lyx = {}
if "刘禹锡" in TAN and "刘禹锡" in YUD:
    lyx = {"author": "刘禹锡", "in_tang_pair": True,
           "nmi_qts_measured": round(TAN["刘禹锡"]["nmi"], 3),
           "nmi_yuding_measured": round(YUD["刘禹锡"]["nmi"], 3),
           "nmi_cf_tang_reprint": stat(nmi_from(obs_tng, "刘禹锡")),
           "per_dim": {d: {"qts_rate": round(TAN["刘禹锡"]["per"][d], 2),
                           "yuding_rate": round(YUD["刘禹锡"]["per"][d], 2),
                           "cf_tang": stat(obs_tng["刘禹锡"][d])} for d in DIMS}}
else:
    lyx = {"author": "刘禹锡", "in_tang_pair": False,
           "note": "不在 唐诗∩御定全唐诗 共有作者 (或 <300字)"}
log(f"[liuyuxi] {json.dumps(lyx, ensure_ascii=False)[:300]}")

# ---------- 输出 ----------
out = {
    "method": ("counterfactual edition-channel projection on the empirical-Bayes "
               "hierarchical model of hb_model.py: latent author dimension rates "
               "alpha_d (posterior combining corpus-side, official-anthology and "
               "private-anthology observations) projected to each channel via its "
               "dimension-level edition effect delta_d^C; NMI_cf = cf_旷达 - cf_悲苦; "
               "uncertainty = parametric bootstrap over (mu, tau, delta) x author "
               "posterior draws, B=300"),
    "design": {"seed": SEED, "n_bootstrap": B, "clustering_inflation_k": K_INFL,
               "minc_chars": MINC,
               "channels": {"official": "御選宋詩 (四庫文淵閣, KR4h0143)",
                            "private": "宋詩鈔 (吳之振, 四庫文淵閣, KR4h0157)",
                            "tang_reprint": "御定全唐诗 vs 全唐诗 (同书重印渠道)"}},
    "n_authors": {"official": len(auth_off), "private": len(auth_prv),
                  "tri_way_author_level": len(auth_tri), "tang": len(auth_tng)},
    "reproducibility_check": repro,
    "channel_deltas": channel_deltas,
    "latent_population_per_dim": latent_pop,
    "queue_authors": queue_rows,
    "author_level_sensitivity": {
        "definition": "observed shift d_i = NMI(官修选本测得) - NMI(全集测得), 256 authors",
        "n_ci_excludes_zero": n_frag,
        "top10_most_edition_fragile": frag,
        "top10_most_stable": stab,
        "ranking_flips_queue8_official_channel": flips_q,
        "ranking_flips_all256": flip_all},
    "liuyuxi_tang": lyx,
}
with open(f"{BASE}/results_counterfactual.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log("wrote results_counterfactual.json")
LOG.close()
print("done")
