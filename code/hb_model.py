# -*- coding: utf-8 -*-
r"""hb_model.py -- Phase 3: empirical-Bayes hierarchical Gaussian noise-floor.

Replaces the poem-level *bootstrap* noise floor (song2_noisecheck.py) with a
fully analytic, reproducible alternative and a generative hierarchical model:

  1. ANALYTIC noise floor.  NMI = 1000*(p_e - p_b), p_e=旷达 rate,
     p_b=悲苦 rate.  By the binomial variance of the two per-1000-char rates,
        sigma_analytic = 1000/sqrt(n) * sqrt( p_e(1-p_e) + p_b(1-p_b) ).
     Poems inside an author are *clustered* (not independent characters), so we
     calibrate a multiplicative clustering-inflation factor k against the
     poem-level bootstrap SD on the 7 TARGET authors:
        k = mean(boot_sd) / mean(sigma_analytic at the anthology sample size).
     The bootstrap is then kept only as an *independent validation*.

  2. HIERARCHICAL model (population scale, 256 shared authors).
     y_i^QSS = alpha_i + eps_i^QSS ,  eps_i^QSS ~ N(0, s_i1^2)
     y_i^anth = alpha_i + delta   + eps_i^anth ,  eps_i^anth ~ N(0, s_i2^2)
     alpha_i ~ N(mu, tau^2)
     with s_i1, s_i2 the (calibrated) analytic noise floors.  Because the QSS
     sample is enormous, s_i1 ~ 0 and y_i^QSS pins alpha_i; the edition-effect
     delta is therefore the *exact* inverse-variance-weighted mean of the
     author-level differences d_i = y_i^anth - y_i^QSS (weights w_i =
     1/(s_i1^2+s_i2^2)), with Var(delta) = 1/sum(w_i) -- no resampling needed.
     mu and tau^2 (the latent author-population mean and heterogeneity) are
     obtained by maximum marginal likelihood; their CIs come from a parametric
     bootstrap.  delta=0 (H0) is the formal test of measurement invariance.

  3. Per-author shrinkage estimates alpha_i | data (posterior mean + 95% CI)
     and a posterior-predictive p-value for 陆游 Lu You's anthology shift.

  4. Per-dimension edition effect delta_d for all five emotional dimensions,
     each an exact inverse-variance-weighted mean -> a formal invariance test
     at the dimension level.

Outputs: results_hb_model.json  (+ console summary).
"""
import os, json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

BASE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, BASE)
import lexicon as L
import song2_xval as S          # 7-author loaders
import phase1b_song as P        # population-scale aggregate()

EASE, BITTER = L.NET             # ('旷达闲适','悲苦孤寂')
EPS = 1e-9

def _laplace_p(rate_per1k, n):
    """Per-1000 rate -> smoothed probability with a Laplace (+0.5) pseudocount,
    so zero-hit authors obtain a finite (small) sampling variance instead of 0,
    which would otherwise give those authors infinite weight in the delta
    estimate."""
    c = rate_per1k / 1000.0 * n          # implied raw count
    cp = c + 0.5
    np_ = n + 1.0
    return cp / np_

def analytic_sigma(per1k, n):
    """Analytic SD of NMI = 1000*(p_e - p_b) at sample size n (Laplace-smoothed)."""
    if n <= 0:
        return float("inf")
    pe = _laplace_p(per1k[EASE], n)
    pb = _laplace_p(per1k[BITTER], n)
    return 1000.0 / np.sqrt(n + 1.0) * np.sqrt(pe * (1 - pe) + pb * (1 - pb))

def analytic_sigma_dim(rate_per1k, n):
    """Analytic SD of a single per-1000-char rate at sample size n (smoothed)."""
    if n <= 0:
        return float("inf")
    p = _laplace_p(rate_per1k, n)
    return 1000.0 / np.sqrt(n + 1.0) * np.sqrt(p * (1 - p))

def agg_nmi(agg):
    out = {}
    for a, texts in agg.items():
        per, nch = L.cat_per1k(texts)
        if nch == 0:
            continue
        out[a] = {"nmi": L.net_index(per), "n": nch, "per": per}
    return out

def weighted_delta(Y1, S1, Y2, S2):
    """Exact inverse-variance-weighted edition effect + SE.
    d_i = Y2_i - Y1_i,  w_i = 1/(S1_i^2 + S2_i^2)."""
    w = 1.0 / (S1 ** 2 + S2 ** 2 + EPS)
    d = Y2 - Y1
    delta = float(np.sum(w * d) / np.sum(w))
    se = float(1.0 / np.sqrt(np.sum(w)))
    return delta, se

def fit_mu_tau(Y1, S1, mu0, tau0):
    """ML of (mu, tau) by maximising the marginal likelihood
    y1_i ~ N(mu, tau^2 + s1_i^2).  Bounded tau via log-transform."""
    def nll(theta):
        mu, logtau = theta
        t2 = np.exp(2 * logtau) + EPS
        v = t2 + S1 ** 2 + EPS
        return 0.5 * np.sum(np.log(2 * np.pi * v) + (Y1 - mu) ** 2 / v)
    r = minimize(nll, [mu0, np.log(tau0)], method="Nelder-Mead",
                 options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 50000})
    mu, logtau = r.x
    return float(mu), float(np.exp(logtau))

def main():
    # ---------- 1. population-scale data (256 shared authors) ----------
    print("aggregating corpora (population scale) ...")
    songshi_agg = P.aggregate(f"{BASE}/data/songshi/poet.song.*.json")
    yusong_agg = P.aggregate(f"{BASE}/data/yusong/poet.song_selection.json")
    A = agg_nmi(songshi_agg)
    B = agg_nmi(yusong_agg)
    MINC = 300
    common = [a for a in A if a in B and A[a]["n"] >= MINC and B[a]["n"] >= MINC]
    common.sort(key=lambda a: -min(A[a]["n"], B[a]["n"]))
    Y1 = np.array([A[a]["nmi"] for a in common])
    Y2 = np.array([B[a]["nmi"] for a in common])
    S1 = np.array([analytic_sigma(A[a]["per"], A[a]["n"]) for a in common])
    S2 = np.array([analytic_sigma(B[a]["per"], B[a]["n"]) for a in common])
    n_auth = len(common)
    print(f"  shared authors (>= {MINC} chars each): {n_auth}")

    # ---------- 2. calibrate clustering-inflation k on 7 TARGET ----------
    print("calibrating clustering-inflation k on 7 TARGET authors ...")
    noise = json.load(open(f"{BASE}/results_song2_noisecheck.json", encoding="utf-8"))
    boot_sd = {r["author"]: r["poem_boot_sd"] for r in noise["rows"]}
    qs_poems = S.load_quansongshi_poems()
    yx = S.load_yuxuan_songshi()
    ks = []
    for a in S.TARGET:
        if a not in boot_sd:
            continue
        per_q, nq = L.cat_per1k(qs_poems[a])
        _, n_anth = L.cat_per1k([yx[a]])
        sig_an = analytic_sigma(per_q, n_anth)
        if sig_an > 0 and n_anth > 0:
            ks.append(boot_sd[a] / sig_an)
    k = float(np.mean(ks))
    print(f"  k = {k:.3f}  (per-author ratios: {', '.join(f'{x:.2f}' for x in ks)})")
    S1k = S1 * k
    S2k = S2 * k

    # ---------- 3. edition effect delta (exact) ----------
    delta, se_d = weighted_delta(Y1, S1k, Y2, S2k)
    d_lo, d_hi = delta - 1.96 * se_d, delta + 1.96 * se_d
    print(f"\n[Edition effect] delta={delta:.3f}  SE={se_d:.3f}  "
          f"95% CI=[{d_lo:.3f},{d_hi:.3f}]  -> "
          f"{'INVARIANCE (H0:delta=0 not rejected)' if d_lo < 0 < d_hi else 'SYSTEMATIC SHIFT'}")
    if "陆游" in common:
        i = common.index("陆游")
        print(f"  [debug 陆游] Y1={Y1[i]:.2f} Y2={Y2[i]:.2f} "
              f"S1={S1k[i]:.3f} S2={S2k[i]:.3f} n_anth={B['陆游']['n']}")

    # ---------- 4. mu, tau (ML) + parametric bootstrap CI ----------
    mu_hat, tau_hat = fit_mu_tau(Y1, S1k, np.mean(Y1), np.std(Y1))
    rng = np.random.default_rng(20260916)
    B_REP = 500
    deltas_b, mus_b, taus_b = [], [], []
    for _ in range(B_REP):
        a_star = rng.normal(mu_hat, tau_hat, n_auth)
        y1s = a_star + rng.normal(0, S1k)
        y2s = a_star + delta + rng.normal(0, S2k)
        db, _ = weighted_delta(y1s, S1k, y2s, S2k)
        deltas_b.append(db)
        mb, tb = fit_mu_tau(y1s, S1k, mu_hat, tau_hat)
        mus_b.append(mb); taus_b.append(tb)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    d_ci = q(deltas_b); m_ci = q(mus_b); t_ci = q(taus_b)
    print(f"[Latent pop.] mu={mu_hat:.3f} CI={m_ci}  "
          f"tau={tau_hat:.3f} CI={t_ci}  tau2={tau_hat**2:.3f}")

    # ---------- 5. per-author shrinkage posterior ----------
    prec_tau = 1.0 / (tau_hat ** 2 + EPS)
    prec1 = 1.0 / (S1k ** 2 + EPS)
    prec2 = 1.0 / (S2k ** 2 + EPS)
    v = 1.0 / (prec_tau + prec1 + prec2)
    m = v * (mu_hat * prec_tau + Y1 * prec1 + (Y2 - delta) * prec2)
    sd = np.sqrt(v)
    rows_hb = []
    for i, a in enumerate(common):
        rows_hb.append({
            "author": a, "n_qss": int(A[a]["n"]), "n_anth": int(B[a]["n"]),
            "nmi_qss": round(float(Y1[i]), 3), "nmi_anth": round(float(Y2[i]), 3),
            "s_qss": round(float(S1k[i]), 3), "s_anth": round(float(S2k[i]), 3),
            "alpha_post": round(float(m[i]), 3),
            "alpha_ci_lo": round(float(m[i] - 1.96 * sd[i]), 3),
            "alpha_ci_hi": round(float(m[i] + 1.96 * sd[i]), 3),
        })

    # ---------- 6. 陆游 posterior predictive ----------
    if "陆游" in common:
        i = common.index("陆游")
        z_ly = (Y2[i] - Y1[i] - delta) / np.sqrt(S1k[i] ** 2 + S2k[i] ** 2 + EPS)
        p_ly = 2 * (1 - norm.cdf(abs(z_ly)))
        print(f"[陆游] posterior-predictive z={z_ly:.2f}, p={p_ly:.4f}")
    else:
        z_ly, p_ly = None, None

    # ---------- 7. per-dimension edition effect delta_d ----------
    print("per-dimension edition effects:")
    dim_eff = {}
    for d in L.CATS:
        X1 = np.array([A[a]["per"][d] for a in common])
        X2 = np.array([B[a]["per"][d] for a in common])
        G1 = np.array([analytic_sigma_dim(A[a]["per"][d], A[a]["n"]) for a in common]) * k
        G2 = np.array([analytic_sigma_dim(B[a]["per"][d], B[a]["n"]) for a in common]) * k
        dd, sed = weighted_delta(X1, G1, X2, G2)
        lo, hi = dd - 1.96 * sed, dd + 1.96 * sed
        dim_eff[d] = {"delta": round(dd, 3), "se": round(sed, 3),
                      "ci": [round(lo, 3), round(hi, 3)],
                      "systematic_shift": not (lo < 0 < hi)}
        print(f"  {d}: delta={dd:.3f} SE={sed:.3f} CI=[{lo:.3f},{hi:.3f}] "
              f"{'*SHIFT*' if not (lo < 0 < hi) else 'ok'}")

    # ---------- 8. 7-author analytic-vs-bootstrap validation ----------
    print("7-author analytic vs bootstrap noise floor:")
    val7 = []
    for a in S.TARGET:
        if a not in boot_sd:
            continue
        per_q, nq = L.cat_per1k(qs_poems[a])
        _, n_anth = L.cat_per1k([yx[a]])
        sig_an = analytic_sigma(per_q, n_anth) * k
        val7.append({"author": a, "n_anth": n_anth,
                     "analytic_sigma": round(float(sig_an), 3),
                     "bootstrap_sd": boot_sd[a],
                     "ratio": round(float(boot_sd[a] / sig_an), 3)})
    mean_an = float(np.mean([v["analytic_sigma"] for v in val7]))
    mean_bo = float(np.mean([v["bootstrap_sd"] for v in val7]))

    # ---------- 9. output ----------
    out = {
        "method": ("empirical-Bayes hierarchical Gaussian noise floor: analytic "
                   "sigma = 1000/sqrt(n)*sqrt(p_e(1-p_e)+p_b(1-p_b)) with "
                   "clustering-inflation k calibrated on the 7-author bootstrap; "
                   "edition effect delta = exact inverse-variance-weighted mean "
                   "of author-level differences; alpha_i~N(mu,tau^2)"),
        "clustering_inflation_k": round(k, 4),
        "k_per_author_ratios": [round(x, 3) for x in ks],
        "population_n_authors": n_auth,
        "edition_effect": {
            "delta": round(delta, 3), "se": round(se_d, 3),
            "ci95_normal": [round(d_lo, 3), round(d_hi, 3)],
            "ci95_bootstrap": [round(x, 3) for x in d_ci],
            "invariance_h0_not_rejected": bool(d_lo < 0 < d_hi)},
        "latent_population": {
            "mu": round(mu_hat, 3), "mu_ci95_boot": [round(x, 3) for x in m_ci],
            "tau": round(tau_hat, 3), "tau_ci95_boot": [round(x, 3) for x in t_ci],
            "tau2": round(tau_hat ** 2, 3)},
        "luyou_posterior_predictive": {
            "z": None if z_ly is None else round(float(z_ly), 3),
            "p_two_sided": None if p_ly is None else round(float(p_ly), 4)},
        "per_dimension_delta": dim_eff,
        "seven_author_validation": {
            "mean_analytic_sigma": round(mean_an, 3),
            "mean_bootstrap_sd": round(mean_bo, 3),
            "rows": val7},
        "author_rows": rows_hb,
    }
    with open(f"{BASE}/results_hb_model.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nwrote results_hb_model.json")

if __name__ == "__main__":
    main()
