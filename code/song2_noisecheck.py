# -*- coding: utf-8 -*-
r"""song2_noisecheck.py -- Song cross-edition NOISE-FLOOR robustness check.

CORRECTED SAMPLING UNIT (P0-2 from six-reviewer audit)
------------------------------------------------------
The previous version drew W random *consecutive character windows* of size
m_a from the full Quan Song Shi text. That is the wrong null model: an
imperial anthology selects *whole poems*, not contiguous character slices.
A contiguous window can split a single poem and mix neighbouring poems, so
its variance underestimates (or mis-estimates) the real selection variance.

The corrected test resamples by the *poem* as the unit: for each overlap
author we draw poems with replacement from that author's complete-collection
poem list until the accumulated character count reaches the anthology's own
size m_a, concatenate the drawn poems, and compute NMI. Repeating this W
times yields the sampling distribution of an anthology-sized sample drawn
from the author's *own complete works* (the correct H0: "the anthology is
just a random anthology-sized selection of this author's poems").

    mu_a    ~ expected NMI at that sample size
    sigma_a ~ the noise floor under POEM-LEVEL selection
We then place the anthology's OBSERVED NMI in this distribution:
    z_a  = (NMI_anth - mu_a) / sigma_a
    pct_a = percentile rank of the observed value
Also: per-author Spearman across the 5 categories (a stricter statistic than
the pooled 35-cell r).

Reading
-------
- Most |z_a| > 2  -> shifts are systematic (selection bias is real).
- Most |z_a| < 2  -> shifts lie within the poem-level noise floor -> the
  absolute-level difference must be reported as NOT distinguishable from
  small-sample selection noise.

Data sources (verbatim, no randomness in the corpora):
  - Quan Song Shi : data/songshi/poet.song.*.json  (Book1Q84; poem-level)
  - Yu Xuan Song Shi : data/song2/KR4h0143_*.txt    (kanripo KR4h0143)
Output: results_song2_noisecheck.json (+ console table)
"""
import os
import sys
import json
import numpy as np
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lexicon as L
import song2_xval as S          # reuse the verified loaders + stats

W = 1000                        # resamples per author
rng = np.random.default_rng(20260915)   # fixed seed -> reproducible


def net_of(text):
    _n, _per1k, net = S.stats(text)
    return net


def main():
    qs_poems = S.load_quansongshi_poems()
    yx = S.load_yuxuan_songshi()

    rows = []
    for a in S.TARGET:
        P = qs_poems[a]
        Y = yx[a]
        m = len(Y)
        if m == 0 or not P:
            continue  # skip authors with no anthology text (e.g. Xin Qiji)

        net_full = net_of("".join(P))
        net_anth = net_of(Y)

        # POEM-LEVEL bootstrap: assemble an anthology-sized sample from whole
        # poems drawn with replacement from the complete collection.
        nets = np.empty(W, float)
        n_poems_sample = np.empty(W, int)
        n_poems_full = len(P)
        for i in range(W):
            chars = 0
            buf = []
            while chars < m:
                j = int(rng.integers(0, n_poems_full))
                buf.append(P[j])
                chars += len(P[j])
            nets[i] = net_of("".join(buf))
            n_poems_sample[i] = len(buf)
        mu = float(nets.mean())
        sd = float(nets.std(ddof=1))
        z = (net_anth - mu) / sd if sd > 0 else float("nan")
        pct = float((nets <= net_anth).mean())
        lo, hi95 = float(np.percentile(nets, 2.5)), float(np.percentile(nets, 97.5))

        # author-level profile agreement over the 5 categories
        _, qp, _ = S.stats("".join(P))
        _, yp, _ = S.stats(Y)
        rho_prof = float(spearmanr([qp[c] for c in L.CATS],
                                   [yp[c] for c in L.CATS])[0])

        rows.append({
            "author": a,
            "n_full_chars": sum(len(p) for p in P),
            "n_full_poems": n_poems_full,
            "n_anth_chars": m,
            "net_full": round(net_full, 3),
            "net_anth": round(net_anth, 3),
            "delta": round(net_anth - net_full, 3),
            "poem_boot_mu": round(mu, 3),
            "poem_boot_sd": round(sd, 3),
            "z": round(z, 3),
            "pct": round(pct, 3),
            "ci95": [round(lo, 3), round(hi95, 3)],
            "rho_profile_5cat": round(rho_prof, 3),
        })

    zs = np.array([r["z"] for r in rows], float)
    dlt = np.array([r["delta"] for r in rows], float)
    sd = np.array([r["poem_boot_sd"] for r in rows], float)

    out = {
        "method": ("noise-floor: W=%d poem-level bootstrap resamples (with "
                   "replacement) from each author's complete-collection poems, "
                   "assembled to the anthology's character size; SEED=20260915"
                   % W),
        "unit": "poem (corrected from consecutive character window)",
        "W": W, "seed": 20260915,
        "n_authors": len(rows),
        "mean_abs_z": round(float(np.nanmean(np.abs(zs))), 3),
        "frac_abs_z_gt2": round(float(np.mean(np.abs(zs) > 2)), 3),
        "frac_abs_z_gt1": round(float(np.mean(np.abs(zs) > 1)), 3),
        "frac_delta_positive": round(float(np.mean(dlt > 0)), 3),
        "mean_poem_boot_sd": round(float(sd.mean()), 3),
        "mean_abs_delta": round(float(np.mean(np.abs(dlt))), 3),
        "mean_rho_profile_5cat": round(float(np.nanmean(
            [r["rho_profile_5cat"] for r in rows])), 3),
        "rows": rows,
    }
    with open(f"{BASE}/results_song2_noisecheck.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{'author':8}{'n_anth':>7}{'net_full':>10}{'net_anth':>10}"
          f"{'delta':>8}{'boot_sd':>9}{'z':>8}{'pct':>7}{'rho5':>8}")
    for r in rows:
        print(f"{r['author']:8}{r['n_anth_chars']:>7}{r['net_full']:>10}"
              f"{r['net_anth']:>10}{r['delta']:>8}{r['poem_boot_sd']:>9}"
              f"{r['z']:>8}{r['pct']:>7}{r['rho_profile_5cat']:>8}")
    print()
    print(f"mean |z|              = {out['mean_abs_z']}")
    print(f"frac |z|>2            = {out['frac_abs_z_gt2']}")
    print(f"frac |z|>1            = {out['frac_abs_z_gt1']}")
    print(f"frac delta>0          = {out['frac_delta_positive']}")
    print(f"mean poem-boot sd     = {out['mean_poem_boot_sd']}")
    print(f"mean |delta|          = {out['mean_abs_delta']}")
    print(f"mean rho(5cat)        = {out['mean_rho_profile_5cat']}")


if __name__ == "__main__":
    main()
