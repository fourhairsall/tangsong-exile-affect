# -*- coding: utf-8 -*-
"""Generate the 5 Experiments figures for the Tang-Song NLP paper.

All numbers are taken verbatim from the paper's tables / results text:
  - fig_abi_authors.pdf      : Table tab:abi (author-level ABI + 95% CI)
  - fig_sushi_period.pdf      : Sec. exile period curve (ABI by period, CI)
  - fig_counterfactual_delta.pdf : Table tab:rep (per-dimension edition effect d)
  - fig_dim_coverage.pdf      : Sec. affective-vs-imagistic (dimension firing %)
  - fig_quintile.pdf          : Table tab:quintile (gold ABI by quintile)

Style matches make_nmi_figure.py (serif, #3b6ea5). Output to ./figures/.
Figures are language-neutral (English labels) and reused by both papers.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
})
BLUE = "#3b6ea5"
RED = "#b5483d"
ORANGE = "#d98859"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Chart 1
# Author-level ABI (Table tab:abi), ordered bleak -> easeful.
authors = [
    ("Han Yu",       -2.491, -3.79, -1.29),
    ("Qin Guan",     -2.113, -4.03, -0.18),
    ("Wang YC",      -0.982, -2.36,  0.29),
    ("Liu Yuxi",     -0.881, -2.38,  0.59),
    ("Ouyang Xiu",   -0.175, -1.35,  1.08),
    ("Liu Zongyuan",  0.221, -2.12,  2.68),
    ("Su Shi",        0.407, -0.36,  1.14),
    ("Lu You",        0.565,  0.02,  1.14),
    ("Huang TJ",      1.344,  0.55,  2.13),
    ("Bai Juyi",      1.762,  0.90,  2.63),
    ("Fan ZY",        3.354,  1.27,  5.57),
    ("Xin Qiji",      4.250,  0.84,  7.69),
]
names = [a[0] for a in authors]
vals = np.array([a[1] for a in authors], float)
lo = np.array([a[2] for a in authors])
hi = np.array([a[3] for a in authors])
err = np.vstack([vals - lo, hi - vals])           # asymmetric 2xN
colors = [RED if v < 0 else BLUE for v in vals]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.bar(range(len(vals)), vals, yerr=err, color=colors, edgecolor="white",
       linewidth=0.4, capsize=2.0, error_kw=dict(ecolor="#666", lw=0.7))
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(names, rotation=40, ha="right")
ax.set_ylabel("ABI (per 1k chars)")
ax.set_title("Author-level ABI with 95% bootstrap CI (n=12 cohort officials)")
ax.grid(axis="y", color="#eee", lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig_abi_authors.pdf")

# ---------------------------------------------------------------- Chart 2
# Su Shi period curve (Sec. exile). Pre has no CI reported.
periods = ["Pre", "Huangzhou", "Yuan-you", "Huizhou", "Danzhou", "Return"]
pv = np.array([-0.63, -3.28, 1.42, 0.59, 2.79, 2.12])
pl = [None, -5.90, 0.12, -2.34, -0.19, -1.76]
pu = [None, -0.81, 2.76, 3.54, 5.70, 6.04]
x = np.arange(len(periods))
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot(x, pv, "-o", color=BLUE, lw=1.4, ms=5, zorder=3)
for i in range(len(periods)):
    if pl[i] is not None:
        ax.errorbar(x[i], pv[i], yerr=[[pv[i] - pl[i]], [pu[i] - pv[i]]],
                    fmt="none", ecolor="#666", lw=0.8, capsize=2.5)
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(periods, rotation=20, ha="right")
ax.set_ylabel("ABI")
ax.set_title("Su Shi: ABI by period (no net exile effect; Huangzhou dip)")
ax.grid(axis="y", color="#eee", lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig_sushi_period.pdf")

# ---------------------------------------------------------------- Chart 3
# Counterfactual editorial signature (Table tab:rep).
dims = ["Sorrow", "Exile", "Easeful", "Nature", "Void"]
official = [1.39, 0.05, 1.67, 16.68, 1.84]
private = [1.53, 0.15, 0.88, 4.43, 0.03]
x = np.arange(len(dims))
w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.bar(x - w / 2, official, w, label="Imperial (Yu Xuan Song Shi)", color=BLUE)
ax.bar(x + w / 2, private, w, label="Private (Song Shi Chao)", color=ORANGE)
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(dims)
ax.set_ylabel(r"$\delta_d$ per 1k chars")
ax.set_title("Editorial signature: per-dimension edition effect")
ax.legend(fontsize=8.5, frameon=False)
ax.grid(axis="y", color="#eee", lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig_counterfactual_delta.pdf")

# ---------------------------------------------------------------- Chart 4
# Dimension firing rate across the shi corpus (Sec. affective-vs-imagistic).
cdims = ["Exile-journey", "Nature-landscape", "Void-temporality"]
cvals = [31.6, 77.4, 62.5]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.bar(range(3), cvals, color=BLUE, edgecolor="white")
ax.set_xticks(range(3))
ax.set_xticklabels(cdims, rotation=15, ha="right")
ax.set_ylabel("% of shi corpus firing")
ax.set_ylim(0, 100)
ax.set_title("Dimension firing rate across the corpus")
ax.grid(axis="y", color="#eee", lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig_dim_coverage.pdf")

# ---------------------------------------------------------------- Chart 5
# Quintile dose-response (Table tab:quintile).
qn = ["Q1", "Q2", "Q3", "Q4", "Q5"]
qv = [-1.34, -0.59, -0.30, -0.02, 0.55]
colors = [RED if v < 0 else BLUE for v in qv]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.bar(range(5), qv, color=colors, edgecolor="white")
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(range(5))
ax.set_xticklabels(qn)
ax.set_ylabel("Gold ABI mean")
ax.set_title("Dose-response: gold ABI by lexicon-ABI quintile")
ax.grid(axis="y", color="#eee", lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig_quintile.pdf")

print("ALL FIGURES DONE")
