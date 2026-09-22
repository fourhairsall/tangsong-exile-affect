# -*- coding: utf-8 -*-
r"""make_nmi_figure.py -- invariance-continuum figure for the NLP paper.

Replaces the earlier bar chart, which (a) hard-coded the Song 7-poet pilot
values and (b) therefore carried the mis-parsed 陆游 artifact (-7.011).  We now
plot the POPULATION-SCALE comparison directly, straight from
cross_edition_points.json (produced by cross_edition_points.py):

  Left : Tang, QTS vs YD  -- two digitizations of the SAME work   (n=479)
  Right: Song, QSS vs YXS -- complete collection vs imperial SELECTION (n=256)

Each point is one author (Net-Mind Index in edition A vs edition B); the
diagonal is perfect agreement.  The contrast -- a tight cloud vs a loose one --
is the paper's central claim made visible.

Outputs (vector + raster) into ./figures/ :
  nmi_cross_edition.pdf / .png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "..", "cross_edition_points.json")
if not os.path.exists(SRC):
    SRC = os.path.join(BASE, "cross_edition_points.json")
D = json.load(open(SRC, encoding="utf-8"))

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.titlesize": 11, "axes.labelsize": 10})

C = "#3b6ea5"


def panel(ax, rec, title, note, lo, hi):
    x = np.array([p["x"] for p in rec["points"]], float)
    y = np.array([p["y"] for p in rec["points"]], float)
    ax.scatter(x, y, s=11, c=C, alpha=0.55, edgecolors="white", linewidths=0.35)
    ax.plot([lo, hi], [lo, hi], "--", color="#999", linewidth=1.0, zorder=0)
    ax.axhline(0, color="#ddd", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#ddd", linewidth=0.7, zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(r"NMI in edition A  (per 1k chars)")
    ax.set_ylabel(r"NMI in edition B  (per 1k chars)")
    ax.text(0.04, 0.955, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.4, color="#222",
            bbox=dict(boxstyle="round,pad=0.32", fc="#f4f4f4", ec="#ccc", lw=0.5))
    ax.grid(color="#eaeaea", linewidth=0.7)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0))

t = D["Tang_same_work"]
panel(axes[0], t, "Tang: QTS vs. Yu Ding (same work)",
      f"$n={t['n']}$ authors\nconcordance $A={t['r_nmi']:.2f}$\nstructure $S={t['S']}/5$",
      -6.0, 6.0)

s = D["Song_corpus_vs_selection"]
panel(axes[1], s, "Song: QSS vs. Yu Xuan (selection)",
      f"$n={s['n']}$ authors\nconcordance $A={s['r_nmi']:.2f}$\nstructure $S={s['S']}/5$",
      -6.0, 6.0)

plt.tight_layout()
os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
fig.savefig(os.path.join(BASE, "figures", "nmi_cross_edition.pdf"),
            bbox_inches="tight", dpi=150)
fig.savefig(os.path.join(BASE, "figures", "nmi_cross_edition.png"),
            bbox_inches="tight", dpi=150)
print("saved figures/nmi_cross_edition.pdf and .png")
