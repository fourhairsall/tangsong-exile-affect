# -*- coding: utf-8 -*-
r"""align_variants.py -- identity baseline for the matched-work realignment.

The matched-work concordance of align_works.py could be trivial if the two
sources carry byte-identical strings for every shared work.  This script is
the guard against that:

  * how many shared works are literally identical strings?
  * how similar are the rest? (difflib ratio on capped text)
  * what is the concordance restricted to the NON-identical works only --
    i.e. on exactly those works where the two editions genuinely disagree?

Output: results_align_variants.json
"""
import json
import math
import os
import random
from difflib import SequenceMatcher

from align_works import (SOURCES, build, rates, pearson, MINC, norm_title)

BASE = os.path.dirname(os.path.abspath(__file__))
CAP = 1200          # cap SequenceMatcher input; long poems would dominate runtime
SAMPLE = 1500       # pairwise similarity is estimated on this seeded sample
MIN_CH = 20         # ignore matched pockets too small to say anything


def sim(a, b):
    return SequenceMatcher(None, a[:CAP], b[:CAP]).ratio()


def _sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def analyse(name, A, B, label):
    authors = sorted(set(A) & set(B))
    ident = diff_authors = 0
    n_pair = 0
    sims = []
    variant_pairs = []           # (ta, tb) for every non-identical shared work
    xs, ys = [], []              # non-identical works only
    xs_all, ys_all = [], []      # all matched
    per_author_diff = {}
    # E7: per-poem paired |Δ NMI| over shared works (+ variant-only subset) and
    # the between-author sd of source-A pooled NMI.  These are the "direct
    # evidence" numbers the reviewers asked for.
    work_deltas = []
    variant_work_deltas = []
    author_pooled_nmi = []
    for a in authors:
        shared = set(A[a]) & set(B[a])
        ident_txt, diff_txt = [], []
        n_ident_here = 0
        la, lb = [], []
        # source-A author-level pooled NMI (for between-author sd)
        pa_nmi, _ = rates([t for lst in A[a].values() for t in lst])
        if pa_nmi is not None:
            author_pooled_nmi.append(pa_nmi)
        for k in shared:
            ta = "".join(A[a][k]); tb = "".join(B[a][k])
            if not ta or not tb:
                continue
            if ta == tb:
                ident += 1; n_ident_here += 1
            else:
                diff_authors += 1
                n_pair += 1
                variant_pairs.append((ta, tb))
                diff_txt.append(tb)
                ident_txt.append(ta)
            la.append(ta); lb.append(tb)
            # E7: |Δ NMI| of the two editions' aggregated text for this work
            va_w, _ = rates([ta]); vb_w, _ = rates([tb])
            if va_w is not None and vb_w is not None:
                d = abs(va_w - vb_w)
                work_deltas.append(d)
                if ta != tb:
                    variant_work_deltas.append(d)
        if not la:
            continue
        per_author_diff[a] = {"n_shared": len(shared), "n_identical": n_ident_here}
        va, na = rates(la); vb, nb = rates(lb)
        if na >= MIN_CH and nb >= MIN_CH:
            xs_all.append(va); ys_all.append(vb)
        vd_a, _ = rates(ident_txt); vd_b, _ = rates(diff_txt)
        if vd_a is not None and vd_b is not None:
            xs.append(vd_a); ys.append(vd_b)
    tot = ident + diff_authors
    # similarity estimated on a seeded sample: exact counts above, cheap estimate below
    rng = random.Random(20260921)
    if len(variant_pairs) > SAMPLE:
        sample = rng.sample(variant_pairs, SAMPLE)
    else:
        sample = variant_pairs
    sims = [sim(a, b) for a, b in sample]
    out = {
        "pair": name, "label": label,
        "shared_works": tot,
        "identical_works": ident,
        "frac_identical": round(ident / tot, 4) if tot else 0.0,
        "variant_works": diff_authors,
        "mean_char_similarity": round(sum(sims) / len(sims), 4) if sims else None,
        "A_all_matched": round(pearson(xs_all, ys_all), 4),
        "n_all_matched": len(xs_all),
        "A_variant_only": round(pearson(xs, ys), 4),
        "n_variant_only": len(xs),
    }
    # E7 summary statistics
    mean_abs_delta = round(sum(work_deltas) / len(work_deltas), 4) if work_deltas else None
    mean_abs_delta_variant = round(sum(variant_work_deltas) / len(variant_work_deltas), 4) \
        if variant_work_deltas else None
    between_author_sd = round(_sd(author_pooled_nmi), 4) \
        if len(author_pooled_nmi) > 1 else None
    ratio = round(mean_abs_delta / between_author_sd, 4) \
        if (mean_abs_delta is not None and between_author_sd) else None
    out.update({
        "mean_abs_delta": mean_abs_delta,
        "mean_abs_delta_variant": mean_abs_delta_variant,
        "between_author_sd": between_author_sd,
        "ratio_mean_abs_over_sd": ratio,
        "n_shared_works_for_delta": len(work_deltas),
    })
    print("[%s]" % label)
    print("  shared works      = %d" % tot)
    print("  identical strings = %d (%.1f%%)" % (ident, 100 * out["frac_identical"]))
    print("  variant works     = %d  mean char similarity = %.4f"
          % (diff_authors, out["mean_char_similarity"] or 0))
    print("  A over ALL matched   = %.4f (n=%d)" % (out["A_all_matched"], out["n_all_matched"]))
    print("  A over VARIANT ONLY  = %.4f (n=%d)" % (out["A_variant_only"], out["n_variant_only"]))
    print("  E7 mean|Δ NMI|/work  = %.4f (variant-only %.4f)  between-author sd = %.4f  ratio = %.4f"
          % (mean_abs_delta or 0, mean_abs_delta_variant or 0,
             between_author_sd or 0, ratio or 0))
    return out


if __name__ == "__main__":
    built = {}
    for name, (pat, kind) in SOURCES.items():
        built[name], _ = build(pat, kind, name=name)
        print("built %s" % name)
    pairs = [
        ("QTS_vs_YD",   "QTS",  "YD",   "Tang: QTS vs YD (same work, 2 digitizations)"),
        ("QSS_vs_YXSS", "QSS",  "YXSS", "Song: QSS vs YuXuan (selection)"),
        ("QSS_vs_SSC",  "QSS",  "SSC",  "Song: QSS vs SongShiChao (selection)"),
        ("YXSS_vs_SSC", "YXSS", "SSC",  "Song: two anthologies"),
    ]
    res = {"pairs": {}}
    ppd = {}
    for key, sa, sb, lab in pairs:
        r = analyse(key, built[sa], built[sb], lab)
        res["pairs"][key] = r
        ppd[key] = {
            "mean_abs_delta": r["mean_abs_delta"],
            "mean_abs_delta_variant": r["mean_abs_delta_variant"],
            "between_author_sd": r["between_author_sd"],
            "ratio_mean_abs_over_sd": r["ratio_mean_abs_over_sd"],
            "n_shared_works_for_delta": r["n_shared_works_for_delta"],
        }
    # E7 top-level key (consensus C17)
    res["per_poem_paired_delta"] = ppd
    json.dump(res, open(os.path.join(BASE, "results_align_variants.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote results_align_variants.json")
