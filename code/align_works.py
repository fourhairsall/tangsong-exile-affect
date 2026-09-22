# -*- coding: utf-8 -*-
r"""align_works.py -- cross-edition work-level realignment (重排对齐).

PURPOSE
-------
The user's methodological requirement: for a given author, a work is the *same
work* whichever version carries it; cross-version differences are at most
scribal (抄写) error, not different content.  Therefore every version must be
re-flown onto a common per-work footing before any cross-edition comparison,
so that each author's corpus is comparable edition by edition.

Operationally three things break cross-source comparability and are repaired
here:

1. SCRIPT.  QTS / YD / QSS ship traditional; YXSS / SSC were opencc-converted
   at parse time (author+title in YXSS, everything in SSC).  All author names,
   titles and body text are pushed to simplified before anything is matched.
2. SEGMENTATION GRANULARITY.  QTS splits a poem series into numbered
   sub-records ('帝京篇十首 一/二/三') while YD carries the same series as one
   record ('帝京篇十首').  Title keys therefore have trailing series numerals
   stripped and all records sharing a key are MERGED, so the two editions are
   re-flown onto the same work boundaries.
3. WORK SET.  Anthologies select subsets; identical titles across sources are
   the only genuinely comparable units.  We therefore also report agreement
   restricted to MATCHED works (titles present in both sources), alongside the
   existing POOLED measure.

MODES
-----
POOLED  : every text parsed for that author in that source (reproduces the
          existing Tables tab:tang / tab:songpop / tab:rep and should
          reproduce their A values -- this is the self-check).
MATCHED : only works whose canonical key exists in BOTH sources.

Output: results_cross_edition_aligned.json
"""
import glob
import json
import math
import os
import random
import re
import sys

try:
    import opencc
    _CC = opencc.OpenCC('t2s')
except Exception:                       # pragma: no cover
    sys.exit("opencc unavailable; cannot normalize script")

import lexicon as L

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MINC = 300                              # same inclusion rule as cross_edition_points.py

SOURCES = {
    "QTS":  ("%s/tang/poet.tang.*.json" % DATA, "glob"),
    "YD":   ("%s/yuding/*.json" % DATA, "glob"),
    "QSS":  ("%s/songshi/poet.song.*.json" % DATA, "glob"),
    "YXSS": ("%s/yusong/poet.song_selection.json" % DATA, "file"),
    "SSC":  ("%s/songchao/poet.song_songchao.json" % DATA, "file"),
}

PUNCT = re.compile(r"[\s，。、！？；：·「」『』（）()《》〈〉\[\]—\-…“”‘’·□]")
# trailing series numeral: '帝京篇十首三' -> '帝京篇十首' ; also 其一 / 之一 forms
TRAIL_NUM = re.compile(r"(其|之)?[一二三四五六七八九十百千]+$")
PAREN = re.compile(r"[（(][^）)]*[）)]")


def t2s(s):
    return _CC.convert(s) if s else ""


def load_raw(pattern, kind):
    if kind == "file":
        for enc in ("utf-8", "utf-8-sig"):
            try:
                return json.load(open(pattern, encoding=enc))
            except UnicodeDecodeError:
                continue
        return []
    out = []
    for fp in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, list):
            out.extend(d)
    return out


def norm_title(title, strip_paren=False):
    """Canonical work key: simplified, punctuation-free, series-number-free."""
    t = t2s(title or "")
    if strip_paren:
        t = PAREN.sub("", t)
    t = PUNCT.sub("", t)
    t = TRAIL_NUM.sub("", t)
    return t


def build(pattern, kind, strip_paren=False, name=""):
    """author -> {work_key: [text, ...]} ; records sharing a key are merged."""
    acc = {}
    n_rec = 0
    n_nokey = 0
    for rec in load_raw(pattern, kind):
        n_rec += 1
        a = t2s((rec.get("author") or "").strip())
        k = norm_title(rec.get("title"), strip_paren)
        paras = rec.get("paragraphs") or []
        txt = t2s("".join(paras)) if isinstance(paras, list) else t2s(str(paras))
        txt = txt.replace("\n", "").replace("\r", "").strip()
        if not a or not txt:
            continue
        if not k:
            # Untitled records still count in POOLED mode (they are real parsed
            # text); give them a SOURCE-UNIQUE key (E1) so they can never
            # cross-match between editions -- previously "__notitle_%d" was reset
            # per source and could spuriously collide.
            n_nokey += 1
            k = "__notitle_%s_%d" % (name, n_nokey)
        acc.setdefault(a, {}).setdefault(k, []).append(txt)
    return acc, {"n_records": n_rec, "n_no_title": n_nokey}


def rates(texts):
    per, n = L.cat_per1k(texts)
    if not n:
        return None, 0
    return L.net_index(per), n


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def concord(A, B, authors, mode, label, minc=MINC, thr_first=False):
    """Agreement over shared authors under POOLED or MATCHED pooling.

    minc      : per-author per-source inclusion threshold (content chars).
    thr_first : if True, filter EACH source's authors to those meeting `minc`
                on their POOLED n INDEPENDENTLY, then intersect, then compute A
                (consensus C2②: threshold-before-intersection path).  Otherwise
                the historical path: intersect authors first, then apply `minc`
                inside the mode loop (threshold-after-intersection).
    """
    if thr_first:
        a_ok = set(); b_ok = set()
        for a in A:
            ta = [t for lst in A[a].values() for t in lst]
            _, n = rates(ta)
            if n >= minc:
                a_ok.add(a)
        for b in B:
            tb = [t for lst in B[b].values() for t in lst]
            _, n = rates(tb)
            if n >= minc:
                b_ok.add(b)
        authors = [a for a in authors if a in a_ok and a in b_ok]
    xs, ys, rows = [], [], []
    for a in authors:
        if a not in A or a not in B:
            continue
        if mode == "pooled":
            ta = [t for lst in A[a].values() for t in lst]
            tb = [t for lst in B[a].values() for t in lst]
        else:
            shared = set(A[a]) & set(B[a])
            ta = [t for k in shared for t in A[a][k]]
            tb = [t for k in shared for t in B[a][k]]
        va, na = rates(ta)
        vb, nb = rates(tb)
        if thr_first:
            if va is None or vb is None:
                continue
        else:
            if va is None or vb is None or na < minc or nb < minc:
                continue
        xs.append(va); ys.append(vb)
        rows.append({"author": a, "x": va, "y": vb, "nx": na, "ny": nb})
    r = pearson(xs, ys)
    print("  %-8s n=%-5d A=%.4f%s" % (mode, len(rows), r,
                                      " [thr-first]" if thr_first else ""))
    return {"mode": mode, "label": label, "n": len(rows), "A": round(r, 4),
            "points": rows}


def compare(name, A, B, label):
    authors = sorted(set(A) & set(B))
    print("[%s] shared authors=%d" % (label, len(authors)))
    out = {"pair": name, "label": label, "shared_authors": len(authors),
           "pooled": concord(A, B, authors, "pooled", label),
           "matched": concord(A, B, authors, "matched", label)}

    # coverage: how much text survives realignment (MATCHED vs POOLED)
    tot_p = tot_m = 0
    for a in authors:
        pa = sum(len(t) for lst in A[a].values() for t in lst)
        pb = sum(len(t) for lst in B[a].values() for t in lst)
        sh = set(A[a]) & set(B[a])
        ma = sum(len(t) for k in sh for t in A[a][k])
        mb = sum(len(t) for k in sh for t in B[a][k])
        tot_p += pa + pb
        tot_m += ma + mb
    out["coverage"] = {
        "pooled_chars": tot_p,
        "matched_chars": tot_m,
        "frac": round(tot_m / tot_p, 4) if tot_p else 0.0,
    }
    print("  coverage matched/pooled chars = %.4f (%d / %d)"
          % (out["coverage"]["frac"], tot_m, tot_p))
    return out


# ---------------------------------------------------------------------------
# Consensus-driven enhancements (E2, E4, E5, E6).  All additive; the legacy
# pooled/matched outputs produced by compare() above are untouched.
# ---------------------------------------------------------------------------
GRID = [20, 50, 100, 200, 300, 500, 1000, 2000]


def _percentile(vals, p):
    """Nearest-rank percentile of sorted-ascending `vals` at percentile p (0-100)."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(math.floor(k)); hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _author_level_points(A, B, mode, minc=MINC):
    """List of (va, vb) for every shared author that survives MINC under `mode`."""
    authors = sorted(set(A) & set(B))
    pts = []
    for a in authors:
        if mode == "pooled":
            ta = [t for lst in A[a].values() for t in lst]
            tb = [t for lst in B[a].values() for t in lst]
        else:
            shared = set(A[a]) & set(B[a])
            ta = [t for k in shared for t in A[a][k]]
            tb = [t for k in shared for t in B[a][k]]
        va, na = rates(ta); vb, nb = rates(tb)
        if va is None or vb is None or na < minc or nb < minc:
            continue
        pts.append((va, vb))
    return pts


# --- E2: work-key record-count constraint (consensus C9①) -------------------
def concord_constrained(A, B, authors, mode, label, max_delta=1):
    xs, ys, rows = [], [], []
    n_before = n_after = 0
    for a in authors:
        if a not in A or a not in B:
            continue
        shared = set(A[a]) & set(B[a])
        kept = [k for k in shared
                if abs(len(A[a][k]) - len(B[a][k])) <= max_delta]
        n_before += len(shared); n_after += len(kept)
        if not kept:
            continue
        if mode == "matched":
            ta = [t for k in kept for t in A[a][k]]
            tb = [t for k in kept for t in B[a][k]]
        else:
            # pooled under constraint: keep the author only if it has a
            # record-count-consistent shared work, but aggregate ALL its texts.
            ta = [t for lst in A[a].values() for t in lst]
            tb = [t for lst in B[a].values() for t in lst]
        va, na = rates(ta); vb, nb = rates(tb)
        if va is None or vb is None or na < MINC or nb < MINC:
            continue
        xs.append(va); ys.append(vb)
        rows.append({"author": a, "x": va, "y": vb, "nx": na, "ny": nb})
    r = pearson(xs, ys)
    print("  %-8s[constrained] n=%-5d A=%.4f" % (mode, len(rows), r))
    return {"mode": mode, "label": label, "n": len(rows), "A": round(r, 4),
            "n_shared_before": n_before, "n_shared_after": n_after}


def compare_constrained(name, A, B, label):
    authors = sorted(set(A) & set(B))
    pooled = concord_constrained(A, B, authors, "pooled", label)
    matched = concord_constrained(A, B, authors, "matched", label)
    tot = 0
    for a in authors:
        if a not in A or a not in B:
            continue
        shared = set(A[a]) & set(B[a])
        kept = [k for k in shared
                if abs(len(A[a][k]) - len(B[a][k])) <= 1]
        tot += sum(len(t) for k in kept for t in A[a][k])
        tot += sum(len(t) for k in kept for t in B[a][k])
    return {"pair": name, "label": label,
            "n_shared_before": matched["n_shared_before"],
            "n_shared_after": matched["n_shared_after"],
            "pooled": {"A": pooled["A"], "n": pooled["n"]},
            "matched": {"A": matched["A"], "n": matched["n"]},
            "coverage_constrained_chars": tot}


# --- E4: threshold sensitivity sweep (consensus C3) --------------------------
def sweep_threshold(built, pairs, grid=GRID):
    sweep = {}
    for key, sa, sb, lab in pairs:
        A, B = built[sa], built[sb]
        authors = sorted(set(A) & set(B))
        sweep[key] = {}
        for mode in ("pooled", "matched"):
            lst = []
            for m in grid:
                pts = _author_level_points(A, B, mode, m)
                r = pearson([x for x, _ in pts], [y for _, y in pts])
                n = len(pts)
                if n < 3:                          # skip if too few authors survive
                    continue
                lst.append({"minc": m, "n": n, "A": round(r, 4)})
            sweep[key][mode] = lst
    return sweep


# --- E5: clustered (author-resampled) bootstrap CI (consensus C6) -----------
def bootstrap_A(A, B, mode, label, B0=2000, seed=20260921):
    data = _author_level_points(A, B, mode, MINC)
    n = len(data)
    A_point = pearson([x for x, _ in data], [y for _, y in data])
    rng = random.Random(seed)
    boot = []
    for _ in range(B0):
        idx = rng.choices(range(n), k=n)
        xs = [data[i][0] for i in idx]; ys = [data[i][1] for i in idx]
        r = pearson(xs, ys)
        if r == r:                                # skip nan realisations
            boot.append(r)
    if not boot:
        return {"A": round(A_point, 4), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": n}
    return {"A": round(A_point, 4),
            "ci_low": round(_percentile(boot, 2.5), 4),
            "ci_high": round(_percentile(boot, 97.5), 4),
            "n": n}


def _author_paired_delta(A, B):
    """Per shared author: NMI(matched, source A) - NMI(pooled, source A)."""
    authors = sorted(set(A) & set(B))
    deltas = []
    for a in authors:
        shared = set(A[a]) & set(B[a])
        if not shared:
            continue
        tp = [t for lst in A[a].values() for t in lst]
        tm = [t for k in shared for t in A[a][k]]
        vp, _ = rates(tp); vm, _ = rates(tm)
        if vp is None or vm is None:
            continue
        deltas.append(vm - vp)
    return deltas


def bootstrap_paired_delta(A, B, B0=2000, seed=20260921):
    deltas = _author_paired_delta(A, B)
    n = len(deltas)
    mean = sum(deltas) / n if n else float("nan")
    rng = random.Random(seed)
    boot = []
    for _ in range(B0):
        idx = rng.choices(range(n), k=n)
        s = [deltas[i] for i in idx]
        boot.append(sum(s) / len(s))
    if not boot:
        return {"mean": round(mean, 4), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": n}
    return {"mean": round(mean, 4),
            "ci_low": round(_percentile(boot, 2.5), 4),
            "ci_high": round(_percentile(boot, 97.5), 4),
            "n": n}


# --- E6: composition / alignment decomposition (consensus C2) ---------------
def decompose(name, A, B, label):
    authors = sorted(set(A) & set(B))
    matched_authors = [a for a in authors if (set(A[a]) & set(B[a]))]
    A_pooled_full = concord(A, B, authors, "pooled", label)["A"]
    A_pooled_on_matched = concord(A, B, matched_authors, "pooled", label)["A"]
    A_matched = concord(A, B, authors, "matched", label)["A"]
    return {"pair": name, "label": label,
            "A_pooled_full": A_pooled_full,
            "A_pooled_on_matched_authors": A_pooled_on_matched,
            "A_matched": A_matched,
            "composition_effect": round(A_pooled_on_matched - A_pooled_full, 4),
            "alignment_effect": round(A_matched - A_pooled_on_matched, 4)}


if __name__ == "__main__":
    strip_paren = "--loose" in sys.argv
    print("build variant: %s" % ("loose (strip parentheticals)" if strip_paren
                                 else "strict"))
    built, meta = {}, {}
    for name, (pat, kind) in SOURCES.items():
        built[name], meta[name] = build(pat, kind, strip_paren, name)
        nw = sum(len(v) for v in built[name].values())
        nc = sum(len(t) for v in built[name].values()
                 for lst in v.values() for t in lst)
        print("%-5s records=%-7d authors=%-6d works=%-7d chars=%d"
              % (name, meta[name]["n_records"], len(built[name]), nw, nc))

    out = {"variant": "loose" if strip_paren else "strict", "meta": meta}
    pairs = [
        ("QTS_vs_YD",  "QTS",  "YD",   "Tang: QTS vs YD (same work, two digitizations)"),
        ("QSS_vs_YXSS", "QSS", "YXSS", "Song: QSS vs YuXuan (independent selection)"),
        ("QSS_vs_SSC",  "QSS", "SSC",  "Song: QSS vs SongShiChao (private selection)"),
        ("YXSS_vs_SSC", "YXSS", "SSC", "Song: two anthologies vs each other"),
    ]
    for key, sa, sb, lab in pairs:
        out[key] = compare(key, built[sa], built[sb], lab)

    # ---- E3: threshold ordering (post- vs pre-intersection) -----------------
    out["threshold_order"] = {}
    for key, sa, sb, lab in pairs:
        A, B = built[sa], built[sb]
        auth = sorted(set(A) & set(B))
        post_p = out[key]["pooled"]["A"]
        pre_p = concord(A, B, auth, "pooled", lab, thr_first=True)["A"]
        post_m = out[key]["matched"]["A"]
        pre_m = concord(A, B, auth, "matched", lab, thr_first=True)["A"]
        out["threshold_order"][key] = {
            "pooled": {"post_intersection": post_p, "pre_intersection": pre_p},
            "matched": {"post_intersection": post_m, "pre_intersection": pre_m},
        }

    # ---- E2: work-key record-count constraint ------------------------------
    out["constrained"] = {key: compare_constrained(key, built[sa], built[sb], lab)
                          for key, sa, sb, lab in pairs}

    # ---- E4: threshold sensitivity sweep -----------------------------------
    out["threshold_sweep"] = sweep_threshold(built, pairs, GRID)

    # ---- E6: composition / alignment decomposition --------------------------
    out["decomposition"] = {key: decompose(key, built[sa], built[sb], lab)
                            for key, sa, sb, lab in pairs}

    # ---- E5: clustered bootstrap CI (per pair x mode; strict/loose variant) -
    out["bootstrap_ci"] = {}
    for key, sa, sb, lab in pairs:
        out["bootstrap_ci"][key] = {
            "pooled": bootstrap_A(built[sa], built[sb], "pooled", lab),
            "matched": bootstrap_A(built[sa], built[sb], "matched", lab),
            "paired_delta": bootstrap_paired_delta(built[sa], built[sb]),
        }

    fn = "results_cross_edition_aligned_%s.json" % out["variant"]
    json.dump(out, open(os.path.join(BASE, fn), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nwrote %s" % fn)
