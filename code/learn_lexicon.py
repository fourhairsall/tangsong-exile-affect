# -*- coding: utf-8 -*-
"""
Phase 1 — Distributional induction of a classical-Chinese emotion lexicon.

Motivation
----------
The published M1 lexicon (lexicon.py) is hand-curated and oriented toward
modern Chinese reading habits; applying it to literary classical Chinese is a
methodological weak point that invites the "you used an off-the-shelf tool"
criticism.  Here we *induce* a corpus-specific lexicon from the target corpus
itself:

  1. Train character-level distributional embeddings on a documented subsample
     of the Tang+Song classical corpus via PPMI + truncated SVD (no external
     dependency, fully reproducible).
  2. Seed each of the five emotion categories with the existing M1 hand lexicon
     (treated as a human prior), and propagate through the embedding space to
     expand each category to a confidence-scored, corpus-grounded lexicon.

Output
------
  * lexicon_learned.py        -- importable module (LEX_LEARNED, score functions)
  * results_learned_lexicon.json -- training summary + validation numbers

Validation reported here
-------------------------
  * Coverage: fraction of held-out poems firing per category (learned vs M1).
  * Convergent validity: Pearson r between author-level M1-NMI and learned-NMI
    across authors present in the training subsample.
(Cross-edition invariance recheck is handled in Phase 1b.)
"""
import json, glob, re, os, math
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

import lexicon as L  # M1 hand lexicon (the human prior / seed)

RNG = np.random.default_rng(20260915)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ----------------------------------------------------------------------------
# 1. Load a documented, deterministic subsample of the classical corpus
# ----------------------------------------------------------------------------
def load_subsample():
    poems = []          # list of raw paragraph strings
    plan = [
        ("data/tang/poet.tang.*.json", 10),
        ("data/songshi/poet.song.*.json", 10),
        ("data/yuding/*.json", 10),
    ]
    for pat, limit in plan:
        fs = sorted(glob.glob(os.path.join(HERE, pat)))[:limit]
        for f in fs:
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(d, list):
                for e in d:
                    ps = e.get("paragraphs", [])
                    if ps:
                        # keep only CJK ideographs; drop punctuation/whitespace
                        # so that co-occurrence statistics are not corrupted by
                        # high-frequency function tokens (e.g. 。，) that would
                        # otherwise sit near every emotion centroid.
                        text = "".join(to_chars("".join(ps)))
                        if text:
                            poems.append(text)
    return poems

CJK = re.compile(r"[\u4e00-\u9fff]")
def to_chars(text):
    return [c for c in text if CJK.match(c)]

# ----------------------------------------------------------------------------
# 2. Vocabulary + co-occurrence (PPMI)
# ----------------------------------------------------------------------------
def build_embeddings(seqs, min_freq=20, W=4, k=100):
    cnt = Counter()
    for s in seqs:
        cnt.update(s)
    vocab = [c for c, f in cnt.most_common() if f >= min_freq]
    vocab.sort()
    idx = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)

    rows, cols = [], []
    for s in seqs:
        ids = np.array([idx[c] for c in s if c in idx], dtype=np.int32)
        Ln = ids.shape[0]
        for d in range(1, W + 1):
            if Ln > d:
                r = np.concatenate([ids[:-d], ids[d:]])
                c = np.concatenate([ids[d:], ids[:-d]])
                rows.append(r); cols.append(c)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    C = coo_matrix((np.ones(rows.shape[0], dtype=np.float32), (rows, cols)),
                   shape=(V, V)).tocsr()
    C = C + C.T
    C_arr = np.asarray(C.todense(), dtype=np.float32)

    N = C_arr.sum()
    row_sum = C_arr.sum(axis=1)  # symmetric => row sum == col sum
    with np.errstate(divide="ignore", invalid="ignore"):
        Pmi = np.log((C_arr * N) / (row_sum[:, None] * row_sum[None, :]))
    Pmi = np.nan_to_num(Pmi, nan=0.0, posinf=0.0, neginf=0.0)
    Pmi = np.maximum(Pmi, 0.0)

    # truncated SVD, scale by singular values
    U, S, _ = svds(Pmi, k=k)
    emb = U * S  # V x k
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return vocab, idx, emb, {"V": V, "min_freq": min_freq, "W": W, "k": k,
                             "n_pairs": int(rows.shape[0]), "N_cooc": float(N)}

# ----------------------------------------------------------------------------
# 3. Seed propagation: expand each M1 category through the embedding space
# ----------------------------------------------------------------------------
def seed_chars_for(cat_words):
    """Flatten M1 entries (single chars + 2-char phrases) to a seed char set."""
    seeds = set()
    for w in cat_words:
        for c in w:
            if CJK.match(c):
                seeds.add(c)
    return seeds

# A small, documented set of clearly NON-emotional, high-frequency characters
# (numerals, grammatical particles, ubiquitous concrete nouns).  Treated as a
# "human prior" exactly like the M1 seeds.  It defines a corpus-neutral baseline
# used by the neutral-contrast gate to suppress *topic/entity leakage*: kinship
# or plant nouns (e.g. 妻 妾 弟 柏) co-occur with elegiac poems and therefore sit
# near the 悲苦 centroid, yet they are also near this neutral baseline, so their
# (best_cos - neutral_cos) differential stays small and they are rejected.
NEUTRAL_SEED = ("一二人四五六七八九十之其而以於兮矣焉哉乃且夫或然已亦復為與"
               "天地日月星辰江河山水風雲雷雨田園草木花鳥蟲魚車馬衣冠金玉")

def build_neutral_centroid(vocab, idx, emb):
    chars = [c for c in NEUTRAL_SEED if c in idx]
    neu_vec = emb[[idx[c] for c in chars]]
    n = neu_vec.mean(axis=0)
    nn = np.linalg.norm(n)
    return n / nn if nn > 0 else n

def induce_lexicon(vocab, idx, emb, neutral=None, top_n=150,
                   th=0.55, delta=0.13, fallback_floor=15):
    """Corpus-adaptive seed propagation with a *neutral-contrast double gate*.

    For every character we keep the single best-matching emotion category
    (mutually exclusive).  A character is admitted to that category only if BOTH

      * specificity : cos(w, C_best) >= th            (close enough to a pole)
      * contrast    : cos(w, C_best) - cos(w, neutral) >= delta
                                                  (clearly more emotional than
                                                   the corpus-neutral baseline)

    The contrast gate is what removes topic/entity leakage (妻/妾/弟/柏 etc.).
    If a dimension induces fewer than `fallback_floor` words it is transparently
    filled with the expert (M1) seeds and flagged, because its vocabulary sits
    too close to the neutral baseline to be isolated distributionally -- a
    finding about the measurability of that abstract concept, not a failure."""
    if neutral is None:
        neutral = build_neutral_centroid(vocab, idx, emb)
    centroids = {}
    for cat in L.CATS:
        seeds = [c for c in seed_chars_for(L.LEX[cat]) if c in idx]
        if not seeds:
            centroids[cat] = None
            continue
        sv = emb[[idx[c] for c in seeds]]
        cen = sv.mean(axis=0)
        nrm = np.linalg.norm(cen)
        centroids[cat] = cen / nrm if nrm > 0 else cen
    S = np.column_stack([emb @ centroids[cat] for cat in L.CATS])  # V x ncat
    best_cat = np.argmax(S, axis=1)
    best_cos = S[np.arange(S.shape[0]), best_cat]
    sim_neu = emb @ neutral
    diff = best_cos - sim_neu
    # seed-locking: an M1 seed character is *owned* by its category and cannot
    # be reassigned elsewhere.  This prevents, e.g., the collapsed 空幻时空 pole
    # from dumping its seeds (幻 妄) into the nearest surviving pole (曠達).
    owner = {}
    for cat in L.CATS:
        for c in seed_chars_for(L.LEX[cat]):
            if c in idx:
                owner[c] = cat
    cat_index = {c: i for i, c in enumerate(L.CATS)}
    lex = {c: [] for c in L.CATS}
    for j, c in enumerate(vocab):
        if c in owner:
            ci = cat_index[owner[c]]; s = float(S[j, ci]); d = s - float(sim_neu[j])
        else:
            ci = best_cat[j]; s = best_cos[j]; d = diff[j]
        if centroids[L.CATS[ci]] is None or s < th or d < delta:
            continue
        lex[L.CATS[ci]].append((c, float(s), float(d)))
    conf = {}; fallback = {}
    for cat in L.CATS:
        lex[cat].sort(key=lambda x: -x[1])
        chosen = lex[cat][:top_n]
        lex[cat] = [w for w, s, d in chosen]
        conf[cat] = float(np.mean([s for w, s, d in chosen])) if chosen else 0.0
        if len(lex[cat]) < fallback_floor:
            extra = [w for w in L.LEX[cat] if w in idx and w not in lex[cat]]
            lex[cat] = lex[cat] + extra
            fallback[cat] = True
            conf[cat] = float("nan")
        else:
            fallback[cat] = False
    return lex, conf, fallback

# ----------------------------------------------------------------------------
# 4. Validation: coverage + convergent validity vs M1
# ----------------------------------------------------------------------------
def count_cat(text, lex):
    return {c: sum(text.count(w) for w in ws) for c, ws in lex.items()}

def norm_cat(raw, n):
    if not n:
        return {k: 0.0 for k in raw}
    return {k: round(v / n * 1000, 3) for k, v in raw.items()}

def net_index(per, pos="旷达闲适", neg="悲苦孤寂"):
    return round(per[pos] - per[neg], 3)

def coverage_on(poems, lex):
    per_cat = {c: 0 for c in lex}
    any_hit = 0
    for t in poems:
        n = len([c for c in t if CJK.match(c)])
        c = count_cat(t, lex)
        fired = False
        for cat, v in c.items():
            if v > 0:
                per_cat[cat] += 1; fired = True
        if fired:
            any_hit += 1
    m = len(poems) or 1
    return {c: round(per_cat[c] / m, 4) for c in per_cat}, round(any_hit / m, 4)

def author_validity(seqs):
    # group poems by author within subsample, author-level NMI via M1 and learned
    by_author = {}
    # rebuild author map from the same files we loaded? simpler: re-load with authors
    return by_author

# ----------------------------------------------------------------------------
def main():
    print("[1/5] loading subsample ...")
    poems = load_subsample()
    print(f"      poems={len(poems)} chars={sum(len(to_chars(p)) for p in poems)}")
    seqs = [to_chars(p) for p in poems]

    print("[2/5] training PPMI-SVD embeddings ...")
    vocab, idx, emb, meta = build_embeddings(seqs)
    print(f"      vocab={meta['V']} pairs={meta['n_pairs']:,} k={meta['k']}")
    # persist embedding for fast downstream diagnostics
    np.savez(os.path.join(HERE, "classical_ppmi_svd.npz"),
             vocab=np.array(vocab, dtype=object), emb=emb)

    print("[3/5] seed propagation (induce lexicon) ...")
    lex, conf, fallback = induce_lexicon(vocab, idx, emb)
    for c in L.CATS:
        tag = "  [expert-seed fallback]" if fallback[c] else ""
        cf = "nan" if conf[c] != conf[c] else f"{conf[c]:.3f}"
        print(f"      {c}: {len(lex[c])} chars  mean_cos={cf}{tag}")

    # ---- write importable module ----
    print("[4/5] writing lexicon_learned.py ...")
    cats = L.CATS
    module = "# -*- coding: utf-8 -*-\n"
    module += '"""Corpus-induced classical-Chinese emotion lexicon (Phase 1).\n'
    module += "Learned via PPMI+SVD character embeddings + M1 seed propagation,\n"
    module += "with a neutral-contrast double gate (specificity AND contrast vs a\n"
    module += "curated non-emotional baseline) that removes topic/entity leakage.\n"
    module += "Categories identical to M1. Dimensions that fail to induce cleanly\n"
    module += "(e.g. the abstract 空幻时空) fall back to the expert M1 seeds and\n"
    module += "are flagged in LEX_FALLBACK.\n"
    module += '"""\n\n'
    module += "CATS = " + repr(cats) + "\n\n"
    module += "LEX_FALLBACK = " + repr(fallback) + "\n\n"
    module += "CATS = " + repr(cats) + "\n\n"
    module += "LEX_LEARNED = {\n"
    for c in cats:
        module += f"    {c!r}: {lex[c]!r},\n"
    module += "}\n\n"
    module += "NET = ('旷达闲适', '悲苦孤寂')\n\n"
    module += "def count_cat(text):\n"
    module += "    return {c: sum(text.count(w) for w in ws) for c, ws in LEX_LEARNED.items()}\n\n"
    module += "def norm(cat_raw, n_char):\n"
    module += "    if not n_char: return {k: 0.0 for k in cat_raw}\n"
    module += "    return {k: round(v / n_char * 1000, 3) for k, v in cat_raw.items()}\n\n"
    module += "def net_index(cat_per1k):\n"
    module += "    return round(cat_per1k[NET[0]] - cat_per1k[NET[1]], 3)\n"
    with open(os.path.join(HERE, "lexicon_learned.py"), "w", encoding="utf-8") as f:
        f.write(module)

    # ---- validation: coverage on held-out files ----
    print("[5/5] validation (coverage + convergent validity) ...")
    held = []
    for pat, start, lim in [("data/songshi/poet.song.*.json", 10, 15)]:
        fs = sorted(glob.glob(os.path.join(HERE, pat)))[start:lim]
        for f in fs:
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(d, list):
                for e in d:
                    ps = e.get("paragraphs", [])
                    if ps:
                        held.append("".join(ps))
    cov_m1, any_m1 = coverage_on(held, L.LEX)
    cov_ln, any_ln = coverage_on(held, lex)

    # convergent validity: author-level NMI across subsample authors
    # re-load with authors from the SAME subsample plan
    auth = {}
    plan = [("data/tang/poet.tang.*.json", 10),
            ("data/songshi/poet.song.*.json", 10)]
    for pat, limit in plan:
        fs = sorted(glob.glob(os.path.join(HERE, pat)))[:limit]
        for f in fs:
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(d, list):
                for e in d:
                    a = e.get("author", "?")
                    ps = e.get("paragraphs", [])
                    if ps:
                        auth.setdefault(a, []).append("".join(ps))
    rows_m1, rows_ln = [], []
    for a, txts in auth.items():
        nch = sum(len(to_chars(t)) for t in txts)
        if nch < 500:
            continue
        m1 = norm_cat(count_cat("".join(txts), L.LEX), nch)
        ln = norm_cat(count_cat("".join(txts), lex), nch)
        rows_m1.append(net_index(m1)); rows_ln.append(net_index(ln))
    if len(rows_m1) > 2:
        r = float(np.corrcoef(rows_m1, rows_ln)[0, 1])
    else:
        r = None

    conf_clean = {c: (None if (conf[c] != conf[c]) else round(conf[c], 4)) for c in cats}
    result = {
        "meta": meta,
        "n_train_poems": len(poems),
        "lexicon_sizes": {c: len(lex[c]) for c in cats},
        "lexicon_mean_cos": conf_clean,
        "lexicon_fallback_to_M1_seeds": fallback,
        "coverage_heldout_M1": cov_m1, "coverage_heldout_learned": cov_ln,
        "any_hit_M1": any_m1, "any_hit_learned": any_ln,
        "convergent_validity_author_pearson_r": (round(r, 4) if r is not None else None),
        "n_authors_validated": len(rows_m1),
        "gate": {"th": 0.55, "delta": 0.13,
                 "note": "specificity (cos to best pole) AND neutral-contrast (cos-best minus cos-neutral); M1 seed chars locked to own category"},
    }
    with open(os.path.join(HERE, "results_learned_lexicon.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
