# -*- coding: utf-8 -*-
"""patch_punct_fix.py -- apply the P0 punctuation-normalisation fix across the
pipeline. Every per-1000-CHAR rate that previously divided by len(text)
(including ~14-16% CBETA punctuation in the full collections) now divides by the
CONTENT character count (punctuation stripped from the denominator only; lexicon
counts stay on the original text so no spurious adjacent bigrams are created).

Each replacement is asserted to exist exactly the expected number of times before
any file is written, so a mismatch fails loudly instead of corrupting a file.
"""
import io, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))

# (relpath, old, new, expected_occurrences) ; use expected=-1 to allow >=1
PATCHES = [
 # ---------------- analyze_v4.py ----------------
 ("analyze_v4.py",
  '''    tot = defaultdict(int); nch = 0
    for t in texts:
        for k, w in L.LEX.items():
            tot[k] += sum(t.count(x) for x in w)
        nch += len(t)
    per = L.norm(dict(tot), nch)
    return {"n":len(texts),"chars":nch,"cat_per1k":per,"净心指数":L.net_index(per)}''',
  '''    per, nch = L.cat_per1k(texts)
    return {"n":len(texts),"chars":nch,"cat_per1k":per,"净心指数":L.net_index(per)}''', 1),

 ("analyze_v4.py",
  '''def weak_label(t):
    c = L.count_cat(t); n = len(t) or 1
    return L.net_index(L.norm(c, n))''',
  '''def weak_label(t):
    return L.net_per1k([t])[0]''', 1),

 ("analyze_v4.py",
  '''        cat = L.count_cat("".join(blk["texts"]))
        per = L.norm(cat, blk["chars"]) if blk["chars"] else {k:0 for k in L.LEX}
        period_stat[a][p] = {"n_works": len(blk["texts"]), "chars": blk["chars"],''',
  '''        per, _nc = L.cat_per1k(blk["texts"])
        period_stat[a][p] = {"n_works": len(blk["texts"]), "chars": _nc,''', 1),

 ("analyze_v4.py",
  '''    cat = L.count_cat("".join(blk["texts"])); per = L.norm(cat, sum(len(x) for x in blk["texts"]))
    letter_stat[a] = {"n": blk["n"], "cat_per1k": per, "净心指数": L.net_index(per)}''',
  '''    per, _nc = L.cat_per1k(blk["texts"])
    letter_stat[a] = {"n": blk["n"], "cat_per1k": per, "净心指数": L.net_index(per)}''', 1),

 # ---------------- cross_edition_points.py ----------------
 ("cross_edition_points.py",
  '''def rates(texts):
    nch = sum(len(t) for t in texts) or 1
    per = L.norm(L.count_cat("".join(texts)), nch)
    return per, nch''',
  '''def rates(texts):
    return L.cat_per1k(texts)''', 1),

 # ---------------- phase1b_invariance.py / phase1b_song.py ----------------
 ("phase1b_invariance.py",
  '''        nch = sum(len(t) for t in texts)
        raw = lexmod.count_cat("".join(texts))
        per = lexmod.norm(raw, nch)''',
  '''        per, nch = lexmod.cat_per1k(texts)''', 1),

 ("phase1b_song.py",
  '''        nch = sum(len(t) for t in texts)
        raw = lexmod.count_cat("".join(texts))
        per = lexmod.norm(raw, nch)''',
  '''        per, nch = lexmod.cat_per1k(texts)''', 1),

 # ---------------- align_works.py (rate denom) ----------------
 ("align_works.py",
  '''def rates(texts):
    joined = "".join(texts)
    n = len(joined)
    if not n:
        return None, 0
    per = L.norm(L.count_cat(joined), n)
    return L.net_index(per), n''',
  '''def rates(texts):
    per, n = L.cat_per1k(texts)
    if not n:
        return None, 0
    return L.net_index(per), n''', 1),

 # ---------------- align_variants.py (variant-only bug) ----------------
 ("align_variants.py",
  '''        if ta == tb:
            ident += 1; n_ident_here += 1
            ident_txt.append(ta); diff_txt.append(tb)
        else:
            diff_authors += 1
            n_pair += 1
            variant_pairs.append((ta, tb))
            diff_txt.append(tb)
            ident_txt.append(ta)
        la.append(ta); lb.append(tb)''',
  '''        if ta == tb:
            ident += 1; n_ident_here += 1
        else:
            diff_authors += 1
            n_pair += 1
            variant_pairs.append((ta, tb))
            diff_txt.append(tb)
            ident_txt.append(ta)
        la.append(ta); lb.append(tb)''', 1),

 # ---------------- hb_model.py ----------------
 ("hb_model.py",
  '''    for a, texts in agg.items():
        nch = sum(len(t) for t in texts)
        if nch == 0:
            continue
        raw = L.count_cat("".join(texts))
        per = L.norm(raw, nch)
        out[a] = {"nmi": L.net_index(per), "n": nch, "per": per}''',
  '''    for a, texts in agg.items():
        per, nch = L.cat_per1k(texts)
        if nch == 0:
            continue
        out[a] = {"nmi": L.net_index(per), "n": nch, "per": per}''', 1),

 ("hb_model.py",
  '''        Pq = "".join(qs_poems[a]); nq = len(Pq)
        per_q = L.norm(L.count_cat(Pq), nq)
        n_anth = len(yx[a])''',
  '''        per_q, nq = L.cat_per1k(qs_poems[a])
        _, n_anth = L.cat_per1k([yx[a]])''', 2),

 # ---------------- counterfactual.py ----------------
 ("counterfactual.py",
  '''    for a, texts in agg.items():
        nch = sum(len(t) for t in texts)
        if nch == 0:
            continue
        per = L.norm(L.count_cat("".join(texts)), nch)
        out[a] = {"chars": nch, "per": per, "nmi": per[EASE] - per[BITTER]}''',
  '''    for a, texts in agg.items():
        per, nch = L.cat_per1k(texts)
        if nch == 0:
            continue
        out[a] = {"chars": nch, "per": per, "nmi": per[EASE] - per[BITTER]}''', 1),

 # ---------------- song3_rep.py ----------------
 ("song3_rep.py",
  '''    for a, texts in agg.items():
        txt = "".join(texts)
        n = len(txt)
        if n == 0:
            continue
        per = L.norm(L.count_cat(txt), n)
        out[a] = {"nmi": L.net_index(per), "per": per, "n": n, "text": txt}''',
  '''    for a, texts in agg.items():
        per, n = L.cat_per1k(texts)
        if n == 0:
            continue
        out[a] = {"nmi": L.net_index(per), "per": per, "n": n, "text": "".join(texts)}''', 1),

 ("song3_rep.py",
  '''        per_q = L.norm(L.count_cat(qs[a]), len(qs[a]))''',
  '''        per_q, _nq = L.cat_per1k([qs[a]])''', 1),

 ("song3_rep.py",
  '''                per = L.norm(L.count_cat(txt), len(txt))''',
  '''                per, _ = L.cat_per1k([txt])''', 1),

 # ---------------- song3_matched.py ----------------
 ("song3_matched.py",
  '''        per = L.norm(L.count_cat(txt), len(txt))
        out[a] = {"nmi": L.net_index(per), "per": per, "n": len(txt)}''',
  '''        per, n = L.cat_per1k([txt])
        out[a] = {"nmi": L.net_index(per), "per": per, "n": n}''', 1),

 # ---------------- gold_validity.py (split-half) ----------------
 ("gold_validity.py",
  '''        po = L.norm(L.count_cat(o), len(o))
        pe = L.norm(L.count_cat(e), len(e))''',
  '''        po, _no = L.cat_per1k([o])
        pe, _ne = L.cat_per1k([e])''', 1),

 # ---------------- methods_compare_v5.py ----------------
 ("methods_compare_v5.py",
  '''def agg_net(texts):
    if not texts:
        return {"n":0,"chars":0,"净心指数":0.0}
    tot = defaultdict(int); nch = 0
    for t in texts:
        for k, w in L.LEX.items():
            tot[k] += sum(t.count(x) for x in w)
        nch += len(t)
    per = L.norm(dict(tot), nch)
    return {"n":len(texts),"chars":nch,"净心指数":L.net_index(per)}''',
  '''def agg_net(texts):
    if not texts:
        return {"n":0,"chars":0,"净心指数":0.0}
    per, nch = L.cat_per1k(texts)
    return {"n":len(texts),"chars":nch,"净心指数":L.net_index(per)}''', 1),

 ("methods_compare_v5.py",
  '''    for t in shi[a]:
        for k, w in L.LEX.items():
            tot[k] += sum(t.count(x) for x in w)
        nch += len(t)
    m1_cat[a] = (dict(tot), nch)''',
  '''    for t in shi[a]:
        for k, w in L.LEX.items():
            tot[k] += sum(t.count(x) for x in w)
        nch += len(L.strip_punct(t))
    m1_cat[a] = (dict(tot), nch)''', 1),

 ("methods_compare_v5.py",
  '''            c = t.count(term)
            if c: bei += c * idf[term]
        nch += len(t)
    M2[a] = (kuang - bei) / (nch / 1000.0) if nch else 0.0''',
  '''            c = t.count(term)
            if c: bei += c * idf[term]
        nch += len(L.strip_punct(t))
    M2[a] = (kuang - bei) / (nch / 1000.0) if nch else 0.0''', 1),

 # ---------------- sign_stability.py ----------------
 ("sign_stability.py",
  '''                c = t.count(w)
                if c:
                    cb[j] += c
            nch += len(t)''',
  '''                c = t.count(w)
                if c:
                    cb[j] += c
            nch += len(L.strip_punct(t))''', 1),

 # ---------------- build_dl_pool.py ----------------
 ("build_dl_pool.py",
  '''        c = L.count_cat(t)
        per = L.norm(c, len(t))
        nmi = L.net_index(per)''',
  '''        per, _ = L.cat_per1k([t])
        nmi = L.net_index(per)''', 1),

 # ---------------- xval_yuding.py ----------------
 ("xval_yuding.py",
  '''    yuding_stat[a] = L.norm(L.count_cat("".join(texts[a])), sum(len(t) for t in texts[a])) if texts[a] else {k:0 for k in L.LEX}
    yuding_stat[a]["__n__"] = len(texts[a])
    yuding_stat[a]["__chars__"] = sum(len(t) for t in texts[a])
    yuding_stat[a]["净心指数"] = L.net_index(yuding_stat[a])''',
  '''    if texts[a]:
        per, nch = L.cat_per1k(texts[a])
        yuding_stat[a] = dict(per)
        yuding_stat[a]["__n__"] = len(texts[a])
        yuding_stat[a]["__chars__"] = nch
    else:
        yuding_stat[a] = {k: 0.0 for k in L.LEX}
        yuding_stat[a]["__n__"] = 0
        yuding_stat[a]["__chars__"] = 0
    yuding_stat[a]["净心指数"] = L.net_index(yuding_stat[a])''', 1),

 # ---------------- song2_xval.py ----------------
 ("song2_xval.py",
  '''def stats(text):
    n = len(text)
    raw = L.count_cat(text)
    per1k = L.norm(raw, n)
    net = L.net_index(per1k)
    return n, per1k, net''',
  '''def stats(text):
    per1k, n = L.cat_per1k([text])
    return n, per1k, L.net_index(per1k)''', 1),

 # ---------------- exile_dose.py ----------------
 ("exile_dose.py",
  '''        t = cc.convert(t_raw)
        n = len(t)
        cnt = np.array([sum(t.count(w) for w in ws) for ws in L.LEX.values()], dtype=float)''',
  '''        t = cc.convert(t_raw)
        n = len(L.strip_punct(t))
        cnt = np.array([sum(t.count(w) for w in ws) for ws in L.LEX.values()], dtype=float)''', 1),
]

fails = 0
ok = 0
for rel, old, new, exp in PATCHES:
    fp = os.path.join(BASE, rel)
    if not os.path.exists(fp):
        print(f"MISSING FILE: {rel}"); fails += 1; continue
    src = open(fp, encoding="utf-8").read()
    cnt = src.count(old)
    if exp == -1:
        ok_exp = cnt >= 1
    else:
        ok_exp = (cnt == exp)
    if not ok_exp:
        print(f"MISMATCH ({cnt}!={exp}): {rel}")
        fails += 1
        continue
    src = src.replace(old, new)
    open(fp, "w", encoding="utf-8").write(src)
    print(f"PATCHED ({cnt}): {rel}")
    ok += 1

print(f"\n=== {ok} patched, {fails} failed ===")
sys.exit(1 if fails else 0)
