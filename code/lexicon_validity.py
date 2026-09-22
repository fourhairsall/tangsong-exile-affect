# -*- coding: utf-8 -*-
r"""lexicon_validity.py -- evidence for the FIVE-DIMENSIONAL EMOTION LEXICON.

Two complementary validity checks the six-reviewer audit asked for (P0-1):

(1) USAGE / RECALL (frequency coverage). A transparent lexicon is only as good
    as the signal it actually fires on the corpus. For each category we report:
      - n_entries : number of lexical entries (incl. multi-char words)
      - total_hits: raw token hits across the unified shi corpus
      - poems_hit : number of poems containing >=1 entry of the category
      - rate/1k   : total_hits per 1000 corpus characters
    This shows the lexicon is not silently empty on any dimension.

(2) CRITERION VALIDITY. The lexicon's Net-Mind Index should reproduce received
    literary scholarship, not contradict it. We report the NMI ranking of the
    Tang exiles and check it against 尚永亮's characterization of the Yuanhe
    exiles as dominated by anxiety/abandonment (a directional, not exact, gold).

Output: results_lexicon_validity.json  (+ console summary)
"""
import os, glob, json
import numpy as np
import opencc
import lexicon as L

BASE = os.path.dirname(os.path.abspath(__file__))
cc = opencc.OpenCC('t2s')
AUTH = ["柳宗元","刘禹锡","韩愈","白居易","苏轼","欧阳修","黄庭坚","秦观","王禹偁","范仲淹","辛弃疾","陆游"]

def load_poems(folder, authors):
    out = {a: [] for a in authors}
    for fp in glob.glob(folder):
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for p in data:
            a = cc.convert(p.get("author", ""))
            if a in out:
                txt = "".join(p.get("paragraphs", []))
                if txt:
                    out[a].append(cc.convert(txt))
    return out

tang = load_poems(f"{BASE}/data/tang/poet.tang.*.json",
                  ["柳宗元","刘禹锡","韩愈","白居易"])
song = load_poems(f"{BASE}/data/songshi/poet.song.*.json",
                  ["苏轼","欧阳修","黄庭坚","秦观","王禹偁","范仲淹","辛弃疾","陆游"])
shi = {**tang, **song}

all_poems = [t for a in AUTH for t in shi[a]]
total_chars = sum(len(t) for t in all_poems)
n_poems = len(all_poems)

# ---- (1) frequency coverage per category ----
cat_stats = {}
for cat in L.CATS:
    entries = L.LEX[cat]
    total_hits = 0
    poems_hit = 0
    for t in all_poems:
        h = sum(t.count(w) for w in entries)
        total_hits += h
        if h > 0:
            poems_hit += 1
    cat_stats[cat] = {
        "n_entries": len(entries),
        "total_hits": total_hits,
        "poems_hit": poems_hit,
        "poems_hit_pct": round(100.0 * poems_hit / n_poems, 1),
        "rate_per_1k": round(1000.0 * total_hits / total_chars, 3),
    }

# ---- (2) criterion validity: NMI ranking vs 尚永亮's Yuanhe characterization ----
def agg_net(texts):
    n = sum(len(L.strip_punct(t)) for t in texts)
    raw = L.count_cat("".join(texts))  # not per-author; recompute properly below
    return n
# compute per-author NMI via lexicon norms
nmi = {}
for a in AUTH:
    n = sum(len(L.strip_punct(t)) for t in shi[a])
    per1k = L.norm(L.count_cat("".join(shi[a])), n)
    nmi[a] = L.net_index(per1k)

# 尚永亮 (2007) frames the Yuanhe exiles (Liu Yuxi, Han Yu, Liu Zongyuan) as
# dominated by melancholy / anxiety / abandonment. Rank them by NMI (more
# negative = bleaker) and verify they occupy the bleak end.
yuanhe = ["柳宗元","刘禹锡","韩愈"]
ranked = sorted(AUTH, key=lambda a: nmi[a])   # most bleak -> most easeful
yuanhe_in_bottom = sum(1 for a in yuanhe if a in ranked[:5])  # 5 of 12 bleakest

criterion = {
    "scholarly_claim": "Yuanhe exiles (Liu Zongyuan, Liu Yuxi, Han Yu) are "
                       "characterised by 尚永亮 as melancholy/anxious/abandoned",
    "nmi_ranking_most_bleak_first": ranked,
    "yuanhe_exiles_in_bleakest_5_of_12": yuanhe_in_bottom,
    "consistent": yuanhe_in_bottom == len(yuanhe),
}

out = {
    "corpus": {"n_poems": n_poems, "total_chars": total_chars},
    "category_frequency": cat_stats,
    "criterion_validity": criterion,
    "nmi_by_author": {a: round(nmi[a], 3) for a in AUTH},
}
with open(f"{BASE}/results_lexicon_validity.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"corpus: {n_poems} poems, {total_chars} chars")
print(f"{'category':10}{'entries':>8}{'hits':>9}{'poems_hit':>11}{'%poems':>8}{'rate/1k':>9}")
for cat, s in cat_stats.items():
    print(f"{cat:10}{s['n_entries']:>8}{s['total_hits']:>9}{s['poems_hit']:>11}"
          f"{s['poems_hit_pct']:>7}%{s['rate_per_1k']:>9}")
print()
print("NMI ranking (most bleak -> most easeful):")
print("  " + " > ".join(f"{a}({nmi[a]:.2f})" for a in ranked))
print(f"Yuanhe exiles in bleakest 5/12: {yuanhe_in_bottom}/3  -> consistent={criterion['consistent']}")
