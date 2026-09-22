# -*- coding: utf-8 -*-
"""analyze_v4.py - Tang-Song exiled-literati psychological big-data study (v4).
Adds: (1) full Tang prose (唐文) layer from kanripo 全唐文; (2) DL attribution
(author 12-way + dynasty 唐/宋 + Tang-prose 4-way) with discriminative-feature PCA.
Unified 诗 layer (Tang=chinese-poetry, Song=Book1Q84 全宋诗), both t2s-normalized.
Outputs results_v4.json (reproducible, fixed seed)."""
import sys, os, json, glob, random
sys.path.insert(0, "D:/databuddy/2026-09-14-10-59-40/tangsong_research")
import opencc
import numpy as np
from collections import defaultdict
import lexicon as L
from dl_model import run_dl, run_attribution
cc = opencc.OpenCC('t2s')

BASE = "D:/databuddy/2026-09-14-10-59-40/tangsong_research"
DATA = f"{BASE}/data"
AUTH = ["柳宗元","刘禹锡","韩愈","白居易","苏轼","欧阳修","黄庭坚","秦观","王禹偁","范仲淹","辛弃疾","陆游"]
DYN = {"柳宗元":"唐","刘禹锡":"唐","韩愈":"唐","白居易":"唐",
       "苏轼":"宋","欧阳修":"宋","黄庭坚":"宋","秦观":"宋","王禹偁":"宋","范仲淹":"宋","辛弃疾":"宋","陆游":"宋"}
PO = ["贬前","初贬","远贬"]
PCN = {"贬前":"贬前(入仕/京师)","初贬":"初贬(首谪)","远贬":"远贬(南荒)"}
GEO = {
    "柳宗元":[("长安",108.9,34.3),("永州",111.6,26.4),("柳州",109.4,24.3)],
    "刘禹锡":[("长安",108.9,34.3),("朗州(武陵)",111.7,29.0),("连州",112.4,24.8),("夔州",109.5,31.0),("和州",118.7,31.7),("洛阳",112.4,34.6)],
    "韩愈":[("长安",108.9,34.3),("阳山",112.9,24.5),("潮州",116.6,23.7)],
    "白居易":[("长安",108.9,34.3),("江州",115.9,29.7),("忠州",108.0,30.3)],
    "苏轼":[("汴京",114.3,34.8),("黄州",114.9,30.4),("惠州",114.4,23.1),("儋州",109.6,19.5)],
    "欧阳修":[("汴京",114.3,34.8),("夷陵",111.3,30.7),("滁州",118.3,32.3)],
    "黄庭坚":[("汴京",114.3,34.8),("黔州",108.5,29.5),("戎州",104.6,28.8),("宜州",108.6,24.0)],
    "秦观":[("汴京",114.3,34.8),("郴州",113.0,25.8),("雷州",110.1,20.9),("藤州",110.9,23.4)],
    "王禹偁":[("汴京",114.3,34.8),("商州",109.9,33.9),("滁州",118.3,32.3)],
    "范仲淹":[("汴京",114.3,34.8),("饶州",117.0,29.0)],
    "辛弃疾":[("汴京",114.3,34.8),("上饶",117.9,28.5),("铅山",117.7,28.3)],
    "陆游":[("临安",120.2,30.3),("夔州",109.5,31.0),("成都",104.1,30.7)],
}

def load_poems(folder, authors):
    out = {a: [] for a in authors}
    for fp in glob.glob(folder):
        d = json.load(open(fp, encoding="utf-8"))
        for p in d:
            a = cc.convert(p.get("author", ""))
            if a in out:
                txt = "".join(p.get("paragraphs", []))
                if txt: out[a].append(cc.convert(txt))
    return out

def agg_net(texts):
    if not texts:
        return {"n":0,"chars":0,"cat_per1k":{k:0 for k in L.LEX},"净心指数":0.0}
    per, nch = L.cat_per1k(texts)
    return {"n":len(texts),"chars":nch,"cat_per1k":per,"净心指数":L.net_index(per)}

# ---------- 1. unified 诗 layer ----------
print("[v4] loading 诗 ...")
tang_poems = load_poems(f"{DATA}/tang/poet.tang.*.json",
                        ["柳宗元","刘禹锡","韩愈","白居易"])
song_poems = load_poems(f"{DATA}/songshi/poet.song.*.json",
                        ["苏轼","欧阳修","黄庭坚","秦观","王禹偁","范仲淹","辛弃疾","陆游"])
shi = {**tang_poems, **song_poems}
# ---------- 2. 词 layer (ci.db) ----------
print("[v4] loading 词 ...")
import sqlite3
con = sqlite3.connect(f"{DATA}/ci.db"); cur = con.cursor()
ci = {}
for a in AUTH:
    cur.execute("SELECT content FROM ci WHERE author=?", (a,))
    rows = [cc.convert(r[0]) for r in cur.fetchall()]
    ci[a] = [x for x in rows if x]
con.close()

# ---------- 3. 唐文 full prose (全唐文) ----------
print("[v4] parsing 唐文 (全唐文) ...")
from parse_qtw import parse_qtw
qtw_dir = f"{DATA}/qtw"
tangwen = parse_qtw(qtw_dir) if os.path.exists(qtw_dir) else {}
for a in tangwen:
    print(f"   唐文 {a}: essays={len(tangwen[a])} chars={sum(len(e['text']) for e in tangwen[a])}")

# ---------- 4. curated period / letters (corpus_labeled_v3) ----------
# NOTE: corpus_labeled_v2.json is METADATA-ONLY (no texts). The real curated
# period/letter texts live in corpus_labeled_v3.json:
#   items: [{a, p, g, t, src, txt}],  letters: {author: [{t, src, txt, p}]}
print("[v4] loading curated period/letters (corpus_labeled_v3) ...")
period = {a: {p: {"texts": [], "chars": 0, "genre": defaultdict(int)} for p in PO} for a in AUTH}
letters = defaultdict(lambda: {"texts": [], "n": 0})
CORP_PATH = f"{BASE}/data/corpus_labeled_v3.json"
if os.path.exists(CORP_PATH):
    corp = json.load(open(CORP_PATH, encoding="utf-8"))
    for it in corp.get("items", []):
        a = it.get("a") or it.get("author"); p = it.get("p") or it.get("period")
        g = it.get("g") or it.get("genre") or "诗"; txt = it.get("txt") or it.get("text") or ""
        if a in period and p in period[a]:
            period[a][p]["texts"].append(txt); period[a][p]["chars"] += len(txt)
            period[a][p]["genre"][g] += 1
    for a, lst in corp.get("letters", {}).items():
        if isinstance(lst, list):
            for lt in lst:
                txt = lt.get("txt") or lt.get("text") or ""
                if txt:
                    letters[a]["texts"].append(txt); letters[a]["n"] += 1

# ---------- aggregate lexicon stats ----------
shi_stat = {a: agg_net(shi[a]) for a in AUTH}
ci_stat = {a: agg_net(ci[a]) for a in AUTH}
tangwen_stat = {a: agg_net([e["text"] for e in tangwen.get(a, [])]) | {"n_essays": len(tangwen.get(a, []))}
                for a in ["柳宗元","刘禹锡","韩愈","白居易"]}
period_stat = {}
for a in AUTH:
    period_stat[a] = {}
    for p in PO:
        blk = period[a][p]
        per, _nc = L.cat_per1k(blk["texts"])
        period_stat[a][p] = {"n_works": len(blk["texts"]), "chars": _nc,
                              "cat_per1k": per, "净心指数": L.net_index(per),
                              "genre_dist": dict(blk["genre"])}
letter_stat = {}
for a, blk in letters.items():
    per, _nc = L.cat_per1k(blk["texts"])
    letter_stat[a] = {"n": blk["n"], "cat_per1k": per, "净心指数": L.net_index(per)}

# ---------- 5. DL regression (净心, weak supervision) ----------
print("[v4] DL regression (净心) ...")
def weak_label(t):
    return L.net_per1k([t])[0]
all_shi_texts = [t for a in AUTH for t in shi[a]]
dl_reg = run_dl(all_shi_texts, weak_label, verbose=True)

# ---------- 6. DL attribution ----------
print("[v4] DL author attribution (12-way) ...")
shix, auth_lab, dyn_lab = [], [], []
for a in AUTH:
    for t in shi[a]:
        shix.append(t); auth_lab.append(AUTH.index(a)); dyn_lab.append(0 if DYN[a]=="唐" else 1)
attr_author = run_attribution(shix, auth_lab, AUTH, verbose=True)
print("[v4] DL dynasty attribution (唐/宋) ...")
attr_dynasty = run_attribution(shix, dyn_lab, ["唐","宋"], verbose=True)

# Tang-prose 4-way attribution (bonus)
print("[v4] DL Tang-prose author attribution (4-way) ...")
tw_texts, tw_lab = [], []
tw_auth = ["柳宗元","刘禹锡","韩愈","白居易"]
for i,a in enumerate(tw_auth):
    for e in tangwen.get(a, []):
        tw_texts.append(e["text"]); tw_lab.append(i)
attr_prose = run_attribution(tw_texts, tw_lab, tw_auth, verbose=True) if tw_texts else None

# ---------- assemble ----------
results = {
    "meta": {
        "dl_reg": dl_reg["metrics"],
        "attr_author": {
            "class_names": AUTH,
            "accuracy": attr_author["metrics"]["accuracy"],
            "macro_F1": attr_author["metrics"]["macro_F1"],
            "weighted_F1": attr_author["metrics"]["weighted_F1"],
            "per_class_F1": attr_author["metrics"]["per_class_F1"],
            "confusion_matrix": attr_author["metrics"]["confusion_matrix"],
        },
        "attr_dynasty": {
            "class_names": ["唐","宋"],
            "accuracy": attr_dynasty["metrics"]["accuracy"],
            "macro_F1": attr_dynasty["metrics"]["macro_F1"],
        },
        "attr_prose": ({"class_names": tw_auth,
                        "accuracy": attr_prose["metrics"]["accuracy"],
                        "macro_F1": attr_prose["metrics"]["macro_F1"]} if attr_prose else None),
        "feature_pca": {
            "author": attr_author["pca"].tolist(),
            "labels_author": auth_lab,
            "dynasty": attr_dynasty["pca"].tolist(),
        },
        "corpus_sizes": {
            "shi_total": sum(shi_stat[a]["n"] for a in AUTH),
            "ci_total": sum(ci_stat[a]["n"] for a in AUTH),
            "tangwen_total_essays": sum(tangwen_stat[a]["n_essays"] for a in tangwen_stat),
            "tangwen_total_chars": sum(tangwen_stat[a]["chars"] for a in tangwen_stat),
        },
    },
    "authors": AUTH, "dynasty": DYN, "period_order": PO, "period_cn": PCN, "geo": GEO,
    "shi_stat": shi_stat, "ci_stat": ci_stat, "tangwen_stat": tangwen_stat,
    "period_stat": period_stat, "letters": letter_stat,
    "tangwen_essays": {a: [{"title": e["title"], "text": e["text"]} for e in tangwen.get(a, [])] for a in tangwen},
}
with open(f"{BASE}/results_v4.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("[v4] wrote results_v4.json")
print("唐文 totals:", results["meta"]["corpus_sizes"]["tangwen_total_essays"], "essays,",
      results["meta"]["corpus_sizes"]["tangwen_total_chars"], "chars")
print("Author attr ACC=%.3f macroF1=%.3f | Dynasty attr ACC=%.3f macroF1=%.3f"
      % (attr_author["metrics"]["accuracy"], attr_author["metrics"]["macro_F1"],
         attr_dynasty["metrics"]["accuracy"], attr_dynasty["metrics"]["macro_F1"]))
