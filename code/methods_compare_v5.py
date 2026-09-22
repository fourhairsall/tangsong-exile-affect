# -*- coding: utf-8 -*-
"""methods_compare_v5.py — 多方法互校 (method triangulation) for the Tang-Song
exiled-literati study. Three independent goals:

  (A) NET-MIND INDEX — compare 3 *different* estimation methods:
        M1 词典计数法           (rule-based additive counting, baseline)
        M2 TF-IDF 显著性加权法  (same lexicon, IDF-weighted salience)
        M3 LSA 分布语义法       (anchor-word cosine in SVD space; NO hand lexicon)
      -> report per-author values + inter-method correlation (Pearson/Spearman).

  (B) AUTHORSHIP / DYNASTY ATTRIBUTION — compare 3 *different* models:
        A1 字符级 CNN          (deep, from results_v4.json — not retrained)
        A2 TF-IDF + 逻辑回归     (shallow linear baseline)
        A3 TF-IDF + 线性 SVM     (shallow linear baseline)
      -> same stratified 85/15 split for fair comparison.

  (C) LDA TOPIC-MODEL VALIDATION — unsupervised check that 旷达/悲苦 poles
      emerge as latent topics; derive per-topic valence from lexicon and
      correlate an LDA-based net-mind with M1.

Outputs: results_v5_methods.json + 3 figures under figures/.
"""
import sys, os, json, glob, random, re
from collections import defaultdict
import numpy as np
import opencc
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, "D:/databuddy/2026-09-14-10-59-40/tangsong_research")
import lexicon as L
cc = opencc.OpenCC('t2s')
CJK = re.compile(r'[\u4e00-\u9fff]')
def clean(t):
    return ''.join(CJK.findall(t))

BASE = "D:/databuddy/2026-09-14-10-59-40/tangsong_research"
DATA = f"{BASE}/data"
AUTH = ["柳宗元","刘禹锡","韩愈","白居易","苏轼","欧阳修","黄庭坚","秦观","王禹偁","范仲淹","辛弃疾","陆游"]
DYN = {"柳宗元":"唐","刘禹锡":"唐","韩愈":"唐","白居易":"唐",
       "苏轼":"宋","欧阳修":"宋","黄庭坚":"宋","秦观":"宋","王禹偁":"宋","范仲淹":"宋","辛弃疾":"宋","陆游":"宋"}
SEED = 20260915
random.seed(SEED); np.random.seed(SEED)

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
        return {"n":0,"chars":0,"净心指数":0.0}
    per, nch = L.cat_per1k(texts)
    return {"n":len(texts),"chars":nch,"净心指数":L.net_index(per)}

print("[v5] loading 诗 ...")
tang_poems = load_poems(f"{DATA}/tang/poet.tang.*.json", ["柳宗元","刘禹锡","韩愈","白居易"])
song_poems = load_poems(f"{DATA}/songshi/poet.song.*.json",
                        ["苏轼","欧阳修","黄庭坚","秦观","王禹偁","范仲淹","辛弃疾","陆游"])
shi = {**tang_poems, **song_poems}
n_total = sum(len(v) for v in shi.values())
print(f"[v5] unified 诗: {n_total} poems across {len(AUTH)} authors")

# ----------  (A) NET-MIND: M1 vs M2 vs M3  ----------
print("[v5] (A) net-mind methods ...")
M1 = {a: agg_net(shi[a])["净心指数"] for a in AUTH}
# M1 raw category counts for later normalization
m1_cat = {}
for a in AUTH:
    tot = defaultdict(int); nch = 0
    for t in shi[a]:
        for k, w in L.LEX.items():
            tot[k] += sum(t.count(x) for x in w)
        nch += len(L.strip_punct(t))
    m1_cat[a] = (dict(tot), nch)

# M2 — TF-IDF salience-weighted lexicon net-mind
all_poems = [t for a in AUTH for t in shi[a]]
# document frequency of each lexicon term across all poems
term_df = defaultdict(int)
all_terms = set()
for k, w in L.LEX.items():
    all_terms.update(w)
for t in all_poems:
    seen = set()
    for term in all_terms:
        if term in t:
            seen.add(term)
    for term in seen:
        term_df[term] += 1
N = len(all_poems)
def math_safe(n, df):
    import math
    return math.log((n + 1) / (1 + df)) + 1.0  # smoothed idf
idf = {term: math_safe(N, term_df[term]) for term in all_terms}

M2 = {}
for a in AUTH:
    kuang = bei = 0.0; nch = 0
    for t in shi[a]:
        for term in L.LEX["旷达闲适"]:
            c = t.count(term)
            if c: kuang += c * idf[term]
        for term in L.LEX["悲苦孤寂"]:
            c = t.count(term)
            if c: bei += c * idf[term]
        nch += len(L.strip_punct(t))
    M2[a] = (kuang - bei) / (nch / 1000.0) if nch else 0.0

# M3 — LSA distributional net-mind (independent method: geometry, not counting)
print("[v5]   fitting TF-IDF + TruncatedSVD (cleaned CJK) ...")
poems_clean = [clean(t) for t in all_poems]
vec = TfidfVectorizer(analyzer="char", ngram_range=(1,2), min_df=3,
                      sublinear_tf=True, max_features=20000)
X = vec.fit_transform(poems_clean)
print(f"[v5]   TF-IDF matrix {X.shape}")
svd = TruncatedSVD(n_components=200, random_state=SEED)
Xs = svd.fit_transform(X)
print(f"[v5]   SVD done, explained variance ratio sum={svd.explained_variance_ratio_.sum():.3f}")

# multi-character affective seed phrases (independent concept anchors, NOT the counting lexicon)
SEED_K = ["闲适","悠然","逍遥","达观","放旷","恬淡","自适","洒脱","旷逸","超然","淡泊","萧散","旷达","乐天"]
SEED_B = ["悲愁","孤寂","愁怨","凄楚","幽怨","悲愤","寂寥","哀怨","孤愤","凄怨","忧愤","惨怛","悲苦","孤寒"]
def anchor_centroid(phrases):
    vecs = []
    for ph in phrases:
        doc = vec.transform([ph])
        if np.asarray(doc.sum(axis=1))[0] > 0:
            vecs.append(svd.transform(doc))
    if not vecs:
        return np.zeros(Xs.shape[1])
    return np.mean(np.vstack(vecs), axis=0)
ck = anchor_centroid(SEED_K); cb = anchor_centroid(SEED_B)
def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na*nb)) if na and nb else 0.0
poem_nm = []; poem_auth = []
for a in AUTH:
    for t in shi[a]:
        v = svd.transform(vec.transform([clean(t)]))[0]
        poem_nm.append(cos(v, ck) - cos(v, cb)); poem_auth.append(a)
M3 = {a: float(np.mean([poem_nm[i] for i in range(len(poem_nm)) if poem_auth[i]==a])) for a in AUTH}

def corr(d1, d2, keys=AUTH):
    x = np.array([d1[k] for k in keys], float)
    y = np.array([d2[k] for k in keys], float)
    pr = pearsonr(x, y)[0]; sr = spearmanr(x, y)[0]
    return float(pr), float(sr)

c12 = corr(M1, M2); c13 = corr(M1, M3); c23 = corr(M2, M3)
print(f"[v5] net-mind M1 vs M2: Pearson={c12[0]:.3f} Spearman={c12[1]:.3f}")
print(f"[v5] net-mind M1 vs M3: Pearson={c13[0]:.3f} Spearman={c13[1]:.3f}")
print(f"[v5] net-mind M2 vs M3: Pearson={c23[0]:.3f} Spearman={c23[1]:.3f}")

# ----------  (C) LDA TOPIC-MODEL VALIDATION  ----------
print("[v5] (C) LDA validation (2-char word topics, cleaned) ...")
lda_vec = TfidfVectorizer(analyzer="char", ngram_range=(2,2), min_df=10, max_features=15000)
Xc = lda_vec.fit_transform(poems_clean)
lda = LatentDirichletAllocation(n_components=10, random_state=SEED, max_iter=20, n_jobs=1)
topic_dist = lda.fit_transform(Xc)
terms = np.array(lda_vec.get_feature_names_out())
SEED_KSET = set(SEED_K); SEED_BSET = set(SEED_B)
topic_top = []
for ti in range(10):
    top = terms[np.argsort(lda.components_[ti])[::-1][:15]].tolist()
    vk = sum(1 for w in top if w in SEED_KSET)
    vb = sum(1 for w in top if w in SEED_BSET)
    # category overlap with each lexicon category (2-char subset) -> validates construct
    cat_overlap = {}
    for cat, wlist in L.LEX.items():
        w2 = set(w for w in wlist if len(w) >= 2)
        if w2:
            cat_overlap[cat] = round(len(set(top) & w2) / len(top), 3)
    best_cat = max(cat_overlap, key=cat_overlap.get) if cat_overlap else "—"
    topic_top.append({"id": ti, "top_terms": top,
                      "valence": (vk - vb), "best_category": best_cat, "cat_overlap": cat_overlap})
# per-author LDA net-mind
M4 = {}
for a in AUTH:
    idx = [i for i in range(len(poem_auth)) if poem_auth[i]==a]
    vals = [sum(topic_dist[i][ti]*topic_top[ti]["valence"] for ti in range(10)) for i in idx]
    M4[a] = float(np.mean(vals)) if vals else 0.0
c14 = corr(M1, M4) if np.std(list(M4.values())) > 0 else (float("nan"), float("nan"))
print(f"[v5] net-mind M1 vs LDA: Pearson={c14[0]:.3f} Spearman={c14[1]:.3f}")

# ----------  (B) ATTRIBUTION: A2/A3 vs A1 (from results_v4.json)  ----------
print("[v5] (B) attribution baselines ...")
def stratified_split(y, test_size=0.15, seed=SEED):
    idx_by = defaultdict(list)
    for i, yi in enumerate(y): idx_by[yi].append(i)
    tr, te = [], []
    random.seed(seed)
    for c, idxs in idx_by.items():
        random.shuffle(idxs)
        k = max(1, int(round(len(idxs)*test_size)))
        te += idxs[:k]; tr += idxs[k:]
    random.shuffle(tr); random.shuffle(te)
    return tr, te

shix, auth_lab, dyn_lab = [], [], []
for a in AUTH:
    for t in shi[a]:
        shix.append(t); auth_lab.append(AUTH.index(a)); dyn_lab.append(0 if DYN[a]=="唐" else 1)

def eval_attr(texts, labels, model_build):
    tr, te = stratified_split(labels)
    Xtr = model_build.fit_transform([texts[i] for i in tr])
    Xte = model_build.transform([texts[i] for i in te])
    clf = model_build.named_steps["clf"] if hasattr(model_build, "named_steps") else None
    # model_build is a Pipeline; fit classifier inside
    raise NotImplementedError

# Use explicit pipeline
from sklearn.pipeline import Pipeline
def attr_task(texts, labels, n_classes, name):
    tr, te = stratified_split(labels)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1,3), min_df=2, sublinear_tf=True, max_features=60000)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0) if name=="logreg"
         else LinearSVC(C=1.0, max_iter=5000))
    ])
    pipe.fit([texts[i] for i in tr], [labels[i] for i in tr])
    pred = pipe.predict([texts[i] for i in te])
    gts = [labels[i] for i in te]
    acc = accuracy_score(gts, pred)
    mf1 = f1_score(gts, pred, average="macro", zero_division=0)
    wf1 = f1_score(gts, pred, average="weighted", zero_division=0)
    return {"accuracy": float(acc), "macro_F1": float(mf1), "weighted_F1": float(wf1), "n_test": len(gts)}

attr_compare = {"author_12way": {}, "dynasty_2way": {}, "prose_4way": {}}
for model in ["logreg", "svm"]:
    mname = "tfidf_logreg" if model=="logreg" else "tfidf_svm"
    attr_compare["author_12way"][mname] = attr_task(shix, auth_lab, 12, model)
    attr_compare["dynasty_2way"][mname] = attr_task(shix, dyn_lab, 2, model)

# prose 4-way (from 全唐文) for CNN comparison
from parse_qtw import parse_qtw
tw_auth = ["柳宗元","刘禹锡","韩愈","白居易"]
tangwen = parse_qtw(f"{DATA}/qtw") if os.path.exists(f"{DATA}/qtw") else {}
tw_texts, tw_lab = [], []
for i,a in enumerate(tw_auth):
    for e in tangwen.get(a, []):
        tw_texts.append(e["text"]); tw_lab.append(i)
for model in ["logreg", "svm"]:
    mname = "tfidf_logreg" if model=="logreg" else "tfidf_svm"
    attr_compare["prose_4way"][mname] = attr_task(tw_texts, tw_lab, 4, model)

# load A1 (char-CNN) from results_v4.json
with open(f"{BASE}/results_v4.json", encoding="utf-8") as f:
    rv4 = json.load(f)
attr_compare["author_12way"]["char_cnn"] = {
    "accuracy": rv4["meta"]["attr_author"]["accuracy"],
    "macro_F1": rv4["meta"]["attr_author"]["macro_F1"],
    "weighted_F1": rv4["meta"]["attr_author"]["weighted_F1"]}
attr_compare["dynasty_2way"]["char_cnn"] = {
    "accuracy": rv4["meta"]["attr_dynasty"]["accuracy"],
    "macro_F1": rv4["meta"]["attr_dynasty"]["macro_F1"]}
if rv4["meta"].get("attr_prose"):
    attr_compare["prose_4way"]["char_cnn"] = {
        "accuracy": rv4["meta"]["attr_prose"]["accuracy"],
        "macro_F1": rv4["meta"]["attr_prose"]["macro_F1"]}

# ---------- SAVE  ----------
out = {
    "net_mind_methods": {
        "authors": AUTH,
        "M1_lexicon": M1, "M2_tfidf": M2, "M3_lsa": M3, "M4_lda": M4,
        "corr_M1_M2": {"pearson": c12[0], "spearman": c12[1]},
        "corr_M1_M3": {"pearson": c13[0], "spearman": c13[1]},
        "corr_M2_M3": {"pearson": c23[0], "spearman": c23[1]},
        "corr_M1_M4": {"pearson": c14[0], "spearman": c14[1]},
    },
    "attribution_compare": attr_compare,
    "lda_topics": topic_top,
    "meta": {"n_poems": n_total, "svd_dims": 200, "n_topics": 10, "seed": SEED},
}
with open(f"{BASE}/results_v5_methods.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("[v5] wrote results_v5_methods.json")

# ---------- FIGURES  ----------
FONT = "C:/Windows/Fonts/simhei.ttf"
if os.path.exists(FONT):
    font_manager.FontProperties(fname=FONT)
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False

# Fig 1: net-mind methods comparison (z-scored per method for visual agreement)
fig, ax = plt.subplots(figsize=(12,5.5))
# build z per method
def zscore(d):
    arr=np.array([d[a] for a in AUTH],float); s=arr.std()
    return {a:((d[a]-arr.mean())/s if s>0 else 0.0) for a in AUTH}
zM = {"M1":zscore(M1),"M2":zscore(M2),"M3":zscore(M3),"M4":zscore(M4)}
x = np.arange(len(AUTH)); w=0.2
for i,(k,col) in enumerate([("M1","#4C72B0"),("M2","#DD8452"),("M3","#55A868"),("M4","#C44E52")]):
    ax.bar(x+(i-1.5)*w, [zM[k][a] for a in AUTH], w, label=k, color=col)
ax.set_xticks(x); ax.set_xticklabels(AUTH, rotation=40, ha="right")
ax.set_ylabel("净心指数（各方法内 z-score）"); ax.set_title("净心指数：四种估计方法排序一致性（z-score 对比）")
ax.axhline(0, color="grey", lw=0.8); ax.legend(ncol=4, loc="upper right")
plt.tight_layout(); plt.savefig(f"{BASE}/figures/v5_netmind_methods.png", dpi=130); plt.close()
print("[v5] fig v5_netmind_methods.png")

# Fig 2: attribution comparison
fig, axes = plt.subplots(1, 2, figsize=(12,5))
tasks = [("author_12way","作者归属 (12类)"), ("dynasty_2way","时期归属 (唐/宋)")]
models = [("char_cnn","字符CNN"),("tfidf_logreg","TF-IDF+LogReg"),("tfidf_svm","TF-IDF+SVM")]
colors = ["#4C72B0","#DD8452","#55A868"]
for ax, (tk, tlabel) in zip(axes, tasks):
    accs = [attr_compare[tk][m[0]]["accuracy"] for m in models]
    f1s = [attr_compare[tk][m[0]]["macro_F1"] for m in models]
    xx = np.arange(len(models)); ww=0.38
    ax.bar(xx-ww/2, accs, ww, label="准确率", color="#4C72B0")
    ax.bar(xx+ww/2, f1s, ww, label="macro-F1", color="#DD8452")
    ax.set_xticks(xx); ax.set_xticklabels([m[1] for m in models]); ax.set_ylim(0,1.05)
    for i,(a,f) in enumerate(zip(accs,f1s)):
        ax.text(i-ww/2,a+0.02,f"{a:.2f}",ha="center",fontsize=8)
        ax.text(i+ww/2,f+0.02,f"{f:.2f}",ha="center",fontsize=8)
    ax.set_title(tlabel); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{BASE}/figures/v5_attr_methods.png", dpi=130); plt.close()
print("[v5] fig v5_attr_methods.png")

# Fig 3: LDA topic valence
fig, ax = plt.subplots(figsize=(10,5))
vals = [t["valence"] for t in topic_top]
cols = ["#C44E52" if v<0 else "#55A868" for v in vals]
ax.bar([f"T{t['id']}" for t in topic_top], vals, color=cols)
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylabel("主题情感极性（负=悲苦 / 正=旷达）")
ax.set_title("LDA 主题模型自动获得的情感极性（lexicon 推导）")
plt.tight_layout(); plt.savefig(f"{BASE}/figures/v5_lda_topics.png", dpi=130); plt.close()
print("[v5] fig v5_lda_topics.png")

print("[v5] DONE")
