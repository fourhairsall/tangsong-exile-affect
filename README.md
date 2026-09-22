# Tang–Song Exile Affect

**Reproduction package** for *Reading the Banished: A Deep-Learning Emotion Analysis of
Tang–Song Exiled Literati* (Shujiang Tang, Hunan University of Science and Engineering).

This repository contains the **instrument**, the **analysis code**, and every **derived
product** needed to reproduce the numbers in the paper, together with the standalone
**Supplementary Material**. Raw third-party corpora are **not** redistributed here; see
[`docs/CORPUS_FETCH.md`](docs/CORPUS_FETCH.md) for how to obtain them.

> The paper's appendices have been moved out of the manuscript and live here as a
> self-contained document: [`supplement/supplement.pdf`](supplement/supplement.pdf).

---

## 1. What the instrument does

A transparent, dictionary-based affect meter for classical Chinese verse, plus a
character-level CNN that reproduces its output.

**Five lexical dimensions** (see `supplement/supplement.pdf`, App. C, for the full word lists):

| Dimension | Register | Role |
|---|---|---|
| 悲苦孤寂 *Bei* | sorrow–loneliness | affective pole |
| 贬谪羁旅 *Bian* | exile–journey | the *named condition* (kept separate from felt affect) |
| 旷达闲适 *Kuang* | detached–easeful | affective pole |
| 自然山水 *Zi* | nature–landscape | imagistic register |
| 空幻时空 *Kong* | void–temporality | imagistic register |

**Net-Mind Index (ABI).** A per-author difference of the two affective pole rates, per
1,000 content characters:

```
ABI(a) = φ_{a,Kuang} − φ_{a,Bei}
```

**Validation.** Poems are scored by two independent coders on an ordinal 0–3 scale for
each dimension (quadratic-weighted κ = 0.72–0.88, ordinal α = 0.93–0.97); the lexicon
agrees with the gold criterion at ρ = 0.46 [0.39, 0.53], and the CNN reproduces the
lexicon at test *r* = 0.934.

---

## 2. Headline findings

- **Transmission stability.** Author-level ABI is highly concordant across editions for
  the Tang (concordance *A* = 0.960, *n* = 432 authors) but markedly less so for the Song
  where a selection filters the corpus (QSS vs. 御選宋詩 *A* = 0.557 / 0.567, *n* = 253;
  QSS vs. 宋詩鈔 *A* = 0.822, *n* = 75). The disagreement is **directional**, not diffuse.
- **Su Shi.** No net exile effect on the index (−0.56 [−2.37, +1.25], p = 0.55); the
  Huangzhou and Danzhou episodes move it in **opposite** directions (+2.79 at Danzhou).
- **Dose–response.** The index tracks the miasma gradient of the place of exile at
  +3.07 per severity grade (p = 0.003).
- **Channel asymmetry.** Official (imperially edited) and private (personal) anthologies
  amplify the nature–landscape register very differently (δ = +16.68 vs. +4.43 per 1,000
  characters), which is what makes the Song concordance edition-dependent.

---

## 3. Repository layout

```
.
├── manuscript/                  # both manuscripts (LaTeX + compiled PDF) and figures
│   ├── tangsong_nlp_paper_en.tex / .pdf      (EN)
│   ├── tangsong_nlp_paper_cn.tex / .pdf      (CN)
│   └── figures/
├── supplement/                  # standalone Supplementary Material
│   ├── supplement.tex
│   └── supplement.pdf
├── code/                        # the analysis pipeline (44 scripts)
├── data/                        # every derived product (no raw corpora)
│   ├── numbers_ledger.json      # <-- traceability ledger: every reported number
│   ├── results_*.json           # per-analysis outputs, one file per stage
│   ├── lexicon_v2_entries.json  # the 214-entry five-dimension lexicon
│   ├── annotations/             # de-identified dual codings (coders A and B)
│   ├── gold_*                   # gold sample, blind coding sheets, reliability
│   ├── register_*               # register-probe sample and codings
│   ├── exile_chronology.csv     # graded exile events per official (A/B/C)
│   ├── poem_period_labels.csv   # poem-level period attachment + coverage
│   ├── cross_edition_points.json
│   ├── classical_ppmi_svd.npz   # frozen PPMI–SVD character embeddings
│   ├── dl_pool*.json            # character-level CNN training pools
│   └── models/af_cnn_30ep.pt    # trained character-CNN checkpoint
├── docs/                        # method memos and reproduction audits
└── requirements.txt
```

### Where the numbers live

`data/numbers_ledger.json` is the **traceability ledger**. Every number that appears in
the manuscript is keyed to the `results_*.json` entry that produced it, so any claim can
be traced to a single JSON key path. Start there when checking a figure.

---

## 4. Reproducing the pipeline

```bash
pip install -r requirements.txt        # torch==2.14.0+cpu, numpy, opencc-python-reimplemented

# 1. Place the raw corpora under ./data/<source>/  (see docs/CORPUS_FETCH.md)
# 2. Re-derive every number in the paper from those corpora:
python code/recompute_all.py
```

`code/recompute_all.py` is the single entry point; it re-derives the corpora statistics,
the cross-edition concordance, the counterfactual channel decomposition, the exile dose
model and the gold-set validity figures, writing each stage to `data/results_*.json`.

Individual stages can be re-run in isolation, e.g.:

| Script | Reproduces |
|---|---|
| `code/recompute_all.py` | all headline numbers (orchestrator) |
| `code/cross_edition_points.py` | per-author ABI in both editions |
| `code/song2_xval.py`, `code/song3_rep.py` | Song cross-edition concordance (*A*) |
| `code/counterfactual.py` | official/private channel δ decomposition |
| `code/exile_dose.py` | miasma dose–response (+3.07 / grade) |
| `code/gold_validity.py`, `code/lexicon_validity.py` | reliability and criterion validity |
| `code/dl_cnn2025_v2.py`, `code/dl_cnn2025_bench.py` | character-CNN (test *r* = 0.934) |
| `code/phase4_m6.py` | invariant-representation probe (supplement App. B) |

### Corpus placement

The scripts expect the raw text under a `data/` directory beside the code, one folder per
source edition: `data/tang/`, `data/yuding/`, `data/songshi/`, `data/song2/`,
`data/songchao/`, `data/qtw/`, `data/qsw/`. Paths are relative to the project root and are
documented at the top of each parsing script.

---

## 5. Data provenance and ethics

- **De-identification.** The released codings under `data/annotations/` are de-identified
  to *annotator A* / *annotator B*. No personally identifying information is included.
- **No raw corpora.** All raw texts (全唐詩, 全宋詩, the kanripo 四庫全書 editions, 全唐文/
  全宋文) remain subject to their own terms and are fetched by the user, not mirrored here.
- **Grade discipline.** Exile events are graded A (histories + dated chronology agree),
  B (scholarly consensus) or C (disputed); grade-C events are never merged into A/B, and a
  ±1-year buffer around every event boundary is excluded from all contrasts.

---

## 6. Scope and limitations

- The **lexicon is the validated object**, not the author ranking. Poem-level word counts
  are sparse, so per-poem rates carry wide uncertainty; per-author aggregates are the
  intended unit of analysis.
- Cross-edition concordance measures **stability across editions**, not agreement with
  ground truth. A low *A* means the two editions disagree, which is itself the finding.
- The gold set is confined to *shi*; nothing here licenses the analyser for *ci* or prose.
- The CNN is a **distillation of the lexicon**, used to show the instrument is learnable;
  it is not an independent theory of affect.

---

## 7. Citation

```bibtex
@article{tang2026banished,
  title  = {Reading the Banished: A Deep-Learning Emotion Analysis of
            Tang--Song Exiled Literati},
  author = {Tang, Shujiang},
  year   = {2026},
  note   = {Hunan University of Science and Engineering},
  url    = {https://github.com/fourhairsall/tangsong-exile-affect}
}
```

## 8. License

Code is released under the **MIT License** (see [`LICENSE`](LICENSE)).
The derived data and documentation are released under **CC BY 4.0**.
Third-party raw corpora retain their original licenses.
