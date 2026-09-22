# Obtaining the raw corpora

This project analyses **third-party** classical-Chinese corpora. They are **not**
redistributed in this repository; the code and derived products are, but the raw text
must be fetched by the user. This document records exactly which sources were used and
the directory layout the pipeline expects.

## Quick start

```bash
python tools/fetch_corpora.py --data-dir data
```

The script clones the upstream providers into a cache and copies the relevant files into
`data/<source>/`. It is idempotent and requires `git` plus network access.

## Sources

| Directory | Edition | Provider | File pattern | Corpus role |
|---|---|---|---|---|
| `data/tang/` | 全唐詩 | [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) — `json/` | `poet.tang.*.json` | Tang *shi* (primary) |
| `data/songshi/` | 全宋詩 | [Book1Q84/Chinese-Poetry-Dataset](https://github.com/Book1Q84/Chinese-Poetry-Dataset) — `json/` | `poet.song.*.json` | Song *shi* (primary) |
| `data/ci.db` | 全宋詞 | chinese-poetry — `ci/` | sqlite | Song *ci* (outside the gold set) |
| `data/song2/` | 御選宋詩 | [kanripo](https://github.com/kanripo) | `KR4h0143_*.txt` | Song *shi*, official anthology (2nd edition) |
| `data/songchao/` | 宋詩鈔 | kanripo | `KR4h0157_*.txt` | Song *shi*, private anthology (3rd edition) |
| `data/qtw/` | 全唐文 | kanripo | `KR4h0168_*.txt` | Tang prose layer |
| `data/songbaijia/` | 宋百家詩存 | kanripo | `KR4h0167_*.txt` | Song *shi*, supplementary |
| `data/songyipu/` | 宋詩紀事 | kanripo | `KR4h0113_*.txt` | Song *shi*, supplementary |
| `data/yuding/` | 御定全唐詩 | *see below* | `*.json` (one per *juan*) | Tang *shi*, independent 2nd edition |
| `data/qsw/` | 全宋文 | *see below* | `*.txt` | Song prose (probe only) |

The three kanripo editions all descend from the **同一** 四庫全書文淵閣本 family and the
same *mandoku* digitisation pipeline, which is what makes the cross-edition comparison in
the paper a like-for-like one: same edition family, same pipeline, same lexicon.

### `data/yuding/` — 御定全唐詩

The 御定全唐詩 (900 *juan*,康熙敕編) is the second, independent Tang edition used for the
Tang cross-edition concordance (*A* = 0.960, *n* = 432 authors). It enters the pipeline as
899 JSON files named `001.json … 899.json`, one per *juan*. No stable GitHub mirror was
available at the time of writing, so `tools/fetch_corpora.py` does **not** automate this
source; supply it manually in the layout above. The corresponding parser is
`code/xval_yuding.py`.

### `data/qsw/` — 全宋文

Only a small probe file (`001.txt`) was used; the Song prose layer does not enter any
headline result. The full 全宋文 is available under its own terms and is not mirrored here.

## Expected layout

```
data/
├── tang/          poet.tang.*.json           (50 files)     全唐詩
├── songshi/       poet.song.*.json          (255 files)     全宋詩
├── ci.db          sqlite                                    全宋詞
├── song2/         KR4h0143_*.txt             (83 files)     御選宋詩
├── songchao/      KR4h0157_*.txt            (109 files)     宋詩鈔
├── qtw/           KR4h0168_*.txt            (177 files)     全唐文
├── songbaijia/    KR4h0167_*.txt             (42 files)     宋百家詩存
├── songyipu/      KR4h0113_*.txt             (24 files)     宋詩紀事
├── yuding/        *.json                    (899 files)     御定全唐詩
└── qsw/           *.txt                                     全宋文
```

Once the corpora are in place:

```bash
python code/recompute_all.py
```

## Normalisation

Every source is normalised before analysis:

1. **Script.** Traditional → simplified via OpenCC `t2s`. The conversion dictionary is
   not interchangeable between distributions — see the note in `requirements.txt`;
   every number in the paper was produced with `opencc-python-reimplemented==0.1.7`.
2. **Punctuation and whitespace.** Stripped before work-title keying.
3. **Work-title normalisation.** Trailing group-sequence suffixes are removed so that
   e.g. 全唐詩's 《帝京篇十首 一/二/三》 and 御定全唐詩's single 《帝京篇十首》 collapse to one
   key. See `code/align_works.py` and `docs/REALIGN_NOTES.md`.

## Licensing

Each source keeps its own licence. The chinese-poetry family is MIT; the kanripo
transcriptions follow the kanripo project's terms (CC BY-SA for the digitised text,
the underlying 四庫全書 text being public domain). Check each provider before
redistributing any raw text.
