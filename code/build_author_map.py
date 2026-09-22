# -*- coding: utf-8 -*-
"""Build an author-aligned map for dl_pool.json (no modification of the shared
pool file). Replicates EXACTLY the poem ordering in build_dl_pool.py so the
emitted author list is position-aligned with dl_pool.json[0..N-1].

OpenCC (trad->simp) lives only in the managed venv, so this script must be run
with the managed python (which has opencc installed).
"""
import sys, os, json, glob, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import opencc
import build_dl_pool as B   # reuse AUTH, load_poems, clean_text

cc = opencc.OpenCC("t2s")

def load_with_author(patterns, authors):
    out = {a: [] for a in authors}
    for pat in patterns:
        for fp in sorted(glob.glob(os.path.join(HERE, pat))):
            try:
                d = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            for p in d:
                a = cc.convert((p.get("author") or "").strip())
                if a in out:
                    txt = cc.convert("".join(p.get("paragraphs", [])))
                    txt = B.clean_text(txt)
                    if txt:
                        out[a].append((a, txt))
    return out

def main():
    tang = load_with_author(["data/tang/poet.tang.*.json"],
                            ["柳宗元", "刘禹锡", "韩愈", "白居易"])
    song = load_with_author(["data/songshi/poet.song.*.json"],
                            ["苏轼", "欧阳修", "黄庭坚", "秦观",
                             "王禹偁", "范仲淹", "辛弃疾", "陆游"])
    shi = {**tang, **song}
    pairs = [(a, t) for au in B.AUTH for (a, t) in shi[au]]   # identical order to build_dl_pool
    authors = [a for a, _ in pairs]
    texts = [t for _, t in pairs]

    # verify alignment against dl_pool.json
    pool = json.load(open(os.path.join(HERE, "dl_pool.json"), encoding="utf-8"))
    assert len(texts) == len(pool), f"len mismatch {len(texts)} vs {len(pool)}"
    mism = sum(1 for i in range(len(pool)) if texts[i] != pool[i]["text"])
    print(f"[map] poems={len(authors)}  text-mismatch-vs-dl_pool={mism}")
    assert mism == 0, "texts not aligned with dl_pool.json -- ordering changed!"

    # count per author
    from collections import Counter
    cnt = Counter(authors)
    print("[map] per-author counts:", dict(cnt))

    out = os.path.join(HERE, "dl_pool_authors.json")
    json.dump(authors, open(out, "w", encoding="utf-8"))
    print(f"[map] wrote {out}")

if __name__ == "__main__":
    main()
