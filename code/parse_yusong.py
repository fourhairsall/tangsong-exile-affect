# -*- coding: utf-8 -*-
"""Parse the 御選宋詩 (Song-selection) portion of data/song2 (御選宋金元明四朝詩,
mandoku txt) into the same {author, title, paragraphs:[text]} schema as songshi.

Author detection uses the FULL songshi author-name set as ground truth: a short
heading (<=12 chars, indent>=2) whose simplified text matches a known songshi
author is treated as an author line; everything else (titles, genre/category
headings, prefaces) is ignored for attribution.  Verse lines are indent<=1
(always the lowest indent in this corpus) or long indent>=2 prose (prefaces).
"""
import json, glob, os, re
import numpy as np
import opencc

cc = opencc.OpenCC('t2s')
BASE = "D:/databuddy/2026-09-14-10-59-40/tangsong_research"
DATA = f"{BASE}/data"
OUTDIR = f"{DATA}/yusong"
os.makedirs(OUTDIR, exist_ok=True)

# ---- build full songshi author set (simplified) ----
print("[1] building songshi author set ...")
songshi_authors = set()
nfiles = 0
for fp in glob.glob(f"{DATA}/songshi/poet.song.*.json"):
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception:
        continue
    if isinstance(d, list):
        for e in d:
            a = e.get("author")
            if a and a.strip():
                songshi_authors.add(cc.convert(a.strip()))
    nfiles += 1
print(f"    scanned {nfiles} files, {len(songshi_authors)} unique songshi authors")

TITLE_PREFIX = ("贈", "寄", "送", "和", "題", "答", "詠", "賦", "書", "呈",
                "次", "挽", "弔", "奉", "謝", "飲", "望", "登", "陪", "酬",
                "憶", "聞", "見", "讀", "詠", "觀", "游", "詠")

def is_author(t):
    if t not in songshi_authors:
        return False
    if len(t) > 6:          # most author names are 2-4 chars; long matches are titles
        return False
    if t.startswith(TITLE_PREFIX):
        return False
    return True

# ---- parse song2 poem volumes ----
print("[2] parsing 御選宋詩 volumes ...")
poems = []
cur_author = None
cur_title = None
cur_body = []
matched_author_lines = 0
heading_lines = 0

def commit():
    global cur_body, cur_author, cur_title
    if cur_author is not None and cur_body:
        poems.append({
            "author": cur_author,
            "title": cur_title or "",
            "paragraphs": ["".join(cur_body)],
        })
    cur_body = []

for fp in sorted(glob.glob(f"{DATA}/song2/*.txt")):
    raw = open(fp, encoding="utf-8").read()
    if not re.search(r"御選宋詩卷", raw):
        continue  # skip preface/bios/jin/ming volumes (those use 姓/金/明 prefixes)
    for line in raw.splitlines():
        if line.startswith("#") or line.startswith("<pb:") or not line.strip():
            continue
        s = line.rstrip("¶").rstrip("\n")
        nlead = len(line) - len(line.lstrip("　"))
        text = s.strip()
        if not text:
            continue
        if nlead <= 1:
            # verse (lowest indent) -> body
            if cur_author is not None:
                cur_body.append(text)
            continue
        # nlead >= 2 : heading candidate or long preface prose
        if len(text) > 12:
            # long indent>=2 line = prose preface -> treat as verse body
            if cur_author is not None:
                cur_body.append(text)
            continue
        # short heading
        heading_lines += 1
        t = cc.convert(text)
        if is_author(t):
            commit()
            cur_author = t
            cur_title = None
            cur_body = []
            matched_author_lines += 1
        else:
            # title or genre/category heading -> start a new poem entry
            commit()
            cur_title = t
            cur_body = []

commit()
print(f"    parsed {len(poems)} poem-entries, {matched_author_lines} author-heading matches")

# stats
from collections import Counter
auth_count = Counter(p["author"] for p in poems)
n_auth = len(auth_count)
overlap = sum(1 for a in auth_count if a in songshi_authors)
print(f"    unique authors in selection: {n_auth}, overlapping songshi authors: {overlap}")
print(f"    top authors: {auth_count.most_common(12)}")

# save
out = f"{OUTDIR}/poet.song_selection.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(poems, f, ensure_ascii=False)
print(f"[3] wrote {out} ({os.path.getsize(out)/1024:.0f} KB)")

summary = {
    "n_poems": len(poems),
    "n_authors": n_auth,
    "n_overlap_authors": overlap,
    "matched_author_lines": matched_author_lines,
    "heading_lines": heading_lines,
    "top_authors": auth_count.most_common(20),
}
with open(f"{BASE}/results_parse_yusong.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("[4] wrote results_parse_yusong.json")
