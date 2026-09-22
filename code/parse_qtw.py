# -*- coding: utf-8 -*-
"""Parse kanripo 全唐文 (KR4h0168) mandoku text into per-essay prose for the
four target Tang authors. Heading format: '　柳宗元(二)' (U+3000 prefix, optional
'(N)' volume part); essay titles: '** 篇名'; body separated by '¶'."""
import re, os, glob
import opencc
cc = opencc.OpenCC('t2s')
TARGETS_TRAD = ["柳宗元", "劉禹錫", "韓愈", "白居易"]
AUTHOR_RE = re.compile(r"^\u3000(" + "|".join(TARGETS_TRAD) + r")")
ESSAY_RE = re.compile(r"^\*\* (.+)$")

def _clean(text):
    text = re.sub(r"<pb:[^>]*>", "", text)
    text = re.sub(r":PROPERTIES:.*?:END:", "", text, flags=re.S)
    text = re.sub(r"#\+[A-Z]+:[^\n]*", "", text)
    return text

def parse_qtw(directory):
    out = {a: [] for a in TARGETS_TRAD}
    for fp in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        raw = open(fp, encoding="utf-8", errors="ignore").read()
        lines = raw.replace("¶", "\n").split("\n")
        cur_author = None; cur_title = None; buf = []
        def flush():
            if cur_author is not None and cur_title is not None:
                t = _clean("".join(buf)).strip()
                if t:
                    out[cur_author].append({"title": cur_title, "text": t})
        for ln in lines:
            ma = AUTHOR_RE.match(ln)
            if ma:
                flush(); buf = []; cur_author = ma.group(1); cur_title = None
                continue
            if cur_author is None:
                continue
            s = ln.strip()
            me = ESSAY_RE.match(s)
            if me:
                flush(); buf = []; cur_title = me.group(1)
                continue
            if s.startswith(":") or s.startswith("#+") or s.startswith("<pb"):
                continue
            if s:
                buf.append(s)
        flush()
    result = {}
    for a in TARGETS_TRAD:
        simp = cc.convert(a)
        result[simp] = [{"title": cc.convert(x["title"]), "text": cc.convert(x["text"])}
                        for x in out[a]]
    return result

if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "D:/databuddy/2026-09-14-10-59-40/tangsong_research/data/qtw_probe"
    r = parse_qtw(d)
    for a, ess in r.items():
        chars = sum(len(e["text"]) for e in ess)
        print(f"{a}: essays={len(ess)} chars={chars}")
        for e in ess[:3]:
            print("   -", e["title"][:30], "len", len(e["text"]))
