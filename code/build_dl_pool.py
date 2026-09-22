# -*- coding: utf-8 -*-
"""Precompute the deep-learning training pool for the head-to-head ablation.

This reproduces EXACTLY the pool + weak labels used by analyze_v4.py:
    all_shi_texts = [t for a in AUTH for t in shi[a]]   (Tang 4 + Song 8 authors)
    weak_label(t) = L.net_index(L.cat_per1k([t])[0])   # NMI scalar (content chars)
plus the 5-dim lexicon profile (per 1000 chars) for the auxiliary head.

The poem JSONs are in traditional Chinese; the M1 lexicon (lexicon.py) is in
simplified. analyze_v4.py normalized via OpenCC t2s. OpenCC lives only in the
managed venv (no torch there), so we do the t2s + labeling here and dump a
self-contained JSON; the torch training script (anaconda) just reads it.

Output: dl_pool.json  [{"text": simplified, "nmi": float, "profile": [5 floats]}, ...]
"""
import sys, os, json, glob, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import opencc
import lexicon as L

cc = opencc.OpenCC('t2s')
AUTH = ["柳宗元","刘禹锡","韩愈","白居易","苏轼","欧阳修","黄庭坚","秦观","王禹偁","范仲淹","辛弃疾","陆游"]

def clean_text(s):
    return re.sub(r"\s+", "", s)

def load_poems(folder, authors):
    out = {a: [] for a in authors}
    for fp in sorted(glob.glob(os.path.join(HERE, folder))):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for p in d:
            a = cc.convert(p.get("author", ""))
            if a in out:
                txt = "".join(p.get("paragraphs", []))
                if txt:
                    out[a].append(cc.convert(txt))
    return out

def main():
    print("[pool] loading 诗 (Tang + Song) ...")
    tang = load_poems("data/tang/poet.tang.*.json",
                      ["柳宗元","刘禹锡","韩愈","白居易"])
    song = load_poems("data/songshi/poet.song.*.json",
                      ["苏轼","欧阳修","黄庭坚","秦观","王禹偁","范仲淹","辛弃疾","陆游"])
    shi = {**tang, **song}
    all_shi = [t for a in AUTH for t in shi[a]]
    print(f"[pool] raw poems: {len(all_shi)}  (Tang {sum(len(v) for v in tang.values())}, "
          f"Song {sum(len(v) for v in song.values())})")

    pool = []
    for raw in all_shi:
        t = clean_text(raw)
        if not t:
            continue
        per, _ = L.cat_per1k([t])
        nmi = L.net_index(per)
        profile = [float(per[cat]) for cat in L.CATS]
        pool.append({"text": t, "nmi": float(round(nmi, 4)), "profile": profile})

    out = os.path.join(HERE, "dl_pool.json")
    json.dump(pool, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    nmi_vals = [p["nmi"] for p in pool]
    import statistics
    print(f"[pool] kept after empty-drop: {len(pool)}")
    print(f"[pool] NMI: mean={statistics.mean(nmi_vals):.3f} sd={statistics.pstdev(nmi_vals):.3f} "
          f"min={min(nmi_vals):.2f} max={max(nmi_vals):.2f}")
    print(f"[pool] wrote {out}")

if __name__ == "__main__":
    main()
