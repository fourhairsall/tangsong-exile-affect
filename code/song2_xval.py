# -*- coding: utf-8 -*-
"""song2_xval.py — 宋诗对称跨版校验 (v5 扩展 2).
主本: Book1Q84《全宋诗》(data/songshi/poet.song.*.json)
第二底本: kanripo KR4h0143《御選宋詩》(data/song2/KR4h0143_*.txt, JUAN=御選宋詩)
方法: 同一透明情感词表(M1), 同口径(t2s + per-1000-char), 对 8 位宋诗人求净心指数与五维频次,
      两侧做 (a) 净心 Pearson/Spearman (b) 五维频次矩阵(8x5=40格) Pearson。
固定: 与 v5 一致, 无随机性。
"""
import os, re, json, glob, sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
from opencc import OpenCC

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lexicon as L

cc = OpenCC('t2s')
TARGET = ["苏轼","欧阳修","黄庭坚","秦观","王禹偁","范仲淹","辛弃疾","陆游"]
SPACE = "\u3000"  # 全角空格

def t2s(s):
    return cc.convert(s)

# ---------- 全宋诗侧 (Book1Q84) ----------
def load_quansongshi():
    txt = {a: [] for a in TARGET}
    for fp in glob.glob(f"{BASE}/data/songshi/poet.song.*.json"):
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for rec in data:
            a = rec.get("author", "")
            a = t2s(a)
            if a not in TARGET:
                continue
            paras = rec.get("paragraphs", []) or []
            s = "".join(paras)
            if s:
                txt[a].append(t2s(s))
    return {a: "".join(v) for a, v in txt.items()}

# ---------- 御選宋詩侧 (kanripo KR4h0143) ----------
# 该底本缩进不可靠(作者/题/句同卷内混用 L2/L3/L4/L0/L1), 故不依赖缩进来区分。
# 可靠锚点: 目标诗人以独立成行本名出现(如 "王禹偁¶")。策略:
#  - 精确匹配目标名 -> 设 cur
#  - 短且前导>=2 的"结构行": 含标题标记(詩/頌/吟...)者视为当前作者诗题(保留 cur);
#    否则视为另一(非目标)作者或分体小注 -> cur=None 以防污染下一目标作者的统计
#  - 其余(长句/低缩进)视为诗句, 归入 cur
TITLE_MARKERS = set("詩頌吟歌辭篇行曲怨操引詠諷謠解弄嘆哀樂賦銘箴誡序説論答送贈別和見題書并（")
def load_yuxuan_songshi():
    txt = {a: [] for a in TARGET}
    for fp in sorted(glob.glob(f"{BASE}/data/song2/KR4h0143_*.txt")):
        raw = open(fp, encoding="utf-8").read()
        if "PROPERTY: JUAN 御選宋詩" not in raw:   # 仅取宋诗卷
            continue
        cur = None
        has_title = False
        for line in raw.split("\n"):
            line = line.replace("¶", "")
            s = line.lstrip(SPACE).strip()
            if not s or s.startswith("<pb:"):
                continue
            if t2s(s) in TARGET:                   # 精确命中目标诗人(简转后) -> 锚定
                cur = t2s(s); has_title = False
                continue
            lead = len(line) - len(line.lstrip(SPACE))
            if lead >= 2 and len(s) <= 8:         # 结构短行: 题/分体/另一作者
                if any(m in s for m in TITLE_MARKERS):
                    has_title = True                # 诗题出现 -> 其后为诗句(剔除序)
                else:
                    cur = None; has_title = False   # 非目标作者或分体 -> 停止归集
                continue
            # 仅归集"诗题之后"的诗句, 跳过作者名与诗题之间的散文小序
            if cur is not None and has_title:
                txt[cur].append(t2s(s))
    return {a: "".join(v) for a, v in txt.items()}

# ---------- 全宋诗侧 (poem-level, for noise-floor bootstrap) ----------
def load_quansongshi_poems():
    """Return {author: [poem_text, ...]} where each poem is one JSON record
    (simplified). Used by the noise-floor test to resample by *whole poems*
    rather than by consecutive character windows."""
    poems = {a: [] for a in TARGET}
    for fp in glob.glob(f"{BASE}/data/songshi/poet.song.*.json"):
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for rec in data:
            a = rec.get("author", "")
            a = t2s(a)
            if a not in TARGET:
                continue
            paras = rec.get("paragraphs", []) or []
            s = "".join(paras)
            if s:
                poems[a].append(t2s(s))
    return poems

def stats(text):
    per1k, n = L.cat_per1k([text])
    return n, per1k, L.net_index(per1k)

def main():
    qs = load_quansongshi()
    yx = load_yuxuan_songshi()

    rows = []
    q_net, y_net = [], []
    q_mat, y_mat = [], []
    for a in TARGET:
        qn, qp, qnet = stats(qs[a])
        yn, yp, ynet = stats(yx[a])
        q_net.append(qnet); y_net.append(ynet)
        q_mat.append([qp[c] for c in L.CATS])
        y_mat.append([yp[c] for c in L.CATS])
        rows.append({
            "author": a,
            "quansongshi": {"n_char": qn, "net": qnet, **{c: qp[c] for c in L.CATS}},
            "yuxuan": {"n_char": yn, "net": ynet, **{c: yp[c] for c in L.CATS}},
            "delta_net": round(ynet - qnet, 3),
        })

    q_net = np.array(q_net, float); y_net = np.array(y_net, float)
    q_mat = np.array(q_mat, float); y_mat = np.array(y_mat, float)

    # 仅取两侧均有文本(字数>0)的诗人做相关
    mask = (q_net != 0) | (y_net != 0)
    # 更严格: 两侧 char 均>0
    qn_arr = np.array([r["quansongshi"]["n_char"] for r in rows])
    yn_arr = np.array([r["yuxuan"]["n_char"] for r in rows])
    both = (qn_arr > 0) & (yn_arr > 0)
    qn_o, yn_o = q_net[both], y_net[both]
    qm_o, ym_o = q_mat[both], y_mat[both]

    pr_net = pearsonr(qn_o, yn_o)[0] if both.sum() > 1 else float("nan")
    sr_net = spearmanr(qn_o, yn_o)[0] if both.sum() > 1 else float("nan")
    # 五维频次矩阵展平相关 (重叠诗人 x 5 格), 与唐诗侧 cat_corr 同口径
    cat_corr = pearsonr(qm_o.flatten(), ym_o.flatten())[0] if both.sum() > 1 else float("nan")

    out = {
        "method": "M1 透明词表 (同 v5); 全宋诗=Book1Q84, 第二底本=kanripo KR4h0143 御選宋詩",
        "target_authors": TARGET,
        "n_authors_overlap": int(both.sum()),
        "net_pearson": round(float(pr_net), 4),
        "net_spearman": round(float(sr_net), 4),
        "cat_freq_pearson": round(float(cat_corr), 4),
        "rows": rows,
    }
    with open(f"{BASE}/results_song2_xval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"重叠诗人(两侧均有文本): {out['n_authors_overlap']}/{len(TARGET)}")
    print(f"{'作者':8}{'全宋诗净心':>12}{'御選宋詩净心':>14}{'Δ净心':>10}{'全宋诗字数':>12}{'御選字数':>10}")
    for r in rows:
        print(f"{r['author']:8}{r['quansongshi']['net']:>12}{r['yuxuan']['net']:>14}{r['delta_net']:>10}"
              f"{r['quansongshi']['n_char']:>12}{r['yuxuan']['n_char']:>10}")
    print(f"\n净心 Pearson r = {out['net_pearson']}")
    print(f"净心 Spearman ρ = {out['net_spearman']}")
    print(f"五维频次矩阵(重叠x5) Pearson = {out['cat_freq_pearson']}")

if __name__ == "__main__":
    main()
