# -*- coding: utf-8 -*-
r"""exile_dose.py — 贬谪剂量—反应的作者内配对分析（C 线数值阶段, 按陈善本裁决执行）.

主分析  = 苏轼(蘇軾) 作者内配对差: exile(473) vs pre+post(1855), NMI 与五维速率
          (每千字), 分层 bootstrap CI(按编年序每 100 首一层, 防同年诗作聚集; seed=20260918)
分期曲线 = pre → 黄州 → 汝州/元祐 → 惠州 → 儋州 → 北归 (按陈善本事件边界锚点切期)
剂量反应 = 黄州(sev2)→惠州(3)→儋州(4), 逐期 NMI + 加权线性斜率 bootstrap CI +
          Spearman 趋势; 若不单调如实报告
条件性佐证 = 白居易(B级): exile(59) vs 合并非贬期(84); 声明 pre 组新乐府选择偏差;
          不分 pre/post、不进剂量梯度
其余四人 = 不跑, 仅在局限中交代 (柳宗元贬前存诗不足、黄庭坚实测推翻等)
稳健性   = 每千字速率 + 逐期字数全报; A级-only; 剔除<40字短诗;
          boundary_buffer 三档: tier0=缓冲诗就近归期(不剔除), tier1=主分析(剔除),
          tier2=再加宽: 剔除距任一边界 <2 序列年(锚点线性插值)的诗

输出: results_exile_dose.json (+ exile_dose_log.txt)
"""
import csv, glob, json, os, re
import numpy as np
import opencc
import lexicon as L

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = f"{BASE}/data"
cc = opencc.OpenCC('t2s')
LOG = open(f"{BASE}/exile_dose_log.txt", "w", encoding="utf-8")
def log(*a):
    LOG.write(" ".join(str(x) for x in a) + "\n"); LOG.flush()

SEED = 20260918
B = 10000
DIMS = list(L.CATS)
EASE, BITTER = L.NET
MIN_SHORT = 40          # 短诗敏感性阈值(字)
BIN = 100               # 编年序分层步长

rng = np.random.default_rng(SEED)

# ---------------- 1. 读标签 ----------------
rows = list(csv.DictReader(open(f"{BASE}/poem_period_labels.csv", encoding="utf-8-sig")))
su_rows = [r for r in rows if r["author"] == "蘇軾"]
bai_rows = [r for r in rows if r["author"] == "白居易"]
log(f"labels: 蘇軾={len(su_rows)} 白居易={len(bai_rows)} (total {len(rows)})")

# ---------------- 2. 载入所需语料文件并建立 (file, corpus_idx) 索引 ----------------
need_files = sorted({r["source_file"] for r in su_rows} | {r["source_file"] for r in bai_rows})
corpus = {}
uuid_idx = {}
for key in need_files:
    for fp in glob.glob(f"{DATA}/**/{key}", recursive=True):
        d = json.load(open(fp, encoding="utf-8"))
        if isinstance(d, list):
            for i, p in enumerate(d):
                corpus[(key, i)] = p
                if "id" in p:
                    uuid_idx[p["id"]] = p
        break
log(f"corpus entries loaded: {len(corpus)} (uuid-indexed {len(uuid_idx)}) from {len(need_files)} files")

def poem_text(r):
    pid = r["poem_id"]
    if "#" in pid:
        m = re.match(r"(.+)#(\d+)$", pid)
        key, i = m.group(1), int(m.group(2))
        return corpus.get((key, i))
    return uuid_idx.get(pid)

# ---------------- 3. 逐诗计数 ----------------
def attach_texts(rows_):
    matched, head_mismatch, missing = 0, 0, 0
    out = []
    for r in rows_:
        p = poem_text(r)
        t_raw = "".join(p.get("paragraphs", [])) if p is not None else ""
        if not t_raw:
            missing += 1
            continue
        head = r["text_head"].strip()
        if head and not t_raw.strip().startswith(head[:6]):
            head_mismatch += 1
        matched += 1
        t = cc.convert(t_raw)
        n = len(L.strip_punct(t))
        cnt = np.array([sum(t.count(w) for w in ws) for ws in L.LEX.values()], dtype=float)
        out.append({"seq": int(r["poem_index"]), "period": r["period"],
                    "basis": r["basis"], "grade": r["evidence_grade"],
                    "file": r["source_file"], "n": n, "cnt": cnt})
    log(f"join: matched={matched} head_mismatch={head_mismatch} missing={missing}")
    return out, {"matched": matched, "head_mismatch": head_mismatch, "missing": missing}

su, su_join = attach_texts(su_rows)
bai, bai_join = attach_texts(bai_rows)

# ---------------- 4. 苏轼分期 ----------------
def su_segment(r):
    p, b, s = r["period"], r["basis"], r["seq"]
    if p == "pre":
        return "pre"
    if p == "post":
        if "北归" in b:
            return "北归"
        if "离开黄州后" in b:
            return "汝州元祐"
        return "post_other"
    if p == "exile":
        if "黄州团练副使" in b:
            return "黄州"
        if "惠州1094" in b:
            return "惠州" if s < 2161 else "儋州"
        if "补编" in b:
            return "exile_B_suppl"
    return None        # boundary_buffer / unknown

for r in su:
    r["seg"] = su_segment(r)
seg_counts = {}
for r in su:
    seg_counts[r["seg"]] = seg_counts.get(r["seg"], 0) + 1
log(f"su segments: {seg_counts}")
EXILE_SEGS = ["黄州", "惠州", "儋州", "exile_B_suppl"]
CTRL_SEGS = ["pre", "汝州元祐", "北归"]
CURVE = ["pre", "黄州", "汝州元祐", "惠州", "儋州", "北归"]
assert sum(seg_counts.get(s, 0) for s in EXILE_SEGS) == 473, "exile count != 473"

# ---------------- 5. bootstrap 机件 ----------------
def count_arrays(rows_, segs):
    segs = list(segs)
    idx = {s: [] for s in segs}
    for r in rows_:
        if r["seg"] in idx and r["n"] > 0:
            idx[r["seg"]].append(r)
    return idx

def seg_rates(sample):
    n = sum(x["n"] for x in sample)
    if n == 0:
        return None
    cnt = np.sum([x["cnt"] for x in sample], axis=0)
    per = cnt / n * 1000.0
    return per, n

def boot_pass(idx, seg_list, n_rep=B, rng=rng):
    """每 segment 分层(按 seq//BIN)重抽; 返回 {seg: (per_array[n_rep,6], n_array)}"""
    out = {}
    for s in seg_list:
        items = sorted(idx[s], key=lambda x: x["seq"])
        if not items:
            out[s] = (np.full((n_rep, 6), np.nan), np.zeros(n_rep))
            continue
        strata = {}
        for x in items:
            strata.setdefault(x["seq"] // BIN, []).append(x)
        keys = sorted(strata)
        cnt_mat = np.array([np.concatenate([x["cnt"], [x["n"]]]) for x in items])
        # 按 strata 记录下标
        pos = {id(x): i for i, x in enumerate(items)}
        stratum_ids = [np.array([pos[id(x)] for x in strata[k]]) for k in keys]
        reps = np.empty((n_rep, 6))
        ns = np.empty(n_rep)
        iE, iB = DIMS.index(EASE), DIMS.index(BITTER)
        for b in range(n_rep):
            pick = np.concatenate([sids[rng.integers(0, len(sids), len(sids))] for sids in stratum_ids])
            tot = cnt_mat[pick].sum(axis=0)
            n = tot[-1]
            ns[b] = n
            if n > 0:
                per = tot[:-1] / n * 1000.0
                reps[b, :5] = per
                reps[b, 5] = per[iE] - per[iB]
            else:
                reps[b] = np.nan
        out[s] = (reps, ns)
    return out

def pct(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

def pack(reps, ns):
    per = {}
    for j, d in enumerate(DIMS + ["NMI"]):
        per[d] = {"rate": round(float(np.nanmean(reps[:, j])), 3), "ci95": pct(reps[:, j])}
    return {"chars_mean": round(float(np.mean(ns)), 0), "rates": per}

# ---------------- 6. 主分析 ----------------
idx_main = count_arrays(su, CURVE + ["exile_B_suppl"])
# tier1 (主): buffer/unknown 不参与 → count_arrays 已只取 CURVE+suppl
bp_main = boot_pass(idx_main, CURVE + ["exile_B_suppl"])
curve_main = {s: pack(*bp_main[s]) for s in CURVE + ["exile_B_suppl"]}

def diff_from(bp, exile_segs, ctrl_segs):
    def merge(segs):
        reps = np.zeros((B, 6)); ns = np.zeros(B)
        for s in segs:
            r_, n_ = bp[s]
            r_safe = np.where(np.isnan(r_), 0.0, r_)
            reps += r_safe * n_[:, None]
            ns += n_
        with np.errstate(invalid="ignore", divide="ignore"):
            return reps / ns[:, None], ns
    er, en = merge(exile_segs)
    cr, cn = merge(ctrl_segs)
    d = er - cr
    out = {"exile_chars_mean": round(float(np.mean(en)), 0),
           "control_chars_mean": round(float(np.mean(cn)), 0)}
    for j, s in enumerate(DIMS + ["NMI"]):
        out[s] = {"exile_rate": round(float(np.nanmean(er[:, j])), 3),
                  "control_rate": round(float(np.nanmean(cr[:, j])), 3),
                  "diff": round(float(np.nanmean(d[:, j])), 3),
                  "ci95": pct(d[:, j]),
                  "ci_excludes_zero": bool(pct(d[:, j])[0] > 0 or pct(d[:, j])[1] < 0)}
    # 两端 bootstrap p (NMI)
    v = d[:, -1]; v = v[~np.isnan(v)]
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    out["NMI"]["boot_p_two_sided"] = round(float(max(p, 1.0 / len(v))), 4)
    return out

main_diff = diff_from(bp_main, EXILE_SEGS, CTRL_SEGS)
log(f"main diff NMI: {main_diff['NMI']}")

# A级-only (剔除 17 首 B 级补编覆写)
a_diff = diff_from(bp_main, ["黄州", "惠州", "儋州"], CTRL_SEGS)

# 剂量—反应
dose_segs = [("黄州", 2), ("惠州", 3), ("儋州", 4)]
nmi_pts = []
for s, sev in dose_segs:
    reps, ns = bp_main[s]
    nmi_pts.append((s, sev, reps[:, -1]))
slope_reps = np.empty(B)
for b in range(B):
    xs = np.array([sev for _, sev, _ in nmi_pts])
    ys = np.array([r[b] for _, _, r in nmi_pts])
    w = np.array([bp_main[s][1][b] for s, _ in dose_segs])
    W = w / w.sum()
    xm, ym = np.sum(W * xs), np.sum(W * ys)
    slope_reps[b] = np.sum(W * (xs - xm) * (ys - ym)) / np.sum(W * (xs - xm) ** 2)
from scipy.stats import spearmanr
xs_full = np.array([sev for _, sev, _ in nmi_pts])
ys_mean = np.array([np.nanmean(r) for _, _, r in nmi_pts])
rho, _ = spearmanr(xs_full, ys_mean)
rho_reps = np.array([spearmanr(xs_full, [r[b] for _, _, r in nmi_pts])[0] for b in range(0, B, 10)])
slope_point = float(np.nanmean(slope_reps))
sl_ci = pct(slope_reps)
sp_lo, sp_hi = pct(rho_reps)
mono = bool(ys_mean[0] >= ys_mean[1] >= ys_mean[2] or ys_mean[0] <= ys_mean[1] <= ys_mean[2])
dose = {"design": "黄州 sev=2 → 惠州 sev=3 → 儋州 sev=4 (exile_chronology.csv severity_grade)",
        "per_period": {s: {"severity": sev, "NMI": curve_main[s]["rates"]["NMI"]}
                       for s, sev in dose_segs},
        "weighted_linear_slope_per_grade": {"point": round(slope_point, 3), "ci95": sl_ci,
                                            "boot_p_two_sided": round(float(2 * min((slope_reps <= 0).mean(),
                                                                                     (slope_reps >= 0).mean())), 4)},
        "spearman_trend_across_3_periods": {"rho": round(float(rho), 3), "ci95": [sp_lo, sp_hi],
                                            "note": "n=3 期, 把握度极低, 仅作方向参考"},
        "monotonic_in_point_estimates": mono}
log(f"dose: {json.dumps(dose, ensure_ascii=False)}")

# ---------------- 7. 稳健性 ----------------
# tier0: buffer 就近归期
su_t0 = [dict(r) for r in su]
buf = sorted([r for r in su_t0 if r["period"] == "boundary_buffer"], key=lambda x: x["seq"])
anchor_segs = sorted([r["seq"] for r in su_t0 if r["seg"] is not None])
import bisect
for r in su_t0:
    if r["period"] == "boundary_buffer":
        i = bisect.bisect_left(anchor_segs, r["seq"])
        cand = [anchor_segs[j] for j in (i - 1, i) if 0 <= j < len(anchor_segs)]
        near = min(cand, key=lambda s: abs(s - r["seq"])) if cand else None
        near_row = next(x for x in su_t0 if x["seq"] == near)
        r["seg"] = near_row["seg"]
idx_t0 = count_arrays(su_t0, CURVE + ["exile_B_suppl"])
bp_t0 = boot_pass(idx_t0, CURVE + ["exile_B_suppl"], n_rep=B)
t0_diff = diff_from(bp_t0, EXILE_SEGS, CTRL_SEGS)

# tier2: 再剔除距边界 <2 序列年
anchors = [(0, 1059.0), (1022, 1080.0), (1190, 1084.4), (2003, 1094.83),
           (2161, 1097.0), (2294, 1100.5), (2386, 1101.7)]
ax = np.array([a[0] for a in anchors]); ay = np.array([a[1] for a in anchors])
bnds = [1080.0, 1084.4, 1094.83, 1097.0, 1100.5]
su_t2 = []
n_dropped_t2 = 0
for r in su:
    if r["seg"] is None or r["seq"] > 2386:
        continue
    yr = float(np.interp(r["seq"], ax, ay))
    if min(abs(yr - b) for b in bnds) < 2.0:
        n_dropped_t2 += 1
        continue
    su_t2.append(r)
idx_t2 = count_arrays(su_t2, CURVE)
bp_t2 = boot_pass(idx_t2, CURVE, n_rep=B)
t2_diff = diff_from(bp_t2, EXILE_SEGS[:3], CTRL_SEGS)
t2_ns = {s: int(round(float(np.mean(bp_t2[s][1])))) for s in CURVE}
log(f"tier2 dropped {n_dropped_t2} poems near boundaries; surviving n per seg: {t2_ns}")

# 短诗敏感性 (<40 字剔除)
su_short = [dict(r) for r in su if r["n"] >= MIN_SHORT and r["seg"] is not None]
idx_sh = count_arrays(su_short, CURVE + ["exile_B_suppl"])
bp_sh = boot_pass(idx_sh, CURVE + ["exile_B_suppl"], n_rep=B)
sh_diff = diff_from(bp_sh, EXILE_SEGS, CTRL_SEGS)
n_short = sum(1 for r in su if r["seg"] is not None and 0 < r["n"] < MIN_SHORT)

# ---------------- 8. 白居易 (条件性佐证, B级) ----------------
def bai_group(r):
    p = r["period"]
    if p == "exile":
        return "exile"
    if p in ("pre", "post"):
        return "nonexile"
    return None
for r in bai:
    r["seg"] = bai_group(r)
idx_b = count_arrays(bai, ["exile", "nonexile"])
bp_b = boot_pass(idx_b, ["exile", "nonexile"], n_rep=B)
bai_diff = diff_from(bp_b, ["exile"], ["nonexile"])
log(f"bai diff NMI: {bai_diff['NMI']}")

# ---------------- 9. 输出 ----------------
def seg_pack(s):
    d = curve_main[s]
    d2 = {"n_poems": seg_counts.get(s, 0)}
    d2.update({k: d[k] for k in ("chars_mean", "rates")})
    return d2

out = {
 "method": ("within-author paired period contrast on the M1 lexicon; rates per 1000 chars; "
            "stratified bootstrap (strata = 100-poem chronological bins, seed 20260918, B=10000); "
            "period boundaries are 陈善本 event anchors (乌台狱995/初到黄州1022/别黄州1190/"
            "初到惠州2003/吾谪海南2161/渡海2294/答径山琳长老2386)"),
 "join_verification": {"su": su_join, "bai": bai_join},
 "su_shi": {
   "segment_counts": {k: v for k, v in sorted(seg_counts.items(), key=lambda x: str(x[0]))},
   "period_curve": {s: seg_pack(s) for s in CURVE + ["exile_B_suppl"]},
   "main_diff_exile_vs_prepost": main_diff,
   "a_grade_only_diff": a_diff,
   "dose_response": dose,
   "robustness": {
      "a_only_diff_NMI": a_diff["NMI"],
      "drop_poems_under_40chars": {"n_dropped": n_short, "diff": {k: sh_diff[k] for k in DIMS + ["NMI"]}},
      "buffer_tier0_nearest_reassign": {k: t0_diff[k] for k in DIMS + ["NMI"]},
      "buffer_tier2_exclude_within_2seqyears": {"n_extra_dropped": n_dropped_t2,
                                                "surviving_n_per_seg": t2_ns,
                                                "diff": {k: t2_diff[k] for k in DIMS + ["NMI"]}},
      "note": ("tier1=主分析(剔除 CSV boundary_buffer 76 首); tier0=缓冲诗按编年序就近归期; "
               "tier2=在 tier1 基础上再剔除距任一事件边界 <2 序列年(锚点线性插值)的诗")}},
 "bai_juyi_conditional": {
   "grade": "B (条件性佐证)",
   "groups": {"exile": {"n": sum(1 for r in bai if r["seg"] == "exile")},
              "nonexile_merged": {"n": sum(1 for r in bai if r["seg"] == "nonexile")}},
   "diff_exile_vs_merged_nonexile": bai_diff,
   "selection_bias": ("pre 组 66 首中 50 首为新乐府讽谕诗(元和初谏官期, 陈善本裁定), 讽谕诗"
                      "本身高悲苦/贬谪语汇, 故非贬期对照被系统性拉向'忧愤'; 因此白居易结果"
                      "只作条件性佐证, 不分 pre/post、不进剂量梯度"),
   "no_dose_gradient": True},
 "others_not_run": {
   "柳宗元": "贬前存诗不足, 无法作者内配对",
   "黄庭坚": "逐诗分期实测样本极少(4首)且实测试点推翻编年挂接, 不跑",
   "欧阳修/王禹偁": "exile 样本 12/29 首, 达不到配对功效, 按裁决不跑"},
}
with open(f"{BASE}/results_exile_dose.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log("wrote results_exile_dose.json")
LOG.close()
print("done")
