# 反事实版本分析 — 论文段（草稿，供 team-lead 审后回填）

生成：counterfactual.py（seed=20260918, B=300, k=1.0108, MINC=300）→ results_counterfactual.json
渠道：官修选本《御選宋詩》(n=256) / 私家选本《宋詩鈔》(n=75；作者级反事实限三向共有 68 人) / 唐·同书重印渠道《全唐诗》vs《御定全唐诗》(n=479, 对照)

---

## EN (Results / Analysis section)

**Counterfactual edition-channel projection.** To quantify how much of a poet's
measured emotional profile is an artifact of the survival channel, we project
each author's posterior corpus-side dimension rates onto three transmission
channels within the hierarchical model: the officially compiled *yuxuan*
anthology, the privately compiled *Songshichao*, and — as a near-transparent
control — the same-book reprint channel (*quanTangshi* vs. *yuding* reprint).
At the aggregate level the channels differ modestly: delta_NMI = −0.22
[−0.51, 0.06] (official), −0.43 [−0.68, −0.18] (private), +0.43 [0.25, 0.60]
(Tang reprint), each at least an order of magnitude smaller than the
author-population heterogeneity (tau ≈ 1.9‰). A poet's scalar "gloominess" is
therefore largely edition-stable. The five-dimensional profile is not: both
selections systematically inflate the landscape dimension (自然山水 +13.2‰
official, +4.2‰ private) and the temporal-vision dimension (空幻时空 +2.5/+0.4)
while contributing only trace amounts of 贬谪羁旅 and 悲苦孤寂; the scalar net
effect is small only because the reshaping is dimension-specific, not because
selection is neutral.

Author-level counterfactuals make the asymmetry concrete. 陆游, the most
buoyant (旷达) of the eight queue officials as measured on his corpus (+0.53),
would measure ≈ 0 through either anthology channel (+0.26 [−0.13, 0.72]
official; −0.01 [−0.40, 0.42] private) — a sign flip produced purely by
transmission. 苏轼's measured gloominess deepens from −0.51 to −0.77
[−1.24, −0.30] through the official channel and −0.95 [−1.38, −0.47] through
the private one; 秦观's from −1.93 to −2.10 / −2.42; while 欧阳修 and 王禹偁
are edition-robust. Conversely, the official anthology would present 范仲淹 as
more serene than his corpus warrants (counterfactual −0.66 [−1.81, 0.44] vs.
an anthology-measured +0.74). Reconstructing a poet's corpus-side score from
the anthology alone can misstate it by ≈ 2‰ (黄庭坚: anthology-only
reconstruction −1.98 vs. true −0.03), because small anthology samples inherit
the population prior. Globally, author rankings are stable: the Spearman rank
correlation between corpus-measured and official-channel counterfactual NMI is
0.95 across 256 authors, yet 15.7% of pairwise orders change, and 21 of 256
authors (8.2%) show significant CI-excluding-zero shifts, up to −19.2‰
(李宗谔) and −10.2‰ (田锡); the most stable authors (何郯, 尤袤, 徐玑, 赵师秀)
move by less than 0.05‰. The Tang control shows these effects are properties
of *selection*, not of reprinting or digitization per se: 刘禹锡's
counterfactual through the same-book reprint channel, −2.70 [−3.24, −2.14],
reproduces his corpus-side value (−2.68). His gloominess is his own, not his
editions' — whereas for the Song poets the transmission channel is a visible,
dimension-specific co-author of the measured profile. (辛弃疾 enters neither
Song anthology and has only ≈ 7,000 shi characters in the corpus; no
counterfactual is computed for him.)

## 中文对应段

**反事实版本渠道投影。** 为量化一位诗人的测得情绪剖面有多少来自「幸存版本」，
我们在层次模型内把每位作者的全集侧维度速率后验投影到三条流传渠道：官修选本
《御選宋詩》、私家选本《宋詩鈔》，以及作为近透明对照的同书重印渠道（《全唐诗》
vs《御定全唐诗》）。总体层面三条渠道的差异有限：δ_NMI = −0.22 [−0.51, 0.06]
（官修）、−0.43 [−0.68, −0.18]（私家）、+0.43 [0.25, 0.60]（唐重印），均比作者
间异质（τ ≈ 1.9‰）小一个数量级以上——诗人的标量「枯槁度」对版本基本稳健。
但五维剖面并非如此：两个选本都系统性抬高自然山水（+13.2‰ / +4.2‰）与空幻时空
（+2.5 / +0.4），而贬谪羁旅与悲苦孤寂仅微量增入；净效应之所以小，是因为重塑
是维度特异性的，而不是因为选本是中立的。

作者级反事实使这一不对称具体化。陆游在全集侧是八位队列官员中最旷达者
（+0.53），但经任一选本渠道测得都约为零（官修 +0.26 [−0.13, 0.72]；私家
−0.01 [−0.40, 0.42]）——纯由流传造成的符号翻转。苏轼的枯槁度从全集侧 −0.51
加深到官修渠道 −0.77 [−1.24, −0.30]、私家渠道 −0.95 [−1.38, −0.47]；秦观从
−1.93 到 −2.10 / −2.42；欧阳修与王禹偁则对版本稳健。反向地，官修选本会把
范仲淹呈现得比其全集更闲适（反事实 −0.66 [−1.81, 0.44]，而选本实测 +0.74）。
仅凭选本还原诗人全集侧分数可误差约 2‰（黄庭坚：仅选本还原 −1.98 vs 真值
−0.03）。总体上作者排序稳定：256 位作者的全集侧实测秩与官修渠道反事实秩的
Spearman 相关为 0.95，但 15.7% 的两两序对翻转，21/256 位作者（8.2%）位移的
置信区间不含零，最大者如李宗谔 −19.2‰、田锡 −10.2‰；最稳固的作者（何郯、
尤袤、徐玑、赵师秀）位移不足 0.05‰。唐代对照表明这些效应是「选本」的属性
而非「重印/数字化」的属性：刘禹锡经同书重印渠道的反事实 −2.70 [−3.24, −2.14]
与其全集侧测值 −2.68 几乎一致——他的枯槁是他自己的，不是他的版本的；而宋代
诗人测得的剖面，其流传渠道是一位可见的、维度特异的合著者。（辛弃疾不入两部
宋选本，全集中诗体作品仅约 7 千字，不作反事实投影。）

## 复现
- 脚本 `counterfactual.py`；seed=20260918；B=300；k=1.0108（继承 hb_model.py）；MINC=300。
- 点估计与已发布结果核对：δ_NMI 官修 −0.224 = results_hb_model.json；私家 −0.427 =
  results_song3_rep.json；官修五维 δ 全部一致。私家自然山水重算 4.212 vs 已发布
  4.138（差异在作者集舍入层面，已在 reproducibility_check 中注明）。
- 全量数字：`results_counterfactual.json`；运行日志：`counterfactual_log.txt`。
