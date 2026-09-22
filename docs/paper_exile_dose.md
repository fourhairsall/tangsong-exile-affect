# 贬谪剂量—反应分析 — 论文段草稿（供 team-lead 审后回填）

生成：`exile_dose.py`（seed=20260918, B=10000, 编年序每100首一层分层bootstrap）→ `results_exile_dose.json`
挂接核验：苏轼 2824/2824、白居易 2931/2931 逐诗按 poem_id 挂接语料，text_head 指纹 0 错配。
分期边界（陈善本锚点）：乌台狱995 / 初到黄州1022 / 别黄州1190 / 初到惠州2003 / 吾谪海南2161 / 渡海2294 / 绝笔2386。
样本：苏轼 exile 473（黄州168·惠州157·儋州131·补编覆写17）vs pre+post 1855；buffer 76 剔除、unknown 420 不计。

---

## EN (Results / Analysis section)

**Within-author exile contrast (Su Shi, grade-A chronology).** Among the six
queue officials only Su Shi permits a fully powered within-author design: his
2,824 *quansongshi* poems are chronologically ordered, and 473 fall inside his
three exile stints (Huangzhou 168, Huizhou 157, Danzhou 131, plus 17
supplementary title-keyword assignments) against 1,855 pre- and post-exile
poems (76 boundary-buffer and 420 undatable poems excluded). The net lexicon
score shows **no significant exile effect**: NMI = −0.36‰ in exile vs. −0.66‰
outside, difference +0.30 [−0.94, +1.54] (year-stratified bootstrap, B =
10,000, seed 20260918). The dimension profile does move: 贬谪羁旅 vocabulary
rises (+0.89 [0.08, 1.74]) — the exile names itself — while 自然山水 (−3.21
[−5.52, −0.91]) and 空幻时空 (−2.62 [−4.17, −1.06]) fall; 悲苦孤寂 (−0.35
[−1.25, 0.58]) and 旷达闲适 (−0.05 [−0.86, 0.80]) do not move at all. Exile
did not make Su Shi's lexicon gloomier.

The period curve locates the single dark stretch: pre −1.65 → **Huangzhou
−2.42 [−4.39, −0.54]** → Yuan-you interlude +0.64 → Huizhou +0.08 → **Danzhou
+1.96 [−0.18, +4.07]** → northward return +0.29. Huangzhou, the immediate
aftermath of the Crow-Terrace Poetry Case, is the only period whose CI
excludes zero; Danzhou is lexically his most buoyant period (旷达闲适 6.6/1,000
chars, a career high). The severity gradient Huangzhou(2) → Huizhou(3) →
Danzhou(4) is monotonic but with the **opposite sign** to a dose-deepens-gloom
model: weighted linear slope +2.20‰ per grade [0.77, 3.65], bootstrap p =
0.002; Spearman ρ = 1.0 (n = 3 periods; low power, direction only). We report
this inversion as found, without adjudicating its mechanism: in lexicon terms
Su Shi's measured buoyancy *rises* with exile severity — consistent with his
documented self-transcendence rhetoric (「兹游奇绝冠平生」) and inconsistent
with reading NMI as a direct situational distress meter. Robustness: A-grade
only (+0.34), dropping poems < 40 chars (+0.48), and reassigning the 76
buffer poems to adjacent periods (+0.35) all leave the null unchanged. A
±2-year boundary-exclusion window erases the short Huizhou/Danzhou stints
entirely (0 surviving chars) and is therefore uninformative rather than
confirmatory.

**Conditional corroboration (Bai Juyi, grade B).** For Bai Juyi — the only
Tang poet with a usable within-author split — exile (59 poems, Jiangzhou/
Zhongzhou) vs. merged non-exile (84) gives ΔNMI = +3.43 [0.72, 6.23]. This
must not be read as "exile cheered him up": 50 of his 66 pre-exile poems are
New Yuefu satires (元和初谏官期), a genre that saturates the control with
悲苦 lexicon (6.2‰ vs. 4.0‰ in exile), so the contrast is confounded by genre
selection. The safe claims are dimensional only: in exile, 贬谪羁旅 self-naming
(+3.66 [1.68, 5.76]) and 自然山水 (+21.8 [15.5, 28.7], the Jiangzhou landscape
mode) rise. Per the adjudication we report no pre/post split and no dose
gradient for Bai.

**Why the other four are not run.** 柳宗元 has no usable pre-exile corpus;
黄庭坚's poem-level attachment pilot yielded only 4 poems and was disproven;
欧阳修 (12 exile poems) and 王禹偁 (29) fall below pairing power. The
dose–response claim of this paper therefore rests on one author executed at
grade A, one conditional grade-B check, and three documented non-runs — an
honest ceiling we keep visible.

## 中文对应段

**作者内贬谪对照（苏轼，A级编年）。** 六位队列官员中唯有苏轼可做全功效的作
者内设计：其《全宋诗》2,824 首依编年排序，473 首落在三次贬期之内（黄州 168、
惠州 157、儋州 131，另 17 首为补编题面覆写），对照为贬前+贬后 1,855 首（剔除
边界缓冲 76 首、不可断代 420 首）。净心指数**无显著贬谪效应**：贬期 −0.36‰ vs
非贬期 −0.66‰，差 +0.30 [−0.94, +1.54]（按年分层 bootstrap，B=10,000，seed
20260918）。但维度剖面在动：贬谪羁旅词汇上升（+0.89 [0.08, 1.74]）——贬谪
自行命名；自然山水（−3.21 [−5.52, −0.91]）与空幻时空（−2.62 [−4.17, −1.06]）
反而下降；悲苦孤寂（−0.35 [−1.25, 0.58]）与旷达闲适（−0.05 [−0.86, 0.80]）
纹丝不动。贬谪没有使苏轼的词汇面貌更悲苦。

分期曲线定位出唯一一段深谷：贬前 −1.65 → **黄州 −2.42 [−4.39, −0.54]** →
元祐间 +0.64 → 惠州 +0.08 → **儋州 +1.96 [−0.18, +4.07]** → 北归 +0.29。
乌台诗案余波中的黄州是唯一置信区间不含零的时期；儋州则是其词汇意义上最
旷达的时期（旷达闲适 6.6/千字，生涯峰值）。瘴疠剂量梯度 黄州(2)→惠州(3)→
儋州(4) 逐期点估计单调，但方向与「剂量加深悲苦」模型**相反**：加权线性斜率
+2.20‰/级 [0.77, 3.65]，bootstrap p=0.002；Spearman ρ=1.0（仅 3 期，把握度
低，仅作方向参考）。我们按原样报告这一反向梯度，不裁定其机制：在词表意义
上，苏轼的测得旷达随贬谪严重度**上升**——与其自述的超脱修辞（「兹游奇绝冠
平生」）一致，与把 NMI 读作情境痛苦直接计的做法不一致。稳健性：仅用 A 级
（+0.34）、剔除 <40 字短诗（+0.48）、76 首缓冲诗就近归期（+0.35），零假设
均不变。±2 年边界剔除窗会把惠州/儋州短贬期整体清空（存字为 0），故该档不
具信息量，如实报告为「不可判」而非「稳健」。

**条件性佐证（白居易，B级）。** 白居易是唯一可做作者内切分的唐人：贬期
（江州/忠州，59 首）vs 合并非贬期（84 首），ΔNMI=+3.43 [0.72, 6.23]。**不得**
读作「贬谪使他旷达」：其贬前 66 首中 50 首为新乐府讽谕诗（元和初谏官期），
讽谕体裁使对照组的悲苦词汇饱和（6.2‰ vs 贬期 4.0‰），对比被体裁选择混淆。
可靠的结论只在维度层面：贬期内贬谪羁旅自指（+3.66 [1.68, 5.76]）与自然山水
（+21.8 [15.5, 28.7]，江州山水模式）上升。按裁决，白居易不分 pre/post、不进
剂量梯度。

**其余四人不跑的理由。** 柳宗元无可用的贬前语料；黄庭坚逐诗挂接试点仅得
4 首且被实测推翻；欧阳修（贬期 12 首）与王禹偁（29 首）低于配对功效。本文
的剂量—反应主张因此建立在：一位作者的 A 级全量执行、一位唐人的 B 级条件性
佐证、三次如实记录的不执行——我们把这个诚实的上限保持在明面上。

## 复现
- 脚本 `exile_dose.py`；seed=20260918；B=10000；分层=编年序每 100 首一层。
- 挂接：苏轼 poem_id=file#corpus_index，白居易 poem_id=UUID→语料 id 字段；
  text_head 12 字指纹 0 错配（2824+2931）。
- 全量数字：`results_exile_dose.json`；日志：`exile_dose_log.txt`。
