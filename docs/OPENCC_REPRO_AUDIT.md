# 证据：原 pin `OpenCC==1.4.2` 无法复现论文的具体数字

> 结论先说：`requirements.txt` 原锁定 `OpenCC==1.4.2`，但用它重跑论文数字的生产脚本
> `cross_edition_points.py`，**唐侧 479 位共享作者会变成 478 位**。论文中「479」共出现
> **12 处**（中稿 5 行、英稿 7 行），在原 pin 下**全部无法复现**。宋侧（256 位、$A{=}0.5149$）
> 不受影响。

## 1. 复现方法（任何人可自行验证）

```bash
pip install --target ./_occ_test "OpenCC==1.4.2"          # 原 pin
PYTHONPATH=./_occ_test  python cross_edition_points.py    # A 组

pip install opencc-python-reimplemented==0.1.7            # 实际产出论文者
python cross_edition_points.py                            # B 组
```

两个包暴露同一个 API `opencc.OpenCC('t2s')`，可互相 shadow，故对照是干净的 A/B。

## 2. 对照结果（生产脚本 `cross_edition_points.py` 实测）

| 源对 | 原 pin `OpenCC==1.4.2` | 实际 `opencc-python-reimplemented==0.1.7` | 判定 |
|---|---|---|---|
| 唐：QTS vs YD | **n=478**, r=0.9494, S=5/5 | **n=479**, r=0.9493, S=5/5 | **不可复现**（少 1 位作者；小數第 4 位偏移） |
| 宋：QSS vs 御選 | n=256, r=0.5149, S=3/5 | n=256, r=0.5149, S=3/5 | 可复现 |

差异根源：两个发行版携带的 t2s 词典不同，导致**少数作者名**转换后不再跨源对齐，
被 $\ge$300 字纳入规则筛掉 1 人；随之净 ABI 的第 4 位小数变动。

## 3. 受影响的具体论文位置（逐条）

### 中文稿 `report_cn/tangsong_nlp_paper_cn.tex`

| 行 | 论文原文（摘录） | 论文数字 | 原 pin 复现 | 判定 |
|---|---|---|---|---|
| L301 | 「扩展到 QTS 与 YD 共享的全部 **479** 位作者，一致性为 $A{=}0.95$……**479** 个作者级点对紧密沿对角线分布」 | 479（本行 2 处） | 478 | **不可复现** |
| L425 | 「（$0.9493\to0.9567$，$n{=}464$，原 **479**）」 | 479 | 478 | **不可复现** |
| L437 | 表 `tab:invariance`：「唐：QTS vs. YD（同书） & M1 & **479** & $0.95$ & $1.00$」 | 479 | 478 | **不可复现** |
| L438 | 表 `tab:invariance`：「唐：QTS vs. YD（同书） & 诱导 & **479** & $0.93$ & $1.00$」 | 479 | 478 | **不可复现** |
| L452 | 图 `fig:abi` caption：「**479** 位作者紧密沿对角线，$A{=}0.95$，$S{=}5/5$」 | 479 | 478 | **不可复现** |
| L415 | 表 `tab:realign` 唐行 pooled | $0.9493$ | $0.9494$ | 第 4 位小数**不可复现** |
| L423 | 自检句：「唐 $0.9493$、宋 $0.5149$ 与 $0.8221$，仅舍入之差」 | $0.9493$ | $0.9494$ | 第 4 位小数**不可复现** |

### 英文稿 `report/tangsong_nlp_paper.tex`

| 行 | 论文原文（摘录） | 论文数字 | 原 pin 复现 | 判定 |
|---|---|---|---|---|
| L765 | 「Scaling to all **479** authors shared by QTS and YD」 | 479 | 478 | **不可复现** |
| L768 | 「the **479** author-level pairs lie tightly along the diagonal」 | 479 | 478 | **不可复现** |
| L1039 | 「$n{=}464$, from **479**)」 | 479 | 478 | **不可复现** |
| L1073 | 表 `tab:invariance`：「Tang: QTS vs. YD (same work) & M1 & **479** & $0.95$ & $1.00$」 | 479 | 478 | **不可复现** |
| L1074 | 表 `tab:invariance`：「… & induced & **479** & $0.93$ & $1.00$」 | 479 | 478 | **不可复现** |
| L1090 | 图 `fig:abi` caption：「**479** authors lie tightly along the diagonal」 | 479 | 478 | **不可复现** |
| L1111 | 「(SE $0.14$, $n{=}$**479**) for the Tang reprint」 | 479 | 478 | **不可复现** |
| L1020 | 表 `tab:realign` 唐行 pooled | $0.9493$ | $0.9494$ | 第 4 位小数**不可复现** |

**合计：中稿 5 行 + 英稿 7 行 = 12 处不可复现**，另有 2 处（两稿 `tab:realign`）为第 4 位小数偏移。

## 4. 哪些数字不受影响（说明，避免过度声称）

- 摘要/引言的 **$A{=}0.95$**（CN L43、EN L84）：$0.9494$ 与 $0.9493$ 舍入后同为 $0.95$ → **不受影响**。
- 唐侧 **$S{=}5/5$** 与逐维 $r$（悲苦 0.95 / 贬谪 0.98 / 旷达 0.95 / 自然 0.97 / 空幻 0.95）：两版一致 → 不受影响。
- 宋侧全部（**256** 位、$A{=}0.5149$、$S{=}3/5$）→ 不受影响。
- 故这不是"整篇论文不可复现"，而是**唐侧纳入作者数这一具体统计量（及 4 位小数精度的 pooled 值）不可复现**。

## 5. 已采取的措施

1. `requirements.txt` 改锁 `opencc-python-reimplemented==0.1.7`，并在文件内写明上述差异与实测数值。
2. 论文全部数字保留在实际产出它们的 0.1.7 口径上（已核对与 `cross_edition_points.json`、
   `numbers_ledger.json` 一致）。
3. `cross_edition_points.json` 已复原为 0.1.7 产物（实验前有备份）。
