# 扩展基准：与 2025/2026 最新 CNN 的 head-to-head 对照

> 训练脚本：`dl_cnn2025_bench.py`；结果：`results_cnn2025_bench.json`
> 全部模型在**完全一致**的条件下训练：池 = `dl_pool.json`（简体诗，NMI + 5 维画像弱标签），
> 3σ 剔除后 **20,847** 首；固定 85/15 切分（seed=20260915）；字符表 ≤8000，MAX_LEN=160；
> **14 epoch**，batch 128，Adam lr=1e-3；多任务头（NMI + 5 维画像辅助），仅 Kim baseline 为 NMI-only。

## 结果表（测试集，NMI 回归）

| 模型 | 年份 / 出处 | 类型 | r | ρ | MAE | 参数量 |
|---|---|---|---|---|---|---|
| Kim CharCNN（基线） | Kim 2014 | NMI-only | 0.9289 | 0.7727 | 3.557 | 564k |
| Label-matched 普通 CNN | — | 多任务 | 0.9335 | 0.7705 | 4.131 | 597k |
| Wide 普通 CNN（容量对照） | — | 多任务 | 0.9766 | 0.7849 | 1.833 | 600k |
| **OverLoCK**（1D 适配） | CVPR 2025 · arXiv 2502.20087 | 纯 ConvNet | 0.9739 | 0.7728 | 1.785 | 896k |
| **PFGNet**（1D 适配） | CVPR 2026 · arXiv 2602.20537 | 纯 ConvNet | 0.9739 | 0.7840 | 1.615 | 739k |
| **AttentionFractalCNN**（本文） | proposed | 多尺度 + 注意力 | 0.9767 | 0.7803 | 1.634 | 631k |
| **TransXNet**（1D 适配） | TNNLS 2025 · arXiv 2310.19380 | CNN-Transformer 混合 | **0.9814** | **0.7896** | **1.474** | 643k |

## 诚实结论（重要）

1. **本文模型在准确率上并不优于近年 SOTA CNN。** 在等算力（14 ep）、等数据、等标签条件下，
   本文 AttentionFractalCNN（r=0.9767）与 OverLoCK（0.9739）、PFGNet（0.9739）、以及容量匹配的
   Wide 普通 CNN（0.9766）**基本持平**，且**略低于** TransXNet（0.9814）。注意力+分形机制并未带来
   准确率上的压倒性优势。

2. **此前"0.93→0.99"的跃升主要来自容量，而非所提机制。** 拆解如下：
   - 更丰富的多任务标签 ≈ +0.005（Label-matched 0.9335 vs 基线 0.9289）；
   - 容量/现代模块（Wide 普通 CNN）≈ +0.043 至 0.977；
   - 分形+注意力在**同等容量**的 Wide 普通 CNN 之上仅 ≈ +0.01。
   因此 Turn 3 中"相对 Kim 基线大幅改进"的观感，主要由容量差距造成，不是机制本身。

3. **单跑方差约 ±0.01。** 本文跑出 0.9767，而 Turn 3 旧脚本跑出 0.9882——差距源于模型创建顺序导致的
   初始化差异（两跑均在 14 ep 附近收敛）。论文中任何数字都**应先做多 seed 置信区间**，不要采用单跑值。

4. **本文模型可主张的、站得住脚的功绩是"可解释性 + 同等竞争力"，而非"更准"：**
   - 多头读出注意力给出**逐字符显著度**（Turn 3 旧脚本已报告词典字符 top-10% 注意力提升），
     这是所有纯准确率 CNN 都不具备的；
   - 多尺度 Fractal Tower 是与诗集层级（字→词→句→联）对应的、容量可比的编码器，达到与
     2025/2026 SOTA CNN 持平的水平。

## 对论文的改写建议

- 新增"与近年 CNN 架构的对照"小节，放入上表 + 上述诚实叙述；**不要宣称 SOTA 准确率**。
- 把贡献重新定位为：**在达到与 2025/2026 SOTA CNN 同等竞争力的同时，提供逐字符可解释性**。
- 如实引用 OverLoCK / TransXNet / PFGNet（arXiv 编号均为真实可查）。
- 在放入论文前，建议对本文模型与最接近的 TransXNet 各跑 5 个 seed 给出 mean±std，以消解单跑方差。

## 复现

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
cd tangsong_research
D:/anaconda/python.exe dl_cnn2025_bench.py   # 需 anaconda(torch 2.9+cu128)；OpenCC 已预计算进 dl_pool.json
```
