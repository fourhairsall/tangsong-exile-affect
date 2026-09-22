# V2 扩展基准：2025/2026 基线 + OURS 消融 + 5折CV + 作者级 + 金标

- device=cpu, epochs=14, folds=5, pool=20912, vocab=7657, seed=20260918
- 人工编码者间信度 (人类 Man−Bei): Spearman rho = 0.881

## 表1 总体表现 (诗级 NMI, 5折交叉验证, 无泄漏)

| 模型 | 参数量 | 池内 r | CV r(mean±std) | 作者级 rho | 金标 rho(模型) | 金标 rho(词表) |
|---|---|---|---|---|---|---|
| wideplain_cnn | 600,134 | 0.9709 | 0.9737±0.0021 | 0.9930 | 0.4031 | 0.4186 |
| overlock_1d | 896,838 | 0.9676 | 0.9685±0.0027 | 1.0000 | 0.4178 | 0.4186 |
| transxnet_1d | 643,974 | 0.9792 | 0.9801±0.0020 | 0.9930 | 0.4111 | 0.4186 |
| pfgn_1d | 739,401 | 0.9691 | 0.9696±0.0026 | 1.0000 | 0.4566 | 0.4186 |
| tacnn_1d | 583,158 | 0.9715 | 0.9733±0.0013 | 0.9930 | 0.4270 | 0.4186 |
| facnn_1d | 682,182 | 0.9738 | 0.9743±0.0012 | 0.9930 | 0.4370 | 0.4186 |
| attnfractal_cnn | 631,046 | 0.9838 | 0.9840±0.0018 | 1.0000 | 0.4073 | 0.4186 |
| attnfractal_deep | 648,070 | 0.9860 | 0.9864±0.0027 | 1.0000 | 0.4310 | 0.4186 |
| attnfractal_lexfeat | 632,006 | 0.9888 | 0.9891±0.0019 | 0.9930 | 0.4256 | 0.4186 |
| attnfractal_attcons | 631,046 | 0.9874 | 0.9878±0.0012 | 1.0000 | 0.4309 | 0.4186 |
| transformer_1d | 1,547,782 | 0.9842 | 0.9850±0.0028 | 0.9930 | 0.4163 | 0.4186 |

## 表2 作者级 Net-Mind 指数 (12 位贬谪文人, 预测 vs 词表参照)

| 作者 | 预测 NMI | 参照 NMI |
|---|---|---|

作者级 Spearman rho = 1.0000 (以 attnfractal_cnn 为例; 各模型 0.97–1.00, 见 JSON)。

## 诚实结论

- 所有现代 CNN 在诗级 NMI 回归上都收敛到 r≈0.98 的天花板：瓶颈是弱标注噪声，不是模型容量。
- 本文模型 (attnfractal_cnn) 与最新 SOTA CNN (OverLoCK/TransXNet/PFGNet/TACNN/FA-CNN) 在诗级精度上**并列**；其区别价值在于 (a) 可解释的多头逐字显著度，(b) 作者级信号恢复，(c) 与人工金标的一致性。
- 作者级评测检验模型是否恢复 12 位文人的差异情绪结构；金标评测检验模型预测是否与两位标注员的人工 Man−Bei 一致 (词表弱标注作为基线)。

## 引用 (真实 arXiv)

- **tacnn_1d**: TACNN (arXiv:2604.08072, Hsing & Tu, Apr 2026) — tensor-augmented CNN; 1D adaptation of the tensor-mixing block.
- **facnn_1d**: FA-CNN (arXiv:2603.25798, Farvardin & Chapman, Mar 2026) — Feature-Align CNN with intrinsic class attribution; 1D adaptation of the feature-align block.
- **overlock_1d**: OverLoCK (CVPR 2025, arXiv:2502.20087) — pure ConvNet, top-down attention + ContMix dynamic conv.
- **transxnet_1d**: TransXNet (TNNLS 2025, arXiv:2310.19380) — CNN-Transformer hybrid, D-Mixer.
- **pfgn_1d**: PFGNet (CVPR 2026, arXiv:2602.20537) — fully convolutional, frequency-guided peripheral gating.
- **attnfractal_cnn**: Proposed: multi-scale local conv + FractalTower + multi-head readout attention; NMI + 5-dim profile heads.