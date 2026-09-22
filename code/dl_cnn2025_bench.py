# -*- coding: utf-8 -*-
"""Extended head-to-head benchmark (Turn 4): our AttentionFractalCNN vs
recent 2025/2026 CNN architectures, on the IDENTICAL char-sequence NMI task.

All models share:
  * pool      = dl_pool.json (t2s-simplified 诗, NMI + 5-dim profile weak labels)
  * 3-sigma NMI drop  -> ~20,847
  * char vocab (<=8000), MAX_LEN=160, FIXED 85/15 split (seed=20260915)
  * 14 epochs, batch 128, Adam lr 1e-3
  * MULTI-TASK heads (NMI scalar + 5-dim profile aux) EXCEPT the faithful
    Kim baseline which regresses NMI only (matches the published pipeline).

The three modern CNNs are HONEST 1D adaptations of their published 2D/3D
core blocks (the original papers are vision backbones); each is given the
same global-pool readout + identical heads so the comparison isolates the
backbone / token-mixer architecture. Citations are real and listed in the
report printed at the end.

Models
  - charcnn            : Kim-style char-CNN (baseline, NMI only)  [dl_model.py]
  - labelmatched_cnn   : same encoder + NMI+profile heads, no fractal/attn (control)
  - overlock_1d        : OverLoCK CVPR2025  (pure ConvNet, ContMix dynamic conv)
  - transxnet_1d       : TransXNet TNNLS2025 (CNN-Transformer hybrid, D-Mixer)
  - pfgn_1d            : PFGNet  CVPR2026    (pure ConvNet, freq-guided gating)
  - attnfractal_cnn    : OURS  (multi-scale local + FractalTower + MHReadout)
  - wideplain_cnn      : capacity-matched plain CNN (control for param count)

Outputs results_cnn2025_bench.json.
"""
import os, json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260915
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MAX_LEN = 160
BATCH = 128
EPOCHS = 14
EMBED = 64
KERNELS = [3, 4, 5]
NCHAN = 64
HIDDEN = 128
LOCAL_K = [3, 5, 7]
D_MODEL = 128
N_HEADS = 4
LAMBDA_PROF = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True

# ----------------------------------------------------------------------------
# data plumbing (identical across all models)
# ----------------------------------------------------------------------------
def build_vocab(texts, max_vocab=8000):
    from collections import Counter
    c = Counter()
    for t in texts:
        c.update(t)
    vocab = ["<PAD>", "<UNK>"] + [w for w, _ in c.most_common(max_vocab - 2)]
    return vocab

def encode(text, vocab, max_len=MAX_LEN):
    ids = [vocab.index(ch) if ch in vocab else 1 for ch in text[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids

class PoemDS(Dataset):
    def __init__(self, x, y, z=None):
        self.x = x; self.y = y; self.z = z
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        if self.z is not None:
            return (torch.tensor(self.x[i], dtype=torch.long),
                    torch.tensor(self.y[i], dtype=torch.float32),
                    torch.tensor(self.z[i], dtype=torch.float32))
        return torch.tensor(self.x[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.float32)

def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return _pearson(ra, rb)

# ----------------------------------------------------------------------------
# shared head: global mean+max pool -> fc -> NMI head + profile head
# ----------------------------------------------------------------------------
class GlobalPoolHead(nn.Module):
    def __init__(self, D, hidden=HIDDEN):
        super().__init__()
        self.fc = nn.Linear(2 * D, hidden)
        self.act = nn.ReLU()
        self.nmi = nn.Linear(hidden, 1)
        self.prof = nn.Linear(hidden, 5)
        self.drop = nn.Dropout(0.3)
    def forward(self, h, return_attn=False):
        r = torch.cat([h.mean(-1), h.amax(-1)], dim=1)   # [B, 2D]
        r = self.act(self.fc(r))
        r = self.drop(r)
        if return_attn:
            return self.nmi(r).squeeze(1), self.prof(r), None
        return self.nmi(r).squeeze(1), self.prof(r)

# ----------------------------------------------------------------------------
# (1) faithful Kim baseline (NMI only)
# ----------------------------------------------------------------------------
class CharCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(EMBED, NCHAN, k) for k in KERNELS])
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(KERNELS) * NCHAN, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 1)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        hs = [self.act(c(x)).max(dim=2)[0] for c in self.convs]
        h = torch.cat(hs, dim=1)
        h = self.drop(h)
        h = self.act(self.fc1(h))
        return self.fc2(h).squeeze(1)

# ----------------------------------------------------------------------------
# (2) label-matched plain CNN (same encoder + NMI+profile heads, no fractal/attn)
# ----------------------------------------------------------------------------
class LabelMatchedCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(EMBED, NCHAN, k) for k in KERNELS])
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(KERNELS) * NCHAN, HIDDEN)
        self.act = nn.ReLU()
        self.head = GlobalPoolHead(HIDDEN)
    def forward(self, x, return_attn=False):
        x = self.emb(x).transpose(1, 2)
        hs = [self.act(c(x)).max(dim=2)[0] for c in self.convs]
        h = torch.cat(hs, dim=1)
        h = self.drop(h)
        h = self.act(self.fc1(h)).unsqueeze(-1)          # [B, HIDDEN, 1]
        return self.head(h, return_attn)

# ----------------------------------------------------------------------------
# (7) capacity-matched wide plain CNN (control for parameter count)
# ----------------------------------------------------------------------------
class PlainCNNBlock(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.c1 = nn.Conv1d(D, D, 3, padding=1, groups=D)
        self.c2 = nn.Conv1d(D, D, 1)
        self.act = nn.GELU()
    def forward(self, x):
        return x + self.act(self.c2(self.act(self.c1(x))))

class WidePlainCNN(nn.Module):
    def __init__(self, vocab_size, D=D_MODEL, n_blocks=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED, padding_idx=0)
        self.proj = nn.Conv1d(EMBED, D, 1)
        self.blocks = nn.ModuleList([PlainCNNBlock(D) for _ in range(n_blocks)])
        self.head = GlobalPoolHead(D)
        self.act = nn.ReLU()
    def forward(self, x, return_attn=False):
        h = self.act(self.proj(self.emb(x).transpose(1, 2)))
        for b in self.blocks:
            h = b(h)
        return self.head(h, return_attn)

# ----------------------------------------------------------------------------
# (3) OverLoCK-1D  -- top-down attention + context-mixing dynamic conv (ContMix)
#     Original: branched Base/Overview/Focus nets; Overview generates a coarse
#     global context prior that drives a *dynamic* convolution (ContMix).
#     1D adaptation: Overview-Net = global-avg -> context c; each block applies
#     (a) depthwise conv (low/mid-level), (b) ContMix dynamic depthwise conv
#     whose per-sample kernels are generated FROM c, (c) a top-down gate from c.
# ----------------------------------------------------------------------------
class OverLoCKBlock(nn.Module):
    def __init__(self, D, K=5):
        super().__init__()
        self.local = nn.Conv1d(D, D, 3, padding=1, groups=D)
        self.act = nn.GELU()
        self.contmix = nn.Linear(D, D * K)     # dynamic depthwise kernels <- context c
        self.K = K
        self.topdown = nn.Sequential(nn.Linear(D, D), nn.Sigmoid())
        self.merge = nn.Conv1d(D, D, 1)
    def forward(self, x, c):
        h = self.act(self.local(x))                      # base / feature extraction
        C = x.size(1)
        w = self.contmix(c).reshape(x.size(0) * C, 1, self.K)
        xr = h.reshape(1, x.size(0) * C, h.size(2))
        hdyn = F.conv1d(xr, w, groups=x.size(0) * C, padding=self.K // 2).reshape_as(h)
        gate = self.topdown(c).unsqueeze(-1)            # top-down modulation
        out = self.merge(self.act(hdyn * gate))
        return x + out

class OverLoCK1D(nn.Module):
    def __init__(self, vocab_size, D=D_MODEL, n_blocks=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED, padding_idx=0)
        self.proj = nn.Conv1d(EMBED, D, 1)
        self.overview = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                      nn.Linear(D, D), nn.GELU())
        self.blocks = nn.ModuleList([OverLoCKBlock(D) for _ in range(n_blocks)])
        self.head = GlobalPoolHead(D)
        self.act = nn.ReLU()
    def forward(self, x, return_attn=False):
        h = self.act(self.proj(self.emb(x).transpose(1, 2)))   # [B, D, L]
        c = self.overview(h)                                   # [B, D] overview-first
        for b in self.blocks:
            h = b(h, c)                                        # look-closely-next (guided)
        return self.head(h, return_attn)

# ----------------------------------------------------------------------------
# (4) TransXNet-1D -- Dual Dynamic token Mixer (D-Mixer)
#     Original: split channels; one half -> input-dependent depthwise conv,
#     other half -> efficient global attention; recombine.
#     1D adaptation: same split logic on the char sequence.
# ----------------------------------------------------------------------------
class DMixerBlock(nn.Module):
    def __init__(self, D, K=5):
        super().__init__()
        self.half = D // 2
        self.dyn = nn.Linear(self.half, self.half * K)   # dynamic depthwise kernels
        self.K = K
        self.merge = nn.Conv1d(D, D, 1)
        self.act = nn.GELU()
    def forward(self, x):
        x1, x2 = x[:, :self.half], x[:, self.half:]
        # branch A: input-dependent dynamic depthwise conv on x1
        cond = x1.mean(-1)                               # [B, half]
        w = self.dyn(cond).reshape(x1.size(0) * self.half, 1, self.K)
        x1r = x1.reshape(1, x1.size(0) * self.half, x1.size(2))
        x1d = F.conv1d(x1r, w, groups=x1.size(0) * self.half,
                       padding=self.K // 2).reshape_as(x1)
        # branch B: global attention on x2
        g = x2.mean(-1, keepdim=True)                    # [B, half, 1] global token
        scores = torch.einsum('bcl,bcx->blx', x2, g) / (self.half ** 0.5)
        attn = torch.softmax(scores, dim=1)              # [B, L, 1]
        ctx = torch.einsum('blx,bcl->bcx', attn, x2)     # [B, half, 1]
        x2a = x2 + ctx                                  # residual global context
        out = torch.cat([x1d, x2a], dim=1)
        return x + self.act(self.merge(out))

class TransXNet1D(nn.Module):
    def __init__(self, vocab_size, D=D_MODEL, n_blocks=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED, padding_idx=0)
        self.proj = nn.Conv1d(EMBED, D, 1)
        self.blocks = nn.ModuleList([DMixerBlock(D) for _ in range(n_blocks)])
        self.head = GlobalPoolHead(D)
        self.act = nn.ReLU()
    def forward(self, x, return_attn=False):
        h = self.act(self.proj(self.emb(x).transpose(1, 2)))
        for b in self.blocks:
            h = b(h)
        return self.head(h, return_attn)

# ----------------------------------------------------------------------------
# (5) PFGNet-1D -- fully convolutional, frequency-guided peripheral gating
#     Original: large-kernel "peripheral" responses + learnable center
#     suppression + frequency-guided gate (FFT spectral cue).
#     1D adaptation: two large-kernel separable convs (peripheral) minus a
#     learnable-scaled center conv, then a gate derived from rFFT spectral
#     energy (low vs high frequency) of the sequence.
# ----------------------------------------------------------------------------
class PFGGate(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, D), nn.ReLU(), nn.Linear(D, D), nn.Sigmoid())
    def forward(self, x):
        spec = torch.fft.rfft(x.mean(1), dim=-1)         # [B, L/2+1]
        mag = spec.abs()
        lo = mag[:, :10].mean(dim=1)                     # low-freq energy
        hi = mag[:, -10:].mean(dim=1)                    # high-freq energy
        gmean = x.mean(dim=(1, 2))
        gmax = x.amax(dim=(1, 2))
        cue = torch.stack([lo, hi, gmean, gmax], dim=1)  # [B, 4]
        gate = self.net(cue).unsqueeze(-1)               # [B, D, 1]
        return x * gate

class PFGBlock(nn.Module):
    def __init__(self, D, k1=7, k2=11):
        super().__init__()
        self.p1 = nn.Conv1d(D, D, k1, padding=k1 // 2, groups=D)
        self.p2 = nn.Conv1d(D, D, k2, padding=k2 // 2, groups=D)
        self.center = nn.Conv1d(D, D, 1)
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.merge = nn.Conv1d(2 * D, D, 1)
        self.act = nn.GELU()
        self.gate = PFGGate(D)
    def forward(self, x):
        center = self.center(x)                                     # [B, D, L]
        p1 = self.p1(x) - self.alpha * center                       # peripheral - center
        p2 = self.p2(x) - self.alpha * center
        periph = torch.cat([p1, p2], dim=1)                         # [B, 2D, L]
        fused = self.merge(self.act(periph))                        # [B, D, L]
        fused = self.gate(fused)                                    # freq-guided gating
        return x + fused

class PFGNet1D(nn.Module):
    def __init__(self, vocab_size, D=D_MODEL, n_blocks=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED, padding_idx=0)
        self.proj = nn.Conv1d(EMBED, D, 1)
        self.blocks = nn.ModuleList([PFGBlock(D) for _ in range(n_blocks)])
        self.head = GlobalPoolHead(D)
        self.act = nn.ReLU()
    def forward(self, x, return_attn=False):
        h = self.act(self.proj(self.emb(x).transpose(1, 2)))
        for b in self.blocks:
            h = b(h)
        return self.head(h, return_attn)

# ----------------------------------------------------------------------------
# (6) OURS: AttentionFractalCNN (multi-scale local + FractalTower + MHReadout)
# ----------------------------------------------------------------------------
class FractalTower(nn.Module):
    def __init__(self, D, dilations=(1, 2, 4)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(D, D, 3, padding=d, dilation=d, groups=D),
                nn.ReLU(), nn.Conv1d(D, D, 1), nn.GELU()))
    def forward(self, x):
        for blk in self.blocks:
            x = x + blk(x)
        return x

class MHReadout(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.queries = nn.Parameter(torch.randn(n_heads, d_model) * 0.1)
    def forward(self, H):
        B, L, D = H.shape
        scores = torch.einsum('bld,hd->blh', H, self.queries) / (D ** 0.5)
        attn = torch.softmax(scores.transpose(1, 2), dim=-1)
        ctx = torch.einsum('bhl,bld->bhd', attn, H).reshape(B, self.n_heads * D)
        saliency = attn.mean(dim=1)
        return ctx, saliency, attn

class AttentionFractalCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED, padding_idx=0)
        self.local = nn.ModuleList([nn.Conv1d(EMBED, NCHAN, k, padding='same')
                                    for k in LOCAL_K])
        self.proj = nn.Conv1d(len(LOCAL_K) * NCHAN, D_MODEL, 1)
        self.tower = FractalTower(D_MODEL)
        self.readout = MHReadout(D_MODEL, N_HEADS)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.nmi_head = nn.Linear(N_HEADS * D_MODEL, 1)
        self.prof_head = nn.Linear(N_HEADS * D_MODEL, 5)
    def forward(self, x, return_attn=False):
        x = self.emb(x).transpose(1, 2)
        hs = [self.act(c(x)) for c in self.local]
        h = torch.cat(hs, dim=1)
        h = self.act(self.proj(h))
        h = self.tower(h)
        Hseq = h.transpose(1, 2)
        ctx, sal, attn = self.readout(Hseq)
        ctx = self.drop(ctx)
        nmi = self.nmi_head(ctx).squeeze(1)
        prof = self.prof_head(ctx)
        if return_attn:
            return nmi, prof, sal
        return nmi, prof

# ----------------------------------------------------------------------------
# training / evaluation
# ----------------------------------------------------------------------------
def count_params(m):
    return sum(p.numel() for p in m.parameters())

def train_model(model, tr_ld, te_ld, prof_mean, prof_std, multitask, epochs=EPOCHS):
    model.to(device)
    pm = prof_mean.to(device); ps = prof_std.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train(); loss_sum = 0
        for xb, yb, zb in tr_ld:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            opt.zero_grad()
            out = model(xb)
            if multitask:
                nmi_p, prof_p = out
                loss = crit(nmi_p, yb) + LAMBDA_PROF * crit(prof_p, (zb - pm) / ps)
            else:
                loss = crit(out, yb)
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(yb)
        if ep % 3 == 0 or ep == epochs - 1:
            m = _eval(model, te_ld, prof_mean, prof_std, multitask)
            print(f"      ep {ep:02d} loss={loss_sum/len(tr_ld.dataset):.3f} "
                  f"r={m['r']:.4f} MAE={m['MAE']:.3f}")
    return _eval(model, te_ld, prof_mean, prof_std, multitask)

def _eval(model, te_ld, prof_mean, prof_std, multitask):
    model.eval(); preds, gts, pp, pg = [], [], [], []
    with torch.no_grad():
        for xb, yb, zb in te_ld:
            out = model(xb.to(device))
            nmi_p = (out[0] if isinstance(out, tuple) else out)
            preds.extend(nmi_p.cpu().numpy()); gts.extend(yb.numpy())
            if multitask:
                pp.extend(out[1].cpu().numpy()); pg.extend(zb.numpy())
    preds = np.array(preds); gts = np.array(gts)
    res = {"r": _pearson(preds, gts), "rho": _spearman(preds, gts),
           "MAE": float(np.mean(np.abs(preds - gts))),
           "RMSE": float(math.sqrt(np.mean((preds - gts) ** 2))),
           "n_test": int(len(gts))}
    if multitask:
        pp = np.array(pp) * prof_std.numpy() + prof_mean.numpy()
        pg = np.array(pg)
        res["profile_pearson_r"] = [_pearson(pp[:, j], pg[:, j]) for j in range(5)]
    return res

# ----------------------------------------------------------------------------
def main():
    print(f"[*] device = {device}")
    pool = json.load(open(os.path.join(HERE, "dl_pool.json"), encoding="utf-8"))
    texts = [p["text"] for p in pool]
    nmi = np.array([p["nmi"] for p in pool], float)
    prof = np.array([p["profile"] for p in pool], float)

    mu, sd = nmi.mean(), nmi.std()
    keep = (nmi >= mu - 3 * sd) & (nmi <= mu + 3 * sd)
    texts = [texts[i] for i in range(len(texts)) if keep[i]]
    nmi = nmi[keep]; prof = prof[keep]
    print(f"[*] after 3-sigma drop: pool={len(texts)} (ledger baseline 20847)")

    vocab = build_vocab(texts)
    X = [encode(t, vocab) for t in texts]
    y = nmi.tolist(); z = prof.tolist()

    n = len(y); idx = list(range(n)); random.shuffle(idx)
    cut = int(n * 0.85)
    tr, te = idx[:cut], idx[cut:]
    tr_ds = PoemDS([X[i] for i in tr], [y[i] for i in tr], [z[i] for i in tr])
    te_ds = PoemDS([X[i] for i in te], [y[i] for i in te], [z[i] for i in te])
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)
    print(f"[*] train={len(tr)} test={len(te)} vocab={len(vocab)}")

    prof_mean = torch.tensor(np.array([z[i] for i in tr]).mean(0), dtype=torch.float32)
    prof_std = torch.tensor(np.array([z[i] for i in tr]).std(0) + 1e-6, dtype=torch.float32)

    V = len(vocab)
    models = {
        "charcnn":          (CharCNN(V),          False),
        "labelmatched_cnn": (LabelMatchedCNN(V),  True),
        "wideplain_cnn":    (WidePlainCNN(V),     True),
        "overlock_1d":      (OverLoCK1D(V),       True),
        "transxnet_1d":     (TransXNet1D(V),      True),
        "pfgn_1d":          (PFGNet1D(V),         True),
        "attnfractal_cnn":  (AttentionFractalCNN(V), True),
    }

    results = {"pool_size": len(texts), "vocab_size": len(vocab),
               "n_train": len(tr), "n_test": len(te), "device": device,
               "epochs": EPOCHS, "budget": "14ep / identical split & pool",
               "models": {}}

    for name, (model, multitask) in models.items():
        print(f"\n=== {name} ({'multi-task' if multitask else 'NMI-only'}) ===")
        nparams = count_params(model)
        m = train_model(model, tr_ld, te_ld, prof_mean, prof_std, multitask, EPOCHS)
        entry = {"params": int(nparams), "multitask": multitask,
                 "r": m["r"], "rho": m["rho"], "MAE": m["MAE"], "RMSE": m["RMSE"],
                 "n_test": m["n_test"]}
        if multitask:
            entry["profile_pearson_r"] = m["profile_pearson_r"]
        results["models"][name] = entry
        print(f"    -> params={nparams:,}  r={m['r']:.4f} rho={m['rho']:.4f} "
              f"MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f}"
              + (f"  prof_r={[round(r,3) for r in m['profile_pearson_r']]}"
                 if multitask else ""))

    # references from prior run (ceiling check, not retrained here)
    results["references"] = {
        "attnfractal_cnn_30ep_prior": {"r": 0.9883, "MAE": 1.11,
            "note": "prior run dl_attn_fractal.py, 30-epoch ceiling (14ep already ~0.988)"},
    }
    results["citations"] = {
        "overlock_1d": "OverLoCK (CVPR 2025), Lou & Yu, arXiv:2502.20087 — pure ConvNet, top-down attention + context-mixing dynamic kernels (ContMix).",
        "transxnet_1d": "TransXNet (TNNLS 2025), Lou et al., arXiv:2310.19380 / DOI 10.1109/TNNLS.2025.3550979 — CNN-Transformer hybrid, Dual Dynamic token Mixer (D-Mixer).",
        "pfgn_1d": "PFGNet (CVPR 2026), Cai et al., arXiv:2602.20537 — fully convolutional, frequency-guided peripheral gating.",
        "charcnn": "Kim (2014) char-CNN baseline, reproduced from dl_model.py / analyze_v4.py.",
        "attnfractal_cnn": "Proposed: multi-scale local conv + FractalTower (self-similar dilated residual) + multi-head readout attention; NMI + 5-dim profile heads.",
    }

    json.dump(results, open(os.path.join(HERE, "results_cnn2025_bench.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[*] wrote results_cnn2025_bench.json")

if __name__ == "__main__":
    main()
