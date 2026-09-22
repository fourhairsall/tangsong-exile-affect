# -*- coding: utf-8 -*-
"""Head-to-head ablation: baseline char-CNN vs Attention+Fractal char-CNN.

Exactly reproduces the baseline setup in dl_model.py / analyze_v4.py:
  * pool = dl_pool.json (21,287 诗, t2s simplified, NMI + 5-dim profile labels)
  * 3-sigma outlier drop on NMI  -> ~20,847 (matches ledger dl_reg.pool_size)
  * char vocabulary (max 8000), MAX_LEN=160, 85/15 split (shared, fixed seed)
  * baseline = Kim-style char-CNN regressing NMI (r=0.929 reported)

New model = AttentionFractalCNN:
  * multi-scale local conv branches (k=2,3,4,5)
  * FRACTAL dilated residual tower: same residual block repeated at exponentially
    increasing dilation (1,2,4) -> self-similar multi-scale aggregation
    (char -> phrase -> line -> couplet hierarchy)
  * MULTI-HEAD READOUT ATTENTION over the character sequence -> per-character
    saliency (interpretability) + pooled context
  * two heads: NMI scalar (head-to-head target) + 5-dim lexicon profile (aux)

Reports, on the IDENTICAL test split:
  * NMI: Pearson r / Spearman rho / MAE / RMSE  (baseline vs new, 14 & 30 ep)
  * profile recovery: per-category Pearson r (auxiliary, interpretability)
  * attention interpretability: lexicon-overlap of top-attended characters
Outputs results_dl_attnfractal.json + a readable report.
"""
import os, json, re, random, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260915
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MAX_LEN = 160
BATCH = 128
EPOCHS = 14
EPOCHS_LONG = 30
EMBED = 64
KERNELS = [3, 4, 5]
NCHAN = 64
HIDDEN = 128
LOCAL_K = [3, 5, 7]         # multi-scale local branches (new model, odd kernels)
D_MODEL = 128
N_HEADS = 4
LAMBDA_PROF = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
# data plumbing (identical for both models)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# BASELINE (faithful copy of dl_model.CharCNN)
# ---------------------------------------------------------------------------
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
        x = self.emb(x)
        x = x.transpose(1, 2)
        hs = [self.act(conv(x)).max(dim=2)[0] for conv in self.convs]
        h = torch.cat(hs, dim=1)
        h = self.drop(h)
        h = self.act(self.fc1(h))
        return self.fc2(h).squeeze(1)

# ---------------------------------------------------------------------------
# NEW: AttentionFractalCNN
# ---------------------------------------------------------------------------
class FractalTower(nn.Module):
    """Self-similar dilated residual tower. The SAME residual block is repeated
    at exponentially increasing dilation (1,2,4): the receptive field grows
    geometrically (char -> 2-gram -> 4-gram -> 8-gram), a fractal-like
    multi-scale aggregation that mirrors the poetic hierarchy
    (character -> phrase -> line -> couplet)."""
    def __init__(self, D, dilations=(1, 2, 4)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(D, D, 3, padding=d, dilation=d, groups=D),
                nn.ReLU(),
                nn.Conv1d(D, D, 1),
                nn.GELU(),
            ))
    def forward(self, x):
        for blk in self.blocks:
            x = x + blk(x)          # residual connection at each scale
        return x

class MHReadout(nn.Module):
    """Multi-head readout attention. Each head has a learnable query; its
    softmax over character positions gives a per-character saliency. The
    attention-weighted contexts are concatenated as the sentence representation.
    `saliency` (mean over heads) is the interpretability signal."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.queries = nn.Parameter(torch.randn(n_heads, d_model) * 0.1)
    def forward(self, H):
        B, L, D = H.shape
        scores = torch.einsum('bld,hd->blh', H, self.queries) / (D ** 0.5)
        attn = torch.softmax(scores.transpose(1, 2), dim=-1)        # [B,H,L]
        ctx = torch.einsum('bhl,bld->bhd', attn, H)                # [B,H,D]
        ctx = ctx.reshape(B, self.n_heads * D)
        saliency = attn.mean(dim=1)                                # [B,L]
        return ctx, saliency, attn

class CharCNNMultiTask(nn.Module):
    """LABEL-MATCHED control for the new model: identical conv encoder to the
    baseline char-CNN (KERNELS=[3,4,5], max-over-time pool, HIDDEN fc) but with
    the SAME two heads (NMI + 5-dim profile) trained on the SAME loss as the new
    model. No attention, no fractal tower. Isolates the effect of the richer
    (multi-task) supervision from the attention/fractal architectural change."""
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(EMBED, NCHAN, k) for k in KERNELS])
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(KERNELS) * NCHAN, HIDDEN)
        self.act = nn.ReLU()
        self.nmi_head = nn.Linear(HIDDEN, 1)
        self.prof_head = nn.Linear(HIDDEN, 5)
    def forward(self, x, return_attn=False):
        x = self.emb(x).transpose(1, 2)
        hs = [self.act(c(x)).max(dim=2)[0] for c in self.convs]
        h = torch.cat(hs, dim=1)
        h = self.drop(h)
        h = self.act(self.fc1(h))
        if return_attn:
            return self.nmi_head(h).squeeze(1), self.prof_head(h), None
        return self.nmi_head(h).squeeze(1), self.prof_head(h)

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
        x = self.emb(x).transpose(1, 2)                  # [B,E,L]
        hs = [self.act(c(x)) for c in self.local]         # [B,NCHAN,L] x K
        h = torch.cat(hs, dim=1)
        h = self.act(self.proj(h))                        # [B,D,L]
        h = self.tower(h)                                # [B,D,L]
        Hseq = h.transpose(1, 2)                         # [B,L,D]
        ctx, sal, attn = self.readout(Hseq)              # [B,H*D], [B,L], [B,H,L]
        ctx = self.drop(ctx)
        nmi = self.nmi_head(ctx).squeeze(1)
        prof = self.prof_head(ctx)
        if return_attn:
            return nmi, prof, sal
        return nmi, prof

# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------
def train_baseline(model, tr_ld, te_ld, epochs=EPOCHS):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train(); loss_sum = 0
        for xb, yb, _ in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step(); loss_sum += loss.item() * len(yb)
        if ep % 3 == 0 or ep == epochs - 1:
            m = _eval_nmi(model, te_ld)
            print(f"    [base] ep {ep:02d} train_loss={loss_sum/len(tr_ld.dataset):.3f} "
                  f"test_r={m['r']:.4f} test_MAE={m['MAE']:.3f}")
    return _eval_nmi(model, te_ld)

def train_new(model, tr_ld, te_ld, prof_mean, prof_std, epochs=EPOCHS):
    model.to(device)
    pm = prof_mean.to(device); ps = prof_std.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train(); loss_sum = 0
        for xb, yb, zb in tr_ld:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            nmi_p, prof_p = model(xb)
            loss = crit(nmi_p, yb) + LAMBDA_PROF * crit(prof_p, (zb - pm) / ps)
            opt.zero_grad(); loss.backward(); opt.step(); loss_sum += loss.item() * len(yb)
        if ep % 3 == 0 or ep == epochs - 1:
            m = _eval_new(model, te_ld, prof_mean, prof_std)
            print(f"    [new ] ep {ep:02d} train_loss={loss_sum/len(tr_ld.dataset):.3f} "
                  f"test_r={m['r']:.4f} test_MAE={m['MAE']:.3f}")
    return _eval_new(model, te_ld, prof_mean, prof_std), model

def _eval_nmi(model, te_ld):
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for xb, yb, _ in te_ld:
            out = model(xb.to(device))
            p = (out[0] if isinstance(out, tuple) else out).cpu().numpy()
            preds.extend(p); gts.extend(yb.numpy())
    preds = np.array(preds); gts = np.array(gts)
    return {"r": _pearson(preds, gts), "rho": _spearman(preds, gts),
            "MAE": float(np.mean(np.abs(preds - gts))),
            "RMSE": float(math.sqrt(np.mean((preds - gts) ** 2))),
            "n_test": int(len(gts))}

def _eval_new(model, te_ld, prof_mean, prof_std):
    model.eval(); preds, gts, pp, pg = [], [], [], []
    with torch.no_grad():
        for xb, yb, zb in te_ld:
            nmi_p, prof_p = model(xb.to(device))
            preds.extend(nmi_p.cpu().numpy()); gts.extend(yb.numpy())
            pp.extend(prof_p.cpu().numpy()); pg.extend(zb.numpy())
    preds = np.array(preds); gts = np.array(gts)
    pp = np.array(pp) * prof_std.numpy() + prof_mean.numpy()   # inverse-standardize
    pg = np.array(pg)
    prof_r = [_pearson(pp[:, j], pg[:, j]) for j in range(5)]
    base = _eval_nmi(model, te_ld)
    base.update({"profile_pearson_r": prof_r})
    return base

# ---------------------------------------------------------------------------
def main():
    print(f"[*] device = {device}")
    pool = json.load(open(os.path.join(HERE, "dl_pool.json"), encoding="utf-8"))
    texts = [p["text"] for p in pool]
    nmi = np.array([p["nmi"] for p in pool], float)
    prof = np.array([p["profile"] for p in pool], float)

    # 3-sigma outlier drop (identical to dl_model.run_dl)
    mu, sd = nmi.mean(), nmi.std()
    keep = (nmi >= mu - 3 * sd) & (nmi <= mu + 3 * sd)
    texts = [texts[i] for i in range(len(texts)) if keep[i]]
    nmi = nmi[keep]; prof = prof[keep]
    print(f"[*] after 3-sigma drop: pool={len(texts)} (ledger baseline pool_size=20847)")

    vocab = build_vocab(texts)
    X = [encode(t, vocab) for t in texts]
    y = nmi.tolist()
    z = prof.tolist()

    # FIXED shared 85/15 split
    n = len(y); idx = list(range(n)); random.shuffle(idx)
    cut = int(n * 0.85)
    tr, te = idx[:cut], idx[cut:]
    tr_ds = PoemDS([X[i] for i in tr], [y[i] for i in tr], [z[i] for i in tr])
    te_ds = PoemDS([X[i] for i in te], [y[i] for i in te], [z[i] for i in te])
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)
    print(f"[*] train={len(tr)} test={len(te)} vocab={len(vocab)}")

    # profile standardization stats from TRAIN only
    prof_mean = torch.tensor(np.array([z[i] for i in tr]).mean(0), dtype=torch.float32)
    prof_std = torch.tensor(np.array([z[i] for i in tr]).std(0) + 1e-6, dtype=torch.float32)

    results = {"pool_size": len(texts), "vocab_size": len(vocab),
               "n_train": len(tr), "n_test": len(te), "device": device}

    # ---- baseline ----
    print("\n=== BASELINE char-CNN (14 ep) ===")
    base = CharCNN(len(vocab))
    m_base = train_baseline(base, tr_ld, te_ld, EPOCHS)
    results["baseline_14ep"] = m_base
    print(f"    baseline: r={m_base['r']:.4f} rho={m_base['rho']:.4f} "
          f"MAE={m_base['MAE']:.3f} RMSE={m_base['RMSE']:.3f}")

    # ---- label-matched control: plain char-CNN, SAME NMI+profile targets ----
    print("\n=== LABEL-MATCHED control: plain CNN (NMI+profile, 14 ep) ===")
    mt = CharCNNMultiTask(len(vocab))
    m_mt, _ = train_new(mt, tr_ld, te_ld, prof_mean, prof_std, EPOCHS)
    results["label_matched_cnn_14ep"] = {k: m_mt[k] for k in ("r", "rho", "MAE", "RMSE", "n_test")}
    results["label_matched_cnn_14ep"]["profile_pearson_r"] = m_mt["profile_pearson_r"]
    print(f"    label-matched: r={m_mt['r']:.4f} rho={m_mt['rho']:.4f} "
          f"MAE={m_mt['MAE']:.3f} RMSE={m_mt['RMSE']:.3f} "
          f"profile_r={[round(r,3) for r in m_mt['profile_pearson_r']]}")

    # ---- new model, 14 ep (equal compute) ----
    print("\n=== NEW AttentionFractalCNN (14 ep) ===")
    newm = AttentionFractalCNN(len(vocab))
    m_new14, newm = train_new(newm, tr_ld, te_ld, prof_mean, prof_std, EPOCHS)
    results["new_14ep"] = {k: m_new14[k] for k in ("r", "rho", "MAE", "RMSE", "n_test")}
    results["new_14ep"]["profile_pearson_r"] = m_new14["profile_pearson_r"]
    print(f"    new(14): r={m_new14['r']:.4f} rho={m_new14['rho']:.4f} "
          f"MAE={m_new14['MAE']:.3f} RMSE={m_new14['RMSE']:.3f} "
          f"profile_r={[round(r,3) for r in m_new14['profile_pearson_r']]}")

    # ---- new model, 30 ep (ceiling check) ----
    print("\n=== NEW AttentionFractalCNN (30 ep, ceiling) ===")
    newm2 = AttentionFractalCNN(len(vocab))
    m_new30, newm2 = train_new(newm2, tr_ld, te_ld, prof_mean, prof_std, EPOCHS_LONG)
    results["new_30ep"] = {k: m_new30[k] for k in ("r", "rho", "MAE", "RMSE", "n_test")}
    results["new_30ep"]["profile_pearson_r"] = m_new30["profile_pearson_r"]
    print(f"    new(30): r={m_new30['r']:.4f} rho={m_new30['rho']:.4f} "
          f"MAE={m_new30['MAE']:.3f} RMSE={m_new30['RMSE']:.3f}")

    # ---- attention interpretability (lexicon overlap of top-attended chars) ----
    print("\n=== attention interpretability ===")
    vocab_inv = {i: c for i, c in enumerate(vocab)}
    import lexicon as L
    lex_charset = set()
    for cat, ws in L.LEX.items():
        for w in ws:
            if len(w) == 1:
                lex_charset.add(w)
    # pool saliencies over the test set (real chars only, excl. pad)
    newm2.eval()
    pooled_chars, pooled_sal = [], []
    sample_vis = []
    with torch.no_grad():
        for xb, yb, zb in te_ld:
            nmi_p, prof_p, sal = newm2(xb.to(device), return_attn=True)
            xb_np = xb.numpy()
            sal_np = sal.cpu().numpy()
            for b in range(xb_np.shape[0]):
                ids = xb_np[b]
                s = sal_np[b]
                chars = [vocab_inv.get(int(i), "") for i in ids if int(i) != 0]
                sv = s[:len(chars)]
                if len(chars) == 0:
                    continue
                pooled_chars.extend(chars); pooled_sal.extend(sv.tolist())
                if len(sample_vis) < 6 and 20 <= len(chars) <= 80:
                    order = np.argsort(-sv)[:8]
                    hi = [(chars[k], round(float(sv[k]), 3)) for k in order]
                    sample_vis.append({"text": "".join(chars), "top_attended": hi,
                                        "true_nmi": float(yb[b].numpy())})
    pooled_chars = np.array(pooled_chars)
    pooled_sal = np.array(pooled_sal)
    base_rate = np.mean([c in lex_charset for c in pooled_chars])
    thr = np.quantile(pooled_sal, 0.90)
    top_mask = pooled_sal >= thr
    top_rate = np.mean([pooled_chars[i] in lex_charset for i in np.where(top_mask)[0]])
    results["attention_interpretability"] = {
        "lexicon_charset_size": len(lex_charset),
        "base_rate_lexicon_chars": float(base_rate),
        "top10pct_attended_lexicon_rate": float(top_rate),
        "lift": float(top_rate / base_rate) if base_rate > 0 else None,
        "samples": sample_vis,
    }
    print(f"    base rate lexicon chars = {base_rate:.4f}")
    print(f"    top-10% attended lexicon rate = {top_rate:.4f}  (lift {top_rate/base_rate:.2f}x)")
    for s in sample_vis[:3]:
        print(f"    NMI={s['true_nmi']:+.2f}  top: {s['top_attended']}")

    json.dump(results, open(os.path.join(HERE, "results_dl_attnfractal.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[*] wrote results_dl_attnfractal.json")

if __name__ == "__main__":
    main()
