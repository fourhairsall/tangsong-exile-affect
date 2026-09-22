# -*- coding: utf-8 -*-
"""Deep-learning psychological signal extraction for classical Chinese poetry.
A character-level CNN is trained with weak supervision: the lexicon-derived
净心指数 (旷达 − 悲苦, per 1000 chars) of each poem serves as the regression
target. The model learns a contextual (non-linear, order-sensitive) mapping that
complements the additive dictionary method. Reproducible: fixed seed, CPU.

Usage: run_dl(pool_texts) -> trained pipeline; predict(model, texts) -> list[float]
"""
import re, random, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 20260915
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MAX_LEN = 160
BATCH = 128
EPOCHS = 14
EMBED = 64
KERNELS = [3, 4, 5]
NCHAN = 64
HIDDEN = 128

def clean_text(s):
    s = re.sub(r"\s+", "", s)
    return s

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
    def __init__(self, x, y):
        self.x = x; self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.x[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.float32)

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
        x = self.emb(x)                      # [B, L, E]
        x = x.transpose(1, 2)                # [B, E, L]
        hs = [self.act(conv(x)).max(dim=2)[0] for conv in self.convs]  # [B, NCHAN]
        h = torch.cat(hs, dim=1)
        h = self.drop(h)
        h = self.act(self.fc1(h))
        return self.fc2(h).squeeze(1)

def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return _pearson(ra, rb)

def run_dl(pool_texts, label_fn, device="cpu", verbose=True):
    """pool_texts: list[str]; label_fn: str->float (weak label). Returns dict."""
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)  # explicit reseed
    import sys
    cleaned = [clean_text(t) for t in pool_texts if clean_text(t)]
    labels = np.array([label_fn(t) for t in cleaned], float)
    # drop extreme outliers beyond 3 std to stabilize training
    mu, sd = labels.mean(), labels.std()
    keep = (labels >= mu - 3 * sd) & (labels <= mu + 3 * sd)
    cleaned = [cleaned[i] for i in range(len(cleaned)) if keep[i]]
    labels = labels[keep]
    if verbose: print(f"[DL] pool size after cleaning: {len(cleaned)}")
    vocab = build_vocab(cleaned)
    X = [encode(t, vocab) for t in cleaned]
    y = labels.tolist()
    # train/test split 85/15
    n = len(y); idx = list(range(n)); random.shuffle(idx)
    cut = int(n * 0.85)
    tr, te = idx[:cut], idx[cut:]
    train_ds = PoemDS([X[i] for i in tr], [y[i] for i in tr])
    test_ds = PoemDS([X[i] for i in te], [y[i] for i in te])
    tr_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
    model = CharCNN(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for ep in range(EPOCHS):
        model.train(); loss_sum = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb); loss.backward(); opt.step()
            loss_sum += loss.item() * len(yb)
        # eval
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(device); p = model(xb).cpu().numpy(); preds.extend(p); gts.extend(yb.numpy())
        mae = float(np.mean(np.abs(np.array(preds) - np.array(gts))))
        if verbose and (ep % 3 == 0 or ep == EPOCHS - 1):
            print(f"[DL] epoch {ep:02d} train_loss={loss_sum/len(tr):.3f} test_MAE={mae:.3f}")
    # final metrics
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            p = model(xb.to(device)).cpu().numpy(); preds.extend(p); gts.extend(yb.numpy())
    preds = np.array(preds); gts = np.array(gts)
    metrics = {
        "pool_size": len(cleaned),
        "vocab_size": len(vocab),
        "test_MAE": float(np.mean(np.abs(preds - gts))),
        "test_RMSE": float(math.sqrt(np.mean((preds - gts) ** 2))),
        "test_Pearson_r": _pearson(preds, gts),
        "test_Spearman_rho": _spearman(preds, gts),
    }
    if verbose: print("[DL] metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    return {"model": model, "vocab": vocab, "metrics": metrics, "device": device}

def predict(dl, texts):
    """dl: returned dict from run_dl. Returns list[float] per text."""
    model = dl["model"].to(dl["device"]); vocab = dl["vocab"]
    out = []
    for t in texts:
        t = clean_text(t)
        if not t:
            out.append(0.0); continue
        x = torch.tensor([encode(t, vocab)], dtype=torch.long).to(dl["device"])
        with torch.no_grad():
            out.append(float(model(x).cpu().numpy()[0]))
    return out

def predict_batch(dl, texts, batch=BATCH):
    model = dl["model"].to(dl["device"]); vocab = dl["vocab"]
    model.eval(); out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        xs = [encode(clean_text(t), vocab) for t in chunk]
        xb = torch.tensor(xs, dtype=torch.long).to(dl["device"])
        with torch.no_grad():
            p = model(xb).cpu().numpy()
        out.extend(p.tolist())
    return out

# ---------------------------------------------------------------------------
# Extension: discriminative attribution (author / dynasty) + feature extraction
# ---------------------------------------------------------------------------
from collections import defaultdict

class ClsDS(Dataset):
    def __init__(self, x, y):
        self.x = x; self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.x[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.long)

class CharCNNClf(nn.Module):
    """Same encoder as CharCNN; classification head instead of regression head.
    Exposes encode_features() for penultimate-layer (discriminative) features."""
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(EMBED, NCHAN, k) for k in KERNELS])
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(KERNELS) * NCHAN, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, n_classes)
        self.act = nn.ReLU()
    def encode_features(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        hs = [self.act(conv(x)).max(dim=2)[0] for conv in self.convs]
        h = torch.cat(hs, dim=1)
        h = self.drop(h)
        return self.act(self.fc1(h))
    def forward(self, x):
        return self.fc2(self.encode_features(x))

def stratified_split(X, y, test_size=0.15, seed=SEED):
    idx_by_class = defaultdict(list)
    for i, yi in enumerate(y):
        idx_by_class[yi].append(i)
    tr, te = [], []
    random.seed(seed)
    for c, idxs in idx_by_class.items():
        random.shuffle(idxs)
        k = max(1, int(round(len(idxs) * test_size)))
        te += idxs[:k]; tr += idxs[k:]
    random.shuffle(tr); random.shuffle(te)
    return tr, te

def run_classification(texts, labels, n_classes, test_size=0.15, device="cpu", verbose=True):
    """texts: list[str]; labels: list[int] class idx. Returns metrics dict + model + vocab."""
    pairs = [(clean_text(t), y) for t, y in zip(texts, labels) if clean_text(t)]
    texts = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    if verbose: print(f"[DL-cls] samples={len(texts)} classes={n_classes}")
    vocab = build_vocab(texts)
    X = [encode(t, vocab) for t in texts]
    tr, te = stratified_split(X, ys, test_size)
    tr_ld = DataLoader(ClsDS([X[i] for i in tr], [ys[i] for i in tr]), batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(ClsDS([X[i] for i in te], [ys[i] for i in te]), batch_size=BATCH, shuffle=False)
    model = CharCNNClf(len(vocab), n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train(); loss_sum = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb); loss.backward(); opt.step()
            loss_sum += loss.item() * len(yb)
        if verbose and (ep % 3 == 0 or ep == EPOCHS - 1):
            model.eval(); preds, gts = [], []
            with torch.no_grad():
                for xb, yb in te_ld:
                    preds.extend(model(xb.to(device)).argmax(1).cpu().numpy().tolist())
                    gts.extend(yb.numpy().tolist())
            from sklearn.metrics import accuracy_score, f1_score as _f1
            acc = accuracy_score(gts, preds)
            f1 = _f1(gts, preds, average="macro", zero_division=0)
            print(f"[DL-cls] epoch {ep:02d} loss={loss_sum/len(tr):.3f} acc={acc:.3f} macroF1={f1:.3f}")
    # final eval
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            preds.extend(model(xb.to(device)).argmax(1).cpu().numpy().tolist())
            gts.extend(yb.numpy().tolist())
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    acc = accuracy_score(gts, preds)
    macro_f1 = f1_score(gts, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(gts, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(gts, preds, labels=list(range(n_classes))).tolist()
    per_class_f1 = f1_score(gts, preds, average=None, zero_division=0).tolist()
    return {"model": model, "vocab": vocab, "device": device,
            "metrics": {"accuracy": float(acc), "macro_F1": float(macro_f1),
                         "weighted_F1": float(weighted_f1),
                         "confusion_matrix": cm, "per_class_F1": per_class_f1},
            "test_idx": te, "test_preds": preds, "test_gts": gts}

def extract_features(model, texts, vocab, device="cpu", batch=BATCH):
    """Penultimate-layer (discriminative) embeddings for each text."""
    model.eval(); feats = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        xs = [encode(clean_text(t), vocab) for t in chunk]
        xb = torch.tensor(xs, dtype=torch.long).to(device)
        with torch.no_grad():
            f = model.encode_features(xb).cpu().numpy()
        feats.extend(f)
    return np.array(feats, dtype=float)

def pca_2d(features, seed=SEED):
    from sklearn.decomposition import PCA
    p = PCA(n_components=2, random_state=seed)
    return p.fit_transform(np.asarray(features, float))

def run_attribution(pool_texts, label_list, class_names, test_size=0.15, device="cpu", verbose=True):
    """End-to-end attribution: train classifier, extract penultimate features, PCA.
    Returns dict with metrics, features (raw), pca coords, and texts/labels for viz."""
    pairs = [(clean_text(t), y) for t, y in zip(pool_texts, label_list) if clean_text(t)]
    texts = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    n_classes = len(class_names)
    clf = run_classification(texts, ys, n_classes, test_size, device, verbose)
    feats = extract_features(clf["model"], texts, clf["vocab"], device)
    coords = pca_2d(feats)
    return {
        "class_names": class_names,
        "metrics": clf["metrics"],
        "features": feats,
        "pca": coords,
        "texts": texts,
        "labels": ys,
        "model": clf["model"], "vocab": clf["vocab"], "device": device,
    }

