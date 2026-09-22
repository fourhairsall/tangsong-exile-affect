# -*- coding: utf-8 -*-
"""Multi-seed robustness (Turn 4 follow-up): re-run the key CNNs across 5 seeds
so the paper reports MEAN +/- STD, not a single lucky run.

Reuses the model classes & training loop from dl_cnn2025_bench.py (imported, not
re-implemented). For each seed we re-shuffle the 85/15 split and re-initialise
the models, then record test Pearson r and MAE.

Outputs results_cnn2025_multiseed.json with per-seed rows + mean/std aggregate.
"""
import os, json, random
import numpy as np
import torch

from dl_cnn2025_bench import (
    build_vocab, encode, PoemDS, train_model,
    WidePlainCNN, OverLoCK1D, TransXNet1D, PFGNet1D, AttentionFractalCNN,
    EPOCHS, MAX_LEN, BATCH,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [20260915, 20260916, 20260917, 20260918, 20260919]
MODELS = {
    "wideplain_cnn":  WidePlainCNN,
    "overlock_1d":    OverLoCK1D,
    "transxnet_1d":   TransXNet1D,
    "pfgn_1d":        PFGNet1D,
    "attnfractal_cnn": AttentionFractalCNN,
}

def run_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pool = json.load(open(os.path.join(HERE, "dl_pool.json"), encoding="utf-8"))
    texts = [p["text"] for p in pool]
    nmi = np.array([p["nmi"] for p in pool], float)
    prof = np.array([p["profile"] for p in pool], float)
    mu, sd = nmi.mean(), nmi.std()
    keep = (nmi >= mu - 3 * sd) & (nmi <= mu + 3 * sd)
    texts = [texts[i] for i in range(len(texts)) if keep[i]]
    nmi = nmi[keep]; prof = prof[keep]
    vocab = build_vocab(texts)
    X = [encode(t, vocab) for t in texts]
    y = nmi.tolist(); z = prof.tolist()
    n = len(y); idx = list(range(n)); random.shuffle(idx)
    cut = int(n * 0.85)
    tr, te = idx[:cut], idx[cut:]
    tr_ds = PoemDS([X[i] for i in tr], [y[i] for i in tr], [z[i] for i in tr])
    te_ds = PoemDS([X[i] for i in te], [y[i] for i in te], [z[i] for i in te])
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_ld = torch.utils.data.DataLoader(te_ds, batch_size=BATCH, shuffle=False)
    prof_mean = torch.tensor(np.array([z[i] for i in tr]).mean(0), dtype=torch.float32)
    prof_std = torch.tensor(np.array([z[i] for i in tr]).std(0) + 1e-6, dtype=torch.float32)
    V = len(vocab)
    rows = {}
    for name, cls in MODELS.items():
        model = cls(V)
        res = train_model(model, tr_ld, te_ld, prof_mean, prof_std, True, EPOCHS)
        rows[name] = {"r": res["r"], "rho": res["rho"], "MAE": res["MAE"]}
        print(f"  [{seed}] {name}: r={res['r']:.4f} MAE={res['MAE']:.3f}")
    return rows

def main():
    all_rows = {}
    for seed in SEEDS:
        print(f"\n##### SEED {seed} #####")
        all_rows[seed] = run_seed(seed)
    # aggregate mean +/- std per model
    agg = {}
    for name in MODELS:
        r = [all_rows[s][name]["r"] for s in SEEDS]
        m = [all_rows[s][name]["MAE"] for s in SEEDS]
        agg[name] = {
            "r_mean": float(np.mean(r)), "r_std": float(np.std(r)),
            "MAE_mean": float(np.mean(m)), "MAE_std": float(np.std(m)),
            "r_per_seed": {str(s): all_rows[s][name]["r"] for s in SEEDS},
            "MAE_per_seed": {str(s): all_rows[s][name]["MAE"] for s in SEEDS},
        }
    out = {"seeds": SEEDS, "n_seeds": len(SEEDS),
           "epochs": EPOCHS, "per_seed": all_rows, "aggregate": agg}
    json.dump(out, open(os.path.join(HERE, "results_cnn2025_multiseed.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n===== AGGREGATE (mean +/- std over %d seeds) =====" % len(SEEDS))
    for name in MODELS:
        a = agg[name]
        print(f"  {name:18s} r = {a['r_mean']:.4f} +/- {a['r_std']:.4f} | "
              f"MAE = {a['MAE_mean']:.3f} +/- {a['MAE_std']:.3f}")
    print("\n[*] wrote results_cnn2025_multiseed.json")

if __name__ == "__main__":
    main()
