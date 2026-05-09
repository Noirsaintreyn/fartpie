"""
run.py — End-to-end pipeline
============================
Usage:
    python run.py                          # train on synthetic data + full eval
    python run.py --csv path/to/data.csv   # train on real timestamps
    python run.py --ablation               # LSTM-only vs attention vs combined
    python run.py --seed 123               # reproducible run

Reproducibility: random seeds are explicit throughout.
"""

import argparse
import random
import math
import numpy as np
import torch
from pathlib import Path

from nhp_model import NeuralHawkesProcess, NHPConfig
from nhp_data import simulate_nhp_dataset, make_loaders, load_timestamps_csv
from nhp_train import train, evaluate
from nhp_policy import RegimeAwarePolicy, PolicyConfig, evaluate_signals, Signal


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def get_intensity_series(
    model: NeuralHawkesProcess,
    dts: list[float],
    device: torch.device,
    dt_grid: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single authoritative forward pass → intensity on a time grid.
    This is the same computation path used during training.

    Returns: (times, lambda_t) arrays
    """
    model.eval()
    if not dts:
        return np.array([]), np.array([])

    # Build padded single-sequence batch
    dts_t = torch.tensor(dts, dtype=torch.float32).unsqueeze(0).to(device)
    types_t = torch.ones(1, len(dts), dtype=torch.long).to(device)
    lengths_t = torch.tensor([len(dts)], dtype=torch.long).to(device)

    # Forward sequence (single authoritative pass)
    hiddens, cells = model.forward_sequence(types_t, dts_t)

    # Build cumulative times
    cum_times = np.concatenate([[0.0], np.cumsum(dts)])
    T = cum_times[-1]

    grid_times = np.arange(0, T + dt_grid, dt_grid)
    lam_grid = []

    for t in grid_times:
        # Find last event before t (causal)
        past_idx = int(np.searchsorted(cum_times, t, side='right')) - 1
        past_idx = max(0, min(past_idx, len(dts) - 1))

        dt_from_last = max(t - cum_times[past_idx], 0.0)
        dt_t = torch.tensor([[dt_from_last]], dtype=torch.float32).to(device)

        h_past = hiddens[:, :past_idx + 1]  # (1, past+1, H)
        c_past = cells[:, :past_idx + 1]

        dt_padded = torch.zeros(1, past_idx + 1, device=device)
        dt_padded[0, past_idx] = dt_from_last
        len_t = torch.tensor([past_idx + 1], dtype=torch.long, device=device)
        lam = model.intensity_at(h_past, c_past, dt_padded, len_t)
        lam_grid.append(lam[0, past_idx].item())

    return grid_times, np.array(lam_grid)


def run_ablation(sequences, loaders, device, seed, epochs=20):
    """
    Ablation: LSTM-only vs attention-only vs combined.
    Reports val LL for each configuration.
    """
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)

    train_l, val_l, test_l = loaders
    results = {}

    configs = {
        'LSTM-only (no attention)': NHPConfig(embed_dim=16, hidden_dim=32, num_heads=0),
        'Attention + LSTM (full NHP)': NHPConfig(embed_dim=16, hidden_dim=32, num_heads=2),
    }

    for name, cfg in configs.items():
        set_seed(seed)
        print(f"\n--- {name} ---")
        model = NeuralHawkesProcess(cfg)

        # For ablation, temporarily disable attention for LSTM-only
        if cfg.num_heads == 0:
            # Monkey-patch: identity attention
            import torch.nn as nn
            model.attn.forward = lambda q, k, mask=None: (q, torch.ones(q.shape[0], k.shape[1], device=q.device) / k.shape[1])

        history = train(
            model, train_l, val_l,
            epochs=epochs, lr=1e-3,
            checkpoint_path=f"nhp_ablation_{name.replace(' ', '_')[:10]}.pt",
            device=device, patience=5,
        )
        val_ll = evaluate(model, val_l, device)
        test_ll = evaluate(model, test_l, device)
        results[name] = {'val_ll': val_ll, 'test_ll': test_ll}
        print(f"  Final val LL:  {val_ll:.4f}")
        print(f"  Final test LL: {test_ll:.4f}")

    print("\nAblation summary:")
    for name, r in results.items():
        print(f"  {name:<35} val={r['val_ll']:.4f}  test={r['test_ll']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None, help='Path to real event CSV')
    parser.add_argument('--ts-col', type=str, default='timestamp')
    parser.add_argument('--group-col', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-seq', type=int, default=500, help='Synthetic sequences')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Seed: {args.seed}")

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.csv:
        print(f"\nLoading real data from {args.csv}...")
        sequences = load_timestamps_csv(args.csv, args.ts_col, args.group_col)
        print(f"Loaded {len(sequences)} sequences")
    else:
        print(f"\nGenerating {args.n_seq} synthetic Hawkes sequences...")
        sequences = simulate_nhp_dataset(
            n_sequences=args.n_seq, T=50.0,
            mu=0.3, alpha=0.5, beta=1.5, seed=args.seed
        )
        print(f"Mean sequence length: {np.mean([len(s) for s in sequences]):.1f} events")

    loaders = make_loaders(sequences, batch_size=args.batch_size)
    train_l, val_l, test_l = loaders

    # ── Ablation (optional) ──────────────────────────────────────────────────
    if args.ablation:
        run_ablation(sequences, loaders, device, args.seed, epochs=min(args.epochs, 25))

    # ── Train full model ──────────────────────────────────────────────────────
    set_seed(args.seed)
    print("\n" + "="*60)
    print("TRAINING FULL NHP MODEL")
    print("="*60)
    cfg = NHPConfig(embed_dim=16, hidden_dim=32, num_heads=2)
    model = NeuralHawkesProcess(cfg)
    history = train(
        model, train_l, val_l,
        epochs=args.epochs, lr=1e-3, weight_decay=1e-4,
        checkpoint_path='nhp_best.pt', device=device, patience=8,
    )

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    test_ll = evaluate(model, test_l, device)
    print(f"Test log-likelihood: {test_ll:.4f}")

    # ── Signal generation + evaluation on test sequences ─────────────────────
    print("\n" + "="*60)
    print("SIGNAL EVALUATION")
    print("="*60)

    policy = RegimeAwarePolicy(PolicyConfig(
        entry_mult=1.4, exit_mult=2.6,
        hysteresis=0.05, cooldown_steps=4,
    ))

    # Evaluate on first 20 test sequences
    test_seqs = sequences[int(len(sequences) * 0.9):][:20]
    all_precisions, all_recalls, all_f1s, all_delays = [], [], [], []

    for seq in test_seqs:
        if len(seq) < 5:
            continue
        dts = list(np.diff(seq, prepend=0.0))
        times, lam = get_intensity_series(model, dts, device)
        if len(lam) == 0:
            continue

        signals = policy.apply(lam, times)

        # Ground-truth clusters: steps where λ > 2× mean (synthetic ground truth)
        mean_lam = np.mean(lam)
        true_starts = [i for i in range(1, len(lam))
                       if lam[i] >= 2 * mean_lam and lam[i-1] < 2 * mean_lam]

        if not true_starts:
            continue

        ev = evaluate_signals(signals, true_starts, lam)
        all_precisions.append(ev.precision)
        all_recalls.append(ev.recall)
        all_f1s.append(ev.f1)
        if not math.isinf(ev.mean_delay):
            all_delays.append(ev.mean_delay)

    if all_f1s:
        print(f"Signal precision:        {np.mean(all_precisions):.3f}")
        print(f"Signal recall:           {np.mean(all_recalls):.3f}")
        print(f"Signal F1:               {np.mean(all_f1s):.3f}")
        print(f"Mean detection delay:    {np.mean(all_delays):.1f} steps")

    # ── Example sequence output ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXAMPLE SEQUENCE — last 5 signal events")
    print("="*60)
    if test_seqs:
        seq = test_seqs[0]
        dts = list(np.diff(seq, prepend=0.0))
        times, lam = get_intensity_series(model, dts, device)
        signals = policy.apply(lam, times)
        for s in signals[-5:]:
            label = {Signal.ENTER: "↑ ENTER", Signal.EXIT: "↓ EXIT ",
                     Signal.HOLD: "— HOLD "}[s.signal]
            print(f"  t={s.time:6.2f}  {label}  λ={s.lambda_t:.4f}  "
                  f"baseline={s.baseline:.4f}  confidence={s.confidence:.3f}")

    print("\nDone. Checkpoint saved to nhp_best.pt")


if __name__ == '__main__':
    main()
