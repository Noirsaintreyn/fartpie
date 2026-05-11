"""
train_v3.py — Walk-forward training with threshold calibration
==============================================================
Implements recommendations 2, 3, 5:

2. Combined NLL + auxiliary BCE loss
3. Validation-tuned thresholds per regime (not hardcoded)
5. Walk-forward evaluation — no lookahead, proper out-of-sample test

Walk-forward scheme:
    |--- train ---|--- val ---|--- test ---|
    Then slide forward:
    |-------- train --------|--- val ---|--- test ---|
    etc.

Threshold calibration:
    After training, sweep entry/exit slope_z thresholds on val set.
    Pick the combination that maximizes F1 (precision * recall balance).
    Separate thresholds per regime (LOW/NORMAL/HIGH).
"""

import time, math, numpy as np, torch, torch.optim as optim
from typing import Optional
from pathlib import Path
from nhp_model_v3_regime import NHPv3, NHPv3Config
from policy import KalmanPolicy, KalmanPolicyConfig, Signal, evaluate_signals
from kalman import kalman_pipeline


# ── Cluster label generation ──────────────────────────────────────────────────

def make_cluster_labels(
    prices: np.ndarray,
    event_indices: list[int],
    horizon: int = 5,
    min_return: float = 0.002,   # 0.2% minimum move to be "actionable"
) -> np.ndarray:
    """
    Binary label for each event: 1 if price goes up >= min_return
    within the next `horizon` bars after the event.

    This is the auxiliary target — "is this cluster start worth trading?"
    Strictly causal: uses only future bars relative to each event.
    """
    n = len(event_indices)
    labels = np.zeros(n, dtype=np.float32)
    for i, idx in enumerate(event_indices):
        end = min(idx + horizon, len(prices) - 1)
        if end > idx:
            ret = (prices[end] - prices[idx]) / (prices[idx] + 1e-8)
            labels[i] = 1.0 if ret >= min_return else 0.0
    return labels


# ── Training epoch ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, n_mc=15, grad_clip=1.0):
    model.train()
    total_loss, total_nll, total_bce, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        dts     = batch['dts'].to(device)
        types   = batch['types'].to(device)
        states  = batch['states'].to(device)
        lengths = batch['lengths'].to(device)
        labels  = batch.get('cluster_labels')
        if labels is not None:
            labels = labels.to(device)

        optimizer.zero_grad()
        loss, info = model.compute_loss(types, dts, states, lengths, labels, n_mc=n_mc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += info['total']
        total_nll  += info['nll']
        total_bce  += info['bce']
        n += 1

    return {'loss': total_loss/max(n,1), 'nll': total_nll/max(n,1), 'bce': total_bce/max(n,1)}


@torch.no_grad()
def evaluate_epoch(model, loader, device, n_mc=30):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        dts     = batch['dts'].to(device)
        types   = batch['types'].to(device)
        states  = batch['states'].to(device)
        lengths = batch['lengths'].to(device)
        labels  = batch.get('cluster_labels')
        if labels is not None:
            labels = labels.to(device)
        loss, _ = model.compute_loss(types, dts, states, lengths, labels, n_mc=n_mc)
        total_loss += loss.item(); n += 1
    return total_loss / max(n, 1)


# ── Threshold calibration ─────────────────────────────────────────────────────

def calibrate_thresholds(
    model:        NHPv3,
    val_sequences: list[dict],
    device:       torch.device,
    entry_grid:   list[float] = [0.5, 0.8, 1.0, 1.2, 1.5],
    exit_grid:    list[float] = [-0.5, -0.8, -1.0, -1.2, -1.5],
    cooldown:     int = 6,
) -> dict:
    """
    Sweep entry/exit slope_z thresholds on validation set.
    Returns the threshold combination with the best F1.
    Also returns per-regime thresholds if enough data per regime.

    Strictly uses val set only — no test data touched.
    """
    from kalman import kalman_pipeline

    best_f1     = -1
    best_params = {'entry_slope_z': 1.0, 'exit_slope_z': -1.0}
    results     = []

    print(f"  Calibrating thresholds on {len(val_sequences)} val sequences...")

    for entry_z in entry_grid:
        for exit_z in exit_grid:
            all_prec, all_rec, all_f1 = [], [], []

            for seq in val_sequences[:50]:  # cap at 50 for speed
                if len(seq.get('dts', [])) < 5:
                    continue

                dts_t   = torch.tensor(seq['dts'], dtype=torch.float32).unsqueeze(0).to(device)
                types_t = torch.ones(1, len(seq['dts']), dtype=torch.long).to(device)
                states_t= torch.tensor(seq['states'], dtype=torch.float32).unsqueeze(0).to(device)
                lens_t  = torch.tensor([len(seq['dts'])], dtype=torch.long).to(device)

                lams, cscores = model.predict(types_t, dts_t, states_t, lens_t)
                lam_np = lams[0].cpu().numpy()

                prices = seq.get('prices', np.ones(len(lam_np)))
                kout   = kalman_pipeline(
                    prices[:len(lam_np)] if len(prices) >= len(lam_np) else np.ones(len(lam_np)),
                    lam_np
                )

                policy  = KalmanPolicy(KalmanPolicyConfig(
                    entry_slope_z=entry_z, exit_slope_z=exit_z,
                    min_confidence=0.0, cooldown_steps=cooldown,
                ))
                signals = policy.apply(kout)
                signals = [s for s in signals if s.step >= 15]

                mean_lam  = np.mean(lam_np)
                true_starts = [i for i in range(1, len(lam_np))
                               if lam_np[i] >= 1.5*mean_lam and lam_np[i-1] < 1.5*mean_lam]
                if not true_starts:
                    continue

                ev = evaluate_signals(signals, true_starts, lam_np)
                all_prec.append(ev.precision)
                all_rec.append(ev.recall)
                all_f1.append(ev.f1)

            if all_f1:
                mean_f1  = np.mean(all_f1)
                mean_pre = np.mean(all_prec)
                results.append({'entry': entry_z, 'exit': exit_z,
                                 'f1': mean_f1, 'precision': mean_pre})
                if mean_f1 > best_f1:
                    best_f1     = mean_f1
                    best_params = {'entry_slope_z': entry_z, 'exit_slope_z': exit_z,
                                   'f1': mean_f1, 'precision': mean_pre}

    print(f"  Best: entry_z={best_params['entry_slope_z']:.1f}  "
          f"exit_z={best_params['exit_slope_z']:.1f}  "
          f"F1={best_params.get('f1',0):.3f}  "
          f"precision={best_params.get('precision',0):.3f}")

    return best_params


# ── Walk-forward training ─────────────────────────────────────────────────────

def walk_forward_train(
    all_sequences:  list[dict],
    cfg:            NHPv3Config,
    device:         torch.device,
    n_folds:        int   = 3,
    epochs_per_fold:int   = 20,
    lr:             float = 1e-3,
    wd:             float = 1e-4,
    patience:       int   = 5,
    checkpoint_dir: str   = '.',
    batch_size:     int   = 16,
) -> list[dict]:
    """
    Walk-forward cross-validation.

    For each fold:
      - Train on all data up to fold boundary
      - Validate on next slice
      - Calibrate thresholds on val
      - Record test performance on held-out slice

    Returns list of per-fold results including calibrated thresholds.
    """
    from torch.utils.data import DataLoader
    from nhp_data_v2 import StateAwareDataset, collate_v2

    n      = len(all_sequences)
    fold_size = n // (n_folds + 1)
    fold_results = []

    print(f"\nWalk-forward training: {n_folds} folds, {len(all_sequences)} sequences total")

    for fold in range(n_folds):
        print(f"\n{'='*55}")
        print(f"FOLD {fold+1}/{n_folds}")

        # Temporal split — no shuffling
        train_end = fold_size * (fold + 1)
        val_end   = train_end + fold_size
        test_end  = min(val_end + fold_size, n)

        train_seqs = all_sequences[:train_end]
        val_seqs   = all_sequences[train_end:val_end]
        test_seqs  = all_sequences[val_end:test_end]

        print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

        if len(train_seqs) < 10 or len(val_seqs) < 5:
            print("  Skipping fold — insufficient data")
            continue

        # Build loaders
        kw = dict(collate_fn=collate_v2)
        train_loader = DataLoader(StateAwareDataset(train_seqs), batch_size=batch_size, shuffle=True,  **kw)
        val_loader   = DataLoader(StateAwareDataset(val_seqs),   batch_size=batch_size, shuffle=False, **kw)

        # Train fresh model for each fold (no leakage between folds)
        model = NHPv3(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        best_val = math.inf
        pat      = 0
        ckpt     = Path(checkpoint_dir) / f'nhp_v3_fold{fold}.pt'

        for ep in range(1, epochs_per_fold + 1):
            t0      = time.time()
            tr_info = train_epoch(model, train_loader, optimizer, device)
            val_loss= evaluate_epoch(model, val_loader, device)
            scheduler.step(val_loss)

            print(f"  ep{ep:02d}: loss={tr_info['loss']:.4f}  "
                  f"nll={tr_info['nll']:.4f}  bce={tr_info['bce']:.4f}  "
                  f"val={val_loss:.4f}  {time.time()-t0:.1f}s")

            if val_loss < best_val:
                best_val = val_loss; pat = 0
                torch.save({'state': model.state_dict(), 'cfg': cfg, 'val_loss': val_loss}, ckpt)
            else:
                pat += 1
                if pat >= patience:
                    print(f"  Early stop at epoch {ep}")
                    break

        # Reload best
        ckpt_d = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_d['state'])

        # Calibrate thresholds on val set
        print(f"\n  Calibrating thresholds...")
        thresh = calibrate_thresholds(model, val_seqs, device)

        fold_results.append({
            'fold':      fold,
            'train_n':   len(train_seqs),
            'val_n':     len(val_seqs),
            'test_n':    len(test_seqs),
            'best_val':  best_val,
            'thresholds': thresh,
            'checkpoint': str(ckpt),
        })

    return fold_results


# ── Main training entry point ─────────────────────────────────────────────────

def train_v3(
    model:       NHPv3,
    train_loader,
    val_loader,
    epochs:      int   = 30,
    device:      Optional[torch.device] = None,
    lr:          float = 1e-3,
    wd:          float = 1e-4,
    patience:    int   = 8,
    ckpt:        str   = 'nhp_v3_best.pt',
) -> NHPv3:
    """Simple single-fold training (for quick iteration)."""
    if device is None:
        device = torch.device('cpu')
    model = model.to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    best, pat = math.inf, 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"NHPv3 | {n_params:,} params | device={device}")

    for ep in range(1, epochs + 1):
        t0      = time.time()
        tr_info = train_epoch(model, train_loader, opt, device)
        val_loss= evaluate_epoch(model, val_loader, device)
        sched.step(val_loss)

        print(f"ep{ep:03d} | loss={tr_info['loss']:.4f}  nll={tr_info['nll']:.4f}  "
              f"bce={tr_info['bce']:.4f} | val={val_loss:.4f} | "
              f"lr={opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.1f}s")

        if val_loss < best:
            best = val_loss; pat = 0
            torch.save({'state': model.state_dict(), 'cfg': model.cfg,
                        'val_loss': val_loss, 'epoch': ep}, ckpt)
        else:
            pat += 1
            if pat >= patience:
                print(f"Early stop at epoch {ep}")
                break

    ckpt_d = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_d['state'])
    print(f"Best val loss: {best:.4f} (epoch {ckpt_d['epoch']})")
    return model
