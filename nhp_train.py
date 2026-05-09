"""
train.py — Training loop for the Neural Hawkes Process
=======================================================
Objective: maximize event-time log-likelihood
  log L = Σᵢ log λ(tᵢ) − ∫₀ᵀ λ(t) dt

No heuristic stability clamps. Stability comes from:
  - Softplus parameterization (always positive intensity)
  - Weight decay / L2 regularization
  - Gradient clipping
  - Learned decay rates bounded below by softplus
"""

import time
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path

from nhp_model import NeuralHawkesProcess, NHPConfig


def train_epoch(
    model: NeuralHawkesProcess,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    n_mc: int = 20,
) -> float:
    model.train()
    total_ll, n_batches = 0.0, 0
    for batch in loader:
        dts = batch['dts'].to(device)
        types = batch['types'].to(device)
        lengths = batch['lengths'].to(device)
        optimizer.zero_grad()
        ll = model.log_likelihood(types, dts, lengths, n_mc=n_mc)
        loss = -ll  # minimize negative log-likelihood
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_ll += ll.item()
        n_batches += 1
    return total_ll / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: NeuralHawkesProcess,
    loader: DataLoader,
    device: torch.device,
    n_mc: int = 50,
) -> float:
    model.eval()
    total_ll, n_batches = 0.0, 0
    for batch in loader:
        dts = batch['dts'].to(device)
        types = batch['types'].to(device)
        lengths = batch['lengths'].to(device)
        ll = model.log_likelihood(types, dts, lengths, n_mc=n_mc)
        total_ll += ll.item()
        n_batches += 1
    return total_ll / max(n_batches, 1)


def train(
    model: NeuralHawkesProcess,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    patience: int = 8,
    checkpoint_path: str = "nhp_best.pt",
    device: Optional[torch.device] = None,
    n_mc_train: int = 20,
    n_mc_val: int = 50,
) -> dict:
    """
    Full training run with early stopping on validation log-likelihood.

    Returns dict of training history.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Training on {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    history = {'train_ll': [], 'val_ll': [], 'epoch_time': []}
    best_val_ll = -math.inf
    patience_count = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_ll = train_epoch(model, train_loader, optimizer, device,
                               grad_clip=grad_clip, n_mc=n_mc_train)
        val_ll = evaluate(model, val_loader, device, n_mc=n_mc_val)
        elapsed = time.time() - t0

        scheduler.step(val_ll)
        history['train_ll'].append(train_ll)
        history['val_ll'].append(val_ll)
        history['epoch_time'].append(elapsed)

        print(f"Epoch {epoch:03d} | train LL={train_ll:.4f} | val LL={val_ll:.4f} "
              f"| lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if val_ll > best_val_ll:
            best_val_ll = val_ll
            patience_count = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'config': model.cfg,
                'val_ll': val_ll,
            }, checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Reload best checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"\nBest val LL: {best_val_ll:.4f} (epoch {ckpt['epoch']})")
    return history
