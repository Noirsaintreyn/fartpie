"""
data_v2.py — State-augmented event sequence dataset
=====================================================
Each sequence now carries:
  - event times (inter-arrival dts)
  - event types
  - state vectors (vol + regime features) at each event time

For synthetic data: vol features are simulated from a GBM price path
that shares the same clustering structure as the Hawkes process.
For real data: pass a price series aligned with the event timestamps.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from dataclasses import dataclass

from nhp_data import simulate_hawkes, temporal_split, collate_fn as base_collate
from nhp_regime import compute_vol_features, RegimeDetector, state_vector_at, VolFeatures


# ── Synthetic price path aligned to Hawkes events ────────────────────────────

def simulate_price_path(
    event_times: list[float],
    T: float,
    dt_bar: float = 0.5,    # price bar interval (same units as event times)
    sigma_base: float = 0.01,
    jump_size: float = 0.005,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a GBM price path with jumps at event times.
    Returns (prices, bar_times) aligned to a uniform grid.
    """
    rng = np.random.RandomState(seed)
    bar_times = np.arange(0, T + dt_bar, dt_bar)
    n = len(bar_times)
    log_price = np.zeros(n)

    for i in range(1, n):
        # Diffusion
        log_price[i] = log_price[i - 1] + rng.normal(0, sigma_base)
        # Jump: any event in this bar?
        t0, t1 = bar_times[i - 1], bar_times[i]
        n_jumps = sum(1 for e in event_times if t0 <= e < t1)
        if n_jumps > 0:
            log_price[i] += rng.choice([-1, 1]) * jump_size * n_jumps

    prices = np.exp(log_price)
    return prices, bar_times


def build_state_vectors(
    event_times: list[float],
    prices: np.ndarray,
    bar_times: np.ndarray,
    regime_detector: Optional[RegimeDetector],
    regimes: Optional[np.ndarray],
    regime_probs: Optional[np.ndarray],
    vf: Optional[VolFeatures],
) -> np.ndarray:
    """
    Build (len(event_times), 6) state array by looking up vol/regime
    features at each event time (interpolated to nearest bar).
    """
    n_ev = len(event_times)
    states = np.zeros((n_ev, 6), dtype=np.float32)

    if vf is None or regimes is None:
        return states

    median_rv = float(np.median(vf.realized_vol[vf.realized_vol > 0]))

    for i, t in enumerate(event_times):
        idx = int(np.searchsorted(bar_times, t, side='right')) - 1
        idx = max(0, min(idx, len(bar_times) - 1))
        sv = state_vector_at(vf, regimes, regime_probs, idx)
        # Normalize realized vol and vol-of-vol by median
        sv[0] = sv[0] / (median_rv + 1e-8)   # normalized realized vol
        sv[1] = sv[1] / (median_rv + 1e-8)   # normalized vol-of-vol
        states[i] = sv

    return states


# ── Dataset ───────────────────────────────────────────────────────────────────

class StateAwareDataset(Dataset):
    def __init__(self, sequences: list[dict], max_len: int = 200):
        self.max_len = max_len
        self.data = []
        for seq in sequences:
            dts   = seq['dts'][:max_len]
            types = seq['types'][:max_len]
            state = seq['states'][:max_len]   # (L, 6)
            L = len(dts)
            if L < 2:
                continue
            self.data.append({'dts': dts, 'types': types, 'states': state, 'length': L})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_v2(batch):
    max_len = max(b['length'] for b in batch)
    B = len(batch)
    state_dim = batch[0]['states'].shape[-1]
    dts    = torch.zeros(B, max_len)
    types  = torch.zeros(B, max_len, dtype=torch.long)
    states = torch.zeros(B, max_len, state_dim)
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        L = b['length']
        dts[i, :L]       = torch.from_numpy(b['dts'])
        types[i, :L]     = torch.from_numpy(b['types'])
        states[i, :L]    = torch.from_numpy(b['states'])
    return {'dts': dts, 'types': types, 'states': states, 'lengths': lengths}


def make_loaders_v2(
    sequences: list[dict],
    batch_size: int = 32,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    max_len: int = 200,
) -> tuple:
    train_s, val_s, test_s = temporal_split(sequences, val_frac, test_frac)
    print(f"Split — train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")
    kw = dict(collate_fn=collate_v2)
    return (
        DataLoader(StateAwareDataset(train_s, max_len), batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(StateAwareDataset(val_s,   max_len), batch_size=batch_size, shuffle=False, **kw),
        DataLoader(StateAwareDataset(test_s,  max_len), batch_size=batch_size, shuffle=False, **kw),
    )
