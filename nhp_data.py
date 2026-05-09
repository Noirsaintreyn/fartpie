"""
data.py — Event sequence dataset utilities
==========================================
Key design decisions:
  - Temporal split (not random shuffle) to test real forward generalization
  - Supports loading real timestamp CSVs or synthetic Hawkes data for unit tests
  - Sequences padded to same length within a batch, with explicit length tracking
"""

import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ── Synthetic data generator (for testing / ablation) ─────────────────────────

def simulate_hawkes(
    mu: float = 0.3,
    alpha: float = 0.6,
    beta: float = 1.5,
    T: float = 100.0,
    seed: Optional[int] = None,
) -> list[float]:
    """
    Simulate a univariate Hawkes process via Ogata's thinning algorithm.
    Returns sorted list of event times in [0, T).

    This is the ground-truth simulation used to validate the NHP model.
    mu:    background rate
    alpha: excitation magnitude (must satisfy alpha < beta for stationarity)
    beta:  exponential decay rate
    """
    assert alpha < beta, "Process is supercritical (alpha >= beta); reduce alpha"
    rng = random.Random(seed)

    events = []
    t = 0.0
    lam_bar = mu  # upper bound on intensity

    while t < T:
        # Draw candidate inter-arrival
        dt = -math.log(rng.random() + 1e-12) / lam_bar
        t += dt
        if t >= T:
            break
        # Compute true intensity at candidate time
        lam_t = mu + alpha * sum(math.exp(-beta * (t - s)) for s in events)
        # Accept/reject
        if rng.random() < lam_t / lam_bar:
            events.append(t)
            # Update bound
            lam_bar = lam_t + alpha  # intensity jumps by alpha at event
        else:
            lam_bar = lam_t

    return events


def simulate_nhp_dataset(
    n_sequences: int = 500,
    T: float = 50.0,
    mu: float = 0.3,
    alpha: float = 0.5,
    beta: float = 1.5,
    seed: int = 42,
) -> list[list[float]]:
    """Generate n_sequences independent Hawkes realizations."""
    return [
        simulate_hawkes(mu=mu, alpha=alpha, beta=beta, T=T, seed=seed + i)
        for i in range(n_sequences)
    ]


# ── Dataset ───────────────────────────────────────────────────────────────────

class EventSequenceDataset(Dataset):
    """
    Wraps a list of event-time sequences.
    Each sequence is stored as inter-arrival times + event types (all type=1 for univariate).
    """
    def __init__(self, sequences: list[list[float]], max_len: int = 200):
        self.max_len = max_len
        self.data = []
        for seq in sequences:
            if len(seq) < 2:
                continue
            times = np.array(seq, dtype=np.float32)
            dts = np.diff(times, prepend=0.0).astype(np.float32)
            dts = dts[:max_len]
            L = len(dts)
            self.data.append({
                'dts': dts,
                'types': np.ones(L, dtype=np.int64),  # type 1 = univariate
                'length': L,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Pad sequences to the longest in the batch."""
    max_len = max(b['length'] for b in batch)
    B = len(batch)
    dts = torch.zeros(B, max_len)
    types = torch.zeros(B, max_len, dtype=torch.long)
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        L = b['length']
        dts[i, :L] = torch.from_numpy(b['dts'])
        types[i, :L] = torch.from_numpy(b['types'])
    return {'dts': dts, 'types': types, 'lengths': lengths}


def temporal_split(
    sequences: list[list[float]],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list, list, list]:
    """
    Split sequences by index order (earliest sequences → train,
    latest → test). This respects temporal ordering and avoids
    lookahead bias that random shuffling would introduce.
    """
    n = len(sequences)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    train = sequences[:n - n_val - n_test]
    val = sequences[n - n_val - n_test:n - n_test]
    test = sequences[n - n_test:]
    return train, val, test


def make_loaders(
    sequences: list[list[float]],
    batch_size: int = 32,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    max_len: int = 200,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_seqs, val_seqs, test_seqs = temporal_split(sequences, val_frac, test_frac)
    print(f"Dataset split — train: {len(train_seqs)}, val: {len(val_seqs)}, test: {len(test_seqs)}")
    train_ds = EventSequenceDataset(train_seqs, max_len=max_len)
    val_ds = EventSequenceDataset(val_seqs, max_len=max_len)
    test_ds = EventSequenceDataset(test_seqs, max_len=max_len)
    kw = dict(collate_fn=collate_fn, num_workers=num_workers)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
    )


# ── Real data loader ──────────────────────────────────────────────────────────

def load_timestamps_csv(
    path: str,
    timestamp_col: str = 'timestamp',
    group_col: Optional[str] = None,
) -> list[list[float]]:
    """
    Load real event timestamps from a CSV file.

    Expected format:
        timestamp (float/datetime), optional group_id

    If group_col is given, each unique group becomes its own sequence.
    Otherwise the entire file is treated as one sequence, split into
    windows of size window_size for batching.

    Example CSV:
        timestamp,symbol
        1700000001.0,AAPL
        1700000002.5,AAPL
        ...
    """
    import csv
    from datetime import datetime

    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row[timestamp_col]
            try:
                t = float(ts)
            except ValueError:
                # Try parsing ISO datetime
                t = datetime.fromisoformat(ts).timestamp()
            group = row[group_col] if group_col and group_col in row else 'all'
            rows.append((group, t))

    from collections import defaultdict
    groups = defaultdict(list)
    for g, t in rows:
        groups[g].append(t)

    sequences = []
    for g in sorted(groups):
        seq = sorted(groups[g])
        sequences.append(seq)
    return sequences


# ── OHLC → Event sequence bridge ─────────────────────────────────────────────

def ohlc_to_event_sequences(
    df,
    vol_threshold: float = 1.5,
    window_size: int = 200,
    stride: int = 50,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Convert an OHLC DataFrame into event sequences for the NHP model.

    Events are generated from bars where activity is "significant":
      - Large absolute returns (|r| > vol_threshold * rolling_std)
      - Volume spikes (volume > vol_threshold * rolling_mean_volume)
      - High-range bars (range / ATR > vol_threshold)

    Each bar's timestamp (as epoch seconds) becomes an event time.
    The result is a list of overlapping windows of event timestamps,
    suitable for NHP training or inference.

    Args:
        df: DataFrame with columns Open, High, Low, Close, Volume
            and a DatetimeIndex.
        vol_threshold: multiplier for triggering an event (default 1.5).
        window_size: max events per sequence window.
        stride: step size for sliding window.

    Returns:
        (sequences, all_timestamps)
        sequences:      list of lists of event times (epoch float)
        all_timestamps: single flat list of all event times (for inference)
    """
    import pandas as pd

    if df is None or len(df) < 10:
        return [], []

    df = df.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    closes = df['Close'].values.astype(np.float64)
    highs = df['High'].values.astype(np.float64)
    lows = df['Low'].values.astype(np.float64)
    volumes = df['Volume'].values.astype(np.float64) if 'Volume' in df.columns else np.ones(len(df))

    # Return-based events
    returns = np.diff(closes, prepend=closes[0]) / (closes + 1e-12)
    roll_std = pd.Series(returns).rolling(20, min_periods=5).std().fillna(returns.std()).values
    return_event = np.abs(returns) > vol_threshold * (roll_std + 1e-12)

    # Volume-spike events
    roll_vol = pd.Series(volumes).rolling(20, min_periods=5).mean().fillna(volumes.mean()).values
    vol_event = volumes > vol_threshold * (roll_vol + 1e-12)

    # Range-based events (high - low relative to ATR)
    ranges = highs - lows
    atr = pd.Series(ranges).rolling(14, min_periods=5).mean().fillna(ranges.mean()).values
    range_event = ranges > vol_threshold * (atr + 1e-12)

    # Union of all event triggers
    is_event = return_event | vol_event | range_event

    # Every bar is at least a candidate; mark significant ones
    # For NHP we use ALL bar timestamps but weight significant ones
    # Actually, use all bars as events — the NHP learns inter-arrival patterns
    timestamps_epoch = np.array([t.timestamp() for t in df.index], dtype=np.float64)

    # Normalize timestamps to start at 0 for numerical stability
    t0 = timestamps_epoch[0]
    timestamps_norm = timestamps_epoch - t0

    # Use all bar timestamps as the event stream
    all_timestamps = timestamps_norm.tolist()

    # Build windowed sequences for training
    sequences = []
    n = len(all_timestamps)
    if n <= window_size:
        sequences.append(all_timestamps)
    else:
        for start in range(0, n - window_size + 1, stride):
            window = all_timestamps[start:start + window_size]
            # Re-normalize window to start at 0
            w0 = window[0]
            sequences.append([t - w0 for t in window])

    # Also create sequences from significant-event-only timestamps
    sig_indices = np.where(is_event)[0]
    if len(sig_indices) >= 5:
        sig_times = timestamps_norm[sig_indices].tolist()
        if len(sig_times) <= window_size:
            sequences.append(sig_times)
        else:
            for start in range(0, len(sig_times) - window_size + 1, stride):
                window = sig_times[start:start + window_size]
                w0 = window[0]
                sequences.append([t - w0 for t in window])

    return sequences, all_timestamps


def ohlc_inter_arrival_times(df) -> list[float]:
    """
    Extract inter-arrival times from OHLC bar timestamps.
    Returns list of dt values (in seconds) for a single sequence pass.
    """
    import pandas as pd

    if df is None or len(df) < 2:
        return []

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    timestamps = np.array([t.timestamp() for t in df.index], dtype=np.float64)
    dts = np.diff(timestamps).tolist()
    return dts
