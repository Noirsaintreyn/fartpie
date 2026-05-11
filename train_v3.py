#!/usr/bin/env python3
"""
Training + backtest for NHP v3 (architectural bug fixes).

v3 fixes from v2:
1. extrapolate() uses correct Mei & Eisner formula with c_bar and output gate
2. IntensityHead uses LayerNorm+SiLU instead of Tanh compression
3. LinearNormAttention preserves magnitude (scales by ||query||)
4. xavier_uniform gain=1.0 on hidden, 0.1 on output projections only
5. forward_sequence returns 4 tensors: (hiddens, cells, cbars, outputs)

No state vectors needed — v3 uses simpler EventEmbedding.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nhp_model_v3 import NeuralHawkesProcess, NHPConfig
from nhp_data import ohlc_inter_arrival_times, collate_fn
from nhp_regime import (
    compute_vol_features, RegimeDetector, RegimeGate, RegimeGateConfig,
    Regime, REGIME_LABELS,
)
from torch.utils.data import Dataset, DataLoader

CHECKPOINT = 'nhp_v3_best.pt'
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'historical_data')

TRAIN_FILES = {
    'NQ': {
        '1h': 'NQ/1H/1H_NQ.csv',
        '4h': 'NQ/4H/4H_NQ.csv',
        '1d': 'NQ/D/D_NQ.csv',
    },
    'ES': {
        '1h': 'ES/1H/1H_ES.csv',
        '4h': 'ES/4H/4H_ES.csv',
        '1d': 'ES/D/D_ES.csv',
    },
}

BACKTEST_FILES = {
    'NQ': {
        '5m': 'NQ/5Min/NQ_5Min.csv',
        '15m': 'NQ/15Min/15Min_NQ.csv',
        '1h': 'NQ/1H/1H_NQ.csv',
        '4h': 'NQ/4H/4H_NQ.csv',
        '1d': 'NQ/D/D_NQ.csv',
    },
    'ES': {
        '5m': 'ES/5Min/5Min_ES.csv',
        '15m': 'ES/15Min/15Min_ES.csv',
        '1h': 'ES/1H/1H_ES.csv',
        '4h': 'ES/4H/4H_ES.csv',
        '1d': 'ES/D/D_ES.csv',
    },
}


def parse_european_number(val):
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace('.', '').replace(',', '.')
    return float(s)


def load_csv(filepath):
    with open(filepath) as f:
        first = f.readline().strip()
    skip = 1 if 'Time Series' in first else 0
    df = pd.read_csv(filepath, sep=';', skiprows=skip)
    df.columns = [c.strip() for c in df.columns]
    for c in ['Open', 'High', 'Low', 'Close']:
        df[c] = df[c].apply(parse_european_number)
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].apply(lambda v: parse_european_number(v) if pd.notna(v) else 0)
    else:
        df['Volume'] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['Datetime'] = pd.to_datetime(df['Date'])
    df = df.set_index('Datetime').sort_index()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df


class SimpleDataset(Dataset):
    def __init__(self, sequences, max_len=20):
        self.data = []
        for seq in sequences:
            dts = seq['dts'][:max_len]
            types = seq['types'][:max_len]
            L = len(dts)
            if L < 2:
                continue
            self.data.append({'dts': dts, 'types': types, 'length': L})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_sequences(df, window_size=20, stride=10):
    dts = ohlc_inter_arrival_times(df)
    n_bars = len(dts)
    dts_arr = np.array(dts, dtype=np.float64)
    scale = np.median(dts_arr)
    if scale <= 0:
        scale = 1.0
    dts_scaled = dts_arr / scale

    sequences = []
    for start in range(0, n_bars - window_size + 1, stride):
        end = start + window_size
        seq_dts = dts_scaled[start:end].astype(np.float32)
        seq_types = np.ones(window_size, dtype=np.int64)
        sequences.append({
            'dts': seq_dts,
            'types': seq_types,
            'start_bar': start,
        })
    return sequences


def train_epoch(model, loader, optimizer, device, n_mc=3):
    model.train()
    total_ll = 0.0
    total_n = 0
    for batch in loader:
        dts = batch['dts'].to(device)
        types = batch['types'].to(device)
        lengths = batch['lengths'].to(device)
        optimizer.zero_grad()
        ll = model.log_likelihood(types, dts, lengths, n_mc=n_mc)
        loss = -ll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_ll += ll.item() * dts.shape[0]
        total_n += dts.shape[0]
    return total_ll / max(total_n, 1)


def eval_model(model, loader, device, n_mc=5):
    model.eval()
    total_ll = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            dts = batch['dts'].to(device)
            types = batch['types'].to(device)
            lengths = batch['lengths'].to(device)
            ll = model.log_likelihood(types, dts, lengths, n_mc=n_mc)
            total_ll += ll.item() * dts.shape[0]
            total_n += dts.shape[0]
    return total_ll / max(total_n, 1)


def compute_price_activity(df):
    closes = df['Close'].values.astype(np.float64)
    highs = df['High'].values.astype(np.float64)
    lows = df['Low'].values.astype(np.float64)
    n = len(closes)
    log_ret = np.diff(np.log(np.maximum(closes, 1e-10)), prepend=0.0)
    abs_ret = np.abs(log_ret)
    tr = (highs - lows) / (closes + 1e-10)
    if 'Volume' in df.columns:
        vol = df['Volume'].values.astype(np.float64)
        vol_norm = vol / (np.median(vol[vol > 0]) + 1e-10) if np.any(vol > 0) else np.ones(n)
        vol_norm = np.clip(vol_norm, 0, 5)
    else:
        vol_norm = np.ones(n)
    activity = 0.4 * abs_ret + 0.3 * tr + 0.3 * vol_norm
    amin, amax = activity.min(), activity.max()
    if amax > amin:
        activity = (activity - amin) / (amax - amin)
    else:
        activity = np.full(n, 0.5)
    return activity


def run_inference(df, regime_detector, checkpoint_path=CHECKPOINT, blend_weight=0.5):
    prices = df['Close'].values.astype(np.float64)
    bar_times = np.arange(len(prices), dtype=np.float64)
    vf = compute_vol_features(prices, bar_times)

    # Regime detection
    try:
        regimes, regime_probs = regime_detector.predict(vf)
    except Exception:
        regimes = np.ones(len(prices), dtype=int)
        regime_probs = np.tile([0.33, 0.34, 0.33], (len(prices), 1))

    dts = ohlc_inter_arrival_times(df)
    n_bars = len(dts)
    dts_arr = np.array(dts, dtype=np.float64)
    scale = np.median(dts_arr)
    if scale <= 0:
        scale = 1.0
    dts_scaled = dts_arr / scale

    device = torch.device('cpu')
    has_ckpt = os.path.exists(checkpoint_path)
    cfg = NHPConfig(embed_dim=16, hidden_dim=32, num_heads=2)
    model = NeuralHawkesProcess(cfg)
    if has_ckpt:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state'])
    model.eval().to(device)

    # Windowed inference with v3 API (4-tuple return)
    window_size = 20
    stride = 10
    model_lam = np.zeros(n_bars, dtype=np.float64)
    lam_counts = np.zeros(n_bars, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n_bars - window_size + 1, stride):
            end = start + window_size
            w_dts = torch.tensor(dts_scaled[start:end], dtype=torch.float32).unsqueeze(0)
            w_types = torch.ones(1, window_size, dtype=torch.long)

            hiddens, cells, cbars, outputs = model.forward_sequence(w_types, w_dts)
            z = torch.zeros_like(w_dts)
            sl = torch.tensor([window_size], dtype=torch.long)
            w_lam = model.intensity_at(hiddens, cells, z, sl, cbars, outputs)[0].cpu().numpy()

            model_lam[start:end] += w_lam
            lam_counts[start:end] += 1.0

        # Handle tail
        if n_bars > window_size:
            remaining_start = max(0, n_bars - window_size)
            w_dts = torch.tensor(dts_scaled[remaining_start:n_bars], dtype=torch.float32).unsqueeze(0)
            w_len = n_bars - remaining_start
            w_types = torch.ones(1, w_len, dtype=torch.long)

            hiddens, cells, cbars, outputs = model.forward_sequence(w_types, w_dts)
            z = torch.zeros_like(w_dts)
            sl = torch.tensor([w_len], dtype=torch.long)
            w_lam = model.intensity_at(hiddens, cells, z, sl, cbars, outputs)[0].cpu().numpy()

            model_lam[remaining_start:n_bars] += w_lam
            lam_counts[remaining_start:n_bars] += 1.0

    lam_counts = np.maximum(lam_counts, 1.0)
    model_lam = model_lam / lam_counts

    pa = compute_price_activity(df)
    if len(pa) == n_bars + 1:
        pa = pa[1:]
    elif len(pa) > n_bars:
        pa = pa[-n_bars:]

    # Z-score normalize
    m_mean, m_std = model_lam.mean(), model_lam.std()
    if m_std > 1e-8:
        model_zscore = (model_lam - m_mean) / m_std
        model_norm = np.clip(model_zscore, -3, 3) / 6.0 + 0.5
    else:
        model_norm = np.full(n_bars, 0.5)

    bw = blend_weight if has_ckpt else 0.2
    blended = bw * model_norm + (1 - bw) * pa
    blended_scaled = 0.1 + blended * 1.9

    # Selective signal generation with price momentum confirmation
    lookback = min(10, max(3, n_bars // 100))
    cooldown = 6
    cooldown_counter = 0

    gate = RegimeGate(RegimeGateConfig(min_confidence=0.55))
    median_rv_pos = vf.realized_vol[vf.realized_vol > 0]
    median_rv = float(np.median(median_rv_pos)) if len(median_rv_pos) > 0 else 1e-8

    closes = df['Close'].values.astype(np.float64)
    ema5 = np.zeros(len(closes))
    ema5[0] = closes[0]
    alpha = 2.0 / 6.0
    for j in range(1, len(closes)):
        ema5[j] = alpha * closes[j] + (1 - alpha) * ema5[j - 1]

    filtered_signals = []
    for i in range(lookback, n_bars):
        if cooldown_counter > 0:
            cooldown_counter -= 1
            continue

        local_window = blended_scaled[max(0, i - lookback):i]
        local_mean = np.mean(local_window)
        local_std = np.std(local_window) if len(local_window) > 1 else 0.01
        current = blended_scaled[i]
        z_val = (current - local_mean) / max(local_std, 1e-6)

        bi = min(i + 1, len(vf.realized_vol) - 1)
        rv = float(vf.realized_vol[bi])
        regime = Regime(regimes[bi])
        regime_conf = float(max(regime_probs[bi]))

        price_idx = min(i + 1, len(closes) - 1)
        prev_idx = max(0, price_idx - 1)

        if z_val > 2.0 and current > local_mean + 1.5 * local_std:
            price_rising = closes[price_idx] > closes[prev_idx]
            above_ema = closes[price_idx] > ema5[price_idx]
            if price_rising and above_ema:
                ok, _ = gate.allows_enter(regime, regime_conf, rv, median_rv)
                if ok:
                    filtered_signals.append({
                        'step': i, 'time': float(i),
                        'signal': 'ENTER', 'bar_index': i,
                    })
                    cooldown_counter = cooldown

        elif z_val < -2.0 and current < local_mean - 1.5 * local_std:
            price_falling = closes[price_idx] < closes[prev_idx]
            below_ema = closes[price_idx] < ema5[price_idx]
            if price_falling and below_ema:
                ok, _ = gate.allows_exit(regime, regime_conf, rv, median_rv)
                if ok:
                    filtered_signals.append({
                        'step': i, 'time': float(i),
                        'signal': 'EXIT', 'bar_index': i,
                    })
                    cooldown_counter = cooldown

    n_enter = sum(1 for s in filtered_signals if s['signal'] == 'ENTER')
    n_exit = sum(1 for s in filtered_signals if s['signal'] == 'EXIT')

    return {
        'signals': filtered_signals,
        'model_trained': has_ckpt,
        'model_lam_std': float(model_lam.std()),
        'model_lam_range': (float(model_lam.min()), float(model_lam.max())),
        'n_raw_signals': n_enter + n_exit,
        'n_filtered_signals': len(filtered_signals),
        'ohlc': {
            'close': df['Close'].values[1:].tolist(),
            'high': df['High'].values[1:].tolist(),
            'low': df['Low'].values[1:].tolist(),
        },
    }


def compute_drawdown_accuracy(closes, highs, lows, signals, target_pct=0.005):
    enter_correct = 0
    enter_total = 0
    exit_correct = 0
    exit_total = 0

    for sig in signals:
        bi = sig['bar_index']
        if bi >= len(closes) - 1:
            continue

        entry_price = closes[bi]
        sig_type = sig['signal']

        if sig_type == 'ENTER':
            enter_total += 1
            target = entry_price * (1 + target_pct)
            for j in range(bi + 1, len(closes)):
                if lows[j] < entry_price:
                    break
                if highs[j] >= target:
                    enter_correct += 1
                    break
        elif sig_type == 'EXIT':
            exit_total += 1
            target = entry_price * (1 - target_pct)
            for j in range(bi + 1, len(closes)):
                if highs[j] > entry_price:
                    break
                if lows[j] <= target:
                    exit_correct += 1
                    break

    overall_total = enter_total + exit_total
    overall_correct = enter_correct + exit_correct

    def fmt_pct(c, t):
        return f"{c/t*100:.1f}%" if t > 0 else "N/A"

    return {
        'overall_acc': fmt_pct(overall_correct, overall_total),
        'overall_correct': overall_correct,
        'overall_total': overall_total,
        'enter_acc': fmt_pct(enter_correct, enter_total),
        'enter_correct': enter_correct,
        'enter_total': enter_total,
        'exit_acc': fmt_pct(exit_correct, exit_total),
        'exit_correct': exit_correct,
        'exit_total': exit_total,
    }


def main():
    print("=" * 60)
    print("  TRAINING NHP v3 (ARCHITECTURE BUGS FIXED)")
    print("=" * 60)

    device = torch.device('cpu')

    # Step 1: Load training data
    print("\n  Loading training data...")
    all_prices = []
    all_bar_times = []
    all_dfs = {}
    offset = 0

    for symbol in ['NQ', 'ES']:
        for tf, rel_path in TRAIN_FILES[symbol].items():
            fp = os.path.join(DATA_ROOT, rel_path)
            if not os.path.exists(fp):
                print(f"    SKIP {symbol} {tf}: file not found")
                continue
            df = load_csv(fp)
            if len(df) > 5000:
                df = df.iloc[-5000:]
            all_dfs[f"{symbol}_{tf}"] = df
            prices = df['Close'].values.astype(np.float64)
            all_prices.append(prices)
            all_bar_times.append(np.arange(len(prices), dtype=np.float64) + offset)
            offset += len(prices) + 100
            print(f"    {symbol} {tf}: {len(df)} bars")

    concat_prices = np.concatenate(all_prices)
    concat_times = np.concatenate(all_bar_times)

    # Step 2: Fit HMM (used for regime gate in backtest, not for training)
    print("\n  Fitting HMM regime detector...")
    train_cutoff = int(len(concat_prices) * 0.8)
    train_vf = compute_vol_features(concat_prices[:train_cutoff], concat_times[:train_cutoff])
    detector = RegimeDetector(n_states=3, n_iter=200, random_state=42)
    detector.fit(train_vf)

    full_vf = compute_vol_features(concat_prices, concat_times)
    regimes, probs = detector.predict(full_vf)
    for r in Regime:
        count = np.sum(regimes == int(r))
        pct = count / len(regimes) * 100
        print(f"    {REGIME_LABELS[r]}: {count} bars ({pct:.1f}%)")

    # Step 3: Build sequences (no state vectors for v3)
    print("\n  Building training sequences...")
    all_sequences = []
    for key, df in all_dfs.items():
        seqs = build_sequences(df, window_size=20, stride=10)
        all_sequences.extend(seqs)
        print(f"    {key}: {len(seqs)} sequences")

    print(f"\n  Total sequences: {len(all_sequences)}")

    # Step 4: Create dataloaders (temporal split)
    n = len(all_sequences)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_seqs = all_sequences[:n_train]
    val_seqs = all_sequences[n_train:n_train + n_val]
    test_seqs = all_sequences[n_train + n_val:]

    print(f"  Split: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")

    train_ds = SimpleDataset(train_seqs, max_len=20)
    val_ds = SimpleDataset(val_seqs, max_len=20)
    kw = dict(collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, **kw)

    # Step 5: Train
    print(f"\n  Training NeuralHawkesProcess v3...")
    cfg = NHPConfig(embed_dim=16, hidden_dim=32, num_heads=2)
    model = NeuralHawkesProcess(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val = float('-inf')
    patience = 0
    max_patience = 7
    best_state = None

    for epoch in range(1, 21):
        t0 = time.time()
        train_ll = train_epoch(model, train_loader, optimizer, device, n_mc=3)
        val_ll = eval_model(model, val_loader, device, n_mc=5)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:03d} | train={train_ll:.4f} | val={val_ll:.4f} | lr={lr:.1e} | {dt:.1f}s")

        scheduler.step(-val_ll)

        if val_ll > best_val:
            best_val = val_ll
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                sample = next(iter(val_loader))
                hiddens, cells, cbars, outputs = model.forward_sequence(
                    sample['types'].to(device),
                    sample['dts'].to(device),
                )
                z = torch.zeros_like(sample['dts'].to(device))
                sl = sample['lengths'].to(device)
                lam = model.intensity_at(hiddens, cells, z, sl, cbars, outputs)
                print(f"    λ check: min={lam.min():.4f} max={lam.max():.4f} std={lam.std():.4f}")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"  Early stop at epoch {epoch}")
                break

    # Save checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({
            'epoch': epoch,
            'state': best_state,
            'cfg': cfg,
            'val_ll': best_val,
        }, CHECKPOINT)
        print(f"\n  Best val LL: {best_val:.4f}")
        print(f"  Checkpoint saved: {CHECKPOINT}")

    # Step 6: Backtest
    print("\n" + "=" * 60)
    print("  BACKTEST WITH TRAINED v3 MODEL")
    print("=" * 60)

    results = []
    for symbol in ['NQ', 'ES']:
        print(f"\n  --- {symbol} ---")
        for tf, rel_path in BACKTEST_FILES[symbol].items():
            fp = os.path.join(DATA_ROOT, rel_path)
            if not os.path.exists(fp):
                print(f"    {tf}: FILE NOT FOUND")
                continue
            try:
                df = load_csv(fp)
                if len(df) > 5000:
                    df = df.iloc[-5000:]

                result = run_inference(df, detector, CHECKPOINT, blend_weight=0.5)
                if result is None:
                    print(f"    {tf}: No result")
                    continue

                acc = compute_drawdown_accuracy(
                    result['ohlc']['close'],
                    result['ohlc']['high'],
                    result['ohlc']['low'],
                    result['signals'],
                )

                print(f"    {tf:>4s}: {len(df)} bars | "
                      f"raw={result['n_raw_signals']} → gated={result['n_filtered_signals']} sigs | "
                      f"Overall: {acc['overall_acc']} | "
                      f"Enter: {acc['enter_acc']} ({acc['enter_correct']}/{acc['enter_total']}) | "
                      f"Exit: {acc['exit_acc']} ({acc['exit_correct']}/{acc['exit_total']}) | "
                      f"λ_std={result['model_lam_std']:.4f}")

                results.append({
                    'symbol': symbol,
                    'tf': tf,
                    'bars': len(df),
                    'raw_signals': result['n_raw_signals'],
                    'gated_signals': result['n_filtered_signals'],
                    **acc,
                    'lam_std': result['model_lam_std'],
                })
            except Exception as e:
                print(f"    {tf}: ERROR - {e}")
                import traceback
                traceback.print_exc()

    if results:
        write_results(results)

    return results


def write_results(results):
    out = "/home/ubuntu/nhp_v3_backtest_results.md"
    with open(out, 'w') as f:
        f.write("# NHP v3 Backtest Results — Architecture Bug Fixes\n\n")
        f.write("**v3 fixes:**\n")
        f.write("- extrapolate() uses correct Mei & Eisner formula with c_bar + output gate\n")
        f.write("- IntensityHead: LayerNorm+SiLU instead of Tanh compression\n")
        f.write("- LinearNormAttention preserves magnitude (scales by ||query||)\n")
        f.write("- xavier_uniform gain=1.0 on hidden, 0.1 on output only\n\n")
        f.write("**Method:** Drawdown-based accuracy\n")
        f.write("- ENTER = win if price rises 0.5%+ without breaking below entry\n")
        f.write("- EXIT = win if price drops 0.5%+ without rallying above exit\n\n")
        f.write("| Symbol | TF | Bars | Signals | Overall | Enter | Exit | λ_std |\n")
        f.write("|--------|-----|------|---------|---------|-------|------|-------|\n")
        for r in results:
            f.write(f"| {r['symbol']} | {r['tf']} | {r['bars']} | "
                    f"{r['gated_signals']} | "
                    f"{r['overall_acc']} ({r['overall_correct']}/{r['overall_total']}) | "
                    f"{r['enter_acc']} ({r['enter_correct']}/{r['enter_total']}) | "
                    f"{r['exit_acc']} ({r['exit_correct']}/{r['exit_total']}) | "
                    f"{r['lam_std']:.4f} |\n")
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
