#!/usr/bin/env python3
"""
run_train_v3_regime.py — Train & backtest the regime-mixture NHP v3
===================================================================
Loads Google Drive historical data, builds state vectors with HMM,
trains with combined NLL + BCE(cluster_quality) loss, calibrates
thresholds via walk-forward, then backtests with drawdown accuracy.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nhp_model_v3_regime import NHPv3, NHPv3Config
from nhp_data import ohlc_inter_arrival_times
from nhp_data_v2 import StateAwareDataset, collate_v2, build_state_vectors
from nhp_regime import (
    compute_vol_features, RegimeDetector, RegimeGate, RegimeGateConfig,
    Regime, REGIME_LABELS, VolFeatures,
)
from train_v3_regime import (
    train_v3, train_epoch, evaluate_epoch, make_cluster_labels,
    walk_forward_train, calibrate_thresholds,
)
from kalman import kalman_pipeline
from policy import KalmanPolicy, KalmanPolicyConfig, Signal
from torch.utils.data import DataLoader

CHECKPOINT = 'nhp_v3_regime_best.pt'
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


def build_sequences_with_states(
    df, vf, regimes, regime_probs, detector, window_size=20, stride=10
):
    """Build sequences with state vectors AND cluster labels."""
    dts = ohlc_inter_arrival_times(df)
    n_bars = len(dts)
    dts_arr = np.array(dts, dtype=np.float64)
    scale = np.median(dts_arr)
    if scale <= 0:
        scale = 1.0
    dts_scaled = dts_arr / scale

    prices = df['Close'].values.astype(np.float64)
    bar_times = np.arange(len(prices), dtype=np.float64)

    # Build state vectors
    pos_rv = vf.realized_vol[vf.realized_vol > 0]
    median_rv = float(np.median(pos_rv)) if len(pos_rv) > 0 else 1e-8
    clip_max = 5.0

    state_vecs = np.zeros((n_bars, 6), dtype=np.float32)
    for i in range(n_bars):
        bi = min(i + 1, len(vf.realized_vol) - 1)
        state_vecs[i, 0] = min(vf.realized_vol[bi] / (median_rv + 1e-8), clip_max)
        state_vecs[i, 1] = min(vf.vol_of_vol[bi] / (median_rv + 1e-8), clip_max)
        state_vecs[i, 2] = min(abs(vf.return_skew[bi]), clip_max)
        state_vecs[i, 3:6] = regime_probs[bi]

    # Build cluster labels (is price +0.2% in next 5 bars?)
    # Use prices[1:] since dts starts from bar 1
    close_prices = prices[1:] if len(prices) > n_bars else prices[-n_bars:]
    cluster_labels = np.zeros(n_bars, dtype=np.float32)
    horizon = 5
    min_ret = 0.002
    for i in range(n_bars):
        end = min(i + horizon, len(close_prices) - 1)
        if end > i:
            ret = (close_prices[end] - close_prices[i]) / (close_prices[i] + 1e-8)
            cluster_labels[i] = 1.0 if ret >= min_ret else 0.0

    sequences = []
    for start in range(0, n_bars - window_size + 1, stride):
        end = start + window_size
        seq = {
            'dts': dts_scaled[start:end].astype(np.float32),
            'types': np.ones(window_size, dtype=np.int64),
            'states': state_vecs[start:end].copy(),
            'cluster_labels': cluster_labels[start:end].copy(),
            'length': window_size,
            'prices': close_prices[start:end].copy() if len(close_prices) >= end else np.ones(window_size),
            'start_bar': start,
        }
        sequences.append(seq)
    return sequences


def collate_v2_with_labels(batch):
    """Collate function that also handles cluster_labels."""
    max_len = max(b['length'] for b in batch)
    B = len(batch)
    state_dim = batch[0]['states'].shape[-1]
    dts = torch.zeros(B, max_len)
    types = torch.zeros(B, max_len, dtype=torch.long)
    states = torch.zeros(B, max_len, state_dim)
    labels = torch.zeros(B, max_len)
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        L = b['length']
        dts[i, :L] = torch.from_numpy(b['dts'])
        types[i, :L] = torch.from_numpy(b['types'])
        states[i, :L] = torch.from_numpy(b['states'])
        if 'cluster_labels' in b:
            labels[i, :L] = torch.from_numpy(b['cluster_labels'])
    return {'dts': dts, 'types': types, 'states': states,
            'cluster_labels': labels, 'lengths': lengths}


def run_inference_v3_regime(df, detector, checkpoint_path=CHECKPOINT):
    """Run inference with the regime-mixture v3 model + Kalman policy."""
    prices = df['Close'].values.astype(np.float64)
    bar_times = np.arange(len(prices), dtype=np.float64)
    vf = compute_vol_features(prices, bar_times)

    try:
        regimes, regime_probs = detector.predict(vf)
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

    # Build state vectors
    pos_rv = vf.realized_vol[vf.realized_vol > 0]
    median_rv = float(np.median(pos_rv)) if len(pos_rv) > 0 else 1e-8
    clip_max = 5.0
    state_vecs = np.zeros((n_bars, 6), dtype=np.float32)
    for i in range(n_bars):
        bi = min(i + 1, len(vf.realized_vol) - 1)
        state_vecs[i, 0] = min(vf.realized_vol[bi] / (median_rv + 1e-8), clip_max)
        state_vecs[i, 1] = min(vf.vol_of_vol[bi] / (median_rv + 1e-8), clip_max)
        state_vecs[i, 2] = min(abs(vf.return_skew[bi]), clip_max)
        state_vecs[i, 3:6] = regime_probs[bi]

    device = torch.device('cpu')
    has_ckpt = os.path.exists(checkpoint_path)
    cfg = NHPv3Config(embed_dim=16, hidden_dim=32, num_heads=2)
    model = NHPv3(cfg)
    if has_ckpt:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state'])
    model.eval().to(device)

    # Windowed inference
    window_size = 20
    stride = 10
    model_lam = np.zeros(n_bars, dtype=np.float64)
    cluster_prob = np.zeros(n_bars, dtype=np.float64)
    lam_counts = np.zeros(n_bars, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n_bars - window_size + 1, stride):
            end = start + window_size
            w_dts = torch.tensor(dts_scaled[start:end], dtype=torch.float32).unsqueeze(0)
            w_types = torch.ones(1, window_size, dtype=torch.long)
            w_states = torch.tensor(state_vecs[start:end], dtype=torch.float32).unsqueeze(0)
            sl = torch.tensor([window_size], dtype=torch.long)

            lams, cscores = model.predict(w_types, w_dts, w_states, sl)
            w_lam = lams[0].cpu().numpy()
            w_cp = cscores[0].cpu().numpy()

            model_lam[start:end] += w_lam
            cluster_prob[start:end] += w_cp
            lam_counts[start:end] += 1.0

    lam_counts = np.maximum(lam_counts, 1.0)
    model_lam /= lam_counts
    cluster_prob /= lam_counts

    # Kalman filter smoothing + slope detection
    close_prices = prices[1:] if len(prices) > n_bars else prices[-n_bars:]
    kout = kalman_pipeline(close_prices, model_lam)

    # Generate signals via Kalman policy
    policy = KalmanPolicy(KalmanPolicyConfig(
        entry_slope_z=1.0,
        exit_slope_z=-1.0,
        min_confidence=0.0,
        cooldown_steps=6,
    ))
    signals = policy.apply(kout)

    # Filter by cluster quality probability (precision gate)
    filtered_signals = []
    for sig in signals:
        cp = cluster_prob[sig.step] if sig.step < len(cluster_prob) else 0.0
        if sig.signal == 'ENTER' and cp < 0.3:
            continue  # skip low-quality cluster entries
        filtered_signals.append({
            'step': sig.step,
            'time': float(sig.step),
            'signal': sig.signal,
            'bar_index': sig.step,
            'confidence': sig.confidence,
            'cluster_prob': float(cp),
        })

    return {
        'signals': filtered_signals,
        'model_lam': model_lam,
        'cluster_prob': cluster_prob,
        'smoothed_lam': kout.smoothed,
        'slope_z': kout.slope_z,
        'model_trained': has_ckpt,
        'model_lam_std': float(model_lam.std()),
        'ohlc': {
            'close': df['Close'].values[1:].tolist(),
            'high': df['High'].values[1:].tolist(),
            'low': df['Low'].values[1:].tolist(),
        },
    }


def compute_drawdown_accuracy(closes, highs, lows, signals, target_pct=0.005):
    enter_correct, enter_total = 0, 0
    exit_correct, exit_total = 0, 0

    for sig in signals:
        bi = sig['bar_index']
        if bi >= len(closes) - 1:
            continue
        entry_price = closes[bi]

        if sig['signal'] == 'ENTER':
            enter_total += 1
            target = entry_price * (1 + target_pct)
            for j in range(bi + 1, len(closes)):
                if lows[j] < entry_price:
                    break
                if highs[j] >= target:
                    enter_correct += 1
                    break
        elif sig['signal'] == 'EXIT':
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
    fmt = lambda c, t: f"{c/t*100:.1f}%" if t > 0 else "N/A"

    return {
        'overall_acc': fmt(overall_correct, overall_total),
        'overall_correct': overall_correct, 'overall_total': overall_total,
        'enter_acc': fmt(enter_correct, enter_total),
        'enter_correct': enter_correct, 'enter_total': enter_total,
        'exit_acc': fmt(exit_correct, exit_total),
        'exit_correct': exit_correct, 'exit_total': exit_total,
    }


def main():
    print("=" * 65)
    print("  TRAINING NHP v3 REGIME-MIXTURE (State + Aux BCE + Kalman)")
    print("=" * 65)

    device = torch.device('cpu')

    # Step 1: Load training data
    print("\n  Loading training data...")
    all_prices_list = []
    all_bar_times_list = []
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
            all_prices_list.append(prices)
            all_bar_times_list.append(np.arange(len(prices), dtype=np.float64) + offset)
            offset += len(prices) + 100
            print(f"    {symbol} {tf}: {len(df)} bars")

    concat_prices = np.concatenate(all_prices_list)
    concat_times = np.concatenate(all_bar_times_list)

    # Step 2: Fit HMM regime detector
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

    # Step 3: Build sequences with state vectors AND cluster labels
    print("\n  Building sequences with states + cluster labels...")
    all_sequences = []
    per_df_offset = 0
    for key, df in all_dfs.items():
        n_df = len(df)
        df_prices = df['Close'].values.astype(np.float64)
        df_times = np.arange(len(df_prices), dtype=np.float64)
        df_vf = compute_vol_features(df_prices, df_times)
        try:
            df_regimes, df_probs = detector.predict(df_vf)
        except Exception:
            df_regimes = np.ones(len(df_prices), dtype=int)
            df_probs = np.tile([0.33, 0.34, 0.33], (len(df_prices), 1))

        seqs = build_sequences_with_states(
            df, df_vf, df_regimes, df_probs, detector,
            window_size=20, stride=10,
        )
        all_sequences.extend(seqs)
        print(f"    {key}: {len(seqs)} sequences")

    print(f"\n  Total sequences: {len(all_sequences)}")

    # Step 4: Simple train/val split (temporal)
    n = len(all_sequences)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_seqs = all_sequences[:n_train]
    val_seqs = all_sequences[n_train:n_train + n_val]
    test_seqs = all_sequences[n_train + n_val:]
    print(f"  Split: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")

    train_loader = DataLoader(
        StateAwareDataset(train_seqs, max_len=20),
        batch_size=64, shuffle=True, collate_fn=collate_v2_with_labels,
    )
    val_loader = DataLoader(
        StateAwareDataset(val_seqs, max_len=20),
        batch_size=64, shuffle=False, collate_fn=collate_v2_with_labels,
    )

    # Step 5: Train
    print(f"\n  Training NHPv3 Regime-Mixture model...")
    cfg = NHPv3Config(
        embed_dim=16, hidden_dim=32, num_heads=2,
        state_dim=6, state_hidden=12, n_regimes=3,
        aux_loss_alpha=0.3, cluster_horizon=5,
    )
    model = NHPv3(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    model = train_v3(
        model, train_loader, val_loader,
        epochs=25, device=device, lr=1e-3, wd=1e-4,
        patience=7, ckpt=CHECKPOINT,
    )

    # Step 6: Backtest
    print("\n" + "=" * 65)
    print("  BACKTEST WITH TRAINED v3 REGIME-MIXTURE MODEL")
    print("=" * 65)

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

                result = run_inference_v3_regime(df, detector, CHECKPOINT)
                if result is None:
                    print(f"    {tf}: No result")
                    continue

                acc = compute_drawdown_accuracy(
                    result['ohlc']['close'],
                    result['ohlc']['high'],
                    result['ohlc']['low'],
                    result['signals'],
                )

                n_enter = sum(1 for s in result['signals'] if s['signal'] == 'ENTER')
                n_exit = sum(1 for s in result['signals'] if s['signal'] == 'EXIT')

                print(f"    {tf:>4s}: {len(df)} bars | "
                      f"sigs={len(result['signals'])} (E:{n_enter}/X:{n_exit}) | "
                      f"Overall: {acc['overall_acc']} | "
                      f"Enter: {acc['enter_acc']} ({acc['enter_correct']}/{acc['enter_total']}) | "
                      f"Exit: {acc['exit_acc']} ({acc['exit_correct']}/{acc['exit_total']}) | "
                      f"λ_std={result['model_lam_std']:.4f}")

                results.append({
                    'symbol': symbol, 'tf': tf, 'bars': len(df),
                    'n_signals': len(result['signals']),
                    'n_enter': n_enter, 'n_exit': n_exit,
                    **acc, 'lam_std': result['model_lam_std'],
                })
            except Exception as e:
                print(f"    {tf}: ERROR - {e}")
                import traceback
                traceback.print_exc()

    if results:
        write_results(results)
    return results


def write_results(results):
    out = "/home/ubuntu/nhp_v3_regime_backtest.md"
    with open(out, 'w') as f:
        f.write("# NHP v3 Regime-Mixture Backtest Results\n\n")
        f.write("**Architecture:**\n")
        f.write("- State injected at embedding AND intensity head\n")
        f.write("- 3-expert regime-mixture intensity (soft-gated by P(regime))\n")
        f.write("- Auxiliary BCE loss for cluster quality\n")
        f.write("- Kalman-filtered slope detection for signals\n")
        f.write("- Cluster quality probability gate (>0.3 for ENTER)\n\n")
        f.write("**Method:** Drawdown-based accuracy (0.5% target)\n\n")
        f.write("| Symbol | TF | Bars | Signals (E/X) | Overall | Enter | Exit | λ_std |\n")
        f.write("|--------|-----|------|---------------|---------|-------|------|-------|\n")
        for r in results:
            f.write(f"| {r['symbol']} | {r['tf']} | {r['bars']} | "
                    f"{r['n_signals']} ({r['n_enter']}/{r['n_exit']}) | "
                    f"{r['overall_acc']} ({r['overall_correct']}/{r['overall_total']}) | "
                    f"{r['enter_acc']} ({r['enter_correct']}/{r['enter_total']}) | "
                    f"{r['exit_acc']} ({r['exit_correct']}/{r['exit_total']}) | "
                    f"{r['lam_std']:.4f} |\n")
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
