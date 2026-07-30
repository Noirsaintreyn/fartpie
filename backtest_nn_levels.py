"""
Backtest the production Causal-CNN + LSTM + MLP level detector
(backend.LevelDetectionNet) on real NQ/ES 1H/4H data, with a proper
chronological train/holdout split (PIT-safe: the model is trained only on
the first `--train-frac` of each file's history and evaluated only on
strictly later bars it never saw during training).

Reuses backend.py's actual class and labeling function (not a
reimplementation) so this measures the real production model, and reuses
backtest_levels.py's evaluate_level/compute_atr so results are directly
comparable to the other 9 methods already backtested there.
"""
import argparse
import contextlib
import io
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backtest_levels import load_csv, infer_timeframe, compute_atr, evaluate_level

with contextlib.redirect_stdout(io.StringIO()):
    import backend


def build_training_samples(opens, highs, lows, closes, volumes, lookback, forward_window):
    X_all, y_all = [], []
    for i in range(lookback, len(closes) - forward_window):
        sl = slice(i - lookback, i)
        ohlcv = np.stack([opens[sl], highs[sl], lows[sl], closes[sl], volumes[sl]], axis=-1)
        ch_mean = ohlcv.mean(axis=0, keepdims=True)
        ch_std = ohlcv.std(axis=0, keepdims=True) + 1e-9
        ohlcv_norm = (ohlcv - ch_mean) / ch_std

        labels = np.zeros(lookback, dtype=np.float32)
        for j in range(lookback):
            global_j = (i - lookback) + j
            labels[j] = backend._label_future_reaction(closes, highs, lows, global_j, forward_window=forward_window)

        X_all.append(ohlcv_norm)
        y_all.append(labels)
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)


def train_nn(opens, highs, lows, closes, volumes, lookback, forward_window, epochs, batch_size, verbose=False):
    X_all, y_all = build_training_samples(opens, highs, lows, closes, volumes, lookback, forward_window)
    pos_rate = float(np.mean(y_all)) if len(y_all) else 0.0
    if verbose:
        print(f"  {len(X_all)} training samples, positive label rate {pos_rate:.2%}")

    split_idx = int(len(X_all) * 0.8)  # chronological internal train/val split, same as backend.py's own trainer
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_val, y_val = X_all[split_idx:], y_all[split_idx:]

    X_train_t, y_train_t = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val_t, y_val_t = torch.FloatTensor(X_val), torch.FloatTensor(y_val)

    model = backend.LevelDetectionNet(lookback=lookback, in_channels=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    pos_weight_val = max(1.0, (1 - pos_rate) / (pos_rate + 1e-9))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        for bs in range(0, len(X_train_t), batch_size):
            be = min(bs + batch_size, len(X_train_t))
            xb, yb = X_train_t[bs:be], y_train_t[bs:be]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs} val_loss={val_loss:.4f}")
        if patience_counter >= 12:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def nn_levels(model, win_o, win_h, win_l, win_c, win_v, threshold=0.5):
    lookback = len(win_c)
    ohlcv = np.stack([win_o, win_h, win_l, win_c, win_v], axis=-1)
    ch_mean = ohlcv.mean(axis=0, keepdims=True)
    ch_std = ohlcv.std(axis=0, keepdims=True) + 1e-9
    ohlcv_norm = (ohlcv - ch_mean) / ch_std
    x = torch.FloatTensor(ohlcv_norm).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].numpy()

    idx = np.where(probs > threshold)[0]
    levels = []
    for i in idx:
        levels.append({'price': float(win_c[i]), 'strength': float(probs[i]), 'category': 'Neural-Network'})
    levels.sort(key=lambda l: l['price'])
    merged = []
    for lv in levels:
        if merged and abs(lv['price'] - merged[-1]['price']) / merged[-1]['price'] < 0.0015:
            if lv['strength'] > merged[-1]['strength']:
                merged[-1] = lv
        else:
            merged.append(lv)
    return merged


def run_file(path, lookback, horizon, step, train_frac, forward_window, nn_epochs, nn_threshold,
             bounce_atr_mult, break_atr_mult, break_confirm_bars, reaction_bars, recovery_bars,
             verbose=False):
    instrument = re.sub(r'^\d+[a-zA-Z]+_', '', os.path.splitext(os.path.basename(path))[0])
    timeframe = infer_timeframe(os.path.basename(path))
    print(f"Loading {path} (instrument={instrument}, timeframe={timeframe})...")
    df = load_csv(path)
    n = len(df)
    opens = df['open'].values.astype(np.float32)
    highs = df['high'].values.astype(np.float32)
    lows = df['low'].values.astype(np.float32)
    closes = df['close'].values.astype(np.float32)
    volumes = df['volume'].values.astype(np.float32)

    train_end = int(n * train_frac)
    print(f"  {n} bars total, training NN on first {train_end} "
          f"({df['datetime'].iloc[0]} -> {df['datetime'].iloc[train_end]})")

    t0 = time.time()
    model = train_nn(opens[:train_end], highs[:train_end], lows[:train_end], closes[:train_end],
                      volumes[:train_end], lookback, forward_window, nn_epochs, batch_size=64, verbose=verbose)
    print(f"  trained in {time.time()-t0:.1f}s")

    starts = [t for t in range(lookback, n - horizon, step) if t > train_end]
    print(f"  {len(starts)} holdout windows to evaluate")

    rows = []
    t0 = time.time()
    for wi, t in enumerate(starts):
        win_o = opens[t - lookback:t]
        win_h = highs[t - lookback:t]
        win_l = lows[t - lookback:t]
        win_c = closes[t - lookback:t]
        win_v = volumes[t - lookback:t]
        current_price = closes[t - 1]
        atr = compute_atr(win_h, win_l, win_c)
        if atr <= 0:
            continue

        fwd_h = highs[t:t + horizon]
        fwd_l = lows[t:t + horizon]
        fwd_c = closes[t:t + horizon]

        levels = nn_levels(model, win_o, win_h, win_l, win_c, win_v, threshold=nn_threshold)
        for lvl in levels:
            price = lvl['price']
            if price <= 0:
                continue
            side = 'support' if price < current_price else 'resistance'
            outcome = evaluate_level(price, side, fwd_h, fwd_l, fwd_c, atr,
                                      bounce_atr_mult, break_atr_mult, break_confirm_bars,
                                      reaction_bars, recovery_bars)
            rows.append({
                'instrument': instrument, 'timeframe': timeframe, 'method': 'Neural-Network',
                't': t, 'price': price, 'side': side,
                'touched': outcome is not None,
                'bounced': outcome['bounced'] if outcome else None,
                'broken': outcome['broken'] if outcome else None,
                'hold_duration': outcome['hold_duration'] if outcome else None,
                'recovered': outcome['recovered'] if outcome else None,
            })
        if verbose and wi % 50 == 0:
            print(f"  window {wi}/{len(starts)} ({time.time()-t0:.1f}s elapsed)")

    return pd.DataFrame(rows)


def summarize(results):
    def agg(g):
        touched = g[g['touched']]
        n_levels = len(g)
        n_touched = len(touched)
        n_bounced = touched['bounced'].sum() if n_touched else 0
        n_broken = touched['broken'].sum() if n_touched else 0
        recovered_subset = touched[touched['broken'] == True]  # noqa: E712
        n_recovered = recovered_subset['recovered'].sum() if len(recovered_subset) else 0
        return pd.Series({
            'n_levels_generated': n_levels, 'n_touched': n_touched,
            'touch_rate': n_touched / n_levels if n_levels else np.nan,
            'support_accuracy': n_bounced / n_touched if n_touched else np.nan,
            'break_rate': n_broken / n_touched if n_touched else np.nan,
            'false_breakout_recovery': n_recovered / n_broken if n_broken else np.nan,
            'avg_hold_duration_bars': touched['hold_duration'].mean() if n_touched else np.nan,
        })
    per_file = results.groupby(['instrument', 'timeframe']).apply(agg).reset_index()
    overall = agg(results)
    return per_file, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=100)
    ap.add_argument('--horizon', type=int, default=40)
    ap.add_argument('--step', type=int, default=20)
    ap.add_argument('--train-frac', type=float, default=0.7)
    ap.add_argument('--forward-window', type=int, default=20)
    ap.add_argument('--nn-epochs', type=int, default=50)
    ap.add_argument('--nn-threshold', type=float, default=0.5)
    ap.add_argument('--bounce-atr-mult', type=float, default=1.0)
    ap.add_argument('--break-atr-mult', type=float, default=0.5)
    ap.add_argument('--break-confirm-bars', type=int, default=2)
    ap.add_argument('--reaction-bars', type=int, default=10)
    ap.add_argument('--recovery-bars', type=int, default=10)
    ap.add_argument('--out', default='backtest_nn_results.csv')
    ap.add_argument('--summary-out', default='backtest_nn_summary.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_results = []
    for path in args.files:
        res = run_file(path, args.lookback, args.horizon, args.step, args.train_frac,
                        args.forward_window, args.nn_epochs, args.nn_threshold,
                        args.bounce_atr_mult, args.break_atr_mult, args.break_confirm_bars,
                        args.reaction_bars, args.recovery_bars, verbose=args.verbose)
        print(f"  -> {len(res)} candidate levels evaluated")
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(args.out, index=False)
    per_file, overall = summarize(results)
    per_file.to_csv(args.summary_out, index=False)

    pd.set_option('display.width', 160)
    print("\n=== Per instrument/timeframe (holdout period only) ===")
    print(per_file.round(3).to_string(index=False))
    print("\n=== Overall ===")
    print(overall.round(3))


if __name__ == '__main__':
    main()
