"""
Backtest the "theoretical HOD/LOD" shown in /api/lstm-forecast's
theoretical_bounds field:
  hod_intraday = current_price + 1.5 * sigma_price
  lod_intraday = current_price - 1.5 * sigma_price
  hod_premarket = current_price + 2.0 * sigma_price
  lod_premarket = current_price - 2.0 * sigma_price
where sigma_price comes from compute_session_volatility (Garman-Klass based).

This can't be backtested the way it's computed live (current_price re-anchors
continuously as new bars arrive, which is why it "prints new levels too
often"). Instead: anchor the band ONCE per calendar day, at that day's
opening price, using sigma_price computed only from bars strictly before
that day (PIT-safe). Then check whether the REST of that day's actual
range breaches the band. This is a standard volatility-containment test,
and it has a real reference point to compare against: for a roughly normal
return distribution, a one-sided 1.5-sigma threshold should be breached
about 100*(1-Phi(1.5)) = 6.68% of the time, and 2.0-sigma about 2.28%. If
the measured breach rate is far from that, sigma_price is miscalibrated
(too narrow = breached too often, too wide = breached too rarely / band is
uselessly loose).
"""
import argparse
import contextlib
import io
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_levels import load_csv, infer_timeframe

with contextlib.redirect_stdout(io.StringIO()):
    import backend

EXPECTED_BREACH_1_5_SIGMA = 1 - norm.cdf(1.5)  # ~6.68%
EXPECTED_BREACH_2_0_SIGMA = 1 - norm.cdf(2.0)  # ~2.28%


def run_file(path, vol_window=60, min_history_bars=100, verbose=False):
    instrument = re.sub(r'^\d+[a-zA-Z]+_', '', os.path.splitext(os.path.basename(path))[0])
    timeframe = infer_timeframe(os.path.basename(path))
    print(f"Loading {path} (instrument={instrument}, timeframe={timeframe})...")
    df = load_csv(path)
    dates = df['datetime'].dt.date.values
    unique_days = sorted(set(dates))

    rows = []
    for day in unique_days:
        day_mask = dates == day
        day_idx = np.where(day_mask)[0]
        if len(day_idx) == 0:
            continue
        first_bar_idx = day_idx[0]
        if first_bar_idx < min_history_bars:
            continue  # not enough PIT-safe history for this day yet

        # PIT-safe: only bars strictly before this day's first bar
        hist_slice = df.iloc[max(0, first_bar_idx - vol_window):first_bar_idx]
        if len(hist_slice) < 20:
            continue
        hist_df = pd.DataFrame({
            'Open': hist_slice['open'].values, 'High': hist_slice['high'].values,
            'Low': hist_slice['low'].values, 'Close': hist_slice['close'].values,
        })
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                vol_result = backend.compute_session_volatility(hist_df, window=vol_window)
            sigma_price = vol_result.get('sigma_price', 0.0)
        except Exception as e:
            if verbose:
                print(f"  vol calc failed for {day}: {e}")
            continue
        if sigma_price <= 0 or not np.isfinite(sigma_price):
            continue

        anchor_price = df['open'].values[first_bar_idx]  # day's opening price - the fixed anchor
        hod_id = anchor_price + 1.5 * sigma_price
        lod_id = anchor_price - 1.5 * sigma_price
        hod_pm = anchor_price + 2.0 * sigma_price
        lod_pm = anchor_price - 2.0 * sigma_price

        day_high = df['high'].values[day_idx].max()
        day_low = df['low'].values[day_idx].min()

        rows.append({
            'instrument': instrument, 'timeframe': timeframe, 'date': day,
            'anchor_price': anchor_price, 'sigma_price': sigma_price,
            'day_high': day_high, 'day_low': day_low,
            'breach_hod_id': day_high > hod_id, 'breach_lod_id': day_low < lod_id,
            'breach_hod_pm': day_high > hod_pm, 'breach_lod_pm': day_low < lod_pm,
        })

    return pd.DataFrame(rows)


def summarize(df):
    def agg(g):
        n = len(g)
        return pd.Series({
            'n_days': n,
            'breach_rate_hod_1.5sigma': g['breach_hod_id'].mean(),
            'breach_rate_lod_1.5sigma': g['breach_lod_id'].mean(),
            'breach_rate_either_1.5sigma': (g['breach_hod_id'] | g['breach_lod_id']).mean(),
            'breach_rate_hod_2.0sigma': g['breach_hod_pm'].mean(),
            'breach_rate_lod_2.0sigma': g['breach_lod_pm'].mean(),
            'breach_rate_either_2.0sigma': (g['breach_hod_pm'] | g['breach_lod_pm']).mean(),
        })
    per_file = df.groupby(['instrument', 'timeframe']).apply(agg).reset_index()
    overall = agg(df)
    return per_file, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--vol-window', type=int, default=60)
    ap.add_argument('--out', default='backtest_theoretical_hodlod_results.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_results = []
    for path in args.files:
        res = run_file(path, vol_window=args.vol_window, verbose=args.verbose)
        print(f"  -> {len(res)} trading days evaluated")
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(args.out, index=False)
    per_file, overall = summarize(results)

    pd.set_option('display.width', 160)
    print("\n=== Per instrument/timeframe ===")
    print(per_file.round(4).to_string(index=False))

    print(f"\n=== Overall ({len(results)} total day-observations across all files) ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n=== Calibration check ===")
    print(f"  Expected one-sided breach rate at 1.5 sigma (if returns were normal): {EXPECTED_BREACH_1_5_SIGMA:.4f} ({EXPECTED_BREACH_1_5_SIGMA*100:.2f}%)")
    print(f"  Expected one-sided breach rate at 2.0 sigma (if returns were normal): {EXPECTED_BREACH_2_0_SIGMA:.4f} ({EXPECTED_BREACH_2_0_SIGMA*100:.2f}%)")
    print(f"  Measured HOD breach @ 1.5sigma: {overall['breach_rate_hod_1.5sigma']:.4f}  vs expected {EXPECTED_BREACH_1_5_SIGMA:.4f}")
    print(f"  Measured LOD breach @ 1.5sigma: {overall['breach_rate_lod_1.5sigma']:.4f}  vs expected {EXPECTED_BREACH_1_5_SIGMA:.4f}")
    print(f"  Measured HOD breach @ 2.0sigma: {overall['breach_rate_hod_2.0sigma']:.4f}  vs expected {EXPECTED_BREACH_2_0_SIGMA:.4f}")
    print(f"  Measured LOD breach @ 2.0sigma: {overall['breach_rate_lod_2.0sigma']:.4f}  vs expected {EXPECTED_BREACH_2_0_SIGMA:.4f}")


if __name__ == '__main__':
    main()
