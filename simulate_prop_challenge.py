"""
Entry/SL/TP trade simulator + prop-firm challenge pass-rate backtest, built
on top of ALL 7 validated ML-filtered level types (score_and_filter_levels
in backend.py - walk-forward validated: GMM, HDBSCAN, Isolation-Forest,
KDE, MeanShift, OPTICS, and TDA all improved in 16/16 folds and cleared
Bonferroni correction once filtered, 0.480-0.497 filtered accuracy).
Originally only used GMM+TDA - expanded after realizing the trade
simulator was never updated when the production filter was.

Entry: price touches the highest-scoring filtered level (any of the 7
types) -> long if support (level below price), short if resistance
(level above price).
Stop:  break_atr_mult * ATR beyond the level (0.5 ATR) - hard intrabar
       stop using highs/lows, not the softer "2 consecutive closes"
       confirmation used to score level VALIDITY - a real stop has to be
       a hard exit, not wait for confirmation while the loss grows.
Target: bounce_atr_mult * ATR from the level (1.0 ATR), same logic.
Only one open position at a time.

Position sizing: account-size- and win-rate-aware, with a hard stop-distance
cap in points, replacing the earlier flat-$-budget version.
  win_rate_estimate = ml_filter_score (the filter's own predicted
    P(bounce) for this specific candidate - it IS a probability estimate,
    reused directly rather than inventing a second one)
  risk_budget_usd = drawdown_limit * win_rate_estimate
    e.g. drawdown_limit=$2000, win_rate_estimate=0.25 -> cap $500 (this is
    the literal example given: "if i have a 25% winrate, i cant risk more
    than 500"). This is a hard ceiling - nothing below scales it UP past
    this, only down.
  stop_cap_points = 35 if (vol regime expanding AND win_rate_estimate is
    high) else 20 - "nothing bigger than a 20 point stop, 35 in volatile +
    high prob scenarios." "Volatile" = gjr_vol_regime_ratio (GJR-GARCH
    forecast vol / Garman-Klass realized vol) above --vol-regime-threshold;
    "high prob" = win_rate_estimate above --high-prob-threshold.
  stop_distance_points = min(break_atr_mult * atr, stop_cap_points) - the
    ATR-derived stop, hard-capped. NOTE this is a deliberate departure from
    the earlier stance of never touching the validated stop distance: the
    user explicitly wants a hard risk ceiling here, so it's implemented,
    but it means realized win rate can run a bit below ml_filter_score
    whenever the cap binds (a tighter-than-validated stop gets touched by
    noise slightly more often) - flagged, not hidden.
  target_distance_points = bounce_atr_mult * atr (unchanged, uncapped)
  realized_rr = target_distance_points / stop_distance_points
  rr_factor = clip(realized_rr / profit_target_ratio, 0.5, 1.0) - a
    HAIRCUT only (capped at 1.0, never boosts size past risk_budget_usd)
    for setups whose actual reward:risk (after the stop cap) falls short
    of the account's own target ratio (profit_target_ratio = account_target
    / drawdown_limit, default 3000/2000 = 1.5 - "account size as context")
  risk_per_contract_usd = stop_distance_points * point_value
  contracts = clip(round(risk_budget_usd * rr_factor / risk_per_contract_usd),
                    min_contracts, max_contracts)
This keeps "smaller stop -> more lenient sizing, bigger stop -> more
conservative" (risk_per_contract_usd is directly in the denominator, and
big stops get capped rather than left to float with a wide-ATR regime),
while win-rate and the 1.5 target ratio are the two levers that set the
ceiling everything else operates under, per instruction.

VWAP bias/stretch, Hurst, and HMM regime-change are NOT separate
multipliers here - they're additional features feeding the SAME
ml_filter_score/win_rate_estimate above (see backtest_ml_filter_v2_events.py)
so "confidence" stays one rigorously-fit probability instead of a stack of
hand-picked weights. Once that v2 model is walk-forward validated, its
pred_proba replaces ml_filter_score with no other change to this formula.

Two things this produces:
  1. A trade list (backtest_prop_trades.csv) - every simulated trade with
     entry/exit/P&L/contracts, for direct inspection.
  2. A prop-firm challenge pass-rate: simulate MANY independent challenge
     attempts starting at different points across the 12yr history (not
     just one), so "does this beat the prop firm" is a measured
     probability, not a single anecdote.

Rules simulated (state explicitly, since these are an interpretation of
what was described, not fetched from an actual firm's rulebook):
  Eval:    reach +$3,000 cumulative P&L over >= 2 trading days, never
           breaching -$2,000 drawdown from the starting balance (static).
  Funded:  >=5 trading days with >=+$150 P&L each, same -$2,000 static
           drawdown limit, timed from the end of the eval phase.
"""
import argparse
import contextlib
import io
import os
import re
import time

import numpy as np
import pandas as pd

from backtest_levels import load_csv, infer_timeframe, compute_atr
from backtest_ml_filter_v2_events import fit_gjr_garch_forecast

with contextlib.redirect_stdout(io.StringIO()):
    import backend

POINT_VALUE = {'NQ': 2.0, 'ES': 5.0}  # micros (MNQ, MES) - minis blew up on price-scale drift, see commit history


def generate_trades(path, lookback=150, step=5,
                     bounce_atr_mult=1.0, break_atr_mult=0.5,
                     max_hold_bars=200, verbose=False, start_date=None, end_date=None,
                     drawdown_limit=2000.0, account_target=3000.0,
                     min_contracts=1, max_contracts=10,
                     stop_cap_points=20.0, stop_cap_points_volatile=35.0,
                     vol_regime_threshold=1.15, high_prob_threshold=0.45):
    """Walk forward, generate ML-filtered GMM/TDA levels, and simulate hard
    intrabar-stop/target trades against them. One trade open at a time."""
    instrument = re.sub(r'^\d+[a-zA-Z]+_', '', os.path.splitext(os.path.basename(path))[0])
    timeframe = infer_timeframe(os.path.basename(path))
    print(f"Loading {path} (instrument={instrument}, timeframe={timeframe})...")
    df = load_csv(path)
    n = len(df)
    opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    volumes = df['volume'].values
    datetimes = df['datetime'].values
    point_value = POINT_VALUE.get(instrument, 20.0)
    profit_target_ratio = account_target / drawdown_limit

    trades = []
    open_trade = None  # dict with entry info while a position is live
    starts = list(range(lookback, n, step))
    start_ts = np.datetime64(start_date) if start_date else None
    end_ts = np.datetime64(end_date) if end_date else None

    t0 = time.time()
    for wi, t in enumerate(starts):
        if verbose and wi % 200 == 0:
            print(f"  window {wi}/{len(starts)} ({time.time()-t0:.1f}s elapsed, {len(trades)} trades so far)")
        # manage an already-open trade first: check if this bar's H/L hits
        # stop or target (hard intrabar exit, checked in time order).
        # Scans only the NEW bars since the last check (last_checked_idx),
        # not from entry every time - the old version re-scanned the whole
        # trade history on every outer iteration while a trade stayed open,
        # O(max_hold_bars^2/step) redundant work per trade instead of
        # O(max_hold_bars).
        if open_trade is not None:
            scan_start = open_trade['last_checked_idx'] + 1
            scan_end = min(t, open_trade['entry_idx'] + max_hold_bars) + 1
            for i in range(scan_start, scan_end):
                if i >= n:
                    break
                h, l = highs[i], lows[i]
                side = open_trade['side']
                if side == 'long':
                    if l <= open_trade['stop_price']:
                        _close_trade(open_trade, i, open_trade['stop_price'], 'stop', datetimes, point_value, trades)
                        open_trade = None
                        break
                    if h >= open_trade['target_price']:
                        _close_trade(open_trade, i, open_trade['target_price'], 'target', datetimes, point_value, trades)
                        open_trade = None
                        break
                else:
                    if h >= open_trade['stop_price']:
                        _close_trade(open_trade, i, open_trade['stop_price'], 'stop', datetimes, point_value, trades)
                        open_trade = None
                        break
                    if l <= open_trade['target_price']:
                        _close_trade(open_trade, i, open_trade['target_price'], 'target', datetimes, point_value, trades)
                        open_trade = None
                        break
                if open_trade is not None:
                    open_trade['last_checked_idx'] = i
            else:
                if open_trade is not None and t >= open_trade['entry_idx'] + max_hold_bars:
                    exit_idx = min(open_trade['entry_idx'] + max_hold_bars, n - 1)
                    _close_trade(open_trade, exit_idx, closes[exit_idx], 'timeout', datetimes, point_value, trades)
                    open_trade = None

        if open_trade is not None:
            continue  # still in a trade, don't look for new entries

        # entries restricted to the target date range - lookback context
        # before it is still used (PIT-safe), we're just gating NEW entries
        # to this window. A trade opened inside the window is still allowed
        # to run to its natural exit even if that lands after end_date.
        if start_ts is not None and datetimes[t - 1] < start_ts:
            continue
        if end_ts is not None and datetimes[t - 1] > end_ts:
            continue

        win_o = opens[t - lookback:t]
        win_h, win_l, win_c = highs[t - lookback:t], lows[t - lookback:t], closes[t - lookback:t]
        win_v, win_dt = volumes[t - lookback:t], datetimes[t - lookback:t]
        current_price = closes[t - 1]
        atr = compute_atr(win_h, win_l, win_c)
        if atr <= 0:
            continue

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                levels_by_category = {
                    'GMM': backend.calculate_gmm_levels(win_h, win_l, win_c),
                    'TDA': backend.persistent_homology_levels(win_h, win_l, win_c, max_levels=8),
                    'HDBSCAN': backend.calculate_hdbscan_levels(win_h, win_l, win_c, timeframe=timeframe),
                    'OPTICS': backend.optics_multi_density_levels(win_h, win_l, win_c),
                    'KDE': backend.kde_based_levels(win_h, win_l, win_c),
                    'MeanShift': backend.calculate_meanshift_levels(win_h, win_l, win_c),
                    'Isolation-Forest': backend.find_pivot_anomalies(win_h, win_l, win_c),
                }
                filtered = backend.score_and_filter_levels(
                    levels_by_category, win_h, win_l, win_c, win_v, current_price, timestamps=win_dt,
                )
        except Exception as e:
            if verbose:
                print(f"  filter failed at t={t}: {e}")
            continue
        if not filtered:
            continue

        # take the single highest-scoring filtered level as this window's
        # entry candidate, only if price is actually near it (within 0.25 ATR)
        # i.e. treat "touch" as the entry trigger, same spirit as the backtest
        best = max(filtered, key=lambda l: l['ml_filter_score'])
        price = best['price']
        if abs(current_price - price) > 0.25 * atr:
            continue  # not close enough to be a real touch/entry right now

        side = 'long' if price < current_price else 'short'
        win_rate_estimate = best['ml_filter_score']

        # vol regime: GJR-GARCH forecast vs Garman-Klass realized, only
        # computed once a real touch is confirmed (keeps the extra GARCH
        # fit off the vast majority of windows that never trigger an entry)
        returns_pct = np.diff(np.log(win_c)) * 100
        with contextlib.redirect_stdout(io.StringIO()):
            gjr_vol_pct = fit_gjr_garch_forecast(returns_pct)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                gk_vol_pct = backend.garman_klass_daily_volatility(win_o, win_h, win_l, win_c) * 100
        except Exception:
            gk_vol_pct = None
        gjr_vol_regime_ratio = (gjr_vol_pct / (gk_vol_pct + 1e-9)) if (gjr_vol_pct and gk_vol_pct) else 1.0

        is_volatile = gjr_vol_regime_ratio > vol_regime_threshold
        is_high_prob = win_rate_estimate >= high_prob_threshold
        stop_cap = stop_cap_points_volatile if (is_volatile and is_high_prob) else stop_cap_points

        stop_distance_points = min(break_atr_mult * atr, stop_cap)
        target_distance_points = bounce_atr_mult * atr
        realized_rr = target_distance_points / stop_distance_points
        rr_factor = float(np.clip(realized_rr / profit_target_ratio, 0.5, 1.0))

        if side == 'long':
            stop_price = price - stop_distance_points
            target_price = price + target_distance_points
        else:
            stop_price = price + stop_distance_points
            target_price = price - target_distance_points

        # win-rate-derived risk CEILING (not the final size): "if i have a
        # 25% winrate, i can't risk more than 500" == drawdown_limit(2000)
        # * win_rate(0.25). rr_factor only ever haircuts this, never boosts
        # past it. The actual contract count used in a given simulated
        # attempt is decided downstream in simulate_prop_challenge_pass_rate.py,
        # as MIN(needed to hit the remaining profit target within the pacing
        # horizon, this ceiling) - that decision needs to know the attempt's
        # running balance/remaining days, which don't exist at this
        # stateless per-window generation stage. This script only precomputes
        # the ceiling and the raw price/point info the downstream sizing needs.
        risk_budget_usd = drawdown_limit * win_rate_estimate
        risk_per_contract_usd = stop_distance_points * point_value
        contracts_max = int(np.clip(
            round(risk_budget_usd * rr_factor / risk_per_contract_usd),
            min_contracts, max_contracts))

        open_trade = {
            'instrument': instrument, 'timeframe': timeframe, 'level_type': best['category'],
            'ml_filter_score': win_rate_estimate, 'gjr_vol_regime_ratio': gjr_vol_regime_ratio,
            'stop_cap_points': stop_cap, 'realized_rr': realized_rr, 'rr_factor': rr_factor,
            'risk_budget_usd': risk_budget_usd, 'contracts_max': contracts_max,
            'point_value': point_value, 'side': side,
            'entry_idx': t, 'entry_price': current_price, 'level_price': price,
            'stop_price': stop_price, 'target_price': target_price,
            'last_checked_idx': t,
        }

    return pd.DataFrame(trades)


def _close_trade(open_trade, exit_idx, exit_price, reason, datetimes, point_value, trades):
    entry_price = open_trade['entry_price']
    side = open_trade['side']
    points = (exit_price - entry_price) if side == 'long' else (entry_price - exit_price)
    trades.append({
        **{k: v for k, v in open_trade.items() if k not in ('entry_idx',)},
        'entry_datetime': datetimes[open_trade['entry_idx']],
        'exit_datetime': datetimes[exit_idx],
        'exit_price': exit_price, 'exit_reason': reason,
        # per-1-contract P&L - final $ depends on the dynamically-chosen
        # contract count, decided downstream (see contracts_max comment above)
        'points': points, 'pnl_per_contract_usd': points * point_value,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=150)
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--out', default='backtest_prop_trades.csv')
    ap.add_argument('--start-date', default=None, help='only open new trades on/after this date (YYYY-MM-DD)')
    ap.add_argument('--end-date', default=None, help='only open new trades on/before this date (YYYY-MM-DD)')
    ap.add_argument('--drawdown-limit', type=float, default=2000.0,
                     help='account max loss - risk_budget_usd = this * win_rate_estimate')
    ap.add_argument('--account-target', type=float, default=3000.0,
                     help='account profit target - profit_target_ratio = this / drawdown-limit (default 1.5)')
    ap.add_argument('--min-contracts', type=int, default=1)
    ap.add_argument('--max-contracts', type=int, default=50,
                     help='absolute sanity ceiling on contracts_max - the real ceiling is risk-budget-derived, '
                          'this just guards against a degenerate tiny-stop edge case producing an absurd count')
    ap.add_argument('--stop-cap-points', type=float, default=20.0,
                     help='hard ceiling on stop distance in price points, normal regime')
    ap.add_argument('--stop-cap-points-volatile', type=float, default=35.0,
                     help='hard ceiling on stop distance in price points, when vol-regime AND win-rate are both high')
    ap.add_argument('--vol-regime-threshold', type=float, default=1.15,
                     help='gjr_vol_regime_ratio above this counts as "volatile"')
    ap.add_argument('--high-prob-threshold', type=float, default=0.45,
                     help='win_rate_estimate (ml_filter_score) above this counts as "high prob"')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_trades = []
    for path in args.files:
        trades = generate_trades(path, lookback=args.lookback, step=args.step, verbose=args.verbose,
                                  start_date=args.start_date, end_date=args.end_date,
                                  drawdown_limit=args.drawdown_limit, account_target=args.account_target,
                                  min_contracts=args.min_contracts, max_contracts=args.max_contracts,
                                  stop_cap_points=args.stop_cap_points,
                                  stop_cap_points_volatile=args.stop_cap_points_volatile,
                                  vol_regime_threshold=args.vol_regime_threshold,
                                  high_prob_threshold=args.high_prob_threshold)
        print(f"  -> {len(trades)} trades simulated")
        all_trades.append(trades)

    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_csv(args.out, index=False)

    print(f"\n=== Trade summary ({len(trades)} total trades) ===")
    print("NOTE: pnl_per_contract_usd/contracts_max are per-1-contract and the risk CEILING respectively -")
    print("      the actual $ results (dynamic min-needed-but-capped-at-ceiling sizing) only exist once run")
    print("      through simulate_prop_challenge_pass_rate.py, which has the running-balance/remaining-days")
    print("      context this script doesn't.")
    print(f"Win rate (per-contract, i.e. per-trade): {(trades['pnl_per_contract_usd'] > 0).mean():.4f}")
    print(f"Avg contracts_max (risk ceiling) per trade: {trades['contracts_max'].mean():.2f}")
    print(f"Avg P&L per trade (1 contract): ${trades['pnl_per_contract_usd'].mean():.2f}")
    print(f"Total P&L (1 contract each): ${trades['pnl_per_contract_usd'].sum():.2f}")
    print(f"By exit reason:\n{trades['exit_reason'].value_counts()}")
    print(f"\nBy instrument:")
    print(trades.groupby('instrument').agg(n=('pnl_per_contract_usd', 'size'),
                                            win_rate=('pnl_per_contract_usd', lambda s: (s > 0).mean()),
                                            avg_contracts_max=('contracts_max', 'mean'),
                                            avg_pnl_per_contract=('pnl_per_contract_usd', 'mean'),
                                            total_pnl_per_contract=('pnl_per_contract_usd', 'sum')).round(2))


if __name__ == '__main__':
    main()
