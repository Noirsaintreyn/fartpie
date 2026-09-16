"""
Paper-tracking ledger for the FROZEN Valiant selection stack. Prospective
only - the first entry this script ever writes is dated today
(2026-09-15) and there is no backfilled history. That is the point: the
in-sample backtest is done and frozen; the only evidence that still
counts from here is what actually happens after this file starts
writing entries.

FROZEN CONFIGURATION (validated in the decomposition/momentum
comparison tests - do not re-tune):
  - Universe: real point-in-time S&P 500 membership (latest available
    snapshot used as "today's" membership)
  - Eligibility: recent bullish flip on the CORRECT engine
    (backtest_pivot_ratchet.run_state_machine, raw OHLC), >=5
    historical flips, positive quality z, SEC EDGAR fundamentals
    strict pass
  - Ranking: top 5 by z-score, 30% sector cap
  - Weighting: inverse-vol (ATR/close)
  - Exposure: the validated, defensible result is FLAT (100%) exposure
    - that's the primary tracked variant. The correlation-based
    exposure dial is ALSO logged each month purely for comparison
    (it measurably hurt Sharpe once properly isolated in-sample) - not
    applied to the primary tracked number.

Usage:
  .venv311/bin/python valiant_paper_tracking.py          # record this month's picks
  .venv311/bin/python valiant_paper_tracking.py report    # show ledger + forward returns so far
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from vector_rebuilt_hmm_model import build_state, eligibility_z, exposure_multiplier, trailing_returns_matrix
from vector_fundamentals_sector_cap import apply_sector_cap
from vector_screener_sp500_sectors import batch_download_cached, load_sectors
from vector_fundamentals_backtest import fundamentals_asof, load_facts_cache

TOP_N = 5
LEDGER_PATH = 'valiant_paper_tracking_ledger.json'


def compute_current_picks(as_of=None):
    as_of = as_of or datetime.now()
    as_of_str = as_of.strftime('%Y-%m-%d')

    with open('sp500_pit_membership.json') as f:
        pit = json.load(f)
    latest_month = sorted(pit.keys())[-1]
    members = set(t.replace('.', '-') for t in pit[latest_month])

    with open('sp500_pit_all_tickers.json') as f:
        all_tickers = json.load(f)
    all_tickers_yf = [t.replace('.', '-') for t in all_tickers]
    raw = batch_download_cached(all_tickers_yf)

    facts_by_ticker = load_facts_cache()
    sectors = load_sectors()

    stocks = {}
    for t, df in raw.items():
        if len(df) < 500 or t not in members:
            continue
        try:
            stocks[t] = build_state(df)
        except Exception:
            continue

    candidates = []
    for t, s in stocks.items():
        idx = len(s['df']) - 1
        gaap = facts_by_ticker.get(t)
        if gaap is None:
            continue
        result = fundamentals_asof(gaap, as_of_str)
        if result is None or not result['strict_pass']:
            continue
        zl = eligibility_z(s, idx, direction=1)
        if zl is None:
            continue
        vol_pct = s['atr'][idx] / s['close'][idx] if not np.isnan(s['atr'][idx]) else np.nan
        candidates.append({'ticker': t, 'vol_pct': vol_pct, 'z': zl, 'sector': sectors.get(t, '?'),
                            'price': float(s['close'][idx])})

    capped = apply_sector_cap(candidates, sectors)
    top5 = sorted(capped, key=lambda c: -c['z'])[:TOP_N]

    dial = 1.0
    if len(capped) >= 2:
        ret_matrix = trailing_returns_matrix([c['ticker'] for c in capped], stocks, pd.Timestamp(as_of))
        if not ret_matrix.empty and len(ret_matrix.columns) >= 2:
            corr = ret_matrix.corr()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            avg_corr = float(np.nanmean(upper.values))
            dial = exposure_multiplier(avg_corr)

    valid = [c for c in top5 if not np.isnan(c['vol_pct']) and c['vol_pct'] > 0]
    weights = []
    if valid:
        inv_vol = np.array([1.0 / c['vol_pct'] for c in valid])
        weights = list(inv_vol / inv_vol.sum())

    picks = []
    for c, w in zip(valid, weights):
        picks.append({'ticker': c['ticker'], 'sector': c['sector'], 'z': round(float(c['z']), 4),
                       'weight_flat': float(w), 'entry_price': round(c['price'], 4)})

    return {
        'as_of': as_of_str,
        'membership_snapshot': latest_month,
        'n_eligible': len(capped),
        'exposure_dial': round(float(dial), 4),
        'picks': picks,
    }


def record_month(as_of=None):
    entry = compute_current_picks(as_of)
    ledger = []
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH) as f:
            ledger = json.load(f)
    entry_month = entry['as_of'][:7]
    ledger = [e for e in ledger if e['as_of'][:7] != entry_month]
    ledger.append(entry)
    ledger.sort(key=lambda e: e['as_of'])
    with open(LEDGER_PATH, 'w') as f:
        json.dump(ledger, f, indent=2)
    return entry


def resolve_and_report():
    if not os.path.exists(LEDGER_PATH):
        print("No ledger yet - run without 'report' first.")
        return
    with open(LEDGER_PATH) as f:
        ledger = json.load(f)

    all_tickers = sorted({p['ticker'] for e in ledger for p in e['picks']})
    if not all_tickers:
        print("Ledger has no picks yet.")
        return
    raw = batch_download_cached(all_tickers)

    print(f"{'='*100}\nVALIANT PAPER-TRACKING LEDGER ({len(ledger)} recorded month(s)) - FLAT EXPOSURE (frozen, primary)\n{'='*100}")
    for entry in ledger:
        print(f"\n  {entry['as_of']}  (membership snapshot {entry['membership_snapshot']}, "
              f"{entry['n_eligible']} eligible, exposure dial would be {entry['exposure_dial']*100:.0f}% - not applied)")
        month_ret_flat = 0.0
        for p in entry['picks']:
            df = raw.get(p['ticker'])
            cur_price = float(df['close'].iloc[-1]) if df is not None and len(df) else None
            fwd_ret = (cur_price / p['entry_price'] - 1) if cur_price else None
            month_ret_flat += (fwd_ret or 0.0) * p['weight_flat']
            now_str = f"${cur_price:.2f}" if cur_price is not None else "n/a"
            fwd_str = f"{fwd_ret*100:+.2f}%" if fwd_ret is not None else "n/a"
            print(f"    {p['ticker']:<6} {p['sector']:<24} z={p['z']:+.2f}  weight={p['weight_flat']*100:4.1f}%  "
                  f"entry=${p['entry_price']:.2f}  now={now_str}  fwd={fwd_str}")
        print(f"    -> basket forward return so far (flat exposure): {month_ret_flat*100:+.2f}%")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        resolve_and_report()
    else:
        entry = record_month()
        print(f"Recorded {len(entry['picks'])} picks for {entry['as_of']} "
              f"(membership snapshot {entry['membership_snapshot']}, {entry['n_eligible']} eligible):")
        for p in entry['picks']:
            print(f"  {p['ticker']:<6} {p['sector']:<24} z={p['z']:+.2f}  "
                  f"weight={p['weight_flat']*100:4.1f}%  entry=${p['entry_price']:.2f}")
