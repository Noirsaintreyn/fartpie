"""
The actual prop-firm challenge simulator - the piece that was missing.
simulate_prop_challenge.py only produces a raw trade list (entry/exit/P&L);
this takes that list and adds the risk-management layer real deployment
needs: daily loss awareness, max trades/day, a cooldown after losses, a
hard per-challenge drawdown floor, and the actual eval/funded pass-fail
rules - then runs MANY independent simulated challenge attempts across the
12-year trade history so "does this beat the prop firm" is a measured pass
rate, not a single equity curve.

Rules simulated (same interpretation stated earlier, repeated here so this
file is self-contained):
  Eval:    reach +$3,000 cumulative P&L over >= 2 distinct trading days,
           never breaching -$2,000 drawdown from the eval starting balance
           (static, not trailing).
  Funded:  starting a FRESH -$2,000 drawdown floor from whatever equity you
           had when eval passed (separate account/phase, standard prop-firm
           structure) - need >=5 distinct trading days with >=+$150 P&L
           each before breaching that floor.

Risk controls (not specified by the user, so defaults are stated explicitly
and easy to change via CLI flags rather than assumed silently):
  --max-trades-per-day 4      stop opening new trades once today already
                               has this many
  --cooldown-after-losses 2   after this many CONSECUTIVE losing trades in
                               a day, stop opening new trades for the rest
                               of that day
  --max-concurrent 3          cap on simultaneously open positions across
                               all instruments/timeframes pooled together
                               (trades from different instrument/timeframe
                               files CAN legitimately overlap in time - a
                               real account can hold NQ and ES positions
                               at once - this just caps how many at a time)
"""
import argparse

import numpy as np
import pandas as pd


def load_trades(paths):
    dfs = [pd.read_csv(p, parse_dates=['entry_datetime', 'exit_datetime']) for p in paths]
    trades = pd.concat(dfs, ignore_index=True)
    trades = trades.sort_values('entry_datetime').reset_index(drop=True)
    return trades


def simulate_one_attempt(trades, start_idx, eval_target=3000, eval_min_days=2,
                          drawdown_limit=2000, funded_days_needed=5, funded_daily_target=150,
                          max_trades_per_day=4, cooldown_after_losses=2, max_concurrent=3):
    """Walk forward through `trades` starting at start_idx, applying risk
    controls, until eval passes/fails, then (if passed) until funded
    passes/fails or data runs out."""
    phase = 'eval'
    phase_start_balance = 0.0
    balance = 0.0
    daily_pnl = {}          # date -> cumulative P&L that day
    daily_trade_count = {}  # date -> count of trades opened that day
    daily_consec_losses = {}  # date -> current losing streak that day
    open_positions = []     # list of (exit_datetime,) for concurrency cap - approximate, doesn't track exact overlap depth precisely but good enough as a cap
    trading_days_with_pnl = {}  # for eval: any day with net nonzero activity; for funded: days meeting the target

    eval_pass_idx = None
    eval_pass_date = None
    result = {'start_idx': start_idx, 'start_date': trades.iloc[start_idx]['entry_datetime'],
              'eval_result': 'incomplete', 'eval_days': None,
              'funded_result': None, 'funded_days': None,
              'max_drawdown': 0.0, 'final_balance': 0.0}

    i = start_idx
    n = len(trades)
    while i < n:
        row = trades.iloc[i]
        day = row['entry_datetime'].date()

        # concurrency cap: drop expired positions, check capacity
        open_positions = [d for d in open_positions if d > row['entry_datetime']]
        if len(open_positions) >= max_concurrent:
            i += 1
            continue

        # daily trade cap
        if daily_trade_count.get(day, 0) >= max_trades_per_day:
            i += 1
            continue

        # cooldown after consecutive losses today
        if daily_consec_losses.get(day, 0) >= cooldown_after_losses:
            i += 1
            continue

        # take the trade
        pnl = row['pnl_usd']
        balance += pnl
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
        daily_trade_count[day] = daily_trade_count.get(day, 0) + 1
        daily_consec_losses[day] = 0 if pnl > 0 else daily_consec_losses.get(day, 0) + 1
        open_positions.append(row['exit_datetime'])

        drawdown = phase_start_balance - balance if balance < phase_start_balance else 0
        # track drawdown as (peak-to-date within this phase is NOT used - static
        # floor from phase start, per the stated rule interpretation)
        loss_from_phase_start = phase_start_balance - balance
        result['max_drawdown'] = max(result['max_drawdown'], loss_from_phase_start)

        if loss_from_phase_start >= drawdown_limit:
            if phase == 'eval':
                result['eval_result'] = 'failed_drawdown'
                result['eval_days'] = len(daily_pnl)
            else:
                result['funded_result'] = 'failed_drawdown'
                result['funded_days'] = len([d for d, p in daily_pnl.items() if d >= eval_pass_date and p >= funded_daily_target])
            result['final_balance'] = balance
            return result

        if phase == 'eval':
            n_days_traded = len({d for d in daily_pnl if d <= day})
            if balance - phase_start_balance >= eval_target and n_days_traded >= eval_min_days:
                eval_pass_idx = i
                eval_pass_date = day
                result['eval_result'] = 'passed'
                result['eval_days'] = n_days_traded
                phase = 'funded'
                phase_start_balance = balance
                daily_consec_losses = {}  # fresh phase, fresh cooldown state
        else:
            qualifying_days = {d for d, p in daily_pnl.items() if d > eval_pass_date and p >= funded_daily_target}
            if len(qualifying_days) >= funded_days_needed:
                result['funded_result'] = 'passed'
                result['funded_days'] = len(qualifying_days)
                result['final_balance'] = balance
                return result

        i += 1

    # ran out of data before resolving
    result['final_balance'] = balance
    if phase == 'eval' and result['eval_result'] == 'incomplete':
        result['eval_days'] = len(daily_pnl)
    elif phase == 'funded' and result['funded_result'] is None:
        result['funded_result'] = 'incomplete'
        result['funded_days'] = len([d for d, p in daily_pnl.items() if d >= eval_pass_date and p >= funded_daily_target])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', nargs='+', required=True)
    ap.add_argument('--attempt-stride', type=int, default=20, help='start a new simulated challenge attempt every N trades in the pooled list')
    ap.add_argument('--eval-target', type=float, default=3000)
    ap.add_argument('--eval-min-days', type=int, default=2)
    ap.add_argument('--drawdown-limit', type=float, default=2000)
    ap.add_argument('--funded-days-needed', type=int, default=5)
    ap.add_argument('--funded-daily-target', type=float, default=150)
    ap.add_argument('--max-trades-per-day', type=int, default=4)
    ap.add_argument('--cooldown-after-losses', type=int, default=2)
    ap.add_argument('--max-concurrent', type=int, default=3)
    ap.add_argument('--out', default='prop_challenge_attempts.csv')
    args = ap.parse_args()

    trades = load_trades(args.trades)
    print(f"Loaded {len(trades)} pooled trades, {trades['entry_datetime'].min()} -> {trades['entry_datetime'].max()}")

    starts = list(range(0, len(trades) - 10, args.attempt_stride))
    print(f"Simulating {len(starts)} independent challenge attempts...")

    results = []
    for start_idx in starts:
        r = simulate_one_attempt(
            trades, start_idx,
            eval_target=args.eval_target, eval_min_days=args.eval_min_days,
            drawdown_limit=args.drawdown_limit,
            funded_days_needed=args.funded_days_needed, funded_daily_target=args.funded_daily_target,
            max_trades_per_day=args.max_trades_per_day, cooldown_after_losses=args.cooldown_after_losses,
            max_concurrent=args.max_concurrent,
        )
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)

    print("\n=== Eval phase ===")
    print(df['eval_result'].value_counts())
    n_complete_eval = (df['eval_result'] != 'incomplete').sum()
    n_passed_eval = (df['eval_result'] == 'passed').sum()
    print(f"Eval pass rate (of resolved attempts): {n_passed_eval}/{n_complete_eval} = {n_passed_eval/max(n_complete_eval,1):.1%}")
    passed = df[df['eval_result'] == 'passed']
    if len(passed) > 0:
        print(f"Median days to pass eval: {passed['eval_days'].median():.0f}")

    print("\n=== Funded phase (of attempts that passed eval) ===")
    print(passed['funded_result'].value_counts())
    n_complete_funded = (passed['funded_result'] != 'incomplete').sum()
    n_passed_funded = (passed['funded_result'] == 'passed').sum()
    if n_complete_funded > 0:
        print(f"Funded pass rate (of resolved attempts): {n_passed_funded}/{n_complete_funded} = {n_passed_funded/n_complete_funded:.1%}")

    print(f"\nMax drawdown experienced (median across all attempts): ${df['max_drawdown'].median():.2f}")
    print(f"Max drawdown experienced (95th percentile): ${df['max_drawdown'].quantile(0.95):.2f}")


if __name__ == '__main__':
    main()
