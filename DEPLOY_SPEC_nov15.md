# Deploy spec (Nov 15 2026): variant I + IFVG 0.5 ct + premium fade 0.2 ct (regime switch)
Sim result (Lucid 50K, one account, 2022-25 starts, report 162): pass 84%, paid once 63%, 2.06 payouts/eval, net $3,180/eval, 28 days to pass (variant I alone: 79% / 51% / 1.78 / $2,695 / 34; + IFVG 0.5 only: 86% / 65% / 1.90 / $2,825 / 32). 2018-21 starts: no gain from the fade (inactive then).

## Leg 1: IFVG, 0.5 ct (5 MNQ), report 144
NQ 5m, first signal per day, entries 09:35-14:00 ET (limit order valid 12 bars after the signal bar).
Long: an earlier BEAR FVG (gap >= 3 pts, formed 3..30 bars before) is inverted by a displacement bar (green, body >= 0.6 * ATR14(5m)) closing above the gap top, with no earlier bar closing above it. Limit buy at the gap midpoint (fills only if price trades 0.25 through). Stop = gap bottom - 2. Risk must be 6..40 pts. Target 2R. Flat 15:55. Short = mirror (bull FVG closed through downward).
Stats (1 ct, cost 0.5): 2018-21 +3.44 pts/trade (t 3.3, n 231) | 2022+ +2.84 (t 3.0, n 425), ~7 trades/month.
Take profit (report 168): 2R is the default; 1R has the highest win rate (64/59%) and t (3.6/4.5); VWAP target is worse (+2.6/+1.9); nearest liquidity uncapped is worse (+1.6/-0.1).
Optional (report 164): trades whose direction agrees with price vs the 5m EMA200 (or 15m EMA21/50/200) do better (+4 to +6 pts both eras) but keeping only those did not improve Lucid pass rates in the earlier context-filter sim; do not gate without a rerun.

## Leg 2: S1 premium fade, 0.2 ct (2 MNQ), ONLY when the regime switch is ON (report 156/161)
Inputs at 09:30 ET: prior RTH day (09:30-16:00) volume profile from 5m bars (bar volume spread over its high-low, 1-pt bins), POC, VAL/VAH = 70% value area; tightness = (VAH-VAL)/day range; ATR14 = mean of the last 14 RTH day ranges.
Trigger: tightness <= 0.49 (2018-21 median) AND the 09:30 open is above prior VAH. Enter SHORT at the open. Stop = open + 0.5 * ATR14. Target = prior VAL. When price touches prior POC the stop moves to open - 0.25 ATR (locked profit). Flat 15:55. Median stop 43 pts (2018-21), 93 pts (2022+) => about $190 per 0.1 ct.
Regime switch: reg60 = mean over the last 60 trading days (shifted one day) of GF = -sign(gap) * (close - open) / ATR14 for days with |gap| >= 5 pts. Trade S1 only when reg60 > 0 (about 46% of S1 days).
Stats: all S1 2018-21 +0.4 (t 0.1, n 229) | 2022+ +41.2 (t 3.7, n 194). Switch ON: +3.5 (t 0.4) | +60.0 (t 3.2). Fails the 2018-21 half: this is a post-2020 regime bet, shadow-track for a few weeks before sizing it.

## Not added (tested, failed the strict split)
Turtle soup fade/continuation, SMT, Goldbach, 15:00 macro, 09:30-10:00 drift, small-gap fill, gap-direction context, RSI/EMA standalone, trend lines (bounce/break/retest), MES dual EMA, Zone Hunter Asia V2.
