# Saved model: live vector screener (locked 2026-10-19, live from Tue 2026-10-20 09:30 ET; breaker added 2026-10-20 12:00)

## Rules
- **Earnings guard (live from Mon 2026-10-26):** the day before a report (after-close today or pre-open tomorrow), if a position is up >= +3%
  but by less than its typical earnings move (mean |gap| of the last 8 reports, from the bot's own bars), it is sold at that day's close.
  Discord warns after the 15:30 bar ("sell at the close, market-on-close"), then confirms after the close. Report dates from Yahoo (cached daily). EARN_GUARD=0 turns it off.
- **Day-extreme alerts (from Mon 2026-10-26):** for held names, from 12:00 ET, when the day's low (or high) looks in (set >= 30 min earlier, price
  >= 0.5 daily sigma off it), Discord posts the odds it holds (~87-89% from 12:00, ~95-96% from 14:00) and the projected other extreme
  (typical / 80% / 95%). Vol = HAR on the daily high/low range (pooled fit) - bands ~8-10% narrower than 20-day std at the same coverage
  (xs/range_models.py). Once per name per side per day. EXTREME_CHANNEL_ID to route them; EXTREME_ALERTS=0 turns them off. Information only - not a trade rule.
- **/pro <ticker>** (any stock): vector state (15m/1h, stop or flip trigger), volatility regime (HAR expected range and its 1-year percentile),
  and whether today's high/low is in with the projected other side. **/brief**: book (P&L, room to stop, vol regime, high/low in),
  universe bull counts, names within 0.6% of a bull flip, sector leadership (universe 5d + sector ETFs). News (Helious) pending the API key/docs.
- **Honorable mentions:** flips skipped because all slots are full are alerted (HM_CHANNEL_ID or the main channel) and shown in /hm; never traded.
- **Universe:** S&P 500 names (point-in-time membership) whose options ratio (option contracts x100 / shares, trailing 252-day median, lagged 1 day) is >= 1.5 (from Oct 22 09:30; 2.5 Oct 20-21). Ratio file: live_data/optvol_ratio_live.csv, refreshed Saturdays by tools/weekly_ratio.sh (needs Theta running).
- **Signal:** 15m pivot-ratchet vector built on premarket + RTH bars (04:00-16:00 ET). Long on a bull flip; entries and exits fill at RTH opens.
- **Exit:** 15m bear flip. Once a trade is >= +2% and the 1h vector has been bull, it switches to the 1h stop and **stays there** (sticky). LOCK5/0.5 profit lock. 1h track runs alongside.
- **Drawdown breaker (Oct 20):** no new entries while the book's realized P&L is more than 8% below its 28-calendar-day high (level at the window start counts). Open positions keep their stops. Discord notice when it trips.
- **No afternoon entries (Oct 20, from 12:45):** no new entries at RTH opens from 14:00 to the close (both tracks); cross alerts for those bars are suppressed. Flips after the close still enter at the next 09:30 open.
- **Book:** 6 equal slots (16.7% each). When more names flip than slots are free, the higher options ratio wins. Long only.
- Trades before 2026-10-20 keep the old setup (8 names at ratio >= 1.5, 8 slots).

## Backtest (honest, Oct 21 2026): point-in-time S&P 500 members, ratio >= 1.5 as known then, 6 slots, sticky, 8% breaker, no entries after 14:00
Daily mark-to-market, 2 bp/trade, 3,720 trades (delisted names included, clean bars). Script: xs/site_backtest_wide.py. Benchmarks through 2026-09-25.
| period | model | SPY | QQQ |
|---|---|---|---|
| 2019-21 | +51.7% / -13% | +26.0% / -34% | +38.1% / -29% |
| 2022-26 | +29.1% / -20% | +12.3% / -24% | +14.9% / -35% |
| 2019-26 | +37.4% / -20% | +17.4% / -34% | +23.3% / -35% |
| 2026 YTD | +54.8% / -12% | +19.7% / -9% | +30.8% / -12% |
Sharpe 1.62 (SPY 0.93, QQQ 1.00). Alpha vs SPY+QQQ +20.8%/yr (t 3.7), betas 0.10 SPY / 0.48 QQQ. $100 on 2019-01-01 -> ~$1,165.
Same universe at ratio >= 2.5: 2019-26 +29.3% / -19% (realized) vs +38.5% / -16% at 1.5 -- 1.5 wins both halves.
Slots / tie-break / swap retested Oct 21 on this universe: 6 slots, highest-ratio tie-break, no swapping all stay.
(The old 45.9%/yr figure used today's 40 names -- survivorship; superseded.)

## Trade ledger (Oct 21 2026)
live/ledger.json (seed live_data/ledger_seed.json): trades are recorded when taken and never recomputed. Bar revisions can't erase an open
position (LLY 10-21) or invent a past one (AVGO 10-13).

## Tested and rejected (did not beat live in both halves)
More slots (7-16), ETFs in the basket, lower ratio cuts, VWAP Bollinger exits, QR exits, after-hours bars, intl ADRs (no gate), non-equity sleeve, shorts on bear flips, options-flow (put-heavy call-share z) across the full universe: only helps in names with ratio >= 1; not adopted.
Also rejected Oct 20: tighter profit locks, buying the intrabar cross, volume confirmation (entry or exit), Hawkes sell-pressure exits, Kalman filters, stretched-entry / resistance filters, VIX short switch, gold/oil in the rotation (with or without VIX), low-beta long/short, fewer slots in bad markets (breaker alone did better).
Open candidate: flat into weekends when VIX > 30.
Also rejected Oct 21 (full-S&P universe, ratio >= 1.5): meta-labeling (gradient boosting on 14 entry features, walk-forward; AUC 0.52-0.56, worse than live as tie-break / skip / sizing),
and four published effects as skip rules and tie-breaks: abnormal option-volume spike (Johnson & So), smooth vs jumpy trends (Da-Gurun-Warachka frog-in-the-pan),
nearness to the 52-week high (George & Hwang), lottery/MAX (Bali-Cakici-Whitelaw). None beat live in 2022-26 (xs/meta_label.py, xs/paper_tests.py).
Also rejected Oct 21: volatility-managed sizing (market vol, the screener's own trade vol, per-stock risk parity), residual-momentum tie-break,
sector-momentum tie-break / skip (xs/equity_models.py). Watch item: trades in names whose sector ETF is down over 60d averaged more per trade
in 7 of 8 years (22-26: +0.40% vs +0.13%), but sizing them up only adds exposure (exposure-neutral 1.25x/0.75x = same as live). Not adopted.
Also rejected Oct 21 (xs/exit_lab2.py, xs/earn_tests.py): time stops (1/2/3/5 days), selling half at +3/5/8%, flat weekends when VIX > 25/30,
skipping entries before earnings, selling before earnings, post-earnings-drift tie-breaks. Holding through earnings is GOOD for this model
(selling before earnings: 22-26 +18.3% vs +28.5%). Candidate, not adopted: tie-break 'earnings within 1 trading day first'
(+60.8/-11, +30.9/-20, 2019-26 +42.1/-20 vs live +38.5/-16): beats both halves on return but inside slot-order noise and with a deeper drawdown.
Oct 25 batch (xs/batch_oct25.py, xs/lod_hod.py, xs/max_pain.py):
- CANDIDATE (passes both halves): implied-move earnings guard - sell at the close before a report when the trade is up >= 3% but less than
  1x the name's typical earnings move (mean |gap| of its last 8 reports): +59.3/-11, +28.9/-16, 2019-26 +40.2/-16, 2026 +40.7/-7 (live +38.5/-16, 2026 +35.6/-10).
- Rejected: flip type skips / tie-breaks (continuation-first +40.4 but DD -20), separate 15m/1h slot pools, pullback limit entries
  (-0.25%/4 bars skip: 19-21 -8.6, 22-26 +8.4 - split), max pain on QQQ 2019-25 (hit rate 42-49%, no pull).
- LOD/HOD: once the low is in (set >= 30 min earlier, price >= 0.5 sigma off it), the day's high lands a median 1.34 daily sigma above the
  low, 80% band 1.9, 95% band 2.8 (fit 2019-21; 2022-26 coverage 83% / 96%). Mirror for the high. The low holds 77% / 87% / 95% at 10:30 / 12:00 / 14:00.
- Live fixes Oct 25: one SELL alert per position, signal log only for posted alerts, ledger keeps closed trades closed, cross/no-buy reasons.
- Options ratio: tools/ibkr_optvol.py + tools/daily_ratio.sh (weekdays 16:20) replace the Theta pull once checked against Theta.
- News exit (xs/news_exit.py): selling in extended hours after a bad earnings reaction is neutral overall; best 'sell 30 min after the
  report if down >= 2% and the trade was up >= 3%': +56.0/-11, +29.8/-16, 2019-26 +39.7/-16 (live +38.5/-16). Small, needs extended-hours orders.
- Correlation caps (xs/sector_cap.py): CANDIDATE 'max 1 semiconductor at a time': +59.6/-10, +31.4/-16, 2019-26 +41.3/-22, 2026 +47.8/-9
  (live +55.1/-11, +28.5/-16, +38.5/-16, +35.6/-10); better in 5 of 8 years, but the full-run drawdown is deeper (-22 vs -16). Max 2 per GICS sector: split halves.
  Max 1 per sector: much worse.
- Correlation rules with a reason (xs/corr_rules.py) all rejected: skip crowded flips (corr > 0.6/0.7/0.8 with 1-2 open), skip after a correlated
  peer lost (much worse: 2022-26 +9..17%), sell correlated peers when one loses, prefer flips whose peers are winning, correlation-aware sizing.
  => the 'max 1 chip' result was luck, not a correlation effect. Not adopted.
- Options sister model (xs/options_model.py, xs/put_sell_prob.py): capped calls don't work - take-profits lift the win rate to 41-48% at most
  (losers hit the vector stop before +1.5%), and a rough 30-day ATM call model loses on average in every variant; higher ratio (>= 3) doesn't fix it.
  Short-put proxy has an edge: after a flip, P(stock ends above entry - 0.5 sigma in 5 days) 81.6% vs 76.8% on random days (19-21), 77.8% vs 72.6% (22-26);
  -1 sigma: 91.2 vs 87.8, 89.0 vs 85.9. Needs real option prices to confirm.
- Stock-specific targets (xs/adaptive_tp.py): typical winner peak ranges from ~3% (LMT, MCD, WMT) to ~9% (MU, APP); AAPL 4.9%, ORCL 7.8%.
  Tightening once a trade reaches its stock's target (vol-scaled or its own median winner peak) lost in almost every version; the only
  near-tie: 2x weekly vol then give back 40% (+58.0/-11, +30.5/-16, 2019-26 +40.2/-19) - deeper full-run drawdown, not adopted.
  Day-range 80% band exit: worse.

