# Saved model: live vector screener (locked 2026-10-19, live from Tue 2026-10-20 09:30 ET; breaker added 2026-10-20 12:00)

## Rules
- **Universe:** S&P 500 names whose options ratio (option contracts x100 / shares, trailing 252-day median, lagged 1 day) is >= 2.5. Ratio file: live_data/optvol_ratio_live.csv, refreshed Saturdays by tools/weekly_ratio.sh (needs Theta running).
- **Signal:** 15m pivot-ratchet vector built on premarket + RTH bars (04:00-16:00 ET). Long on a bull flip; entries and exits fill at RTH opens.
- **Exit:** 15m bear flip. Once a trade is >= +2% and the 1h vector has been bull, it switches to the 1h stop and **stays there** (sticky). LOCK5/0.5 profit lock. 1h track runs alongside.
- **Drawdown breaker (Oct 20):** no new entries while the book's realized P&L is more than 8% below its 28-calendar-day high (level at the window start counts). Open positions keep their stops. Discord notice when it trips.
- **No afternoon entries (Oct 20, from 12:45):** no new entries at RTH opens from 14:00 to the close (both tracks); cross alerts for those bars are suppressed. Flips after the close still enter at the next 09:30 open.
- **Book:** 6 equal slots (16.7% each). When more names flip than slots are free, the higher options ratio wins. Long only.
- Trades before 2026-10-20 keep the old setup (8 names at ratio >= 1.5, 8 slots).

## Backtest (6 slots, ratio >= 2.5, sticky, 8% breaker, no entries after 14:00; daily mark-to-market; benchmarks through 2026-09-25)
| period | model | SPY | QQQ |
|---|---|---|---|
| 2019-21 | +72.5% / -13% | +26.0% / -34% | +38.1% / -29% |
| 2022-26 | +31.3% / -22% | +12.3% / -24% | +14.9% / -35% |
| 2019-26 | +45.9% / -22% | +17.4% / -34% | +23.3% / -35% |
| 2026 YTD | +50.0% / -16% | +19.7% / -9% | +30.8% / -12% |
Sharpe 1.88 (SPY 0.93, QQQ 1.00). Alpha vs SPY+QQQ exposure +26.1%/yr (t 4.4). Breaker only: +40.7% / -23%, Sharpe 1.69. Neither: +38.2% / -25%, Sharpe 1.50.
$100 on 2019-01-01 -> ~$1,828 by Oct 2026 (SPY ~$288, QQQ ~$445). Caveat: the afternoon rule was found after ~20 tests on Oct 20; expect some give-back.

## Tested and rejected (did not beat live in both halves)
More slots (7-16), ETFs in the basket, lower ratio cuts, VWAP Bollinger exits, QR exits, after-hours bars, intl ADRs (no gate), non-equity sleeve, shorts on bear flips, options-flow (put-heavy call-share z) across the full universe: only helps in names with ratio >= 1; not adopted.
Also rejected Oct 20: tighter profit locks, buying the intrabar cross, volume confirmation (entry or exit), Hawkes sell-pressure exits, Kalman filters, stretched-entry / resistance filters, VIX short switch, gold/oil in the rotation (with or without VIX), low-beta long/short, fewer slots in bad markets (breaker alone did better).
Open candidate: flat into weekends when VIX > 30.
