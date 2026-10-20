# Saved model: live vector screener (locked 2026-10-19, live from Tue 2026-10-20 09:30 ET)

## Rules
- **Universe:** S&P 500 names whose options ratio (option contracts x100 / shares, trailing 252-day median, lagged 1 day) is >= 2.5. Ratio file: live_data/optvol_ratio_live.csv, refreshed Saturdays by tools/weekly_ratio.sh (needs Theta running).
- **Signal:** 15m pivot-ratchet vector built on premarket + RTH bars (04:00-16:00 ET). Long on a bull flip; entries and exits fill at RTH opens.
- **Exit:** 15m bear flip. Once a trade is >= +2% and the 1h vector has been bull, it switches to the 1h stop and **stays there** (sticky). LOCK5/0.5 profit lock. 1h track runs alongside.
- **Book:** 6 equal slots (16.7% each). When more names flip than slots are free, the higher options ratio wins. Long only.
- Trades before 2026-10-20 keep the old setup (8 names at ratio >= 1.5, 8 slots).

## Backtest (6 slots, ratio >= 2.5, sticky, daily mark-to-market)
| period | CAGR | max DD |
|---|---|---|
| 2019-21 | +58.4% | -25% |
| 2022-26 | +25.0% | -23% |
| 2019-26 | +37.0% | -25% |
Sharpe 1.47 (SPY 0.86, QQQ 0.97). Alpha vs holding the same names equal-weight: +14.8%/yr (t 2.9).
$100 on 2019-01-01 -> ~$1,140 by Oct 2026 (SPY ~$288, QQQ ~$445).

## Tested and rejected (did not beat live in both halves)
More slots (7-16), ETFs in the basket, lower ratio cuts, VWAP Bollinger exits, QR exits, after-hours bars, intl ADRs (no gate), non-equity sleeve, shorts on bear flips, options-flow (put-heavy call-share z) across the full universe: only helps in names with ratio >= 1; not adopted.
