# Saved model: live vector screener (locked 2026-10-19, live from Tue 2026-10-20 09:30 ET; breaker added 2026-10-20 12:00)

## Rules
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
