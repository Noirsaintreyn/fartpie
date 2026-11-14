# Persistence thesis (saved Nov 13 2026, Rey's direction)

**Thesis.** Many models we tested had edge in *some slice* (an era, a session, a context). If one model of the market is real, those slices should line up and explain *why each model works when it does*. Candidate unifier: **price is a persistence process**. Moves continue when order-flow imbalance persists (returns positively autocorrelated at the model's horizon, VR>1) and revert when it does not (VR<1). The vector (pivot ratchet = first-passage of a trailing-drawdown process) is a *state estimator* of that persistence, so its edge should exist when VR>1. Variance ratio is one estimator; wavelet/multi-scale energy, Hurst, signed-volume imbalance and overnight-inventory continuation are others. Do not tie the thesis to one estimator.

## What we know (each is a slice that should be explained)
| Finding | Where it worked | Where it didn't |
|---|---|---|
| Vector flip, over 1h Maximin pivot (A) | 1h bars: +36bp/8 bars, +50bp/16, both eras (9 LSE names) | 15m (net negative after 20bp), daily large caps (flips lag the universe by 0.2-0.6%) |
| Pivot cross while over vector (B) | nothing beyond bull-bar drift | - |
| ICT chain (sweep->MSS+FVG->limit) | only when overnight inventory is strong (>=90% bars one side): +6.1/+6.4 both eras | balanced overnight: ~-1 |
| Chain vs inventory direction | with-inventory +11.5 (2018-21), against +14.2 (2022+): the SIGN flips by regime | - |
| VA-B (go with value-area break) | + both eras, longs, touch before 10:30; speeds the eval | shorts weak; hurts 2022-25 pass rate (79%->70%) |
| VA fade, VWAP-z fade, IB, shapes, poor H/L | - | lose or flip between eras (price persists past value, doesn't revert to it) |
| FVG-2R in :50-:10 macro window | +4.4/+9.3 | other windows |
| Small-cap runners | volume flag + gain-so-far says what to watch | chasing after the alert loses 4-7%; gap-ups fade |
| Fundamentals as fair value | raw P/S cheap>rich +1.1%/63d in 2022+ | macro-conditioned z = own-history z = 0 OOS |
| Anchored VWAP (W/M/Q/Y, sector) context for flips | - | null/sign flips |
| Persistence VR(7) bucket on 1h flips | persistent beats reverting in both eras (+29/+8bp at 8 bars; +301/+72 hold-to-bear) | t 0.3-2.0, VR vs return corr ~0; NQ n too small |
| Yearly mean VR(7) | 2023 1.03; 2018-19 ~1.0; 2020, 2024, 2025 ~0.95 | regime moves by year |

## Predictions to check (alignment tests; each could kill the thesis)
1. VA-B (continuation) should work when VR>1 and VA-F (fade) when VR<1, at the same bars.
2. With-vs-against overnight inventory should follow the sign of recent daily persistence (trailing counter-auction rate = persistence at 1-day horizon).
3. Vector 15m fails because VR(15m)<1 on most names/years; 1h works because VR(1h..1d)>1. Check VR by horizon (spectrum).
4. Chain edge on strong-inventory days = days with high signed-flow persistence.
5. Runners: after a spike VR<1 (fade); the rare continuation days should show VR>1 before the alert.
6. Main NQ legs by year: leg PnL vs the persistence regime at the leg's horizon; may explain 2018-21 vs 2022+ pass-rate differences.
7. Eval vs funded algos can be gated by the regime separately.

## Next
Regime atlas: VR/Hurst/wavelet by horizon, per name and day (PIT), over the 1,250-name 1h universe and NQ; then re-cut every model above by regime. Keep the strict split (2018-21 / 2022+) and hypothesis-first rules. Scripts: xs/report_30_persistence_vector.py (first test).

## Alignment tests run (Nov 14 2026, xs/report_31_alignment.py): thesis as stated NOT supported
- #1 VA-B vs VA-F by VR / inventory-continuation: signs flip between eras, all |t|<2 -> no alignment.
- #2 chain with vs against overnight by persistence: opposite of predicted in 2018-21, mixed in 2022+ -> no.
- #3 spectrum: NQ and 9 stocks are MEAN-REVERTING at every horizon in 2018-21 (VR 0.83-0.95; daily 0.71-0.78) and ~random-walk in 2022+ (VR 0.99-1.01). 15m and 1h look the same; "15m reverts, 1h persists" is not there. The era split itself is a persistence-level shift (2020 is part of it).
- #4 strong-inventory days are not more efficient (eff 0.125 vs 0.119 / 0.123 vs 0.130); inventory continuation +0.058 vs -0.026 in 2018-21, 0 in 2022+.
- #5 runners: not testable (only 60d of 5m history).
- #6 NQ legs by regime: no leg has |t|>=2 with the same sign in both eras. Weak same-sign hints: M1, V15a, RA better when reverting (VR); R1, V15a(22+) better with inventory continuation.
- #7 regime gating sim: not run (no regime separates the legs consistently).
Takeaway: VR level is an ERA axis, not a clean per-model switch. Next candidates for the state variable: realized-vol / VIX change, signed-volume imbalance, dispersion, rather than autocorrelation alone.

## Refinement (Rey, Nov 13 2026): regimes, not one narrative
The market's participants and plumbing changed (stimulus/retail, tightening, 0DTE, AI bull, tariffs/policy shocks), so a model can work in the regimes it was "made for" and fail elsewhere; the same applies to the prop algo (made for some days, not the other half). Persistence not working in 2018-22 doesn't kill it, it may belong to a different regime. Standard for an explanation: quantitative regime features + a mechanism; a model that works in separate regimes must share a measurable feature, mapping fit on early regimes and confirmed on later ones. The calendar used so far (declared, event-based): R1 2018-01..2020-02 | R2 2020-03..2021-12 | R3 2022-01..2023-03 | R4 2023-04..2025-03 | R5 2025-04..now.

### Regime atlas v1 (xs/report_32_regime_atlas.py)
- NQ avg RTH range (pts): R1 99, R2 205, R3 287, R4 236, R5 349. VR(7): 1.03/0.97/1.02/1.01/**0.92** (R5 is the most mean-reverting). ON->RTH continuation: 0/+.04/**+.09**/-.03/-.04.
- Main algo mean pts/day: +1.8, +3.6, +7.5, +5.2, +7.0 (scales with range). Legs differ: RA best R3 (+14.8), ON+ best R5 (+5.8) but -4.6 in R3, V15a ~0 in R1 then +2 to +6, M1 weakest in R5 (+0.18).
- VA-B: +5.6, +5.6, +8.3, +7.9, **-1.6 (R5)**; VA-F: -7.7, -8.0, -10.3, -10.7, **-0.4 (R5)**; INV-F turns positive in R5 (+1.3). Continuation-from-value dies and fades stop losing exactly in the most mean-reverting regime (consistent with the persistence idea at REGIME level). VA-80 and IB-B are positive in R2-R3 and negative/flat in R4-R5.
- ICT chain, strong overnight: +8.8, +2.1, +1.6, +6.7, +12.8 (positive in all five regimes). With/against overnight matches the sign of regime ON-continuation in only 2 of 5.
- 1h vector flip A (9 stocks, bp @8 bars): +41, +58, +15, +41, +16 vs bull-bar control +9, +25, +17, +16, +7: edge in R1, R2, R4, R5, none in R3 (tightening/bear).
- H-b (declared AFTER seeing the atlas; mechanism: flips are upside-trend continuation): flip A with QQQ above its 200d SMA (PIT) beats the control in ALL five regimes (@8 bars +24/+53/+42/+40/+25; @16 +14/+85/+133/+52/+38; R3 n=31). Below the 200d it is inconsistent (R3 +10 vs control +32; R5 -33). Needs out-of-sample confirmation on a wider universe.

### Expected range vs prior value (Rey's idea, xs/report_34_range_vs_value.py, Nov 14 2026)
rho = ADR20 / prior-day VA width, PIT expanding terciles. LOW rho (expected range fits inside value) -> open inside prior VA 43% vs 22% (HIGH), close inside VA 27% vs 12%, BUT the day's range is larger vs ADR (1.1-1.25x vs 0.84-0.91x): rho is partly a vol-clustering proxy (wide yesterday = wide today). Models: IB-B (continuation) HIGH > LOW in both eras (+6.2/+2.8 vs -2.2/-2.3); main algo pts/day HIGH > LOW both eras (RULES, M1 same); VA-80 (rotation) LOW > HIGH both eras (+6.9/+3.3 vs -4.9/-7.8) but it fails in R4-R5 and t~1; VA-F (fade) loses in every bucket, SHAPE weakly LOW>HIGH; chain positive in both. So "range exceeds value -> continuation" partially holds; "range fits value -> fade" holds only for the rotation model VA-80, not the plain VA-edge fade.

## Nov 14 2026 addendum (reports 35-38)
- HAR range forecast / VA-80 gate (r35), high-rho legs in the pairing sim (r36): INV-F[HIGH] funded-only helps 2018-21, hurts 2022-25; IB-B[HIGH] eval-only lowers pass. Nothing beats current best on 2022-25.
- 200 EMA filter (r37): stock-level not supported (signs inconsistent, small t); QQQ>200d same sign across regimes, pooled t~1.4.
- 15m flip +/- 1h bull, +/- 200d EMA, 9 LSE names (r38): 1h-bull flips sign by era (-17bp 18-21, +12.7bp 22+); EMA adds nothing; 1h bull & <200d EMA positive both eras but small n (lead only).

## Theory v2 (Nov 14 2026): persistence is real but state-conditional; each model is a conditional bet on a state
Mechanism (literature: Bouchaud-Farmer-Lillo 2008, Toth et al. 2011): large orders are split over time, so order flow is persistent; a vector holds while imbalance is still being worked and flips when it is exhausted. Persistence shows up in returns only when the horizon, participation and market state let it (time-series momentum 1-12 months; intraday momentum stronger on volatile, high-volume, news days; momentum after up markets, not down).
Map of our pieces: vector flips = continuation (QQQ above 200d, volume spike; no edge in the 2022-23 bear); VA-B = continuation from accepted value (R1-R4, dies in R5, the most mean-reverting regime); INV-F / fades = reversion (positive only in R5); ICT chain on strong-overnight = inventory unwinding (positive in all five regimes); gap sizing = opening information scales with the range.
Why add-ons help when they align: independent reads of the same state agree (align x1.25 helps, x1.5 hurts: the confidence is real but small). Why win-rate leverage works for VA-B: a 70% win-rate leg has low variance for its edge, so a little size adds return without adding much risk of ruin.
Falsified so far: variance ratio as a per-model switch (era axis), fundamental / VWAP fair value, TDA landscape norm as a switch (report_41: high-low sign flips between eras on the 16-bar exit, and the residual beyond vol and correlation flips too).
Next: PIT state from realized vol, dispersion, volume (fit R1-R2, confirm R3-R5) before any gate.

## Nov 14 2026 addendum 2 (reports 42-62): what the persistence thesis says about WHICH flips become the tail
- Whole book is the tail: top decile of flips carries ~4x (large-cap 15m) to >10x (small-cap 1h) of total long return; median flip loses ~5% (daily wide). Selection of the tail is the only question.
- Post-entry commitment (up-volume share, efficiency over first 4 bars) is NOT an entry/exit gate: exit-if-fail and late-entry both lose to holding (report_49, large caps; report_50 small caps: weak starts recover). Usable at most as a size tilt.
- Bar-level participant classifier (17 proxy features, logit/SVM/boosting, strict 2019-21 -> 2022+): raw label dominated by ATR%; vol-normalized AUC ~0.52 longs / ~0.56 shorts (report_53/53b). Price bars alone cannot identify who is trading.
- ICT/overnight-book ideas on stock flips: premarket strength weak (pooled t +2.2, same sign both eras); room-to-prior-high and premarket return only 2022+; gap and prior-day-low sweep do not transfer (report_48).
- Shorts: negative everywhere (large -0.17%, small -0.73% per flip); the tail exists but is not selectable.
- Catalyst/participant-reason layer on small/mid: 8-K catalyst presence no lift on large caps (54); PEAD gate and financing-8-K skip looked good on 19 months of 1h data (56-58) but FAILED strict-split replication on ~1,650 names daily 2019-2026Q1 (report_61): PEAD gate +1.6t in 2019-21 and nothing in 2022+, financing skip wrong sign. Not adopted.
- Small-cap anomalies tried (report_58/59): lottery rejected, low price / illiquidity positive but cost-sensitive, short interest and FRED macro gates no usable signal (macro = few independent regimes).
- KEPT (decision Nov 14): flip-bar volume >= 1.5x (wide daily +3.19% vs +1.91%, t +2.0, both eras) and >= 1 officer/director open-market purchase (SEC Form 4 code P) in the prior 30 days ON TOP of volume (+8.31% vs +3.19%, both eras positive, pooled t ~+1.6, strongest 2022+). A tilt, not a proven gate: 423 signals over 7 years, lumpy by year, survivorship bias in universe, insider data only through 2026-03 (bulk files for 2026 Q2+ empty; fill via per-filing Form 4 XML). 2-insider cluster rule FAILED (n=134, -0.98%). Officer/director/dollar-size splits do not sharpen it (report_62).
- Still open: out-of-sample year + point-in-time universe for the insider+volume tilt; exit/sizing design (how winners are held, slot allocation) rather than more entry filters; options ratio remains the only validated large-cap discriminator.
