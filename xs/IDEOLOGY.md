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
