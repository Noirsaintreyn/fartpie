# Edge inventory across the 3 algos (step a+b of the regime programme, Nov 15 2026)
Train = 2018/19-21 (NQ R1-R2), Test = 2022+ (NQ R3-R5). Tag: BOTH = positive in both; TRAIN-ONLY; TEST-ONLY. Class: T trend/continuation, MR mean-reverting, V vol/event, S supply/liquidity (mostly regime-agnostic), ? unclear.

## A. NQ prop algo (pts per trade, weighted R1-2 vs R3-5; source report_32 atlas)
| Edge | Train | Test | R5 (latest) | Tag | Class |
|---|---|---|---|---|---|
| RA (opening-range rule) | +3.6 | +9.3 | +6.8 | BOTH, stronger test | T (needs range expansion; peaks in R3 bear +14.8) |
| M1 | +1.3 | +1.7 | +0.2 | BOTH, fading R5 | MR/mixed |
| R1 | +1.3 | +4.1 | +2.4 | BOTH | T |
| ON+ (overnight long) | +1.3 | +2.1 | +5.8 | BOTH; fails R3 (-4.6) | T/inventory, breaks in high-vol bear |
| ON- | +0.1 | +0.2 | +1.3 | ~0 | - |
| RULES | +1.2 | +1.6 | +1.4 | BOTH, steady | mixed |
| V15a | +1.0 | +4.9 | +2.6 | mostly TEST | T/vol (R3 +6.4, R4 +4.8) |
| VA-B | +5.6 | +6.2 | -1.6 | BOTH, dead in R5 | MR (value-area acceptance) |
| VA-F | -7.9 | -8.6 | -0.4 | negative both (inverse of VA-B) | - |
| VA-80 | +3.2 | -4.0 | -17.5 | TRAIN-ONLY | MR, breaks R4-R5 |
| IB-B | +1.7 | +1.2 | 0.0 | BOTH, faded | MR |
| INV-F | -1.5 | -2.3 | +1.3 | negative; HIGH-inventory filter was used | MR |
| SHAPE | -0.2 | +3.3 | -2.7 | TEST-ONLY (R3,R4) | T |
| ICT chain strong-overnight | +5.7 | +6.8 | +12.8 | BOTH | T (overnight inventory) |
| ICT chain with-overnight | +0.4 | +4.7 | -6.4 | unstable | - |
| ICT chain against-overnight | +3.5 | +2.3 | +11.3 | BOTH (flips sign in R3) | MR |
Other NQ: 3-state HMM noise engine: with-regime minus against-regime +8.3 pts (t 1.9) in 18-21, -6.3 (t -0.8) in 22-26 = regime map NOT stable. GMM K=4-8 trained cluster signs: +2.8/d (t 1.1) train, +10-12/d (t 2.3-2.5) test (above random 95th pct). NQ models-of-regime bake-off (report_regime_models): no learned regime feature stable across both eras (HMM AUC ~0.5; RF/SVM AUC 0.7+ but payoff separation inconsistent). Gate results (nq_regime_gate): post-hoc labels, in-sample by construction. Rule A by year: 2024 and 2026 near zero (+0.07/+0.12 R), 2023/2025 strong.

## B. Mid/small-cap vector book (daily flips 30,323; survivorship-biased; test 2022+)
| Edge | Train | Test | Tag | Class |
|---|---|---|---|---|
| Price < $5 / low dollar volume (huge-move rate 3.5-16% vs ~1%) | strong | strong | BOTH | S |
| Low market cap, high turnover (flip vol / shares) | IC +0.17 excursion | +0.17 | BOTH | S |
| ATRP (ATR14/price) | IC +0.28 | +0.26 | BOTH | V (volatility-expansion) |
| EXT (extension off 60d low) | + | + (coef +0.5) | BOTH | T (momentum already in motion) |
| Flip-bar volume ratio | +7pp top tercile | +3.4 | BOTH, weak | T/attention |
| Frozen stacked score top 20% | +2.8pp all, $1-10 +9.2pp (test t 2.8) | | TEST-validated (trained on train) | T |
| OS outlier score (ATRP+EXT+turnover) with price<$5 | shortlist ~+10-13%/trade test, median -10% | | BOTH | T/V/S |
| Range expansion (ATR10/ATR60 high) | +1.2pp | +1.3pp | BOTH, small | V |
| Isolated flips beat crowded themes | -2.5pp crowded | | BOTH | ? |
| Day-3 distance above ratchet predicts survival (AUC .73) | | | BOTH, but no return edge | state |
| Insider P (officer/director) + volume candidate | lumpy (top3 = 62-68%) | | weak lead | event |
| Squeeze regime tag | wired, gate OFF | | lead | regime |
| 2y yield 60d change | | | lead (only fast regime driver found) | macro |
| Slot breadth N / queue K=5-10 | | | portfolio lever, not alpha | - |
Dead: 13F, index/13D, EMA/trend gates, vector as bias, shorts, hard stops, ML bake-off, 1H pennies, seasonal (Dec first-half weak lead only). Market-level: 2019-21 universe mean +4.6%/flip (VR 0.7-0.9 mean-reverting boom) vs ~0 in 2022+ (the tail selection is what survives).

## C. Large-cap / mega-cap vector screener (6-slot S&P book, 15m/1H)
| Edge | Evidence | Tag | Class |
|---|---|---|---|
| Large-cap daily/1H flips | drift only, no flip alpha | none | - |
| ATRP on 1H flips | t +2.49 in 2025+ only (one regime, 730d) | lead, no split | V |
| Chandelier vs vector exit | halves 1H losses, still negative | none | - |
| QQQ > 200d regime on LSE-1H flip A (atlas H-b) | above-200d: +42/+34 train/test (exit 8), below: +68 train, +6 test (R5 -33) | above BOTH; below TRAIN-ONLY | T |
Caveat: only ONE regime for all intraday stock results (60-730d).

## D. Grouping (step b) -- which regime each edge needs
- TREND / expansion regime (needs directional follow-through, higher vol): NQ RA, R1, V15a, SHAPE, ICT strong-overnight, ON+ (not in high-vol bear), stock EXT, flip-bar volume, stacked score, range-expansion/ATRP, QQQ>200d flips.
- MEAN-REVERT / balanced regime: NQ VA-B, IB-B, M1, INV-F (as filter), ICT against-overnight, VA-80 (TRAIN).
- Regime-agnostic supply/liquidity (stay on in every state): price<$5, low mcap, high turnover, dollar volume. These are cross-sectional, not time-series regimes.
- Unclear: ON-, RULES (flat, steady), isolated-vs-crowded flips, insider P.
- Stable-across-eras but regime-sensitive within: ON+ (dies R3), VA-B (dies R5), VA-80 (dies R4-R5), M1 (fades R5).

## E. Proposed regime identification (step c, not run)
Ranked by prior evidence: (1) ATR/realized-vol level and change (RA/V15a love expansion, VA-B loves quiet), (2) ratio of vol at flip to trailing, (3) volume EMA / relative volume, (4) overnight inventory (strong-overnight share, ON->RTH continuation), (5) index trend (QQQ vs 200d), (6) breadth/flip count (market-wide share of names in bull vector) and 2y yield change. Compare: own rule-based state machine on (1)-(5) vs GMM vs HMM, all FIT ON TRAIN ONLY, judged on whether they sort leg P&L the same direction in R1-2 and R3-5. Prior art says HMM alone sorts poorly (AUC ~0.5).
