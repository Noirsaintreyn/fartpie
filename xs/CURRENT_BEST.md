# NQ prop algo: current best (saved 2026-11-10)

**System:** S6 / V2 + room rule E2 — legs RA, M1, R1, ON+, ON-, RULES, V15a; IV-jump skip + HAR sizing; mv30 skip on RA and V15a.
**New (variant I):** gap-size sizing on the non-overnight legs (RA, M1, R1, RULES, V15a only): x0.5 when |gap| / 20d mean range ranks in the bottom third of its trailing 252 days (known at 09:30). ON+/ON- untouched (their P&L is the gap: look-ahead).

Flex 50K, one account, cost 0.5 pt (xs/report_gap.py, report_gap_50K.log):
| | 2018-21 paid once | net/eval | 2022-25 paid once | net/eval |
|---|---|---|---|---|
| A. S6 + room rule | 77% | $1,907 | 51% | $2,435 |
| **I. + x0.5 bottom-third gap** | 85% | $2,094 | 51% | $2,695 |

Not adopted: skip bottom-third gap days (H, K: hurts 2022-25), x1.25 top third (J: ~0), noise leg in funded accounts (variant D: lowers paid-once), squeeze skips (noise leg / ON- only, unproven).
Evidence level: gap-size spread t = 3.1/4.0 (S6 without overnight legs), but the sim gain is ~+10% net per eval: treat as a low-risk tweak, not a new edge.
Pending: 150K and 10-account fleet runs of I; NQ 5m data after 2026-01-30 for 2026.

## Variant J (saved 2026-11-14): variant I + add-on stack + win-rate leverage + align boost (xs/report_39_addon_stack.py)
Added to I (everything that beat I in BOTH eras on net per eval when combined; singles were era-specific or flat):
- Chain strong-overnight leg, 0.25 contract, FUNDED accounts only (no leverage: its win rate does not beat the base)
- VA-B leg, 0.375 contract (0.25 x1.5 win-rate leverage: win rate 72%/68% vs base 51%/54%), EVAL accounts only
- INV-F[HIGH] leg, 0.25 contract, FUNDED accounts only (expected range above prior value)
- Align boost: RA and V15a sized x1.25 on days an add-on signal points the same way (x1.5 hurts)
50K, one account, Lucid caps, cost 0.5 pt (I -> J): pass 95% -> 95% (18-21), 79% -> 80% (22-25); paid once 85% -> 79% and 51% -> 54%; payouts/eval 1.84 -> 2.58 and 1.78 -> 1.80;
net/eval $2,094 -> $2,914 and $2,695 -> $2,767; days start->pass 60 -> 51 and 34 -> 31; start->1st payout 86 -> 72 and 44 -> 45; avg payout unchanged (cap-limited).
Not adopted: IB-B[HIGH] (cuts pass), VA-B at 0.5 eval-only (pass 70% in 22-25), month-1 cut rule (small, depends on reset price; xs/report_40), QQQ>200d tag (pooled t 1.4), TDA norm (xs/report_41, fails).
Evidence level: the gain is mostly 2018-21; 22-25 is +3% net. Treat as a low-risk stack, not a new edge. VA-B was negative in the latest regime (R5).

## Nov 14 2026: mid/small screener candidate tilt (NOT yet wired into live code)
Candidate: prefer/size-up flips with flip-bar volume >= 1.5x AND >= 1 officer/director open-market buy (Form 4 'P') in the prior 30 days. Evidence: reports 60-62 (wide daily universe, strict split, survivorship caveat). PEAD gate and financing-8-K skip were tested and dropped (report_61). Keep the existing earnings guard. The Mac live code (midsmall_screener.py) was not edited this session (device link unavailable).
