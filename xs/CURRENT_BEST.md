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
