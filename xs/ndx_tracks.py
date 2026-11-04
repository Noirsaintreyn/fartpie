"""Nov 4 2026: Nasdaq-100 (and S&P, Dow for reference) by track: 15m only / 1h only / both. DECLARED BEFORE RESULTS.
Each book alone, 6 slots, ratio>=1.5 (Dow >=1.0), 40 matched runs; reported vs live S&P both-tracks baseline."""
import sys
sys.argv = sys.argv[:1] + ['live']
_s = open('xs/uni_slots_ratio.py').read(); exec(_s[:_s.index("base = evN(")])
base = evN(book('live', 1.5), 6)
for nm, F in (('S&P (live)', W.sp & (W.ratio >= 1.5)), ('Nasdaq-100', W.ndx & (W.ratio >= 1.5)), ('Dow', W.dow & (W.ratio >= 1.0))):
    for tr, M in (('both', None), ('15m only', ~W.h1.astype(bool)), ('1h only', W.h1.astype(bool))):
        Y = W[F if M is None else (F & M)].reset_index(drop=True)
        show(f'{nm} {tr}', Y, 6, base)
