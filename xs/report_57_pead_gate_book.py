"""Nov 14 2026 PEAD GATE + EARNINGS-GUARD ON/OFF on the small/mid long book (declared before results). Reading of Rey's 'add PEAD, drop earnings': add the PEAD gate (skip a flip that goes AGAINST a big earnings reaction, |z|>=2, in the first 20 trading days after it) and test REMOVING the earnings guard (skip flips with earnings reaction due within 5 trading days), since the guard was the earnings layer before. Universe: 374 small/mid names, 1h, Apr 2025-Nov 2026, LONG flips only (the live book is long + half-size shorts; shorts shown separately), base = flip-bar volume >= 1.5x. Variants compared per-flip (equal weight, 15bp round trip subtracted): BASE | +EARN-GUARD | +PEAD(6-20d) | +PEAD(1-20d) | +PEAD(1-20d) no guard | +PEAD(1-20d)+guard. Halves by time (exploratory: 19 months)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('/home/claude/fartpie/xs/report_56_pead_flips.py').read().split('# ---- large caps')[0]; exec(src)
sm = src.split('def show(')[0]  # functions only
exec(open('/home/claude/fartpie/xs/report_56_pead_flips.py').read().split('def tag_pre')[1].split('def show_pre')[0].join(['def tag_pre', '']))
K2 = pd.read_csv('/home/claude/fartpie/xs/ext/8k_items_midsmall.csv'); K2['items'] = K2['items'].astype(str); P2 = pd.read_pickle(SC + 'part_small.pkl'); D = pd.read_pickle('live/midsmall_bars.pkl')['d']; dc2 = {tk: B.close.groupby(B.index.normalize()).last() for tk, B in D.items()}
ev2 = events(K2, dc2); P = tag_pre(tag(P2, ev2, dc2), ev2, dc2); P['net'] = P.ret - 0.0015; mid = pd.Timestamp('2026-03-01')
for d, nm in ((1, 'LONG'), (-1, 'SHORT')):
    S = P[(P.d == d) & (P.BV >= 1.5)].copy(); big_ag = (S.z.abs() >= 2) & (S.ALIGN < 0); guard = S.dto.between(1, 5)
    V = {'BASE (vol>=1.5x)': np.ones(len(S), bool), '+EARN-GUARD': ~guard, '+PEAD 6-20d': ~(big_ag & S.lag.between(6, 20)), '+PEAD 1-20d': ~(big_ag & S.lag.between(1, 20)), '+PEAD 1-20d, no guard': ~(big_ag & S.lag.between(1, 20)), '+PEAD 1-20d + guard': ~(big_ag & S.lag.between(1, 20)) & ~guard}
    print(f'\n{nm}: per-flip net mean % | win% | n | (first half / second half mean) | skipped')
    for k, m in V.items():
        s = S[np.asarray(m)]; a = s[s.t < mid]; b = s[s.t >= mid]; print(f' {k:24s} {s.net.mean()*100:+.2f}% win {(s.net>0).mean()*100:4.1f}% n {len(s):5d} | {a.net.mean()*100:+.2f} / {b.net.mean()*100:+.2f} | skipped {len(S)-len(s)}')
    sk = S[big_ag & S.lag.between(1, 20)]; print(f'   skipped by PEAD 1-20d: n {len(sk)} mean {sk.net.mean()*100:+.2f}% | skipped by guard: n {int(guard.sum())} mean {S[guard].net.mean()*100:+.2f}%')
