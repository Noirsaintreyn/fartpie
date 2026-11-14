"""Nov 14 2026 SMALL-CAP ENTRY TIMING + SHORTS (declared before results). Universe: all 374 names in live/midsmall_bars.pkl (1h RTH, 2025-04..2026-11, ~19 months: too short for a 2018-21 split, so split by halves and treat as exploratory). Vector 1h flips, entry next-bar open, exit open after the opposite flip. d = +1 long on bull flip, -1 short on bear flip. Commitment = direction share of volume over the first 2 bars from entry (UVS; for shorts the DOWN-volume share), known at bar 2 close. Fail = bottom PIT-expanding tercile. A hold all | B exit at bar-2 close if fail | C enter at bar-2 close only if pass. Returns direction-adjusted, no costs (15bp round trip would be subtracted from each taken trade)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
D = pd.read_pickle('live/midsmall_bars.pkl')['d']; rows = []
for tk, B in D.items():
    B = B.dropna(); B = B[~B.index.duplicated()]
    if len(B) < 300 or B.close.median() < 0.3: continue
    reg = run_state_machine(B.assign(datetime=B.index))[0]; n = len(B); o = B.open.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index
    ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]
    for a in range(len(ch) - 1):
        i, j = ch[a], ch[a + 1]; st, en = i + 1, j + 1; d = int(reg[i])
        if en >= n or st + 2 >= en: continue
        ent = o[st]; p2 = c[st + 1]; ex = o[en]
        if ent <= 0 or v[st:st + 2].sum() <= 0: continue
        up = (c[st:st + 2] > o[st:st + 2]); sh = (v[st:st + 2] * (up if d > 0 else ~up)).sum() / v[st:st + 2].sum()
        rows.append(dict(t=idx[st], d=d, full=d * (ex / ent - 1), r2=d * (p2 / ent - 1), rem=d * (ex / p2 - 1), UV=sh, px=ent))
P = pd.DataFrame(rows).sort_values('t').reset_index(drop=True); print('flips', len(P), 'names', len(D), 'long', (P.d > 0).sum(), 'short', (P.d < 0).sum())
for dd, nm in ((1, 'LONG'), (-1, 'SHORT')):
    S = P[P.d == dd].copy().reset_index(drop=True); v_ = S.UV.values; f = np.zeros(len(v_), bool); ok = np.zeros(len(v_), bool)
    for i in range(100, len(v_)): ok[i] = True; f[i] = v_[i] <= np.quantile(v_[:i], 1 / 3)
    S = S[ok]; f = f[ok]; mid = S.t.iloc[len(S) // 2]
    for lab, m in (('all', np.ones(len(S), bool)), ('half1', (S.t < mid).values), ('half2', (S.t >= mid).values)):
        s = S[m]; ff = f[m]
        A = s.full; B_ = np.where(ff, s.r2, s.full); C = np.where(ff, 0, s.rem)
        print(f'{nm:5s} {lab:5s} n {len(s):5d} fail {ff.mean():.2f} win {(s.full>0).mean():.2f} | A {A.mean()*100:+.2f} | B {B_.mean()*100:+.2f} | C {C.mean()*100:+.2f} (taken {s.rem[~ff].mean()*100:+.2f}) | fail grp: full {s.full[ff].mean()*100:+.2f} r2 {s.r2[ff].mean()*100:+.2f} rem {s.rem[ff].mean()*100:+.2f} | pass grp full {s.full[~ff].mean()*100:+.2f}')
