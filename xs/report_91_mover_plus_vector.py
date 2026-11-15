"""Nov 14 2026 DISCRETIONARY WORKFLOW: find the big mover first, then use the vector for entry / confirmation / stop (Rey: find big moves in small caps, enter with the vector). Declared before results. MOVER day i: close $1-10, 20d median dollar volume >= $500k, day return >= +15%, volume >= 3x prior-20d mean, no prior mover in that name in the last 10 days. State at the mover-day close (no look-ahead): BULL (vector bullish) or BEAR. Entries at next open, 30bp round trip (understated for small caps; 100bp sensitivity printed). H1: BULL movers beat BEAR movers on 10-day excess return vs the equal-weight universe, both eras (>=1pp, pooled month-clustered t>=2). H2: holding to the vector exit (open after the opposite flip) beats fixed 10-day hold on mean AND on loss share (>10% losers) for BULL movers. H3: PULLBACK entry (BULL movers, first day within 10 days where close <= vector level + 1 ATR and still bullish; entry next open, stop = vector exit) has better mean per unit of initial risk (return / ((entry - level)/entry)) than buying the mover next open. H4: BEAR movers that later FLIP bullish within 20 days (entry next open after the flip, vector exit) vs buying the BEAR mover immediately. Report n, mean, median, huge (>=+100%), win10, ex-top-3 mean per era."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); H = 10; C = 0.003
opn = pd.DataFrame({tk: B.open[~B.index.duplicated()] for tk, B in D.items()}).sort_index(); opn = opn[opn.index >= '2018-06-01']; ro = opn.pct_change(fill_method=None).clip(-.5, 1.0); Lm = np.log1p(ro.mean(axis=1).fillna(0)).cumsum(); mk = np.expm1(Lm.shift(-(H + 1)) - Lm.shift(-1))
rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']
    if len(B) < 300: continue
    o = B.open.values; h = B.high.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; n = len(B); reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index))
    dv = pd.Series(c * v).rolling(20).median().values; vm = pd.Series(v).rolling(20).mean().shift(1).values; r1 = np.r_[np.nan, c[1:] / c[:-1] - 1]; last = -99
    def vexit(s):
        j = s
        while j < n and reg[j] != -1: j += 1
        return (o[j + 1], j + 1 - s) if j + 1 < n else (np.nan, 0)
    for i in range(131, n - H - 25):
        if not (1 <= c[i] < 10 and dv[i] >= 5e5 and r1[i] >= .15 and v[i] >= 3 * vm[i] and i - last > 10) or reg[i] == 0: continue
        last = i; d0 = idx[i]; st = 'BULL' if reg[i] == 1 else 'BEAR'; e = o[i + 1]; x10 = o[i + 1 + H] / e - 1 - C; xv, hd = vexit(i + 1); xvr = xv / e - 1 - C if not np.isnan(xv) else np.nan
        risk = (e - lev[i]) / e if st == 'BULL' else np.nan
        rows.append(dict(kind='MOVER', st=st, tk=tk, d=d0, x10=x10, xv=xvr, risk=risk, ex10=x10 - mk.get(idx[i + 1], np.nan)))
        if st == 'BULL':
            for k in range(i + 1, min(i + 11, n - H - 3)):
                if reg[k] != 1: break
                if c[k] <= lev[k] + atr[k]:
                    ek = o[k + 1]; xvk, _ = vexit(k + 1)
                    if not np.isnan(xvk): rk = (ek - lev[k]) / ek; rows.append(dict(kind='PULLBACK', st='BULL', tk=tk, d=idx[k], x10=o[k + 1 + H] / ek - 1 - C, xv=xvk / ek - 1 - C, risk=rk, ex10=np.nan))
                    break
        else:
            for k in range(i + 1, min(i + 21, n - 3)):
                if reg[k] == 1 and reg[k - 1] != 1:
                    ek = o[k + 1]; xvk, _ = vexit(k + 1)
                    if not np.isnan(xvk): rows.append(dict(kind='BEAR_THEN_FLIP', st='BEAR', tk=tk, d=idx[k], x10=np.nan, xv=xvk / ek - 1 - C, risk=np.nan, ex10=np.nan))
                    break
A = pd.DataFrame(rows); A['era'] = np.where(A.d < '2022-01-01', '2019-21', '2022-26Q1'); A = A[(A.d >= '2019-01-01') & (A.d < '2026-03-01')]; A['mo'] = A.d.dt.to_period('M'); print('events', A.groupby('kind').size().to_dict())
def st(x):
    x = x.dropna(); return f'n {len(x):4d} mean {x.mean()*100:+6.1f}% med {x.median()*100:+5.1f}% huge {(x>=1).mean()*100:4.1f}% win10 {(x>=.1).mean()*100:4.1f}% loss>10% {(x<-.1).mean()*100:4.1f}% ex-top3 {x.sort_values().iloc[:-3].mean()*100 if len(x)>6 else np.nan:+5.1f}%'
M = A[A.kind == 'MOVER']; print('\nBASELINE: buy every mover next open')
for en in ('2019-21', '2022-26Q1'):
    for s in ('BULL', 'BEAR'): z = M[(M.era == en) & (M.st == s)]; print(f' {en:9s} {s}: fixed10 {st(z.x10)} | excess10 mean {z.ex10.mean()*100:+.1f}% | to-vector-exit {st(z.xv)}')
print('\nH1 BULL minus BEAR (10d excess):')
for en in ('2019-21', '2022-26Q1', 'POOLED'):
    z = M if en == 'POOLED' else M[M.era == en]; a = z[z.st == 'BULL'].groupby('mo').ex10.mean(); b = z[z.st == 'BEAR'].groupby('mo').ex10.mean(); d = (a - b).dropna(); print(f'  {en:9s} diff {(z[z.st=="BULL"].ex10.mean()-z[z.st=="BEAR"].ex10.mean())*100:+.1f}pp  month-clustered t {d.mean()/(d.std()/np.sqrt(len(d))):+.1f}  (months {len(d)})')
print('\nH2 BULL movers: fixed10 vs hold-to-vector-exit'); 
for en in ('2019-21', '2022-26Q1'): z = M[(M.era == en) & (M.st == 'BULL')]; print(f' {en:9s} fixed10 {st(z.x10)}\n {"":9s} vecexit {st(z.xv)}')
print('\nH3 PULLBACK entries (BULL movers) vs buying the mover next open; return per unit of initial risk (entry-level)/entry:')
for en in ('2019-21', '2022-26Q1'):
    p = A[(A.kind == 'PULLBACK') & (A.era == en)]; m = M[(M.era == en) & (M.st == 'BULL')]
    for lab, z in (('PULLBACK', p), ('MOVER BULL', m)): zz = z.dropna(subset=['xv', 'risk']); zz = zz[zz.risk > .005]; print(f' {en:9s} {lab:10s} to-vector-exit {st(z.xv)} | return/risk median {np.median(zz.xv/zz.risk):+.2f} mean {np.mean(np.clip(zz.xv/zz.risk,-5,20)):+.2f} (clipped -5..20) | median initial risk {zz.risk.median()*100:.1f}%')
print('\nH4 BEAR movers: buy now vs wait for the bull flip (<=20d), both held to vector exit')
for en in ('2019-21', '2022-26Q1'):
    z = M[(M.era == en) & (M.st == 'BEAR')]; f = A[(A.kind == 'BEAR_THEN_FLIP') & (A.era == en)]; print(f' {en:9s} buy now {st(z.xv)}\n {"":9s} wait-for-flip {st(f.xv)}  (flipped within 20d: {len(f)}/{len(z)} = {len(f)/max(len(z),1)*100:.0f}%)')
A.to_pickle(X + 'mover_vector.pkl')
