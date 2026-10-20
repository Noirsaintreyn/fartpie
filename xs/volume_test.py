"""Volume confirmation on the live model. RVOL = bar volume / average volume of the same 15m time slot over the prior 20 sessions.
  entry filter: only buy a 15m flip if the flip bar's RVOL >= E
  exit confirm: a bear 15m state only exits on a bar with RVOL >= X (quiet bear bars are held; 1h stop / lock unchanged)
Stage 1 TK..: -> xs/ext/volx; stage 2: sims (ratio >= 2.5, 6 slots)."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
V = [(0, 0), (0.8, 0), (1.0, 0), (1.5, 0), (2.0, 0), (0, 0.8), (0, 1.0), (0, 1.5), (1.0, 1.0)]
if sys.argv[1] == '1':
    exec(open('xs/ah_sticky.py').read().split('def run(')[0])                     # bars()
    def run(H, h1track, E, X):
        reg, s1, o, c, hi, d, m, rv = H.reg.values, H.s1.values, H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values, H.rv.values
        g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False
        KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]; RV = [rv[j - 1] for j in g]
        for p in range(1, len(g)):
            j = g[p]
            if not pos:
                sig = (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1 and (E == 0 or RV[p] >= E))
                if sig: pos, e, px, b1, peak, h1, rve = True, j, o[j], False, o[j], False, RV[p]
                continue
            if j == e: continue
            k = g[p - 1]
            if k >= e: peak = max(peak, hi[k])
            b1 = b1 or KS[p] == 1; h1 = h1 or (c[j - 1] / px - 1 >= .02 and b1)
            ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1 and (X == 0 or RV[p] >= X)))
            pg = peak / px - 1
            if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex = True
            if ex: T.append((d[e], d[j], o[j] / px - 1 - 2e-4, rve)); pos = False
        return T
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    for tk in sys.argv[2:]:
        H = bars(fs[tk], 960)
        P = H.pivot_table(index=H.dt.dt.normalize(), columns='m', values='volume')
        avg = P.rolling(20, min_periods=10).mean().shift(1).stack().rename('av').reset_index(); avg.columns = ['d', 'm', 'av']
        H['d'] = H.dt.dt.normalize(); H = H.merge(avg, on=['d', 'm'], how='left'); H['rv'] = (H.volume / H.av).fillna(1.0)
        rows = []
        for E, X in V: rows += [(tk, f'{E}/{X}', h1t) + t for h1t in (False, True) for t in run(H, h1t, E, X)]
        pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret', 'rv']).to_parquet(f'xs/ext/volx/{tk}.parquet'); print(tk, end=' ', flush=True)
else:
    exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])                  # RAT, last
    T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/volx/*.parquet')); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
    T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
    T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5]
    B = T[(T['var'] == '0/0') & ~T.h1]; q = pd.cut(B.rv, [0, .5, .8, 1, 1.5, 2, 3, 99])
    print('avg 15m-flip trade by flip-bar RVOL (2019-21 | 2022-26):')
    g = B.groupby([q, B.entry.dt.year >= 2022]).ret.agg(['mean', 'count']); g['mean'] *= 100; print(g.unstack().round(2).to_string())
    def sim(Z, n=6):
        Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, ev = [], []
        for e, x, r in zip(Z.entry.values, Z.exit.values, Z.ret.values):
            op = [o for o in op if o > e]
            if len(op) < n: op.append(x); ev.append((x, r / n))
        return pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
    def st(s):
        e = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
        return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min()
    P = (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))
    print(f'\n{"6 slots, ratio >= 2.5, sticky":<40}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
    for v in dict.fromkeys(T['var']):
        E, X = map(float, v.split('/')); Z = T[T['var'] == v]; res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
        nm = 'live' if E == 0 and X == 0 else ' + '.join(s for s in (f'buy only if flip RVOL >= {E}' if E else '', f'exit only if RVOL >= {X}' if X else '') if s)
        print(f'  {nm:<38}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res), flush=True)
