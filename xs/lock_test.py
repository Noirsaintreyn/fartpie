"""Tighter profit locks so winners don't round-trip. Live model (premarket+RTH 15m vector, 15m + 1h tracks, sticky 1h, entries/exits at RTH opens).
Lock rule: once the peak gain >= K, exit (next open) when a 15m close falls to <= (1-G) x peak gain.  live = K 5%, G 0.5.
G = 1 means a breakeven stop once the trade has been up K.  Stage 1 TK..: -> xs/ext/lockx; stage 2: sims."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
V = [(.05, .5), (.04, .5), (.03, .5), (.05, .6), (.04, .6), (.03, .6), (.05, .7), (.03, .7), (.02, 1.0), (.03, 1.0)]
if sys.argv[1] == '1':
    exec(open('xs/ah_sticky.py').read().split('def run(')[0])                     # bars()
    def run(H, h1track, K, G):
        reg, s1, o, c, hi, d, m = H.reg.values, H.s1.values, H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
        g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False
        KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
        for p in range(1, len(g)):
            j = g[p]
            if not pos:
                sig = (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1)
                if sig: pos, e, px, b1, peak, h1 = True, j, o[j], False, o[j], False
                continue
            if j == e: continue
            k = g[p - 1]
            if k >= e: peak = max(peak, hi[k])
            b1 = b1 or KS[p] == 1; h1 = h1 or (c[j - 1] / px - 1 >= .02 and b1)
            ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1))
            pg = peak / px - 1
            if pg >= K and (c[j - 1] / px - 1) <= (1 - G) * pg: ex = True
            if ex: T.append((d[e], d[j], o[j] / px - 1 - 2e-4, pg)); pos = False
        return T
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    for tk in sys.argv[2:]:
        H = bars(fs[tk], 960); rows = []
        for K, G in V:
            rows += [(tk, f'{K:.0%}/{G}', h1t) + t for h1t in (False, True) for t in run(H, h1t, K, G)]
        pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret', 'peak']).to_parquet(f'xs/ext/lockx/{tk}.parquet'); print(tk, end=' ', flush=True)
else:
    exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])                  # RAT, last
    T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/lockx/*.parquet')); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
    T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
    T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5]
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
    print(f'{"lock: peak >= K, keep G of it":<34}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}  round-trips*')
    for v in dict.fromkeys(T['var']):
        Z = T[T['var'] == v]; res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
        rt = ((Z.peak >= .03) & (Z.ret <= 0)).mean()
        K, G = v.split('/'); nm = f"K {K}, keep {float(G):.0%}" + ('  (live)' if v == '5%/0.5' else '') if G != '1.0' else f'breakeven stop after +{K}'
        print(f'  {nm:<32}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res) + f'{rt:>10.1%}', flush=True)
    print('* share of trades that were up >= 3% at some point and closed at a loss')
