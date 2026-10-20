"""Hawkes-style sell pressure for exits. Events on 15m bars (premarket+RTH):
  sell burst = bar return < -0.5 ATR(14) with RVOL >= 1.5;  buy burst = mirror.  RVOL = volume / same-slot 20-session average.
  intensity S_t = S_{t-1} * exp(-1/tau) + sell_burst_t  (self-exciting clusters stack; tau in bars).  B_t likewise; net = S - B.
Exit variants on the live model (15m track while not on the 1h stop; 1h track and the profit lock unchanged):
  early  : exit when net >= T before the 15m break        confirm: a 15m break only exits if S >= T (quiet breaks held)
Stage 1 TK..: -> xs/ext/hawkx; stage 2: sims (ratio >= 2.5, 6 slots)."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
V = [('live', 0, 0), ('early', 4, 1.0), ('early', 4, 1.5), ('early', 8, 1.5), ('early', 8, 2.0), ('confirm', 4, 0.3), ('confirm', 8, 0.3), ('confirm', 8, 0.6)]
if sys.argv[1] == '1':
    exec(open('xs/ah_sticky.py').read().split('def run(')[0])                     # bars()
    def run(H, h1track, mode, NET, S, TH=0):
        reg, s1, o, c, hi, d, m = H.reg.values, H.s1.values, H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
        g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False
        KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]; KN = [NET[j - 1] for j in g]; KX = [S[j - 1] for j in g]
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
            if h1track or h1: ex = KS[p] == -1
            elif mode == 'early': ex = KR[p] == -1 or KN[p] >= TH
            elif mode == 'confirm': ex = KR[p] == -1 and KX[p] >= TH
            else: ex = KR[p] == -1
            pg = peak / px - 1
            if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex = True
            if ex: T.append((d[e], d[j], o[j] / px - 1 - 2e-4)); pos = False
        return T
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    for tk in sys.argv[2:]:
        H = bars(fs[tk], 960)
        P = H.pivot_table(index=H.dt.dt.normalize(), columns='m', values='volume')
        avg = P.rolling(20, min_periods=10).mean().shift(1).stack().rename('av').reset_index(); avg.columns = ['d', 'm', 'av']
        H['d'] = H.dt.dt.normalize(); H = H.merge(avg, on=['d', 'm'], how='left'); rv = (H.volume / H.av).fillna(1.0).values
        tr = np.maximum(H.high - H.low, np.maximum((H.high - H.close.shift()).abs(), (H.low - H.close.shift()).abs())); atr = tr.rolling(14).mean().values
        r = ((H.close - H.open) / atr).values; sb = ((r < -.5) & (rv >= 1.5)).astype(float); bb = ((r > .5) & (rv >= 1.5)).astype(float)
        rows = []
        for mode, tau, TH in V:
            a = np.exp(-1 / tau) if tau else 0; S = np.zeros(len(H)); B = np.zeros(len(H))
            for i in range(1, len(H)): S[i] = S[i - 1] * a + sb[i]; B[i] = B[i - 1] * a + bb[i]
            rows += [(tk, f'{mode}/{tau}/{TH}', h1t) + t for h1t in (False, True) for t in run(H, h1t, mode, S - B, S, TH)]
        pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(f'xs/ext/hawkx/{tk}.parquet'); print(tk, end=' ', flush=True)
else:
    exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])                  # RAT, last
    T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/hawkx/*.parquet')); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
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
    print(f'{"6 slots, ratio >= 2.5, sticky":<40}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}  avg trade')
    for v in dict.fromkeys(T['var']):
        Z = T[T['var'] == v]; res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
        mode, tau, th = v.split('/')
        nm = 'live' if mode == 'live' else (f'early exit: net sell >= {th} (fade {tau} bars)' if mode == 'early' else f'break needs sell >= {th} (fade {tau} bars)')
        print(f'  {nm:<38}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res) + f'  {Z.ret.mean():+.2%}', flush=True)
