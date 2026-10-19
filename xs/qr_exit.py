"""QR exit test. Live model trades (ratio >= 2.5 names), at every RTH bar a trade is open we predict the 10/50/90% quantiles of the
forward return over the next H=8 bars (open of next bar -> open 8 bars later) with gradient-boosted quantile regression, walk-forward
by year (trained only on years before the test year; 2019 = no model). Exit at the next open when the rule fires.
  stage a: build per-bar dataset   stage b YEAR: fit + predict that year   stage c: portfolio sim (6 slots, ratio tie-break)"""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
STAGE = sys.argv[1]; H8 = 8; OUT = 'xs/ext/qrx'
FE = ['up', 'peak', 'give', 'held', 'atrp', 'tod', 'reg', 's1', 'dstop', 'r1', 'r4', 'r16', 'vz', 'h1']
if STAGE == 'a':
    exec(open('xs/ctx_trades.py').read().split('fs = {')[0])
    exec(open('xs/ctx_sim.py').read().split('def sim(')[0])
    from vector_pine import run_state_machine as RSM
    T = T[(T['mode'] == 'live') & (T.ratio >= 2.5)]; fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    rows = []
    for tk in sorted(set(T.tk)):
        H = bars(fs[tk]); n = len(H); o, c, hi, lo, v = H.open.values, H.close.values, H.high.values, H.low.values, H.volume.values.astype(float)
        _, lv, atr, _ = RSM(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt)); lv = np.asarray(lv, float); atr = np.asarray(atr, float)
        g = np.array([j for j in range(1, n) if H.m.values[j] >= 570]); pos = {j: p for p, j in enumerate(g)}
        idx = {d: i for i, d in enumerate(H.dt.values)}
        vz = (pd.Series(np.log1p(v)) - pd.Series(np.log1p(v)).rolling(130).mean()) / pd.Series(np.log1p(v)).rolling(130).std()
        for t in T[T.tk == tk].itertuples():
            e, x = idx.get(np.datetime64(t.entry)), idx.get(np.datetime64(t.exit))
            if e is None or x is None: continue
            px = o[e]; pk = px; pe, px_ = pos[e], pos[x]
            for p in range(pe + 1, px_ + 1):                                   # decision at the open of g[p] using bars up to g[p]-1
                j = g[p]; k = j - 1; pk = max(pk, hi[k]) if k >= e else pk
                fwd = o[g[p + H8]] / o[j] - 1 if p + H8 < len(g) else np.nan
                rows.append((tk, t.entry, t.exit, t.h1, H.dt.values[j], c[k] / px - 1, pk / px - 1, (pk - c[k]) / px, p - pe, atr[k] / c[k],
                             H.m.values[j], H.reg.values[k], H.s1.values[k], (c[k] - lv[k]) / atr[k] if atr[k] > 0 else 0,
                             c[k] / c[k - 1] - 1, c[k] / c[max(0, k - 4)] - 1, c[k] / c[max(0, k - 16)] - 1, vz.values[k], float(t.h1), fwd, o[j] / px - 1))
        print(tk, len(rows), flush=True)
    D = pd.DataFrame(rows, columns=['tk', 'entry', 'exit', 'h1t', 'dt', 'up', 'peak', 'give', 'held', 'atrp', 'tod', 'reg', 's1', 'dstop', 'r1', 'r4', 'r16', 'vz', 'h1', 'fwd', 'xret'])
    D['ratio'] = D.tk.map(lambda t: 0); D.to_parquet(f'{OUT}/rows.parquet'); print(len(D))
elif STAGE == 'b':
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    Y = int(sys.argv[2]); D = pd.read_parquet(f'{OUT}/rows.parquet'); D['y'] = pd.to_datetime(D.dt).dt.year
    tr = D[(D.y < Y) & D.fwd.notna()]; te = D[D.y == Y].copy()
    tr = tr.sample(min(len(tr), 250000), random_state=0)
    for q in (0.1, 0.5, 0.9):
        m = HGB(loss='quantile', quantile=q, max_depth=4, max_iter=200, learning_rate=0.05, min_samples_leaf=200).fit(tr[FE].values, tr.fwd.values)
        te[f'q{int(q * 100)}'] = m.predict(te[FE].values)
    te[['tk', 'entry', 'exit', 'dt', 'up', 'xret', 'q10', 'q50', 'q90']].to_parquet(f'{OUT}/pred_{Y}.parquet'); print(Y, len(tr), len(te), te[['q10', 'q50', 'q90']].mean().round(4).to_dict())
else:
    exec(open('xs/ctx_sim.py').read().split('def sim(')[0]); T = T[(T['mode'] == 'live') & (T.ratio >= 2.5)].copy()
    P = pd.concat(pd.read_parquet(f) for f in glob.glob(f'{OUT}/pred_*.parquet')); P['entry'] = pd.to_datetime(P.entry); P['dt'] = pd.to_datetime(P.dt)
    RULES = {'q50 < 0': lambda z: z.q50 < 0, 'q50 < 0 and up >= 2%': lambda z: (z.q50 < 0) & (z.up >= .02),
             'q10+q90 < 0 (skew down) and up >= 2%': lambda z: (z.q10 + z.q90 < 0) & (z.up >= .02),
             'q90 < 0.5% and up >= 3%': lambda z: (z.q90 < .005) & (z.up >= .03), 'q50 < -0.3%': lambda z: z.q50 < -.003, 'q50 < -0.2%': lambda z: z.q50 < -.002, 'q50 < -0.4%': lambda z: z.q50 < -.004, 'q50 < -0.5%': lambda z: z.q50 < -.005, 'q50 < -0.3% and up >= 2%': lambda z: (z.q50 < -.003) & (z.up >= .02)}
    def sim(Z, n=6):
        Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, ev = [], []
        for e, x, r in zip(Z.entry.values, Z.exit.values, Z.ret.values):
            op = [o for o in op if o > e]
            if len(op) < n: op.append(x); ev.append((x, r / n))
        return pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
    def st(s):
        e = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
        return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min()
    PER = (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2020-01-01', '2027-01-01'))
    def show(nm, X, extra=''):
        res = [st(sim(X[(X.entry >= a) & (X.entry < b)])) for a, b in PER]
        print(f'  {nm:<40}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res) + extra, flush=True)
    print(f'{"6 slots, ratio >= 2.5 (2019 = no model)":<42}{"2019-21":>16}{"2022-26":>16}{"2020-26 (OOS)":>16}')
    show('live exits', T)
    key = ['tk', 'entry']
    for nm, f in [(k, v) for k, v in RULES.items() if os.getenv('ONLY', '') in k]:
        hit = P[f(P)].sort_values('dt').groupby(key).head(1)[key + ['dt', 'xret']]
        X = T.merge(hit, on=key, how='left'); m = X.dt.notna()
        X.loc[m, 'exit'] = X.loc[m, 'dt']; X.loc[m, 'ret'] = X.loc[m, 'xret'] - 2e-4
        show(nm, X, f'   exits changed {m.mean():.0%}, avg {T.ret[m.values].mean():+.2%} -> {X.ret[m].mean():+.2%}')
