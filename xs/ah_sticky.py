"""(1) vector on 04:00-20:00 bars (incl. after-hours, like TradingView ETH) vs 04:00-16:00 (live);  (2) sticky 1h mode: once a trade is
>= +2% with the 1h vector seen bull, it stays on the 1h stop even if the gain later drops below 2% (live re-checks every bar).
Live model otherwise (15m + 1h tracks, LOCK5/0.5, entries/exits at RTH opens). Stage 1 TK..: trades -> xs/ext/ahx; stage 2: sim."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
def sm(x): return np.asarray(run_state_machine(x[['open', 'high', 'low', 'close']].assign(datetime=x.dt))[0])
def bars(f, end):
    H = pd.read_parquet(f); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 240) & (m < end) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; H['reg'] = sm(H)
    x = H[H.m < 960].set_index('dt')[['open', 'high', 'low', 'close']]; g = x.groupby([x.index.normalize(), pd.Grouper(freq='1h', origin='start_day', offset='9h30min')])
    r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna(); r.index = r.index.get_level_values(1); r = r.reset_index(); r.columns = ['dt', 'open', 'high', 'low', 'close']
    s = pd.DataFrame({'known': r.dt + pd.Timedelta('1h'), 'r': sm(r)}).sort_values('known')
    H['s1'] = pd.merge_asof(pd.DataFrame({'t': H.dt + pd.Timedelta('15min')}), s, left_on='t', right_on='known', direction='backward').r.fillna(0).values
    return H
def run(H, h1track, sticky):
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
        b1 = b1 or KS[p] == 1; now = c[j - 1] / px - 1 >= .02 and b1; h1 = (h1 or now) if sticky else now
        ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1))
        pg = peak / px - 1
        if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex = True
        if ex: T.append((d[e], d[j], o[j] / px - 1 - 2e-4)); pos = False
    return T
if sys.argv[1] == '1':
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
    for tk in sys.argv[2:]:
        rows = []
        for vn, end in (('base', 960), ('ah', 1200)):
            H = bars(fs[tk], end)
            for sticky in (False, True):
                rows += [(tk, vn + ('+sticky' if sticky else ''), h1t) + t for h1t in (False, True) for t in run(H, h1t, sticky)]
        pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(f'xs/ext/ahx/{tk}.parquet'); print(tk, len(rows), flush=True)
else:
    exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])                  # RAT, last
    T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ahx/*.parquet')); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
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
    P = (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'))
    print(f'{"6 slots, ratio >= 2.5":<34}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}  trades  avg')
    for v in ('base', 'base+sticky', 'ah', 'ah+sticky'):
        Z = T[T['var'] == v]; res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
        print(f'  {v:<32}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res) + f'{len(Z):>8,} {Z.ret.mean():+.2%}', flush=True)
