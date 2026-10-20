"""Low-beta long + short book: S&P names with median 252d beta < 0.8 (2019-26). Longs = live model (15m + 1h tracks, sticky, lock);
shorts = mirror of the 15m track (short the open after a bear flip, cover after the bull flip). Stage 0: pick names; 1 TK..: trades; 2: results."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
if sys.argv[1] == '0':
    D = pd.read_parquet('etf_daily/SPY.parquet'); D['d'] = pd.to_datetime(D.datetime).dt.tz_localize(None).dt.normalize(); spy = np.log(D.drop_duplicates('d').set_index('d').adjclose).diff()
    ETF = {'QQQ', 'SPY', 'IWM', 'DIA', 'SMH', 'XLE', 'XLF', 'XLK', 'GLD', 'SLV', 'TLT', 'IEF', 'GDX', 'HYG'}; res = {}
    for f in glob.glob('xs/ext/rot_trades/*.parquet'):
        tk = os.path.basename(f)[:-8]
        if tk in ETF or tk not in fs: continue
        H = pd.read_parquet(fs[tk], columns=['datetime', 'close']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)]; r = np.log(H.groupby(H.dt.dt.normalize()).close.last()).diff()
        J = pd.concat([r, spy], axis=1, keys=['s', 'm']).dropna(); J = J[J.index >= '2019-01-01']
        b = J.s.rolling(252, min_periods=120).cov(J.m) / J.m.rolling(252, min_periods=120).var(); res[tk] = b.median()
    s = pd.Series(res).sort_values(); s.to_csv('xs/ext/lowb/betas.csv'); low = s[s < .8]
    print(len(s), 'names;', len(low), 'with median beta < 0.8:'); print(' '.join(low.index))
elif sys.argv[1] == '1':
    exec(open('xs/ah_sticky.py').read().split("if sys.argv[1] == '1':")[0])        # bars(), run()
    exec(open('xs/flow_test.py').read().split('def run_short(H):')[1].split('fs = {')[0].join(['def run_short(H):', '']))
    for tk in sys.argv[2:]:
        H = bars(fs[tk], 960)
        rows = [(tk, 'long', h1t) + t for h1t in (False, True) for t in run(H, h1t, True)] + [(tk, 'short', False) + t for t in run_short(H)]
        pd.DataFrame(rows, columns=['tk', 'side', 'h1', 'entry', 'exit', 'ret']).to_parquet(f'xs/ext/lowb/{tk}.parquet'); print(tk, end=' ', flush=True)
else:
    T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/lowb/*.parquet')); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
    T = T[T.entry >= '2019-01-01']; T['ratio'] = 0.0; T['per'] = np.where(T.entry.dt.year >= 2022, '22-26', '19-21')
    g = T.groupby(['side', 'per']).ret.agg(['mean', 'count', lambda s: (s > 0).mean()]); g.columns = ['avg', 'n', 'win']; g['avg'] *= 100
    print(f'{T.tk.nunique()} low-beta names'); print(g.unstack().round(3).to_string())
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
    print(f'\n{"low-beta book, 6 slots":<40}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
    def show(nm, Z):
        res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
        print(f'  {nm:<38}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res), flush=True)
    show('long only', T[T.side == 'long']); show('short only', T[T.side == 'short']); show('long + short', T)
