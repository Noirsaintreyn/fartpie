"""Oct 26 2026: the five ideas from Rey's research note, tested as information on the live vector trades (all known before entry).
 1 duration   age of the 1h vector uptrend at entry (1h bars since it flipped bull) and age of the daily vector uptrend (days)
 2 factor     20-day trend split into market (SPY), sector (its SPDR ETF net of SPY) and stock-specific residual (net of both, betas
              from the prior 120 days) - Mapping Momentum style
 3 changepoint  Page-Hinkley changepoint count over the last 5 days (how recently the 15m return process broke) - selective forgetting
 4 fast/slow  daily vector state at entry vs the 15m / 1h entry (disagreement)
 5 earnings clock  trading days since the last earnings release and the stock's return since the close before it (event momentum)
Stage 1 (per name, sharded): python3 xs/trend_features.py 1 BARS_DIR ETF_DIR EARN_DIR TRADES_GLOB OUT_DIR
Stage 2 (report):            python3 xs/trend_features.py 2 OUT_DIR"""
import sys, os, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A_ = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
if A_[0] == '1':
    exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
    BARS, ETF, EARN, TG, OUT = A_[1:6]; os.makedirs(OUT, exist_ok=True)
    SEC = json.load(open(os.path.join(HERE, '..', 'sp500_sectors.json')))
    SE = {'Information Technology': 'XLK', 'Health Care': 'XLV', 'Financials': 'XLF', 'Consumer Discretionary': 'XLY', 'Communication Services': 'XLC',
          'Industrials': 'XLI', 'Consumer Staples': 'XLP', 'Energy': 'XLE', 'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB'}
    def etf(t):
        f = glob.glob(f'{ETF}/{t}*.parquet')[0]; d = pd.read_parquet(f)
        if 'datetime' in d: d = d.set_index('datetime')
        d.index = pd.to_datetime(d.index); d.index = (d.index.tz_localize(None) if d.index.tz is not None else d.index).normalize()
        c = d['Close'] if 'Close' in d else d['close']; c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c; c = c.sort_index(); return c[~c.index.duplicated(keep='last')]
    SPY = np.log(etf('SPY')).diff()
    T = pd.concat(pd.read_parquet(f) for f in glob.glob(TG)); T = T[T['var'] == 'vec']
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
    SH, NS = int(os.getenv('SHARD', 0)), int(os.getenv('NSHARD', 1))
    for tk in sorted(T.tk.unique())[SH::NS]:
        fo = f'{OUT}/{tk}.parquet'
        if os.path.exists(fo) or tk not in fs: continue
        H = load(fs[tk]); reg, s1 = states(H, vec); n = len(H); idx = np.arange(n)
        f1 = np.r_[False, (s1[1:] == 1) & (s1[:-1] != 1)]; lastf1 = pd.Series(np.where(f1, idx, np.nan)).ffill().values
        R = H[H.m >= 570]; D = R.groupby(R.dt.dt.normalize()).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'))
        sd = pd.Series(vec(D.reset_index().rename(columns={'dt': 'd'})), index=D.index)
        dflip = (sd == 1) & (sd.shift(1) != 1); dage = pd.Series(np.where(sd == 1, 1, 0), index=D.index)
        dage = dage.groupby((dage == 0).cumsum()).cumsum()                       # consecutive bull days
        lr = np.log(D.close).diff(); sec = SE.get(SEC.get(tk, ''), None)
        rs = np.log(etf(sec)).diff() if sec else pd.Series(0.0, index=lr.index)
        J = pd.concat([lr.rename('s'), SPY.rename('m'), rs.rename('x')], axis=1).dropna()
        bx = J.x.rolling(120).cov(J.m) / J.m.rolling(120).var(); xe = J.x - bx.shift(1) * J.m          # sector net of market
        J['xe'] = xe; cov = lambda a, b: J[a].rolling(120).cov(J[b])
        vm, ve, cme = J.m.rolling(120).var(), J.xe.rolling(120).var(), cov('m', 'xe')
        det = vm * ve - cme ** 2; bm = (cov('s', 'm') * ve - cov('s', 'xe') * cme) / det; be = (cov('s', 'xe') * vm - cov('s', 'm') * cme) / det
        J['e'] = J.s - bm.shift(1) * J.m - be.shift(1) * J.xe
        F20 = J[['m', 'xe', 'e', 's']].rolling(20).sum()
        z = np.log(H.close).diff().values; sdz = pd.Series(z).rolling(260, min_periods=60).std().values; z = z / sdz
        sp = sm = 0.0; cp = np.zeros(n)
        for i in range(n):                                                     # Page-Hinkley on the 15m mean, both directions
            if np.isnan(z[i]): continue
            sp = max(0, sp + z[i] - .5); sm = max(0, sm - z[i] - .5)
            if sp > 8 or sm > 8: cp[i] = 1; sp = sm = 0.0
        cp5 = pd.Series(cp).rolling(130, min_periods=1).sum().values           # breaks in the last ~5 sessions
        try:
            E = pd.read_csv(f'{EARN}/{tk}.csv'); E['t'] = pd.to_datetime(E['Earnings Date'], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
            E = E[E['Reported EPS'].notna()].sort_values('t'); ed = E.t.values
        except Exception: ed = np.array([], dtype='datetime64[ns]')
        days = D.index.values; rows = []
        for t in T[T.tk == tk].itertuples():
            e = np.datetime64(t.entry); k = np.searchsorted(H.dt.values, e) - 1
            if k < 1: continue
            d0 = np.datetime64(pd.Timestamp(t.entry).normalize()); pd_ = days[days < d0]
            if not len(pd_): continue
            pdy = pd.Timestamp(pd_[-1])                                          # prior session (all daily features known at its close)
            a1 = (k - lastf1[k]) if s1[k] == 1 and lastf1[k] == lastf1[k] else -1
            j = ed[ed < e]; dse = rse = np.nan
            if len(j):
                le = pd.Timestamp(j[-1]); before = days[days < np.datetime64(le.normalize() if le.hour >= 9 else le.normalize())]
                b0 = pd.Timestamp(before[-1]) if len(before) else None
                dse = int(((days > np.datetime64(b0)) & (days <= np.datetime64(pdy))).sum()) if b0 is not None else np.nan
                rse = D.close[pdy] / D.close[b0] - 1 if b0 is not None else np.nan
            f20 = F20.loc[:pdy].iloc[-1] if len(F20.loc[:pdy]) else pd.Series(np.nan, index=F20.columns)
            rows.append((tk, t.h1, t.entry, t.exit, t.ret, a1, int(dage.get(pdy, 0)), int(sd.get(pdy, 0)), f20.m, f20.xe, f20.e, f20.s, cp5[k], dse, rse))
        pd.DataFrame(rows, columns=['tk', 'h1', 'entry', 'exit', 'ret', 'age1h', 'dage', 'dstate', 'mkt20', 'sec20', 'res20', 'stk20', 'cp5', 'dse', 'rse']).to_parquet(fo); print(tk, len(rows), flush=True)
else:
    OUT = A_[1]
    exec(open(os.path.join(HERE, 'book_mtm.py')).read().split("if __name__")[0].replace('/tmp/claude-0/ml/stopbook.py', os.path.join(HERE, 'stop_book.py')) if False else '')
