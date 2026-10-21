"""Oct 21 2026: concepts from equity models with lasting out-of-sample evidence, on the full-S&P screener (ratio >= 1.5, live rules).
 A  volatility-managed book (Barroso & Santa-Clara; Moreira & Muir): position size scaled by target / recent volatility
    A1 market: QQQ 20d realized vol vs its trailing 1y median    A2 strategy: the screener's own realized trade P&L vol (closed trades, 60d)
    A3 stock: each name's 20d vol vs the median name (risk parity per slot)
 B  residual momentum (Blitz, Huij & Martens): 120d return net of SPY beta (skip last 5d), scaled by residual vol -> tie-break
 C  industry momentum (Moskowitz & Grinblatt): the stock's sector ETF 20d / 60d return -> tie-break, and skip names in the weakest sectors
Sizes are capped 0.5x-1.5x of a 1/6 slot. All inputs through the prior close."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from collections import deque
exec(open('xs/slot_tests.py').read().split('P = [(np.datetime64')[0])
T = T.sort_values('entry').reset_index(drop=True); d = T.entry.dt.normalize()
def etf(t):
    x = pd.read_parquet(f'etf_daily/{t}.parquet'); x['d'] = pd.to_datetime(x.datetime).dt.tz_localize(None).dt.normalize(); return x.drop_duplicates('d').set_index('d').adjclose
qqq = etf('QQQ'); spy_ = etf('SPY'); qv = qqq.pct_change().rolling(20).std()
T['mvol'] = (qv.rolling(252).median() / qv).shift(1).reindex(d, method='ffill').values           # >1 = calmer than usual
med = T.vol20.median(); T['svol'] = med / T.vol20
r = CL.pct_change(); sr = spy_.reindex(CL.index).pct_change()
beta = r.rolling(252, min_periods=120).cov(sr).div(sr.rolling(252, min_periods=120).var(), axis=0)
res = r.sub(beta.mul(sr, axis=0))
RM = (res.rolling(115).sum().shift(5) / (res.rolling(120).std() * np.sqrt(115))).shift(1)
T['resmom'] = [RM[t].asof(x) if t in RM.columns else np.nan for t, x in zip(T.tk, d)]
SEC = json.load(open('sp500_sectors.json')); EM = {'Information Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV', 'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP',
    'Industrials': 'XLI', 'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB', 'Energy': 'XLE', 'Communication Services': 'XLC'}
SECF = {'GOOG': 'XLC', 'META': 'XLC', 'FB': 'XLC', 'BRKB': 'XLF', 'NFLX': 'XLC', 'TSLA': 'XLY', 'AMZN': 'XLY', 'COIN': 'XLF', 'HOOD': 'XLF', 'PLTR': 'XLK', 'CRWD': 'XLK', 'APP': 'XLK', 'CVNA': 'XLY'}
E = {v: etf(v) for v in EM.values()}
def secmom(tk, x, n):
    e = SECF.get(tk) or EM.get(SEC.get(tk, ''), None)
    if not e: return np.nan
    p = E[e]; p = p[p.index < x]
    return p.iloc[-1] / p.iloc[-1 - n] - 1 if len(p) > n else np.nan
T['sec20'] = [secmom(t, x, 20) for t, x in zip(T.tk, d)]; T['sec60'] = [secmom(t, x, 60) for t, x in zip(T.tk, d)]
print('coverage', {c: f'{T[c].notna().mean():.0%}' for c in ('mvol', 'svol', 'resmom', 'sec20', 'sec60')})
T['sec60rk'] = T.sec60                                                         # relative to the other sectors that day
def sim3(Z, a, b, key='ratio', size=None, stratvol=False, n=6, brk=.08):
    Z = Z[(Z.entry >= a) & (Z.entry < b)].sort_values(['entry', key, 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; recent = deque()
    W = (1 / n) * np.clip(Z[size].fillna(1).values, .5, 1.5) if size else np.full(len(Z), 1 / n)
    for e, x, r_, w in zip(Z.entry.values, Z.exit.values, Z.ret.values, W):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr, ww in done: real += np.log1p(rr * ww); hist.append((xt, real)); win.append((xt, real)); recent.append((xt, rr))
        op = [o for o in op if o > e]
        if len(op) >= n: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        if stratvol:
            while recent and recent[0][0] < e - np.timedelta64(60, 'D'): recent.popleft()
            if len(recent) >= 10:
                v = np.std([q for _, q in recent]); w = (1 / n) * float(np.clip(STRAT_MED / v, .5, 1.5)) if v > 0 else w
        op.append(x); pend.append((x, r_, w))
    for xt, rr, ww in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr * ww); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); yy = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / yy) - 1, (eq / eq.cummax() - 1).min()
STRAT_MED = T[T.entry < '2020-01-01'].ret.std()                                 # target = 2019 trade-return vol (known by 2020)
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z=None, **kw):
    Z = T if Z is None else Z
    print(f'  {nm:<52}' + ''.join(f'{c:>+9.1%} /{dd:>5.0%}' for c, dd in (sim3(Z, a, b, **kw) for a, b in P)), flush=True)
print(f'\n{"S&P, ratio >= 1.5, live rules":<54}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live')
show('A1 size by market vol (QQQ 20d vs 1y median)', size='mvol')
show('A2 size by the screener\'s own trade vol (60d)', stratvol=True)
show('A3 size by stock vol (risk parity per slot)', size='svol')
show('B  tie-break: residual momentum', key='resmom')
show('C  tie-break: sector 20d momentum', key='sec20')
show('C  tie-break: sector 60d momentum', key='sec60')
rk = T.groupby(d).sec60.rank(pct=True)
weak = T.sec60 < T.groupby(d.dt.to_period('M')).sec60.transform(lambda s: s.quantile(.25))
show('C  skip names whose sector 60d return < 0', T[~(T.sec60 < 0)])
show('C  skip names whose sector 20d return < 0', T[~(T.sec20 < 0)])
