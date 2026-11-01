"""Nov 1 2026: the short side. Bear-flip shorts on the live universe (S&P, ratio >= 1.5), its own 6-slot book (ratio tie-break, 8% breaker).
Gates known at entry: SPY downtrend (daily vector bear AND under 200d) / SPY under 200d / QQQ 20d return < 0 / stock's daily vector bear /
options: ratio >= 2.5, call share of options volume low (23 names). Then the combined long + short book."""
import os, glob, numpy as np, pandas as pd
from collections import deque
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
import sys; sys.path.insert(0, '/home/claude/fartpie'); cwd = os.getcwd(); os.chdir('/home/claude/fartpie'); import live_screener as LS; os.chdir(cwd)
def simq(Z, n=6, brk=.08):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r_ in zip(Z.entry.values, Z.exit.values, Z.ret.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= n: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append(x); pend.append((x, r_))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    return s.groupby(s.index.normalize()).last()                                 # cumulative log equity by day
HALF = [('2019-21', '2019-01-01', '2022-01-01'), ('2022-26', '2022-01-01', '2027-01-01')]
def stat(eq):                                                                     # eq = cumulative log equity (daily, realized)
    out = []
    for h, a, b in HALF:
        e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); y = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / y) - 1; d = (e / e.cummax() - 1).min(); out.append((c, d))
    return out
def fmt(o): return '   '.join(f'{c:+.1%}/yr DD {d:.0%} ({c / -d if d else 0:.2f})' for c, d in o)
Ls = prep(glob.glob('tm/*.parquet')); Ls = Ls[Ls['var'] == 'vec']
Sh = prep(glob.glob('sh/*.parquet'))
E = '/mnt/user-data/uploads/fartpie/etf_daily'
def dly(t): x = pd.read_parquet(f'{E}/{t}.parquet'); x['dt'] = pd.to_datetime(x.datetime).dt.normalize(); return x.set_index('dt')
S = dly('SPY'); V = LS.vec(S.reset_index()[['dt', 'open', 'high', 'low', 'close']].assign(volume=1.0))
mk = pd.DataFrame({'dn': (V.reg.values == -1) & (S.close.values < S.close.rolling(200).mean().values), 'u200': (S.close < S.close.rolling(200).mean()).values}, index=S.index)
Qd = dly('QQQ'); mk['q20neg'] = (Qd.close / Qd.close.shift(20) - 1 < 0).reindex(mk.index).fillna(False); mk = mk.shift(1).fillna(False)
Sh['d'] = Sh.entry.dt.normalize(); Sh = Sh.join(mk.reindex(Sh.d.values).reset_index(drop=True).set_index(Sh.index))
vd = {}
for tk in Sh.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')[0]; H = pd.read_parquet(f); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; R = H[(m >= 570) & (m < 960)]
    D = R.set_index('dt').resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
    vd[tk] = pd.Series(LS.vec(D).reg.values, index=D.dt).shift(1)
Sh['vDbear'] = [vd[t].asof(d) == -1 for t, d in zip(Sh.tk, Sh.d)]
cs = {}
for f in glob.glob('/mnt/user-data/uploads/fartpie/xs/ext/opt_summary/*.csv'):
    o = pd.read_csv(f, parse_dates=['date']).set_index('date'); c = o.call_vol + o.put_vol; cs[os.path.basename(f)[:-4]] = (o.call_vol.rolling(5).sum() / c.rolling(5).sum()).shift(1)
Sh['cshare'] = [cs[t].asof(d) if t in cs else np.nan for t, d in zip(Sh.tk, Sh.d)]
for c in ('dn', 'u200', 'q20neg'): Sh[c] = Sh[c].fillna(False).astype(bool)
Sh['h'] = np.where(Sh.entry.dt.year <= 2021, '2019-21', '2022-26')
print(f'short trades {len(Sh)}; avg {Sh.ret.mean():+.2%}, win {(Sh.ret > 0).mean():.0%}')
G = {'all bear flips': Sh.index == Sh.index, 'SPY downtrend': Sh.dn, 'SPY under 200d': Sh.u200, 'QQQ 20d down': Sh.q20neg, 'stock daily vector bear': Sh.vDbear,
     'ratio >= 2.5': Sh.ratio >= 2.5, 'ratio >= 2.5 + SPY under 200d': (Sh.ratio >= 2.5) & Sh.u200, 'daily bear + SPY under 200d': Sh.vDbear & Sh.u200,
     'low call share (<0.5)': Sh.cshare < .5, 'low call share + SPY under 200d': (Sh.cshare < .5) & Sh.u200, 'daily bear + QQQ 20d down': Sh.vDbear & Sh.q20neg}
G = {k: pd.Series(v, index=Sh.index).fillna(False).astype(bool) for k, v in G.items()}
print('\nA) avg short trade % (n) by gate          2019-21              2022-26')
for nm, g in G.items():
    t = Sh[g].groupby('h').ret.agg(['size', 'mean']); print(f'  {nm:<34}' + '   '.join(f'{t.loc[h, "mean"] * 100:+.2f}% ({int(t.loc[h, "size"])})' if h in t.index else '   -   ' for h in ('2019-21', '2022-26')))
print('\nB) short book alone (6 slots), return / DD (return per DD)')
for nm, g in G.items(): print(f'  {nm:<34}{fmt(stat(simq(Sh[g])))}', flush=True)
LE = simq(Ls); print(f'\nC) long book alone                 {fmt(stat(LE))}')
idx = pd.bdate_range('2019-01-01', LE.index.max()); le = LE.reindex(idx, method='ffill').fillna(0).diff().fillna(0)
for nm in ('all bear flips', 'SPY downtrend', 'SPY under 200d', 'daily bear + SPY under 200d', 'daily bear + QQQ 20d down'):
    se = simq(Sh[G[nm]]).reindex(idx, method='ffill').fillna(0).diff().fillna(0)
    print(f'   corr(long, short {nm}) {le.corr(se):+.2f}')
    for w in (.5, 1.0):
        print(f'   long + {w:.0%} short [{nm}]: {fmt(stat((le + w * se).cumsum()))}', flush=True)
