"""Fewer positions when conditions are bad: max open positions = N_good normally, N_bad when the market is weak.
Each position stays 1/6 of the book (cash for the rest) unless noted.  Bad = SPY below its 50-day (prior close) or VIX (prior close) > 25.
Optional: the Oct 20 drawdown breaker (no new entries while realized book < -8% vs its 28-day high, window-start level counted).
Live trades (xs/ext/ahx 'base+sticky'), ratio >= 2.5."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.argv = ['x', '2']
exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])
T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ahx/*.parquet')); T = T[T['var'] == 'base+sticky']
T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5]
D = pd.read_parquet('etf_daily/SPY.parquet'); D['d'] = pd.to_datetime(D.datetime).dt.tz_localize(None).dt.normalize(); c = D.drop_duplicates('d').set_index('d').adjclose
vix = pd.read_parquet('etf_daily/VIX.parquet').close; vix.index = pd.to_datetime(vix.index).normalize()
dd_ = T.entry.dt.normalize()
T['spy_bad'] = (~(c > c.rolling(50).mean())).shift(1).reindex(dd_, method='ffill').fillna(False).values.astype(bool)
T['vix_bad'] = (vix > 25).shift(1).reindex(dd_, method='ffill').fillna(False).values.astype(bool)
Z0 = T.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
E, X, R, SB, VB = (Z0[k].values for k in ('entry', 'exit', 'ret', 'spy_bad', 'vix_bad'))
def sim(a, b, ng=6, nb=6, cond='spy', w=6, brk=0.0, ret_series=False):
    op = []; ev = []; pend = []; real = 0.0; hist = []
    for e, x, r, sb, vb in zip(E, X, R, SB, VB):
        if e < a or e >= b: continue
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / w); hist.append((xt, real))
        op = [o for o in op if o > e]
        bad = sb if cond == 'spy' else vb if cond == 'vix' else (sb or vb)
        if len(op) >= (nb if bad else ng): continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D'); inw = [h for t_, h in hist if t_ >= t0]; pre = [h for t_, h in hist if t_ < t0]
            hi = max([pre[-1] if pre else 0.0] + inw)
            if real - hi < np.log1p(-brk): continue
        op.append(x); pend.append((x, r)); ev.append((x, r / w))
    s = pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
    eq = (1 + s).groupby(s.index.normalize()).prod().cumprod()
    if ret_series: return eq
    y = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min()
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
if __name__ == '__main__':
    print(f'{"max positions (good / bad market)":<46}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
    def show(nm, **kw):
        print(f'  {nm:<44}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(a, b, **kw) for a, b in P)), flush=True)
    show('live: 6 / 6')
    show('live + breaker 8%', brk=.08)
    for cond, cn in (('spy', 'SPY < 50d'), ('vix', 'VIX > 25'), ('either', 'SPY < 50d or VIX > 25')):
        print(f'  -- bad = {cn}')
        for nb in (4, 3, 2, 0): show(f'6 / {nb}', nb=nb, cond=cond)
        show('8 / 4 (12.5% each)', ng=8, nb=4, w=8, cond=cond)
        show('6 / 3 + breaker 8%', nb=3, cond=cond, brk=.08)
        show('6 / 4 + breaker 8%', nb=4, cond=cond, brk=.08)
