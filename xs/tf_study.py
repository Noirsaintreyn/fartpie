"""Nov 4 2026: is 15m special? DECLARED BEFORE RESULTS. Bare vector (no stops/slots/ratio) on 5m-cache names, premarket+RTH bars aggregated to
5/10/15/20/30/45/60m time bars, Heikin-Ashi 15m/30m, and 15m with pivot length scaled to constant clock time. Signals use the last completed bar before each RTH bar
(entry/exit at RTH-bar opens, no entries after 14:00, 2bp cost). Metrics per period (2019-21, 2022-26) pooled over tickers: avg trade, win, share >=+15%,
trades/ticker-yr, compounded return per ticker-year while in market (mean of per-ticker CAGR of the always-follow-signal equity). Pivot-in-bars fixed unless stated."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
sys.path.insert(0, 'xs'); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open('xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
D = '/mnt/user-data/uploads/fartpie/lse_ohlcv_cache'; C2 = 2e-4
def aggm(H, mins):
    x = H.set_index('dt')[['open', 'high', 'low', 'close']]
    g = x.groupby([x.index.normalize(), pd.Grouper(freq=f'{mins}min', origin='start_day', offset='4h')])
    r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna()
    r.index = r.index.get_level_values(1); r = r.reset_index(); r.columns = ['dt', 'open', 'high', 'low', 'close']; return r
def ha(r):
    c = (r.open + r.high + r.low + r.close) / 4; o = np.zeros(len(r)); o[0] = (r.open[0] + r.close[0]) / 2
    for i in range(1, len(r)): o[i] = (o[i - 1] + c[i - 1]) / 2
    out = r.copy(); out['open'] = o; out['close'] = c; out['high'] = np.maximum(r.high, np.maximum(o, c)); out['low'] = np.minimum(r.low, np.minimum(o, c)); return out
VAR = [('5m', 5, 0, {}), ('10m', 10, 0, {}), ('15m', 15, 0, {}), ('20m', 20, 0, {}), ('30m', 30, 0, {}), ('45m', 45, 0, {}), ('60m', 60, 0, {}),
       ('HA15', 15, 1, {}), ('HA30', 30, 1, {}), ('5m piv=15min', 5, 0, dict(PIVOT_LEN=9)), ('30m piv=15min', 30, 0, dict(PIVOT_LEN=3))]  # base PIVOT_LEN=VC.PIVOT_LEN
def trades(r, st, mins):
    # RTH bar opens only; state from last completed bar
    d = r.dt.values; m = (r.dt.dt.hour * 60 + r.dt.dt.minute).values; o = r.open.values
    idx = [i for i in range(1, len(r)) if 570 <= m[i] < 960]; T = []; pos = False; prev = 0
    for i in idx:
        s = st[i - 1]
        if not pos:
            if s == 1 and prev != 1 and m[i] < 840: pos, e = True, i
        elif s == -1: T.append((d[e], d[i], o[i] / o[e] - 1 - C2)); pos = False
        prev = s
    return T
def one(f):
    tk = os.path.basename(f).split('_5m_')[0]; H = load(f); out = []
    for nm, mins, hflag, kw in VAR:
        r = aggm(H, mins)
        if hflag: r = ha(r)
        st = vec(r, **kw)
        for a, b, x in trades(r, st, mins): out.append((nm, tk, a, b, x))
    return out
if __name__ == '__main__':
    fs = sorted(glob.glob(f'{D}/*_5m_2018-01-01_*.parquet')); fs = [f for f in fs if os.path.basename(f).split('_5m_')[0] not in ('SPY', 'QQQ')]
    with Pool(8) as p: R = sum(p.map(one, fs), [])
    T = pd.DataFrame(R, columns=['v', 'tk', 'entry', 'exit', 'ret']); T.to_parquet('/tmp/claude-0/ml/tf_trades.parquet')
    T['per'] = np.where(T.entry.dt.year <= 2021, '19-21', '22-26'); nt = T.tk.nunique(); yrs = {'19-21': 3, '22-26': 4.8}
    print('tickers', nt)
    print(f'{"variant":<16}' + ''.join(f'| {p}: avg  win  >=15%  tr/tk-yr  ret/tk-yr(sum)  ' for p in yrs))
    for nm, _, _, _ in VAR:
        row = f'{nm:<16}'
        for p in yrs:
            z = T[(T.v == nm) & (T.per == p)]
            row += f'| {z.ret.mean()*100:+.2f}% {z.ret.gt(0).mean():.0%} {z.ret.ge(.15).mean():.1%} {len(z)/nt/yrs[p]:6.1f}  {z.ret.sum()/nt/yrs[p]*100:+6.1f}%   '
        print(row)
