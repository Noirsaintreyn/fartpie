"""MULTI-ASSET, PURE VECTOR EXITS (no added stop, no target: position = the vector regime; enter next bar open after a flip, exit at the next opposite flip's next open; ETFs flat at the RTH close, futures run through). Nov 11 2026.
Part A: ETF proxies 15m RTH 2018-2026 (cost 1bp, FX/USO 3bp). Part B: yfinance =F futures 1h, last ~730 days (cost 1bp), all hours."""
import sys, glob, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']; NSL = {}
exec(open('/home/claude/fartpie/xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0], NSL); sys.argv = _a; vec = NSL['vec']
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
def trades(x, cost, flat_daily):
    reg = np.asarray(vec(x[['open', 'high', 'low', 'close']])); o, c = x.open.values, x.close.values; dy = x.day.values; rows = []; pos = 0; e = None; ed = None
    for i in range(1, len(x) - 1):
        newday = dy[i] != dy[i - 1]
        if flat_daily and pos != 0 and newday:                       # close at previous bar's close
            rows.append((ed, pos * (c[i - 1] - e) / e * 1e4 - cost)); pos = 0
        if reg[i] != reg[i - 1] and reg[i] != 0 and not (flat_daily and dy[i + 1] != dy[i]):
            if pos != 0: rows.append((ed, pos * (o[i + 1] - e) / e * 1e4 - cost))
            pos = int(reg[i]); e = o[i + 1]; ed = x.index[i + 1] if not flat_daily else pd.Timestamp(dy[i])
    return pd.DataFrame(rows, columns=['t', 'bps'])
def report(name, T, split):
    a = T[T.t < split].bps; b = T[T.t >= split].bps
    print(f'{name:10} n {len(T):5d} bps/trade {T.bps.mean():+6.1f} win {np.mean(T.bps>0):3.0%} t {tstat(T.bps):+5.1f} | first {len(a):4d} {a.mean():+6.1f} ({tstat(a):+.1f}) | second {len(b):4d} {b.mean():+6.1f} ({tstat(b):+.1f})')
print('== A. ETF proxies, 15m RTH, pure vector exits')
for t, c in (('QQQ', 1), ('SPY', 1), ('IWM', 1), ('DIA', 1), ('GLD', 1), ('SLV', 2), ('USO', 3), ('TLT', 1), ('FXE', 3), ('FXY', 3), ('FXB', 3)):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{t}_15m_*.parquet')[0]; x = pd.read_parquet(f); dt = pd.to_datetime(x.datetime)
    x['dt'] = (dt.dt.tz_localize('UTC') if dt.dt.tz is None else dt).dt.tz_convert('America/New_York').dt.tz_localize(None); x['day'] = x.dt.dt.normalize(); x['m'] = x.dt.dt.hour * 60 + x.dt.dt.minute
    x = x[(x.m >= 570) & (x.m < 960)].sort_values('dt').reset_index(drop=True); report(t, trades(x, c, True), pd.Timestamp('2022-01-01'))
print('\n== B. yfinance futures 1h (~730d), pure vector exits, run through')
for t in ('NQ=F', 'ES=F', 'GC=F', 'SI=F', 'CL=F', 'HG=F', 'NG=F', 'ZN=F', '6E=F', '6J=F', 'ZC=F'):
    d = yf.download(t, period='729d', interval='1h', progress=False, auto_adjust=False)
    if d.empty: print(t, 'no data'); continue
    d.columns = [c[0].lower() for c in d.columns]; d.index = pd.to_datetime(d.index).tz_localize(None); d['day'] = d.index.normalize()
    T = trades(d[['open', 'high', 'low', 'close', 'day']], 1, False); report(t, T, T.t.iloc[len(T) // 2])
