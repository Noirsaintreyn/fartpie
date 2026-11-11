"""LSE 15m FUTURES (GOLD.F CRUDEOIL.F SILVER.F ... + NASDAQ100.F control), 2017-04..2026-11, ALL hours, continuous. Vector flips (signal_lab.vec) on 15m bars, enter next bar open, exit at the opposite flip's next open (PURE vector exit) or at a stop of k*ADR20 (ADR = prior 20 days' mean daily high-low) checked on bar high/low. Cost 1 bp round trip. bps per trade; eras 2017-21 / 2022-26. Nov 11 2026."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']; NSL = {}
exec(open('/home/claude/fartpie/xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0], NSL); sys.argv = _a; vec = NSL['vec']
D = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/lsefut/lse_ohlcv_cache/'; COST = 1.0
def tstat(x): x = np.asarray(x, float); x = x[~np.isnan(x)]; return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
def load(s):
    x = pd.read_parquet(glob.glob(D + f'{s}_15m_*.parquet')[0]); x['dt'] = pd.to_datetime(x.datetime); x = x.sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
    x['dt'] = x.dt.dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None); x['day'] = x.dt.dt.normalize(); x['m'] = x.dt.dt.hour * 60 + x.dt.dt.minute; return x
def trades(x, k, lo=570, hi=960):
    x = x[(x.m >= lo) & (x.m < hi)].reset_index(drop=True)
    reg = np.asarray(vec(x[['open', 'high', 'low', 'close']])); o, h, l, c = x.open.values, x.high.values, x.low.values, x.close.values; dt = x.dt.values
    dr_ = x.groupby('day').apply(lambda g: g.high.max() - g.low.min()); adr = dr_.shift(1).rolling(20).mean(); dyv = x.day.values; rows = []; n = len(x); i = 1
    while i < n - 1:
        if dyv[i] != dyv[i - 1] or reg[i] == reg[i - 1] or reg[i] == 0 or x.m.values[i] + 15 < lo + 15 or dyv[min(i + 1, n - 1)] != dyv[i]: i += 1; continue
        a = adr.get(pd.Timestamp(dyv[i]), np.nan)
        if not np.isfinite(a): i += 1; continue
        d = int(reg[i]); e = o[i + 1]; S = e - d * k * a if k else None; j = i + 1; ex = None; wr = 0.0
        while j < n - 1 and dyv[j + 1] == dyv[i]:
            wr = min(wr, (l[j] - e) if d > 0 else (e - h[j]))
            if S is not None and ((d > 0 and l[j] <= S) or (d < 0 and h[j] >= S)): ex = S; break
            if reg[j] == -d and reg[j - 1] != reg[j]: ex = o[j + 1]; break
            j += 1
        if ex is None:
            ex = c[j]  # flat at window end
        rows.append((pd.Timestamp(dt[i + 1]), d * (ex - e) / e * 1e4 - COST, wr / e * 1e4)); i = max(j, i + 1)
    return pd.DataFrame(rows, columns=['t', 'bps', 'mae'])
def line(name, T):
    a = T[T.t < '2022-01-01'].bps; b = T[T.t >= '2022-01-01'].bps
    return f'{name:22} n {len(T):5d} bps/trade {T.bps.mean():+6.1f} win {np.mean(T.bps>0):3.0%} t {tstat(T.bps):+5.1f} | 2017-21 {a.mean():+6.1f} ({tstat(a):+.1f}) | 2022-26 {b.mean():+6.1f} ({tstat(b):+.1f}) | worst {T.bps.min():+.0f}bps'
WIN = {'NASDAQ100.F': ((570, 960),), 'GOLD.F': ((570, 960), (500, 810)), 'CRUDEOIL.F': ((570, 960), (540, 870))}
for s in sys.argv[1:] or list(WIN):
    x = load(s); vol = x.groupby('m').volume.mean(); print(f'\n== {s}: busiest ET minutes {list(vol.nlargest(3).index)} (570=09:30)', flush=True)
    for lo, hi in WIN.get(s, ((570, 960),)):
        print(f' -- window {lo//60}:{lo%60:02d}-{hi//60}:{hi%60:02d} ET, flat at window end')
        for nm, k in (('pure vector exit', None), ('stop 0.25 ADR', .25), ('stop 0.40 ADR', .40), ('stop 0.15 ADR', .15)): print(' ' + line(nm, trades(x, k, lo, hi)), flush=True)
