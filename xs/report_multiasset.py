"""MULTI-ASSET VECTOR FLIPS (ETF proxies for GC/CL etc: GLD SLV USO TLT FXE FXY FXB SPY IWM DIA QQQ-control), Nov 11 2026. No GC/CL futures data in the repo, ETFs 15m RTH 2018-2026.
Same recipe as NQ V15a: vector regime flips on 15m bars (signal_lab.vec), enter next bar open during 09:45-15:30, stop 0.15*ADR20, exit on opposite flip else the close, max 4 flips/day. Cost 1bp (USO/FX 3bp).
Questions: (1) edge per asset, (2) daily correlation with the NQ system and behavior on NQ BAD days / NQ-skipped (small-gap) days, (3) would a regime switch help."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']; NSL = {}
exec(open('/home/claude/fartpie/xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0], NSL); sys.argv = _a; vec = NSL['vec']
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; DIR = '/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/'
import glob
def load(t):
    f = glob.glob(DIR + f'{t}_15m_*.parquet')[0]; x = pd.read_parquet(f)
    x['dt'] = pd.to_datetime(x.datetime).dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None) if pd.to_datetime(x.datetime).dt.tz is None else pd.to_datetime(x.datetime).dt.tz_convert('America/New_York').dt.tz_localize(None)
    x['day'] = x.dt.dt.normalize(); x['m'] = x.dt.dt.hour * 60 + x.dt.dt.minute; x = x[(x.m >= 570) & (x.m < 960)].sort_values('dt').reset_index(drop=True); return x
def run(t, cost_bps):
    x = load(t); x['reg'] = np.asarray(vec(x[['open', 'high', 'low', 'close']]))
    rng = x.groupby('day').apply(lambda g: g.high.max() - g.low.min()); adr = rng.shift(1).rolling(20).mean()
    o, h, l, c = x.open.values, x.high.values, x.low.values, x.close.values; reg = x.reg.values; dy = x.day.values; m = x.m.values; rows = []
    i = 1
    while i < len(x) - 1:
        if dy[i] != dy[i - 1] or reg[i] == reg[i - 1] or reg[i] == 0 or not (585 <= m[i] + 15 <= 930) or dy[i + 1] != dy[i]: i += 1; continue
        d = pd.Timestamp(dy[i]); a = adr.get(d, np.nan)
        if not np.isfinite(a): i += 1; continue
        dr = int(reg[i]); e = o[i + 1]; S = e - dr * .15 * a; j = i + 1; ex = None; cnt = 0
        while j < len(x) and dy[j] == dy[i]:
            if (dr > 0 and l[j] <= S) or (dr < 0 and h[j] >= S): ex = S; break
            if j + 1 < len(x) and dy[j + 1] == dy[i] and reg[j] != reg[j - 1] and j > i + 0 and reg[j] == -dr: ex = o[j + 1]; break
            j += 1
        if ex is None: ex = c[min(j, len(x) - 1) if dy[min(j, len(x) - 1)] == dy[i] else j - 1]
        rows.append((d, int(m[i]), dr, (dr * (ex - e) / e) * 1e4 - cost_bps)); i = max(j, i + 1)
    return pd.DataFrame(rows, columns=['day', 'm', 'dir', 'bps'])
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
TD = pd.read_pickle(SC + 'tiers_days.pkl'); TD.index = pd.to_datetime(TD.index).normalize(); TD['T'] = TD.tier.str[0].astype(int)
R = {}
print('asset | trades | bps/trade | win | t | 2018-21 (n, bps, t) | 2022-26 (n, bps, t) | bps/day | daily corr w/ NQ system | on NQ BAD days(tier1-2): bps/day (t) | on NQ small-gap days')
for t, c in (('QQQ', 1), ('SPY', 1), ('IWM', 1), ('DIA', 1), ('GLD', 1), ('SLV', 2), ('USO', 3), ('TLT', 1), ('FXE', 3), ('FXY', 3), ('FXB', 3)):
    try: T = run(t, c)
    except Exception as e: print(t, 'ERR', e); continue
    R[t] = T; a = T[T.day < '2022-01-01'].bps; b = T[T.day >= '2022-01-01'].bps
    dd = T.groupby('day').bps.sum(); J = TD.join(dd.rename('a'), how='left').fillna({'a': 0}); bad = J[J['T'] <= 2]; sg = J[J.gap_rel <= J.gap_rel.quantile(1 / 3)]
    print(f'{t:4} | {len(T):5d} | {T.bps.mean():+6.1f} | {np.mean(T.bps>0):3.0%} | {tstat(T.bps):+5.1f} | {len(a):4d} {a.mean():+6.1f} {tstat(a):+5.1f} | {len(b):4d} {b.mean():+6.1f} {tstat(b):+5.1f} | {dd.mean():+5.1f} | {J.pts.corr(J.a):+.2f} | {bad.a.mean():+5.1f} ({tstat(bad.a):+.1f}) | {sg.a.mean():+5.1f} ({tstat(sg.a):+.1f})')
pd.to_pickle(R, SC + 'multiasset_trades.pkl')
