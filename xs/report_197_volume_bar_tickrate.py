"""Report 197: volume bars + tick-rate z-score (friend's idea). DECLARED before results.
Data: MNQ 1-min with n_trades, buy/sell vol (mnq_orderflow_1min.parquet, 2024-12..2026-10 only, ~1.8y: NO 2018-21 train era, so the strict split is not possible; split = first half | second half, a lead at best).
Volume bars: cumulative volume cut every V contracts (V=1000,2000,5000; boundaries at 1-min resolution, an approximation to tick-built bars). RTH 09:30-16:00 ET only, buckets do not cross days.
Feature: tpv = ticks (n_trades) / volume of the bucket; z = (tpv - mean of prior 50 buckets)/std of prior 50 (shifted, no lookahead). High z = more ticks per contract = smaller average trades.
Hypotheses (mechanism: small average size = retail / sliced flow; large average size = block/institutional): H1 low z (large trades) -> the bucket's direction CONTINUES next bucket; H2 high z (small trades) -> fades or nothing.
Outcome: sign(bucket return) x (return over next bucket / next 3 buckets), points, entry at the END of the bucket (the 1-min close that completes it). Cost shown vs 1.0 pt NQ round trip."""
import numpy as np, pandas as pd
d = pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet'); d['dt'] = pd.to_datetime(d.datetime, utc=True).dt.tz_convert('America/New_York')
d['vol'] = d.buy_vol + d.sell_vol; m = d.dt.dt.hour * 60 + d.dt.dt.minute; d = d[(m >= 570) & (m < 960) & (d.vol > 0)].copy(); d['day'] = d.dt.dt.date
print('minutes', len(d), d.dt.iloc[0], d.dt.iloc[-1], 'median vol/min', d.vol.median())
def tt(x): x = np.asarray(x, float); x = x[~np.isnan(x)]; return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 5 and x.std() > 0 else np.nan
def bars(V):
    rows = []
    for day, g in d.groupby('day'):
        cum = g.vol.cumsum().values; b = (cum // V).astype(int); g = g.assign(b=b)
        for k, h in g.groupby('b'):
            rows.append(dict(day=day, t=h.dt.iloc[-1], ticks=h.n_trades.sum(), vol=h.vol.sum(), sv=h.signed_vol.sum(), op=h.last_price.iloc[0], cl=h.last_price.iloc[-1], n=len(h)))
    B = pd.DataFrame(rows); return B
for V in (1000, 2000, 5000):
    B = bars(V); B = B.sort_values('t').reset_index(drop=True); B['tpv'] = B.ticks / B.vol
    mu = B.tpv.rolling(50).mean().shift(1); sd = B.tpv.rolling(50).std().shift(1); B['z'] = (B.tpv - mu) / sd
    B['ret'] = B.cl - B.op; B['dir'] = np.sign(B.ret)
    same = B.day.shift(-1) == B.day; same3 = B.day.shift(-3) == B.day
    B['f1'] = np.where(same, (B.cl.shift(-1) - B.cl) * B.dir, np.nan); B['f3'] = np.where(same3, (B.cl.shift(-3) - B.cl) * B.dir, np.nan)
    B = B.dropna(subset=['z']); half = B.t.iloc[len(B) // 2]; B['era'] = np.where(B.t < half, 'H1', 'H2')
    print(f'\n=== V={V}: {len(B)} buckets, median {B.n.median():.0f} min each, median ticks/contract {B.tpv.median():.2f}; halves split at {half}')
    q = B.z.quantile([1 / 3, 2 / 3]).values; B['g'] = np.where(B.z <= q[0], 'low z (big trades)', np.where(B.z >= q[1], 'high z (small trades)', 'mid'))
    for g in ('low z (big trades)', 'mid', 'high z (small trades)'):
        s = B[B.g == g]; print(f'  {g:24s}', ' | '.join(f'{e}: n{(s.era==e).sum():4d} next1 {s[s.era==e].f1.mean():6.2f}pt t{tt(s[s.era==e].f1):5.1f}  next3 {s[s.era==e].f3.mean():6.2f} t{tt(s[s.era==e].f3):5.1f}' for e in ('H1', 'H2')))
    for e in ('H1', 'H2'):
        s = B[B.era == e]; print(f'  spearman z vs next1 continuation ({e}): {s.z.corr(s.f1, method="spearman"):+.3f}; corr z vs |ret| {s.z.corr(s.ret.abs(), method="spearman"):+.3f}')
