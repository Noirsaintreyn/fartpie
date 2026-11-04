"""Nov 4 2026: can order flow (CVD) or a lower timeframe turn the NY AM vector into a prop-eval trade (75-150 NQ points, 1-2 trades, $1,500/day consistency cap)? DECLARED BEFORE RESULTS.
DATA  MNQ 1-minute order flow (signed volume, last price), Dec 2024 - Nov 2025 (about 230 sessions: small sample). Vector on 5m / 15m bars built from the minute prints (04:00-16:00 ET).
      Gamma = QQQ net GEX at the prior close (negative = < 0). Long history check without order flow: QQQ 5m bars 2018-2026 scaled to NQ points.
SIGNALS (NY AM: signal bar closes 09:30-11:30; entry at the next minute's price; first signal of the day only = 'once a day')
  S0  15m vector flip, either direction, negative gamma (the version from the last test)
  S1  S0 + CVD: signed volume over the 30 minutes before entry agrees with the direction
  S2  S0 + the flip bar's own delta (signed volume of that 15m bar) agrees with the direction
  S3  lower timeframe: 5m vector flip in the direction of the 15m state, negative gamma
  S4  S3 + CVD 30-minute agreement
  S5  S3 on ALL days (no gamma filter)
TRADE  1 NQ ($20/pt), fixed target / stop in points: 75/40, 75/50, 100/50, 150/75; flat at 12:00; checked on minute prints; cost 1 pt round trip.
PLACEBO  same entry minutes, coin-flip direction (30 draws).
PROP  Topstep-style 50K eval: +$3,000, $2,000 trailing max loss (end of day), $1,000 daily loss, consistency: best day <= 50% of total profit when passing."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/stop_1h.py').read(); exec(src[:src.index("if __name__ == '__main__':")])
import yfinance as yf
ROOT = '/mnt/user-data/uploads/fartpie'
G = pd.read_csv(f'{ROOT}/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index()
def neg(day): v = G[G.index < day]; return bool(len(v) and v.iloc[-1] < 0)
M = pd.read_parquet(f'{ROOT}/mnq_orderflow_1min.parquet'); M['dt'] = pd.to_datetime(M.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
M = M.dropna(subset=['last_price']).sort_values('dt').drop_duplicates('dt').reset_index(drop=True); M['m'] = M.dt.dt.hour * 60 + M.dt.dt.minute
M = M[(M.m >= 240) & (M.m < 960) & (M.dt.dt.dayofweek < 5)].reset_index(drop=True)
M['signed_vol'] = -M.signed_vol      # the file's buy/sell labels are swapped: signed volume moves AGAINST price (r = -0.68 per minute), so the true delta is -signed_vol
px = M.last_price.values; sv = M.signed_vol.fillna(0).values; mt = M.dt.values; mm = M.m.values; md = M.dt.dt.normalize().values
def ohlc(freq):
    x = M.set_index('dt'); g = x.last_price.groupby([x.index.normalize(), pd.Grouper(freq=freq, origin='start_day', offset='4h')])
    r = pd.DataFrame({'open': g.first(), 'high': g.max(), 'low': g.min(), 'close': g.last(), 'delta': x.signed_vol.groupby([x.index.normalize(), pd.Grouper(freq=freq, origin='start_day', offset='4h')]).sum()}).dropna()
    r.index = r.index.get_level_values(1); r = r.reset_index().rename(columns={'dt': 'dt'}); r.columns = ['dt', 'open', 'high', 'low', 'close', 'delta']
    r['end'] = r.dt + pd.Timedelta(freq); r['m_end'] = r.end.dt.hour * 60 + r.end.dt.minute; return r
B15, B5 = ohlc('15min'), ohlc('5min')
B15['reg'] = np.asarray(vec(B15[['open', 'high', 'low', 'close']])); B5['reg'] = np.asarray(vec(B5[['open', 'high', 'low', 'close']]))
s15 = pd.merge_asof(pd.DataFrame({'t': B5.end}), pd.DataFrame({'t': B15.end, 'r15': B15.reg}), on='t', direction='backward').r15.values   # 15m state known at each 5m close
mi = pd.Series(np.arange(len(M)), index=pd.DatetimeIndex(mt))
def cand(B, kind):
    """(day, entry minute index, direction, delta-of-bar) for flips whose bar closes 09:30-11:30"""
    out = []; r = B.reg.values
    for k in range(2, len(B)):
        if r[k] == r[k - 1] or r[k - 1] == 0: continue
        if not (570 < B.m_end.iloc[k] <= 690): continue
        d = int(r[k])
        if kind == 'ltf' and s15[k] != d: continue
        j = mi.index.searchsorted(np.datetime64(B.end.iloc[k]))
        if j >= len(M) or md[j] != np.datetime64(B.dt.iloc[k].normalize()): continue
        out.append((pd.Timestamp(md[j]), int(j), d, float(B.delta.iloc[k])))
    return pd.DataFrame(out, columns=['day', 'j', 'dir', 'bar_delta'])
def cvd30(j): a = mi.index.searchsorted(mt[j] - np.timedelta64(30, 'm')); return sv[a:j].sum()
def run(C, T, S, per_day=1):
    rows = []
    for day, g in C.groupby('day'):
        busy_until = -1; n = 0
        for _, c in g.sort_values('j').iterrows():
            if n >= per_day or c.j <= busy_until: continue
            j = c.j; e = px[j]; d = c.dir; res = None
            for k in range(j + 1, len(M)):
                if md[k] != md[j] or mm[k] >= 720: res = ((px[k - 1] - e) * d, k - 1); break
                mv = (px[k] - e) * d
                if mv <= -S: res = (-S, k); break
                if mv >= T: res = (T, k); break
            if res is None: continue
            rows.append((day, j, d, res[0] - 1.0)); busy_until = res[1]; n += 1
    return pd.DataFrame(rows, columns=['day', 'j', 'dir', 'pts'])
def prop(R, days, usd=20, target=3000, mll=2000, dll=1000):
    v = R.groupby('day').pts.sum().reindex(days).fillna(0).values * usd; res = []
    for s in range(len(v)):
        bal, peak, floor, best = 0.0, 0.0, -mll, 0.0; out = None
        for k in range(s, len(v)):
            day = max(v[k], -dll); bal += day; best = max(best, day)
            if bal <= floor: out = ('fail', k - s + 1); break
            if bal >= target and best <= .5 * bal: out = ('pass', k - s + 1); break
            peak = max(peak, bal); floor = min(max(floor, peak - mll), 0.0)
        res.append(out or ('open', len(v) - s))
    Rr = pd.DataFrame(res, columns=['r', 'n']); dn = Rr[Rr.r != 'open']
    return (dn.r == 'pass').mean() if len(dn) else np.nan, dn[dn.r == 'pass'].n.median() if len(dn) else np.nan, len(dn)
if __name__ == '__main__':
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(md[mm >= 570]))))
    C15, C5 = cand(B15, 'htf'), cand(B5, 'ltf')
    for C in (C15, C5): C['neg'] = [neg(d) for d in C.day]; C['cvd30'] = [cvd30(j) for j in C.j]
    SIG = {'S0 15m flip, neg gamma': C15[C15.neg], 'S1 S0 + CVD 30m agrees': C15[C15.neg & (np.sign(C15.cvd30) == C15.dir)],
           'S2 S0 + flip-bar delta agrees': C15[C15.neg & (np.sign(C15.bar_delta) == C15.dir)], 'S3 5m flip with 15m state, neg gamma': C5[C5.neg],
           'S4 S3 + CVD 30m agrees': C5[C5.neg & (np.sign(C5.cvd30) == C5.dir)], 'S5 S3, all days': C5}
    print(f'MNQ sessions {len(days)}  ({days[0]:%Y-%m-%d} - {days[-1]:%Y-%m-%d}), neg-gamma sessions {np.mean([neg(d) for d in days]):.0%}', flush=True)
    rng = np.random.default_rng(0)
    for T, S in ((75, 40), (75, 50), (100, 50), (150, 75)):
        print(f'\n== target {T} / stop {S} pts, 1 NQ, one trade a day', flush=True)
        for nm, C in SIG.items():
            R = run(C, T, S)
            if not len(R): print(f'  {nm:<38} no trades'); continue
            pl = []
            for _ in range(30):
                Z = C.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); pl.append(run(Z, T, S).pts.mean())
            pr, mdn, na = prop(R, days)
            print(f'  {nm:<38} n{len(R):>4} ({len(R) / len(days) * 5:.1f}/wk) win {R.pts.gt(0).mean():.0%} avg {R.pts.mean():+6.1f} pts (coin-flip {np.median(pl):+.1f}, 90% {np.percentile(pl, 5):+.1f}..{np.percentile(pl, 95):+.1f}) total {R.pts.sum():+6.0f} | eval pass {pr:.0%} median {mdn:.0f} sessions', flush=True)
