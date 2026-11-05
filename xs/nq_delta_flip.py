"""Nov 5 2026: DELTA FLIP vs VECTOR CROSS on real MNQ order flow (1-minute buy / sell volume, Dec 2024 - Nov 2025, ~235 sessions). DECLARED BEFORE RESULTS.
All signals on 15m RTH bars closing 09:45-11:30, entry at the next minute, 75 / 50, flat 12:00, <= 2 a day, 1 pt cost, path on 1m bars; coin-flip direction for each.
  VEC        RTH 15m vector flip (the current signal; +10.9 pts / trade in nq_vector_session)
  CVD0       session cumulative delta (reset 09:30) changes sign at a 15m close -> trade that way
  BARD       15m bar delta flips sign vs the previous bar and |delta| >= its 20-day median
  CVDX       CVD's 3-bar EMA crosses its 8-bar EMA (CVD trend flip)
  VEC+D      vector flip whose flip-bar delta agrees      VEC-D   vector flip whose flip-bar delta disagrees
  VEC+C      vector flip with session CVD on the same side VEC-C   vector flip with CVD against
  DIV        vector flip where price made a new session extreme against CVD on the flip bar side (CVD divergence) - shown for interest"""
import numpy as np, pandas as pd, sys, os, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
M = pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet'); M['dt'] = pd.to_datetime(M.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
M = M.dropna(subset=['last_price']).sort_values('dt').drop_duplicates('dt').reset_index(drop=True); M = M[M.dt.dt.dayofweek < 5].reset_index(drop=True)
M['m'] = M.dt.dt.hour * 60 + M.dt.dt.minute; M['sd'] = (M.dt + pd.Timedelta(hours=6)).dt.normalize()
c = M.last_price.values; o = np.r_[c[0], c[:-1]]; vw = M.vwap.fillna(M.last_price).values
M['open'], M['high'], M['low'], M['close'] = o, np.maximum.reduce([o, c, vw]), np.minimum.reduce([o, c, vw]), c
M['delta'] = -M.signed_vol.fillna(M.buy_vol.fillna(0) - M.sell_vol.fillna(0))     # the file's buy / sell sides are swapped: buy - sell has corr -0.68 with the 1m price change, so flip it
t1, m1, d1 = M.dt.values, M.m.values, M.sd.values; h1, l1, c1, o1 = M.high.values, M.low.values, M.close.values, M.open.values
R = M[(M.m >= 570) & (M.m < 960)].set_index('dt')
B = R.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'delta': 'sum'}).dropna(subset=['close']).reset_index()
B['reg'] = np.asarray(vec(B[['open', 'high', 'low', 'close']])); B['end'] = B.dt + pd.Timedelta('15min'); B['day'] = B.dt.dt.normalize()
B['cvd'] = B.groupby('day').delta.cumsum(); B['me'] = B.end.dt.hour * 60 + B.end.dt.minute
B['absmed'] = B.delta.abs().rolling(26 * 20, min_periods=26 * 5).median().shift(1)
e3 = B.groupby('day').cvd.transform(lambda s: s.ewm(span=3, adjust=False).mean()); e8 = B.groupby('day').cvd.transform(lambda s: s.ewm(span=8, adjust=False).mean()); B['cx'] = np.sign(e3 - e8)
B['shi'] = B.groupby('day').high.cummax(); B['slo'] = B.groupby('day').low.cummin()
def entry(k):
    e = B.end.values[k]; j = np.searchsorted(t1, e)
    return j if j < len(t1) and (t1[j] - e) <= np.timedelta64(5, 'm') else None
S = {k: [] for k in ('VEC', 'CVD0', 'BARD', 'CVDX', 'VEC+D', 'VEC-D', 'VEC+C', 'VEC-C', 'DIV')}
r, dl, cv, cx, me, dy = B.reg.values, B.delta.values, B.cvd.values, B.cx.values, B.me.values, B.day.values
for k in range(2, len(B)):
    if not (585 <= me[k] <= 690): continue
    j = entry(k)
    if j is None: continue
    day = pd.Timestamp(d1[j]); first = dy[k] != dy[k - 1]                     # 09:45 bar: vector compares with yesterday's last bar; CVD signals need a prior bar today
    if r[k] != r[k - 1] and r[k - 1] != 0:
        d = int(r[k]); S['VEC'].append((day, j, d)); S['VEC+D' if np.sign(dl[k]) == d else 'VEC-D'].append((day, j, d)); S['VEC+C' if np.sign(cv[k]) == d else 'VEC-C'].append((day, j, d))
        newx = (B.high.values[k] >= B.shi.values[k - 1]) if d < 0 else (B.low.values[k] <= B.slo.values[k - 1])
        if np.sign(cv[k]) == d and newx: S['DIV'].append((day, j, d))
    if first: continue
    if np.sign(cv[k]) != np.sign(cv[k - 1]) and np.sign(cv[k]) != 0: S['CVD0'].append((day, j, int(np.sign(cv[k]))))
    if np.sign(dl[k]) != np.sign(dl[k - 1]) and np.isfinite(B.absmed.values[k]) and abs(dl[k]) >= B.absmed.values[k]: S['BARD'].append((day, j, int(np.sign(dl[k]))))
    if cx[k] != cx[k - 1] and cx[k] != 0 and k >= 1: S['CVDX'].append((day, j, int(cx[k])))
def run1(X):
    out = []
    for day, g in X.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for j, d in zip(g.j.values, g.dir.values):
            if n >= 2 or j <= busy: continue
            e = o1[j]; res = None
            for q in range(j, len(t1)):
                if d1[q] != d1[j] or m1[q] >= 720: res = ((c1[q - 1] - e) * d, q - 1); break
                if ((l1[q] if d > 0 else h1[q]) - e) * d <= -50: res = (-50, q); break
                if ((h1[q] if d > 0 else l1[q]) - e) * d >= 75: res = (75, q); break
            if res: out.append((day, res[0] - 1.0)); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['day', 'net'])
rng = np.random.default_rng(11); nd = B.day.nunique()
print(f'MNQ order flow {B.dt.min():%Y-%m-%d} -> {B.dt.max():%Y-%m-%d}, {nd} sessions, 15m RTH bars {len(B)}; delta = aggressor buys - sells per bar (sign-corrected)', flush=True)
for nm, L in S.items():
    X = pd.DataFrame(L, columns=['day', 'j', 'dir'])
    if len(X) < 10: print(f'  {nm:<7} only {len(X)} signals', flush=True); continue
    Rr = run1(X); h = Rr.day < '2025-06-01'; cf = np.median([run1(X.assign(dir=rng.choice([-1, 1], len(X)))).net.mean() for _ in range(20)])
    t = Rr.net.mean() / (Rr.net.std() / np.sqrt(len(Rr)))
    print(f'  {nm:<7} signals/day {len(X) / nd:4.2f} trades {len(Rr):>4} win {Rr.net.gt(0).mean():4.0%} avg {Rr.net.mean():+6.1f} pts (Dec-May {Rr[h].net.mean():+5.1f} / Jun-Nov {Rr[~h].net.mean():+5.1f}) | coin-flip {cf:+5.1f} | t {t:+.1f} | total {Rr.net.sum():+6.0f}', flush=True)
