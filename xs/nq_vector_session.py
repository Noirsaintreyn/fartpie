"""Nov 5 2026: does the SESSION the vector is built on matter? DECLARED BEFORE RESULTS.
(A) real MNQ minutes Dec 2024 - Nov 2025 (near 24h; 1m bar = open at the prior close, high / low = max / min of open, close, minute VWAP). 15m vector state built from:
    RTH 09:30-16:00 only | PRE 04:00-16:00 | GLOBEX all hours (overnight included) | BOTH = RTH flips only when the GLOBEX state agrees | BOTH2 = GLOBEX flips only when the RTH state agrees.
    Trades: flips of 15m bars closing 09:30-11:30, entry at the next minute, 75 / 50, flat 12:00, <= 2 a day, 1 pt cost, path on 1m bars. Coin-flip direction for each.
(B) QQQ 2018 - 2026: RTH-only vector vs the current state (premarket bars are sparse in the file, so this mostly checks nothing breaks)."""
import numpy as np, pandas as pd, sys, os, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
ROOT = '/mnt/user-data/uploads/fartpie'
M = pd.read_parquet(f'{ROOT}/mnq_orderflow_1min.parquet'); M['dt'] = pd.to_datetime(M.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
M = M.dropna(subset=['last_price']).sort_values('dt').drop_duplicates('dt').reset_index(drop=True); M['m'] = M.dt.dt.hour * 60 + M.dt.dt.minute
M = M[M.dt.dt.dayofweek < 5].reset_index(drop=True)
M['sd'] = (M.dt + pd.Timedelta(hours=6)).dt.normalize()                         # session date: 18:00 belongs to the next day
c = M.last_price.values; o = np.r_[c[0], c[:-1]]; vw = M.vwap.fillna(M.last_price).values
M['open'], M['high'], M['low'], M['close'] = o, np.maximum.reduce([o, c, vw]), np.minimum.reduce([o, c, vw]), c
t1, m1, d1 = M.dt.values, M.m.values, M.sd.values; h1, l1, c1, o1 = M.high.values, M.low.values, M.close.values, M.open.values
def bars15(mask):
    x = M[mask].set_index('dt')[['open', 'high', 'low', 'close']]
    g = x.resample('15min', label='left', closed='left').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
    g['reg'] = np.asarray(vec(g[['open', 'high', 'low', 'close']])); g['end'] = g.dt + pd.Timedelta('15min'); return g
SESS = {'RTH': (M.m >= 570) & (M.m < 960), 'PRE': (M.m >= 240) & (M.m < 960), 'GLOBEX': ~((M.m >= 1020) & (M.m < 1080))}
B = {k: bars15(v) for k, v in SESS.items()}
def state_at(Bk, when): i = np.searchsorted(Bk.end.values, when, side='right') - 1; return np.where(i >= 0, Bk.reg.values[np.clip(i, 0, None)], 0)
def cands(Bk, agree_with=None):
    r = Bk.reg.values; me = (Bk.end.dt.hour * 60 + Bk.end.dt.minute).values; out = []
    for k in range(2, len(Bk)):
        if r[k] == r[k - 1] or r[k - 1] == 0 or not (570 < me[k] <= 690): continue
        e = Bk.end.values[k]; j = np.searchsorted(t1, e)
        if j >= len(t1) or (t1[j] - e) > np.timedelta64(5, 'm'): continue
        if agree_with is not None and state_at(agree_with, [e])[0] != r[k]: continue
        out.append((pd.Timestamp(d1[j]), j, int(r[k])))
    return pd.DataFrame(out, columns=['day', 'j', 'dir'])
def run1(X):
    out = []
    for day, g in X.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for j, d in zip(g.j.values, g.dir.values):
            if n >= 2 or j <= busy: continue
            e = o1[j]; res = None
            for q in range(j, len(t1)):
                if d1[q] != d1[j] or m1[q] >= 720: res = ((c1[q - 1] - e) * d, q - 1); break
                lo_ = ((l1[q] if d > 0 else h1[q]) - e) * d; hi_ = ((h1[q] if d > 0 else l1[q]) - e) * d
                if lo_ <= -50: res = (-50, q); break
                if hi_ >= 75: res = (75, q); break
            if res: out.append((day, res[0] - 1.0)); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['day', 'net'])
rng = np.random.default_rng(31); nd = len(np.unique(d1[(m1 >= 570) & (m1 < 960)]))
print(f'(A) real MNQ minutes {M.dt.min():%Y-%m-%d} -> {M.dt.max():%Y-%m-%d}, {nd} sessions; 15m bars RTH {len(B["RTH"])}, PRE {len(B["PRE"])}, GLOBEX {len(B["GLOBEX"])}', flush=True)
SETS = {'RTH vector': cands(B['RTH']), 'PRE (04:00-16:00) vector': cands(B['PRE']), 'GLOBEX (overnight incl.) vector': cands(B['GLOBEX']),
        'BOTH: RTH flip, GLOBEX agrees': cands(B['RTH'], B['GLOBEX']), 'BOTH2: GLOBEX flip, RTH agrees': cands(B['GLOBEX'], B['RTH'])}
for nm, X in SETS.items():
    R = run1(X); h = R.day < '2025-06-01'; cf = np.median([run1(X.assign(dir=rng.choice([-1, 1], len(X)))).net.mean() for _ in range(10)])
    print(f'  {nm:<32} flips/day {len(X) / nd:.2f} trades {len(R):>3} win {R.net.gt(0).mean():.0%} avg {R.net.mean():+5.1f} pts (Dec-May {R[h].net.mean():+5.1f} / Jun-Nov {R[~h].net.mean():+5.1f}) | coin-flip {cf:+.1f} | total {R.net.sum():+6.0f}', flush=True)
s15 = {k: state_at(B[k], B['RTH'].end.values) for k in B}
print(f'  15m state agreement at RTH bar ends: RTH vs PRE {np.mean(s15["RTH"] == s15["PRE"]):.0%}, RTH vs GLOBEX {np.mean(s15["RTH"] == s15["GLOBEX"]):.0%}', flush=True)
# ---------------------------------------------------------------- (B) QQQ RTH-only vs current
import glob
Hq = load(glob.glob(f'{ROOT}/lse_ohlcv_cache/QQQ_5m_*.parquet')[0]); print(f'\n(B) QQQ 5m bars before 09:30: {np.mean(Hq.m < 570):.1%} of bars ({(Hq.m < 570).sum()} over {Hq.dt.dt.normalize().nunique()} days)', flush=True)
for nm, mk in (('current (04:00-16:00 bars)', np.ones(len(Hq), bool)), ('RTH only', (Hq.m >= 570).values)):
    x = Hq[mk].set_index('dt')[['open', 'high', 'low', 'close']]
    g = x.groupby([x.index.normalize(), pd.Grouper(freq='15min', origin='start_day', offset='4h')]).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    g.index = g.index.get_level_values(1); g = g.reset_index(); g.columns = ['dt', 'open', 'high', 'low', 'close']; g['reg'] = np.asarray(vec(g[['open', 'high', 'low', 'close']]))
    me = (g.dt.dt.hour * 60 + g.dt.dt.minute + 15).values; r = g.reg.values; fl = sum(1 for k in range(2, len(g)) if r[k] != r[k - 1] and r[k - 1] != 0 and 570 < me[k] <= 690)
    print(f'  {nm:<28} AM flips {fl} ({fl / g.dt.dt.normalize().nunique():.2f}/day)', flush=True)
