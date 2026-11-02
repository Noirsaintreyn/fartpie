"""Nov 2 2026: noise audit. Structural traits known at entry; a bucket counts as NOISE only if (1) its avg trade is negative in BOTH
2019-21 and 2022-26 and (2) it holds clearly fewer of the big winners (top 5% of trades) than its share of trades (jump ratio < 0.7)
in both periods. Survivors get the fixed-budget book test (6 slots, 40 matched runs) incl. neighbouring thresholds.
Traits: whip (losing trades on the same ticker in the last 10 sessions), tkrec (ticker's trailing-1y avg trade), gap (entry-day gap in
daily sigma, flips on the opening bar), runup (5-day return before entry in sigma), z20 (close vs 20d mean in sigma), prior (prior day
in sigma), crowd (other flips across the universe the same day), qqq (QQQ 15m vector state at entry), tod (entry minute), dow, margin
(how far the entry bar opened past the prior close in ATR terms)."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
Q = load(glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/QQQ_15m_*.parquet')[0]); qr, qs = states(Q, vec)
QS = pd.Series(np.asarray(qr), index=Q.dt.values).shift(1)                                  # state known at the bar open
_load = load
exec(open(f'{HERE}/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
_W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); _W['d'] = _W.entry.dt.normalize()
RT = _W.groupby(['tk', 'd']).agg(ratio=('ratio', 'first'), sp=('sp', 'first')).reset_index().sort_values('d')
S_ = pd.concat(pd.read_parquet(f) for f in glob.glob('/tmp/claude-0/ml/rx/*.parquet')); S_ = S_[S_['var'] == 'live']; S_['d'] = S_.entry.dt.normalize()
S_ = S_[(S_.entry >= '2019-01-01') & ((S_.entry.dt.hour * 60 + S_.entry.dt.minute) < 840)].sort_values('d')
S_ = pd.merge_asof(S_, RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); X = S_[(S_.sp == True) & (S_.ratio >= 1.5)].sort_values('entry').reset_index(drop=True)
X['h'] = np.where(X.entry.dt.year <= 2021, '2019-21', '2022-26')
# --- traits
F = {c: np.full(len(X), np.nan) for c in ('gap', 'runup', 'z20', 'prior', 'margin')}
for tk, g in X.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = _load(f[0]); H = H[(H.m >= 570) & (H.m < 960)].reset_index(drop=True); H['day'] = H.dt.dt.normalize()
    D = H.groupby('day').agg(o=('open', 'first'), c=('close', 'last'), hi=('high', 'max'), lo=('low', 'min'))
    r = D.c.pct_change(); sd = r.rolling(20).std(); atr = (D.hi - D.lo).rolling(14).mean()
    pc, sdp, atrp = D.c.shift(1), sd.shift(1), atr.shift(1)
    T = {'gap': (D.o / pc - 1) / sdp, 'runup': (pc / D.c.shift(6) - 1) / (sdp * np.sqrt(5)), 'z20': (pc - D.c.rolling(20).mean().shift(1)) / (D.c.rolling(20).std().shift(1)),
         'prior': r.shift(1) / sd.shift(2)}
    t = H.dt.values; op = H.open.values; dy = H.day.values
    for i, e in zip(g.index, g.entry.values):
        j = np.searchsorted(t, e)
        if j >= len(t): continue
        dd = dy[j]
        for k, s in T.items(): F[k][i] = s.get(dd, np.nan)
        F['margin'][i] = (op[j] / pc.get(dd, np.nan) - 1) * pc.get(dd, np.nan) / atrp.get(dd, np.nan)
for k, v in F.items(): X[k] = v
X['tod'] = X.entry.dt.hour * 60 + X.entry.dt.minute; X['dow'] = X.entry.dt.dayofweek; X['open_bar'] = (X.tod == 570).astype(float)
X['qqq'] = QS.reindex(X.entry.values, method='ffill').values
X['crowd'] = X.groupby('d').tk.transform('size') - 1
wh, rec = [], []
for i, (tk, e) in enumerate(zip(X.tk.values, X.entry.values)):
    p = X[(X.tk == tk) & (X.exit < e)]
    wh.append(((p.exit > e - np.timedelta64(14, 'D')) & (p.ret < 0)).sum()); q = p[p.exit > e - np.timedelta64(365, 'D')]
    rec.append(q.ret.mean() if len(q) >= 5 else np.nan)
X['whip'] = wh; X['tkrec'] = rec
X.to_parquet('/tmp/claude-0/ml/noise_X.parquet')
# --- audit
top = X.groupby('h').ret.transform(lambda s: s >= s.quantile(.95))
def bucket_stats(m):
    out = []
    for h in ('2019-21', '2022-26'):
        z = X.h == h; mm = m & z; n = mm.sum()
        out.append((n / z.sum(), X.ret[mm].mean(), (top[mm].sum() / top[z].sum()) / (n / z.sum()) if n else np.nan, n))
    return out
CAND = []
print('trait bucket                share of trades   avg trade 19-21 / 22-26   big-winner ratio 19-21 / 22-26   NOISE?')
def show(nm, m):
    m = m.fillna(False).astype(bool) if hasattr(m, 'fillna') else m; s = bucket_stats(m)
    noise = all(a[1] < 0 and a[2] < .7 for a in s); flag = '  <== NOISE' if noise else ''
    print(f'  {nm:<26} {s[0][0]:>5.1%} / {s[1][0]:>5.1%}   {s[0][1]:+.2%} / {s[1][1]:+.2%}          {s[0][2]:.2f} / {s[1][2]:.2f}{flag}', flush=True)
    if noise: CAND.append((nm, m))
for c in ('gap', 'runup', 'z20', 'prior', 'margin', 'tkrec', 'crowd'):
    q = X[c].rank(pct=True)
    for lo, hi in ((0, .1), (0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01), (.9, 1.01)): show(f'{c} pct {lo:.0%}-{min(hi, 1):.0%}', (q > lo) & (q <= hi))
for k in (0, 1, 2, 3): show(f'whip = {k}' if k < 3 else 'whip >= 3', (X.whip == k) if k < 3 else (X.whip >= 3))
for nm, m in (('qqq 15m bearish', X.qqq == -1), ('qqq 15m bullish', X.qqq == 1), ('opening-bar entry', X.open_bar == 1), ('entry 10:00-11:45', (X.tod >= 600) & (X.tod < 720)),
              ('entry 12:00-13:45', (X.tod >= 720) & (X.tod < 840)), ('1h track', X.h1 == True), ('15m track', X.h1 == False)): show(nm, m)
for k, nm in enumerate(('Mon', 'Tue', 'Wed', 'Thu', 'Fri')): show(f'entry {nm}', X.dow == k)
print(f'\n{len(CAND)} bucket(s) pass both noise tests')
if CAND:
    rng = np.random.default_rng(97); DR = [rng.random(len(X)) < .05 if i else np.zeros(len(X), bool) for i in range(41)]
    def ev(Z): return np.array([[v_ for cd in stat(dret(simw(Z[~dl[Z.index]])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in [pd.Series(d_, X.index) for d_ in DR]])
    def line(nm, a, b=None):
        m = np.median(a[1:], axis=0); s = f'  {nm:<34} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
        if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}  (return {(a[1:, 0] > b[1:, 0]).mean():.0%} / {(a[1:, 3] > b[1:, 3]).mean():.0%})'
        print(s, flush=True)
    print('\nbook test, removing each noise bucket (fixed budget, 6 slots)'); B0 = ev(X); line('live', B0)
    for nm, m in CAND: line(f'drop {nm}', ev(X[~m]), B0)
    if len(CAND) > 1: allm = np.logical_or.reduce([m for _, m in CAND]); line('drop all noise buckets', ev(X[~allm]), B0)
