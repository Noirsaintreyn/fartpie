"""Nov 4 2026: opening-range breakout (ORB), 15-minute and 30-minute ranges, and whether the ORB idea helps the vector. DECLARED BEFORE RESULTS.
PART A standalone long ORB, every stock-day in the 54-name universe (and the ratio >= 1.5 subset), 2019-2026:
   range = high/low of the first 15 min (bar 09:30) or first 30 min (bars 09:30, 09:45); entry = open of the bar AFTER the first 15m bar that CLOSES above the range high
   (entries until 14:00), exit = close of the day's last bar; cost 2 bp round trip.  Variant S: exit early at the range low (stop) if touched.
   Baseline for 'excess': buy the open of the bar after the range completes on every stock-day, hold to the close (drift control).
   Buckets (within-year terciles): 20d realized vol, first-15m volume vs its 20-day mean (relative open volume), range width / 14-day average daily range.
PART B vector entries (17,447 backtest trades): position of the entry open against the same-day 15m / 30m range: above high / inside / below low / entry before the range completes;
   avg trade and big-winner ratio (share of top-5% trades / share of trades) in 2019-21 and 2022-26."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W['d'] = W.entry.dt.normalize()
RT = W.groupby(['tk', 'd']).agg(ratio=('ratio', 'first'), sp=('sp', 'first')).reset_index().sort_values('d')
rows = []; DAYF = {}
for tk in sorted(X.tk.unique()):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = load(f[0]); H = H[(H.m >= 570) & (H.m < 960)].reset_index(drop=True); H['d'] = H.dt.dt.normalize()
    dd = H.groupby('d').agg(o=('open', 'first'), c=('close', 'last'), hi=('high', 'max'), lo=('low', 'min'), v0=('volume', 'first'))
    rv = dd.c.pct_change().rolling(20).std().shift(1); rng = (dd.hi - dd.lo).rolling(14).mean().shift(1); rvol = dd.v0 / dd.v0.rolling(20).mean().shift(1)
    for d, g in H.groupby('d'):
        if len(g) < 20: continue
        t = g.dt.values; o = g.open.values; c = g.close.values; hi = g.high.values; lo = g.low.values; m = g.m.values; n = len(g)
        base = {}
        for nm, nb in (('15', 1), ('30', 2)):
            orh, orl = hi[:nb].max(), lo[:nb].min(); k0 = nb  # first bar after the range completes
            r_all = c[-1] / o[k0] - 1 - 2e-4; ent = None
            for b in range(k0, n - 1):
                if m[b + 1] > 840: break
                if c[b] > orh: ent = b + 1; break
            rec = dict(tk=tk, d=d, nm=nm, base=r_all, rv=rv.get(d, np.nan), rvol=rvol.get(d, np.nan), width=(orh - orl) / rng.get(d, np.nan) if rng.get(d, np.nan) == rng.get(d, np.nan) else np.nan, brk=0, ret=np.nan, retS=np.nan)
            if ent is not None:
                px = o[ent]; rec['brk'] = 1; rec['ret'] = c[-1] / px - 1 - 2e-4; ex = c[-1]
                for b in range(ent, n):
                    if lo[b] <= orl: ex = min(o[b], orl) if o[b] < orl else orl; break
                rec['retS'] = ex / px - 1 - 2e-4
            rows.append(rec)
A = pd.DataFrame(rows); A['yr'] = A.d.dt.year; A['p'] = np.where(A.yr <= 2021, '2019-21', '2022-26')
A = pd.merge_asof(A.sort_values('d'), RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); A['univ'] = (A.sp == True) & (A.ratio >= 1.5)
for c in ('rv', 'rvol', 'width'): A[c + 't'] = A.groupby(['yr', 'nm'])[c].transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False) if s.notna().sum() > 30 else s * np.nan)
A.to_parquet('/tmp/claude-0/ml/orb_A.parquet')
def tstat(v): return v.mean() / (v.std() / np.sqrt(len(v))) if len(v) > 30 else np.nan
print('PART A. standalone long ORB.  avg trade | excess vs the same stock-days bought at the range completion | win rate | n | t   (2019-21 || 2022-26)')
for scope, mask in (('ALL stock-days', np.ones(len(A), bool)), ('ratio >= 1.5 universe', A.univ.values)):
    for nm in ('15', '30'):
        for var, col in (('exit at close', 'ret'), ('with range-low stop', 'retS')):
            line = []
            for p in ('2019-21', '2022-26'):
                z = A[mask & (A.nm == nm).values & (A.p == p).values]; b = z[z.brk == 1]
                line.append(f'{b[col].mean():+.3%} | exc {b[col].mean() - z.base.mean():+.3%} | win {np.mean(b[col] > 0):.0%} | n{len(b)} | t{tstat(b[col]):+.1f}')
            print(f'  {scope:<22} OR{nm}m {var:<20} ' + '  ||  '.join(line))
print('\nby volatility tercile (within year), ALL stock-days, exit at close: avg trade | excess | n   (2019-21 || 2022-26)')
for nm in ('15', '30'):
    for col, lab in (('rvt', '20d realized vol'), ('rvolt', 'relative open volume'), ('widtht', 'range width / avg daily range')):
        for q in (0, 1, 2):
            line = []
            for p in ('2019-21', '2022-26'):
                z = A[(A.nm == nm) & (A.p == p) & (A[col] == q)]; b = z[z.brk == 1]; line.append(f'{b.ret.mean():+.3%} | exc {b.ret.mean() - z.base.mean():+.3%} | n{len(b)}')
            print(f'  OR{nm}m {lab:<28} tercile {q + 1}: ' + '  ||  '.join(line))
print('\nsame, ratio >= 1.5 universe only (the screener names):')
for nm in ('15', '30'):
    for q in (0, 1, 2):
        line = []
        for p in ('2019-21', '2022-26'):
            z = A[(A.nm == nm) & (A.p == p) & (A.rvt == q) & A.univ]; b = z[z.brk == 1]; line.append(f'{b.ret.mean():+.3%} | exc {b.ret.mean() - z.base.mean():+.3%} | n{len(b)}')
        print(f'  OR{nm}m realized vol tercile {q + 1}: ' + '  ||  '.join(line))
# ---------------- PART B
DAYR = {}
for tk in sorted(X.tk.unique()):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = load(f[0]); H = H[(H.m >= 570) & (H.m < 960)].reset_index(drop=True); H['d'] = H.dt.dt.normalize(); G = H.groupby('d')
    DAYR[tk] = {d: (g.high.values[:1].max(), g.low.values[:1].min(), g.high.values[:2].max(), g.low.values[:2].min(), g.dt.values, g.open.values) for d, g in G}
st15 = np.empty(len(X), object); st30 = np.empty(len(X), object); pos30 = np.full(len(X), np.nan)
for i, x in enumerate(X.itertuples()):
    r = DAYR.get(x.tk, {}).get(x.entry.normalize())
    if r is None: st15[i] = st30[i] = 'na'; continue
    h15, l15, h30, l30, t, o = r; j = np.searchsorted(t, np.datetime64(x.entry)); px = o[j] if j < len(o) else np.nan; tod = x.entry.hour * 60 + x.entry.minute
    st15[i] = 'pre' if tod < 585 else ('above' if px > h15 else 'below' if px < l15 else 'inside'); st30[i] = 'pre' if tod < 600 else ('above' if px > h30 else 'below' if px < l30 else 'inside')
    if tod >= 600 and h30 > l30: pos30[i] = (px - l30) / (h30 - l30)
X['st15'], X['st30'], X['pos30'] = st15, st30, pos30; X['p'] = np.where(X.entry.dt.year <= 2021, '2019-21', '2022-26'); top = X.groupby('p').ret.transform(lambda s: s >= s.quantile(.95))
print('\nPART B. vector entries by position vs the same-day opening range (avg trade | win | big-winner ratio | n)   2019-21 || 2022-26')
for col in ('st15', 'st30'):
    for s in ('pre', 'above', 'inside', 'below'):
        line = []
        for p in ('2019-21', '2022-26'):
            z = X[(X.p == p)]; m = z[col] == s; jr = (top[z.index][m].sum() / top[z.index].sum()) / max(m.mean(), 1e-9)
            line.append(f'{z.ret[m].mean():+.3%} | win {np.mean(z.ret[m] > 0):.0%} | j{jr:.2f} | n{m.sum()}')
        print(f'  {col} {s:<7} ' + '  ||  '.join(line))
X.to_parquet('/tmp/claude-0/ml/orb_X.parquet')
