"""Nov 2 2026: outside review #1 (decision-focused labels) and #2 (congestion), DIAGNOSTICS ONLY. A policy is built only if the
target is predictable. DECLARED: proceed to a policy only if the walk-forward rank correlation of predicted vs actual branch
advantage is >= +0.10 in BOTH 2020-21 and 2022-26 at the primary horizon (60 trading days).
 #1 For every oversubscribed same-bar group in the live book: branch = swap the last baseline pick for the best displaced flip
    (ratio order); everything after follows the unchanged live policy. Label A_H = log equity(branch) - log equity(live) at H = 20 /
    60 / 120 trading days. Features: the swap pair's differences (ratio, 1h, entry strength, gap), free slots, group size, open-position
    count, average age and 1h share of open positions, time of day, flips in the last 5 sessions, QQQ state.
 #2 Congestion: can the number of eligible flips in the NEXT session be forecast at today's close? (walk-forward Poisson GBM)."""
import copy, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
import lightgbm as lgb
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
X['h1f'] = X.h1.astype(float); X['margin'] = X.margin.fillna(0); X['gap'] = X.gap.fillna(0)
o = np.lexsort((X.tk.values, X.h1f.values, -X.ratio.values, X.entry.values)); ids = X.index.values[o]
E = X.entry.values[ids]; EXT = X.exit.values; RET = X.ret.values; ENT = X.entry.values
groups = []; i = 0
while i < len(ids):
    j = i
    while j < len(ids) and E[j] == E[i]: j += 1
    groups.append(list(ids[i:j])); i = j
def step(st, g, choice=None):
    """advance state st through group g; returns the picked ids. st = dict(op, pend, real, hist, win, lvl0)"""
    e = ENT[g[0]]
    done = sorted([p for p in st['pend'] if p[0] <= e], key=lambda z: z[0]); st['pend'] = [p for p in st['pend'] if p[0] > e]
    for xt, rr in done: st['real'] += np.log1p(rr / 6); st['hist'].append((xt, st['real'])); st['win'].append((xt, st['real']))
    st['op'] = [q for q in st['op'] if q[0] > e]; free = 6 - len(st['op'])
    if free <= 0: return [], free
    if st['hist']:
        t0 = e - np.timedelta64(28, 'D')
        while st['win'] and st['win'][0][0] < t0: st['lvl0'] = st['win'].popleft()[1]
        if st['real'] - max([st['lvl0']] + [h for _, h in st['win']]) < np.log1p(-.08): return [], 0
    pick = (choice if choice is not None else g)[:free]
    for k in pick: st['op'].append((EXT[k], k)); st['pend'].append((EXT[k], RET[k]))
    return pick, free
def eq_at(st, t):
    """realized log equity at time t, settling pending exits up to t (open positions marked at entry = 0)"""
    v = st['real']
    for xt, rr in sorted(st['pend']):
        if xt <= t: v += np.log1p(rr / 6)
    return v
st = dict(op=[], pend=[], real=0.0, hist=[], win=deque(), lvl0=0.0); rows = []
day_counts = X.groupby('d').size()
for gi, g in enumerate(groups):
    snap = None
    e = ENT[g[0]]
    # peek: oversubscribed?
    pre = copy.deepcopy(st); _, free_now = step(copy.deepcopy(st), g)
    if 0 < free_now < len(g):
        snap = pre; base = g[:free_now]; alt = base[:-1] + [g[free_now]]
        op_ages = [(e - ENT[k]) / np.timedelta64(1, 'D') for _, k in pre['op'] if ENT[k] <= e]
        d0 = pd.Timestamp(e).normalize(); prior = day_counts[(day_counts.index < d0)].tail(5)
        a, b = base[-1], g[free_now]
        rows.append(dict(gi=gi, e=e, a=a, b=b, alt=alt, free=free_now, n=len(g), nopen=6 - free_now,
                         age=np.mean([x for x in op_ages if x == x]) if op_ages else 0.0, h1open=np.mean([X.h1f.values[k] for _, k in pre['op']]) if pre['op'] else 0.0,
                         d_ratio=X.ratio.values[b] - X.ratio.values[a], d_h1=X.h1f.values[b] - X.h1f.values[a], d_margin=X.margin.values[b] - X.margin.values[a],
                         d_gap=X.gap.values[b] - X.gap.values[a], tod=X.tod.values[a], rate5=prior.mean() if len(prior) else np.nan, qqq=X.qqq.values[a]))
    step(st, g)
print(f'{len(rows)} oversubscribed decisions', flush=True)
R = pd.DataFrame(rows); HZ = (20, 60, 120); bd = pd.bdate_range('2019-01-01', '2027-12-31')
pos = {gi: k for k, gi in enumerate(range(len(groups)))}
# baseline states per group index for branching: re-run once storing snapshots only at decision groups
need = set(R.gi); SN = {}; st = dict(op=[], pend=[], real=0.0, hist=[], win=deque(), lvl0=0.0)
for gi, g in enumerate(groups):
    if gi in need: SN[gi] = copy.deepcopy(st)
    step(st, g)
def horizon_t(e, h):
    k = bd.searchsorted(pd.Timestamp(e).normalize()); return np.datetime64(bd[min(k + h, len(bd) - 1)])
lab = {h: [] for h in HZ}
for r in R.itertuples():
    tH = {h: horizon_t(r.e, h) for h in HZ}; tmax = max(tH.values())
    res = {}
    for nm, choice in (('base', None), ('alt', r.alt)):
        s = copy.deepcopy(SN[r.gi]); step(s, groups[r.gi], choice); out = {}
        gj = r.gi + 1
        for h in sorted(HZ):
            while gj < len(groups) and ENT[groups[gj][0]] <= tH[h]: step(s, groups[gj]); gj += 1
            out[h] = eq_at(s, tH[h])
        res[nm] = out
    for h in HZ: lab[h].append(res['alt'][h] - res['base'][h])
    if r.Index % 200 == 199: print(f'  {r.Index + 1} branches', flush=True)
for h in HZ: R[f'A{h}'] = lab[h]
R['yr'] = pd.to_datetime(R.e).dt.year; R.drop(columns=['alt']).to_parquet('/tmp/claude-0/ml/branch_R.parquet')
print('\nbranch labels: share where the swap helped / mean advantage (log, x1e4 = bp of book)')
for h in HZ: print(f'  H={h}: helped {(R[f"A{h}"] > 0).mean():.0%}, hurt {(R[f"A{h}"] < 0).mean():.0%}, mean {R[f"A{h}"].mean() * 1e4:+.1f} bp, sd {R[f"A{h}"].std() * 1e4:.0f} bp')
F = ['free', 'n', 'nopen', 'age', 'h1open', 'd_ratio', 'd_h1', 'd_margin', 'd_gap', 'tod', 'rate5', 'qqq']
print('\nwalk-forward predictability (rank corr of predicted vs actual advantage), 2020-21 / 2022-26')
for h in HZ:
    R['p'] = np.nan
    for Y in range(2020, 2027):
        tr = pd.to_datetime(R.e) < pd.Timestamp(f'{Y}-01-01') - pd.Timedelta(days=200); te = R.yr == Y
        if tr.sum() < 100 or te.sum() == 0: continue
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=.02, num_leaves=4, min_child_samples=40, subsample=.8, subsample_freq=1, reg_lambda=10, verbose=-1)
        m.fit(R.loc[tr, F], R.loc[tr, f'A{h}']); R.loc[te, 'p'] = m.predict(R.loc[te, F])
    c = [R[(R.yr <= 2021) & (R.yr >= 2020)], R[R.yr >= 2022]]
    print(f'  H={h}: {c[0].p.corr(c[0][f"A{h}"], method="spearman"):+.3f} / {c[1].p.corr(c[1][f"A{h}"], method="spearman"):+.3f}   (n {len(c[0])} / {len(c[1])})')
print('single-feature rank corr with A60 (2019-21 / 2022-26):')
for f in F:
    a, b = R[R.yr <= 2021], R[R.yr >= 2022]; print(f'  {f:<9} {a[f].corr(a.A60, method="spearman"):+.3f} / {b[f].corr(b.A60, method="spearman"):+.3f}')
# --- #2 congestion
dc = X.groupby('d').size().reindex(pd.bdate_range(X.d.min(), X.d.max()), fill_value=0)
C = pd.DataFrame({'y': dc.shift(-1), 'c0': dc, 'c5': dc.rolling(5).mean(), 'c20': dc.rolling(20).mean(), 'dow': dc.index.dayofweek})
q = X.groupby('d').qqq.mean().reindex(C.index); C['qqq'] = q.ffill(); C = C.dropna(); C['yr'] = C.index.year; C['p'] = np.nan
for Y in range(2020, 2027):
    tr = C.index < pd.Timestamp(f'{Y}-01-01'); te = C.yr == Y
    m = lgb.LGBMRegressor(objective='poisson', n_estimators=200, learning_rate=.03, num_leaves=4, min_child_samples=30, verbose=-1).fit(C.loc[tr, ['c0', 'c5', 'c20', 'dow', 'qqq']], C.loc[tr, 'y'])
    C.loc[te, 'p'] = m.predict(C.loc[te, ['c0', 'c5', 'c20', 'dow', 'qqq']])
for nm, z in (('2020-21', C[(C.yr >= 2020) & (C.yr <= 2021)]), ('2022-26', C[C.yr >= 2022])):
    naive = z.c5; r2 = 1 - ((z.y - z.p) ** 2).sum() / ((z.y - z.y.mean()) ** 2).sum(); r2n = 1 - ((z.y - naive) ** 2).sum() / ((z.y - z.y.mean()) ** 2).sum()
    print(f'#2 congestion {nm}: next-session flip count, model R2 {r2:.2f} (rank corr {z.p.corr(z.y, method="spearman"):+.2f}); naive 5-day average R2 {r2n:.2f}')
