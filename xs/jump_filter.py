"""Nov 2 2026: binary jump model (nt_test H5 ablation) used as a FILTER instead of a ranker. Same walk-forward model, same pass rule."""
"""Nov 2 2026: tests H6, H2, H3, H1, H5 (hypotheses + pass rule declared in xs/nt_feat.py before any result).
Fixed budget, 6 slots at 1/6, 8% breaker; 40 matched runs (the same signals, by ticker + entry time, are dropped in every variant);
periods 2020-21 / 2022-26 (walk-forward models need 2019 to train). PART = h6 | h23 | h1 | h5."""
import os, zlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
import lightgbm as lgb
X = pd.read_parquet('/tmp/claude-0/ml/nt_X.parquet').sort_values('entry').reset_index(drop=True)
X = X[X.entry >= '2020-01-01'].reset_index(drop=True); X['yr'] = X.entry.dt.year
X['h2'] = np.where(X.yr <= 2021, '2020-21', '2022-26')
HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(X.tk, X.entry.astype('int64'))], dtype=np.uint64)
DROPS = [np.zeros(len(X), bool)] + [((HSH * 2654435761 + i * 40503) % 1000) < 50 for i in range(1, 41)]
IDX = pd.bdate_range('2020-01-01', X.exit.max())
P2 = [(np.datetime64('2020-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def stat2(eq):
    eq = eq.reindex(IDX, method='ffill').fillna(0); out = []
    for a, b in P2:
        e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / yr) - 1; dd = (e / e.cummax() - 1).min(); out += [c, dd, c / -dd]
    return out
def sim(E, Xx, R, K, TK, n=6, brk=.08, pick=None):
    """entries grouped by timestamp; within a group, order by K (desc) or by pick(cands, held) when slots are short"""
    o = np.lexsort((TK, -K, E)); E, Xx, R, K, TK = E[o], Xx[o], R[o], K[o], TK[o]
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; i = 0; N = len(E)
    while i < N:
        j = i
        while j < N and E[j] == E[i]: j += 1
        e = E[i]
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q[0] > e]; free = n - len(op)
        ok = free > 0
        if ok and brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): ok = False
        if ok:
            cand = list(range(i, j))
            if len(cand) > free and pick is not None: cand = pick(cand, [q[1] for q in op], e, free, TK, K)
            for k in cand[:free]: op.append((Xx[k], TK[k])); pend.append((Xx[k], R[k]))
        i = j
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); return s.groupby(s.index.normalize()).last()
def ev(key=None, exits=None, rets=None, pick=None, Z=None):
    Z = X if Z is None else Z; K = (Z.ratio if key is None else key).fillna((Z.ratio if key is None else key).median()).values.astype(float)
    Xx = Z.exit.values if exits is None else exits; R = Z.ret.values if rets is None else rets
    return np.array([stat2(sim(Z.entry.values[~dl], Xx[~dl], R[~dl], K[~dl], Z.tk.values[~dl], pick=pick)) for dl in DROPS])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<44} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None:
        w1, w2 = (a[1:, 2] > b[1:, 2]).mean(), (a[1:, 5] > b[1:, 5]).mean(); s += f'   beats live {w1:.0%} / {w2:.0%}' + ('   PASS' if w1 >= .7 and w2 >= .7 else '')
    print(s, flush=True)
pr = lambda s: s.groupby(X.yr).rank(pct=True).fillna(.5)
top = X.groupby('h2').ret.transform(lambda s: s >= s.quantile(.95))
def buckets(nm, s):
    q = s.groupby(X.yr).rank(pct=True); out = []
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        m = (q > lo) & (q <= hi); row = []
        for h in ('2020-21', '2022-26'):
            z = (X.h2 == h) & q.notna(); mm = m & z; row.append(f'{X.ret[mm].mean():+.2%} j{(top[mm].sum() / top[z].sum()) / (mm.sum() / z.sum()):.2f}')
        out.append('/'.join(row))
    print(f'  {nm:<12} q1..q5 (avg trade, big-winner ratio; 2020-21/2022-26): ' + '  |  '.join(out), flush=True)
Xa = pd.read_parquet('/tmp/claude-0/ml/nt_X.parquet').sort_values('entry').reset_index(drop=True); Xa['yr'] = Xa.entry.dt.year
thr = Xa.groupby('yr').ret.transform(lambda s: s.quantile(.9)); jump = Xa.ret >= thr
Xa['cls'] = np.where(~jump, 0, np.where(Xa.q_up == 1, 3, np.where(Xa.lg_on >= Xa.lg_in, 1, 2)))
print('class shares:', Xa.cls.value_counts(normalize=True).sort_index().round(3).to_dict(), '(0 none, 1 gap-led, 2 intraday, 3 market-wide)')
Xa['h1f'] = Xa.h1.astype(float)
F = ['ratio', 'gap', 'runup', 'z20', 'prior', 'margin', 'tod', 'whip', 'tkrec', 'qqq', 'h1f', 'gskew', 'gskew_raw', 'res_now', 'res1', 'res5', 'pjump', 'dow']
Xa['p_any'] = np.nan; Xa['p_bin'] = np.nan
for Y in range(2020, 2027):
    tr = Xa.exit < pd.Timestamp(f'{Y}-01-01'); te = Xa.yr == Y
    kw = dict(n_estimators=300, learning_rate=.02, num_leaves=8, min_child_samples=150, subsample=.8, subsample_freq=1, reg_lambda=5, verbose=-1)
    mc = lgb.LGBMClassifier(objective='multiclass', **kw).fit(Xa.loc[tr, F], Xa.loc[tr, 'cls']); pm = mc.predict_proba(Xa.loc[te, F])
    Xa.loc[te, 'p_any'] = 1 - pm[:, list(mc.classes_).index(0)]
    mb = lgb.LGBMClassifier(**kw).fit(Xa.loc[tr, F], (Xa.loc[tr, 'cls'] > 0).astype(int)); Xa.loc[te, 'p_bin'] = mb.predict_proba(Xa.loc[te, F])[:, 1]
k = Xa.entry.astype('int64').astype(str) + Xa.tk; mp = dict(zip(k, Xa.p_any)); mb_ = dict(zip(k, Xa.p_bin))
kx = X.entry.astype('int64').astype(str) + X.tk; X['p_any'] = kx.map(mp); X['p_bin'] = kx.map(mb_)

print("binary jump model as a FILTER (drop the lowest-scored flips; walk-forward by year; 2020-21 / 2022-26)")
B0 = ev(); line('live', B0)
pb = X.p_bin.groupby(X.yr).rank(pct=True)
for q in (.1, .2, .3):
    keep = ~(pb <= q).fillna(False)
    Z = X[keep].reset_index(drop=True); Zd = [d[keep.values] for d in DROPS]
    A = np.array([stat2(sim(Z.entry.values[~dl], Z.exit.values[~dl], Z.ret.values[~dl], Z.ratio.values[~dl].astype(float), Z.tk.values[~dl])) for dl in Zd])
    tb = top[keep]; print(f'  drop lowest {q:.0%}: removes {(~keep).mean():.0%} of flips, {(top & ~keep).sum() / top.sum():.0%} of the top-5% winners')
    line(f'  filter: drop lowest {q:.0%} by P(jump)', A, B0)
