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
PART = os.getenv('PART', 'h6'); print(f'=== {PART} ===', flush=True)
B0 = ev(); line('live (ratio tie-break)', B0)
if PART == 'h6':
    CJ = pd.read_pickle('/tmp/claude-0/ml/nt_cj.pkl'); months = sorted(CJ)
    def cj_at(e):
        k = np.searchsorted(np.array(months, dtype='datetime64[ns]'), e, side='right') - 1; return CJ[months[max(k, 0)]]
    def make(sign, mode='cj'):
        def pick(cand, held, e, free, TK, K):
            M = cj_at(e); H = list(held); out = []; rest = list(cand)
            while rest and len(out) < free:
                def score(k):
                    t = TK[k]
                    if not H or t not in M.index: return 0.0
                    v = [M.at[t, h] for h in H if h in M.columns]; return np.mean(v) if v else 0.0
                rest.sort(key=lambda k: (sign * score(k), -K[k])); k = rest.pop(0); out.append(k); H.append(TK[k])
            return out
        return pick
    print('tie-break among same-bar flips when slots are short (greedy, updates as each is taken)')
    line('H6 least co-jump with held (primary)', ev(pick=make(+1)), B0)
    line('   most co-jump with held (opposite)', ev(pick=make(-1)), B0)
    print(f'  (same-bar groups larger than the free slots decide only these trades; share of entries in a tied group: {(X.groupby("entry").tk.transform("size") > 1).mean():.0%})')
elif PART == 'h23':
    for c in ('gskew', 'gskew_raw', 'p_y_up', 'res_now', 'res1', 'res5'): buckets(c, X[c])
    print(f'  pjump (a sector peer had a big up day yesterday, this name did not): share {X.pjump.mean():.0%}; avg trade yes/no: ' +
          '  '.join(f'{h}: {X.ret[(X.h2 == h) & (X.pjump == 1)].mean():+.2%} / {X.ret[(X.h2 == h) & (X.pjump == 0)].mean():+.2%}' for h in ('2020-21', '2022-26')))
    print('H2 signed gap tails as ranker')
    line('H2 ratio + gap skew forecast (primary)', ev(pr(X.ratio) + pr(X.gskew)), B0)
    line('   gap skew forecast alone', ev(X.gskew), B0)
    line('   ratio + raw up-minus-down gap freq', ev(pr(X.ratio) + pr(X.gskew_raw)), B0)
    print('H3 peer lead-lag as ranker')
    line('H3 ratio + peer residual at entry (primary)', ev(pr(X.ratio) + pr(X.res_now)), B0)
    line('   ratio + sector-ETF residual 1d', ev(pr(X.ratio) + pr(X.res1)), B0)
    line('   ratio + sector-ETF residual 5d', ev(pr(X.ratio) + pr(X.res5)), B0)
    line('   ratio + peer jumped yesterday first', ev(pr(X.ratio) + X.pjump.fillna(0) * 10), B0)
elif PART == 'h1':
    CK = pd.read_parquet('/tmp/claude-0/ml/nt_CK.parquet'); Xa = pd.read_parquet('/tmp/claude-0/ml/nt_X.parquet').sort_values('entry').reset_index(drop=True)
    CK = CK.merge(Xa[['ratio', 'h1', 'gskew', 'res_now', 'res1', 'qqq', 'entry', 'exit', 'ret', 'tk']].reset_index().rename(columns={'index': 'i'}), on='i')
    CK['h1'] = CK.h1.astype(float); CK['yr'] = pd.to_datetime(CK.ck).dt.year
    F = ['nd', 'cum', 'pk', 'today', 'gap', 'ratio', 'h1', 'gskew', 'res_now', 'res1', 'qqq']
    CK['p_hold'] = np.nan; CK['p_big'] = np.nan
    for Y in range(2020, 2027):
        tr = CK.exit < pd.Timestamp(f'{Y}-01-01'); te = CK.yr == Y
        if tr.sum() < 1000 or te.sum() == 0: continue
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=.02, num_leaves=8, min_child_samples=300, subsample=.8, subsample_freq=1, reg_lambda=5, verbose=-1)
        m.fit(CK.loc[tr, F], CK.loc[tr, 'hold'].clip(-.1, .2)); CK.loc[te, 'p_hold'] = m.predict(CK.loc[te, F])
        c = lgb.LGBMClassifier(n_estimators=300, learning_rate=.02, num_leaves=8, min_child_samples=300, subsample=.8, subsample_freq=1, reg_lambda=5, verbose=-1)
        c.fit(CK.loc[tr, F], CK.loc[tr, 'futbig']); CK.loc[te, 'p_big'] = c.predict_proba(CK.loc[te, F])[:, 1]
    T = CK[CK.ck >= np.datetime64('2020-01-01')]
    print(f'checkpoints {len(T)}; rank corr predicted vs actual hold value: ' + '  '.join(f'{h}: {g.p_hold.corr(g.hold, method="spearman"):+.3f}' for h, g in T.groupby(T.yr <= 2021)) +
          f';  AUC-ish corr p_big vs futbig: ' + '  '.join(f'{g.p_big.corr(g.futbig, method="spearman"):+.3f}' for _, g in T.groupby(T.yr <= 2021)))
    key = dict(zip(Xa.entry.astype('int64').astype(str) + Xa.tk, range(len(Xa))))
    def policy(col, thr, below=True):
        ex, rt = X.exit.values.copy(), X.ret.values.copy()
        Tt = T.sort_values('ck'); first = Tt[(Tt[col] < thr) if below else (Tt[col] > thr)].groupby('i').first()
        pos = {k: v for k, v in zip(X.entry.astype('int64').astype(str) + X.tk, range(len(X)))}
        n = 0
        for i_, r in first.iterrows():
            k = str(pd.Timestamp(r.entry).value) + r.tk
            if k in pos: j = pos[k]; ex[j] = r.ck; rt[j] = r.exnow; n += 1
        return ex, rt, n / len(X)
    for nm, col, thr in (('H1 exit when predicted hold value < 0 (primary)', 'p_hold', 0.0), ('   ... < -0.5%', 'p_hold', -.005), ('   ... < +0.5%', 'p_hold', .005)):
        ex, rt, sh = policy(col, thr); line(f'{nm} [cuts {sh:.0%}]', ev(exits=ex, rets=rt), B0)
    for q in (.1, .2):
        thr = T.p_big.quantile(q); ex, rt, sh = policy('p_big', thr); line(f'   exit when P(jump before exit) in bottom {q:.0%} [cuts {sh:.0%}]', ev(exits=ex, rets=rt), B0)
elif PART == 'h5':
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
    buckets('p_any', X.p_any); buckets('p_bin', X.p_bin)
    line('H5 ratio + P(any jump type) multiclass (primary)', ev(pr(X.ratio) + pr(X.p_any)), B0)
    line('   P(any jump type) alone', ev(X.p_any), B0)
    line('   ratio + single binary jump model (ablation)', ev(pr(X.ratio) + pr(X.p_bin)), B0)
