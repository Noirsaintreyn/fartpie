"""Nov 4 2026: the NQ NY AM trade for a ONE-MONTH prop eval, 2022 - Sep 2026 only (QQQ proxy, NQ points, 1 pt cost NQ / 1.2 pt micros). DECLARED BEFORE RESULTS.
Eval: +$3,000 within 21 sessions, $2,000 trailing max loss (end of day), $1,000 daily loss, consistency: best day <= 50% of the profit when passing; otherwise it expires.
Sizes chosen so one target hit is about $1,500 (the consistency cap):  A 1 NQ 75/40   B 1 NQ 75/50   C 5 MNQ 150/75   D 7 MNQ 100/50
Signal sets: S0 (15m flip, neg gamma, first of the day) | S0 + SPY 15m agrees | S0 + SPY 20d up | S0, up to 2 trades a day | ALL15 = 15m flip on any gamma day | ALL15, 2 a day
Per set x size: eval pass / fail / expire within a month (every session a start date), monthly P&L in $ (mean, median, % months up, worst), coin-flip pass rate (10 draws)."""
_s = open('xs/nq_htf.py').read(); exec(_s[:_s.index("def report(")])
P0 = pd.Timestamp('2022-01-01'); dpost = days[days >= P0]
r = H15.reg.values; AL = []                                                    # 15m flips 09:30-11:30 on any day (all of them, not only the first)
for k in range(2, len(H15)):
    if r[k] == r[k - 1] or r[k - 1] == 0 or not (570 < H15.me.iloc[k] <= 690): continue
    day = H15.dt.iloc[k].normalize(); j = np.searchsorted(t5, np.datetime64(H15.end.iloc[k]))
    if j < len(H5) and d5[j] == np.datetime64(day):
        g = G[G.index < day]; AL.append((day, int(j), int(r[k]), bool(len(g) and g.iloc[-1] < 0)))
AL = pd.DataFrame(AL, columns=['day', 'j', 'dir', 'neg']); AL['k'] = NQs.reindex(pd.DatetimeIndex(AL.day) - pd.Timedelta(days=1), method='ffill').values / o5[AL.j.values]
def run_n(Cc, T, S, per_day):
    """like run_q but up to per_day trades a day, no overlap (next trade only after the previous one closed)"""
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for _, c in g.iterrows():
            if n >= per_day or c.j <= busy: continue
            e = o5[c.j]; d = c.dir; kk = c.k; res = None
            for q in range(c.j, len(H5)):
                if d5[q] != d5[c.j] or m5[q] >= 720: res = ((c5[q - 1] - e) * d * kk, q - 1); break
                lo = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
                if lo <= -S: res = (-S, q); break
                if hi >= T: res = (T, q); break
            if res: out.append((day, res[0])); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['day', 'pts'])
def evalm(R, usd, cost, n=21, target=3000, mll=2000, dll=1000):
    v = (R.assign(p=(R.pts - cost) * usd).groupby('day').p.sum()).reindex(days).fillna(0.0); v = v[v.index >= P0].values; res = {'pass': 0, 'fail': 0, 'expire': 0}
    for s in range(len(v) - n):
        bal, peak, floor, best, out = 0.0, 0.0, -mll, 0.0, 'expire'
        for k in range(s, s + n):
            day = max(v[k], -dll); bal += day; best = max(best, day)
            if bal <= floor: out = 'fail'; break
            if bal >= target and best <= .5 * bal: out = 'pass'; break
            peak = max(peak, bal); floor = min(max(floor, peak - mll), 0.0)
        res[out] += 1
    tot = sum(res.values()); return {k: v_ / tot for k, v_ in res.items()}
def monthly(R, usd, cost):
    x = R[R.day >= P0].assign(p=lambda z: (z.pts - cost) * usd); mo = x.groupby(x.day.dt.to_period('M')).p.sum()
    allm = pd.period_range(P0, dpost[-1], freq='M'); mo = mo.reindex(allm).fillna(0.0); return mo
SETS = {'S0': (C2, 1), 'S0 + SPY agrees': (C2[C2.SPY], 1), 'S0 + SPY 20d up': (C2[C2.MKT20 == 'SPY 20d up'], 1), 'S0, 2 a day': (AL[AL.neg], 2),
        'ALL15 (any gamma)': (AL.groupby('day').head(1), 1), 'ALL15, 2 a day': (AL, 2)}
SIZES = [('A 1 NQ 75/40', 20, 75, 40, 1.0), ('B 1 NQ 75/50', 20, 75, 50, 1.0), ('C 5 MNQ 150/75', 10, 150, 75, 1.2), ('D 7 MNQ 100/50', 14, 100, 50, 1.2)]
rng = np.random.default_rng(21)
print(f'2022 - {dpost[-1]:%Y-%m}: {len(dpost)} sessions, {len(pd.period_range(P0, dpost[-1], freq="M"))} months', flush=True)
for sn, (Cc, pdn) in SETS.items():
    Cc = Cc[Cc.day >= P0]; print(f'\n== {sn}: {len(Cc.day.unique())} signal days ({len(Cc.day.unique()) / len(pd.period_range(P0, dpost[-1], freq="M")):.1f} per month)', flush=True)
    for lab, usd, T, S, cost in SIZES:
        R = run_n(Cc, T, S, pdn); e = evalm(R, usd, cost); mo = monthly(R, usd, cost); cf = []
        for _ in range(10):
            Z = Cc.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); cf.append(evalm(run_n(Z, T, S, pdn), usd, cost)['pass'])
        print(f'  {lab:<16} trades {len(R):>4} win {R.pts.gt(0).mean():.0%} avg {R.pts.mean() - cost:+5.1f} pts | 1-month eval: pass {e["pass"]:.0%} fail {e["fail"]:.0%} expire {e["expire"]:.0%} (coin-flip pass {np.median(cf):.0%}) | month $: mean {mo.mean():+6,.0f} median {mo.median():+6,.0f} up {mo.gt(0).mean():.0%} worst {mo.min():+7,.0f} best {mo.max():+7,.0f}', flush=True)
# ---------------------------------------------------------------- ensemble tree on all the context (asked Nov 4): walk-forward, never scored by a model that saw the trade
from sklearn.ensemble import HistGradientBoostingClassifier
print('\n== ENSEMBLE: gradient-boosted trees on all context, walk-forward by year (train on every earlier year, score the next), S0 trades, 150/75', flush=True)
R0 = run_q(C2, 150, 75).reset_index(drop=True); Z = pd.concat([C2.reset_index(drop=True), R0[['pts']]], axis=1).dropna(subset=['pts']).reset_index(drop=True)
Z['win'] = (Z.pts > 0).astype(int); Z['yr'] = Z.day.dt.year; Z['withgap'] = (Z.gap_dir == Z.dir).astype(float); Z['mins'] = m5[Z.j.values]
FEAT = ['bull', 'stress', 'z20', 'r20', 'vixp', 'net_gex', 'pcr', 'withgap', 'H1', 'H4', 'SPY', 'MEGA', 'dir', 'mins']
X = Z[FEAT].astype(float); Z['p'] = np.nan
for y in range(2020, 2027):
    tr, te = Z.yr < y, Z.yr == y
    if te.sum() == 0 or tr.sum() < 80: continue
    mdl = HistGradientBoostingClassifier(max_depth=2, learning_rate=.05, max_iter=150, min_samples_leaf=20, l2_regularization=2.0, random_state=0).fit(X[tr], Z.win[tr])
    Z.loc[te, 'p'] = mdl.predict_proba(X[te])[:, 1]
from sklearn.metrics import roc_auc_score
W = Z[Z.p.notna() & (Z.day >= P0)].copy(); thr = W.p.median()
print(f'  2022-26 scored trades {len(W)}: AUC {roc_auc_score(W.win, W.p):.3f} | top half (p >= median) n{(W.p >= thr).sum()} win {W[W.p >= thr].win.mean():.0%} avg {W[W.p >= thr].pts.mean():+.1f} pts | bottom half win {W[W.p < thr].win.mean():.0%} avg {W[W.p < thr].pts.mean():+.1f} | all win {W.win.mean():.0%} avg {W.pts.mean():+.1f}', flush=True)
for y, g in W.groupby('yr'): print(f'    {y}: AUC {roc_auc_score(g.win, g.p) if g.win.nunique() > 1 else float("nan"):.2f}  top-half avg {g[g.p >= thr].pts.mean():+.1f} vs bottom {g[g.p < thr].pts.mean():+.1f} (n {len(g)})', flush=True)
top = C2.reset_index(drop=True).loc[Z.index[Z.p.notna() & (Z.p >= thr)]]
for lab, usd, T, S, cost in SIZES[2:3]:
    R = run_n(top[top.day >= P0], T, S, 1); e = evalm(R, usd, cost); mo = monthly(R, usd, cost)
    print(f'  model top half, {lab}: trades {len(R)} | 1-month eval pass {e["pass"]:.0%} fail {e["fail"]:.0%} expire {e["expire"]:.0%} | month $ mean {mo.mean():+,.0f} up {mo.gt(0).mean():.0%}', flush=True)
