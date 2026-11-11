"""MIX SCREEN (Nov 11 2026, declared before results). Every family in one table, per trade of the two scored legs (V15a, RA).
Families: ICT (sweeps/MSS/disp/FVG/PD/opens/SMT/session), VP/AMT, OPT (greeks), ECON (econometrics: vol regime, variance ratio, autocorr, trend t, gap z), TA (SMAs, Donchian, ROC, daily RSI, intraday RSI/ADX/BB/ER), CAL (day of week, month turn, opex, NFP week), XA (mega-cap breadth, ES smt).
Candidate conditions: signed ternary -> ==+1 and ==-1; continuous -> top/bottom third (cut from era A 2014-21 only); directional ones are made 'with trade'.
Step 1 select conditions on era A only (|t|>=1.5, n1,n0>=20), oriented so that cond=1 is the good side. Step 2 score = count of selected conditions in a family subset (127 subsets) and tier by era-A terciles.
Step 3 validate on era B (2022+). Placebo: same pipeline with pts shuffled (200x) -> null of the best-of-127 B result. Macro limit: no econ-calendar/yields/VIX data uploaded -> only calendar proxies."""
import numpy as np, pandas as pd, itertools, warnings, glob, pickle; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
rng = np.random.default_rng(7)
D = pd.read_pickle(SC + 'nq_days.pkl'); D.index = pd.to_datetime(D.index)
C = D.C; ret = np.log(C / C.shift(1)); P = lambda s: s.shift(1)          # prior-day values only
rv5, rv60 = ret.abs().rolling(5).mean(), ret.abs().rolling(60).mean()
ew = np.sqrt((ret ** 2).ewm(alpha=.06).mean());
vr = (np.log(C / C.shift(5)).rolling(60).var() / (5 * ret.rolling(60).var()))
ar1 = ret.rolling(60).corr(ret.shift(1)); tr20 = ret.rolling(20).mean() / (ret.rolling(20).std() / np.sqrt(20))
sma = {k: C.rolling(k).mean() for k in (20, 50, 200)}; hi20, lo20 = C.rolling(20).max(), C.rolling(20).min()
dl = C.diff(); up = dl.clip(lower=0).ewm(alpha=1 / 14).mean(); dn = (-dl.clip(upper=0)).ewm(alpha=1 / 14).mean(); rsi14 = 100 - 100 / (1 + up / dn)
DAY = pd.DataFrame({'volx': P(rv5 / rv60), 'ewx': P(ew / ew.rolling(60).mean()), 'vr': P(vr), 'ar1': P(ar1), 'tr20': P(tr20), 'rsi14': P(rsi14), 'roc10': P(np.log(C / C.shift(10))),
    'd20': P(C / sma[20] - 1), 'd50': P(C / sma[50] - 1), 'd200': P(C / sma[200] - 1), 'tr2050': P(sma[20] / sma[50] - 1), 'don': P((C - lo20) / (hi20 - lo20)), 'ret1': P(ret),
    'gapz': D.gap / D.rng20, 'rng_x': P(D.rng / D.rng20)})
# calendar
idx = D.index; dd = pd.DataFrame(index=idx); dd['dow'] = idx.dayofweek; ym = idx.to_period('M')
dd['tdom'] = pd.Series(1, index=idx).groupby(ym).cumsum().values; dd['tdoe'] = pd.Series(1, index=idx).groupby(ym).cumsum().groupby(ym).transform('max').values - dd.tdom
d3 = [(ts.day, ts.dayofweek) for ts in idx]; dd['opex'] = [(ts.dayofweek == 4 and 15 <= ts.day <= 21) for ts in idx]
dd['opexw'] = [(15 <= (ts + pd.Timedelta(days=(4 - ts.dayofweek))).day <= 21) for ts in idx]; dd['nfp'] = [(ts.dayofweek == 4 and ts.day <= 7) for ts in idx]
dd['qend'] = [(ts.month in (3, 6, 9, 12)) and (ts.day >= 24) for ts in idx]
# mega-cap breadth: share of 8 names above their prior daily close at the bar before entry (ET 15m, 2018+)
MC = {}
for t in ('AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA', 'AVGO'):
    f = glob.glob(U + f'lse_clean/{t}_15m_*.parquet')[0]; x = pd.read_parquet(f); x['day'] = x.datetime.dt.normalize(); x['m'] = x.datetime.dt.hour * 60 + x.datetime.dt.minute
    x = x[(x.m >= 570) & (x.m < 960)]; MC[t] = x
def breadth(day, m, dr):
    s = []
    for t, x in MC.items():
        pc = x[x.day < day].close.iloc[-1] if (x.day < day).any() else np.nan
        b = x[(x.day == day) & (x.m + 15 <= m)]
        if len(b) and pc == pc: s.append(np.sign(b.close.iloc[-1] - pc) * dr)
    return np.mean(s) if len(s) >= 5 else np.nan
LEGS = {}
for nm in ('V15a', 'RA'):
    N = pd.read_pickle(SC + f'narr_{nm}.pkl'); N = N.drop_duplicates(['day', 'm']).copy(); N['day'] = pd.to_datetime(N.day)
    A = pd.read_pickle(SC + f'amt_{nm}.pkl'); A = A.drop_duplicates(['day', 'm']); A['day'] = pd.to_datetime(A.day)
    S = pd.read_pickle(SC + f'stack_{nm}.pkl'); S = S.drop_duplicates(['day', 'm']); S['day'] = pd.to_datetime(S.day)
    O = pd.read_pickle(SC + f'opt_{nm}.pkl'); O = O.drop_duplicates(['day', 'm']); O['day'] = pd.to_datetime(O.day)
    T = N.set_index(['day', 'm']).join(A.set_index(['day', 'm'])[['open_val', 'px_val', 'rule80', 'migr', 'dev_poc', 'dev_va', 'ib_break', 'open_drive', 'px_poc1']], how='left')
    T = T.join(S.set_index(['day', 'm'])[['vflow', 'gex']], how='left').join(O.set_index(['day', 'm'])[['dex_al', 'net_vanna_al', 'net_charm_al', 'd_call', 'skew_25d', 'ivchg', 'pcv']], how='left').reset_index()
    X = {}; fam = {}; dr = T.dir.values; ok = lambda s: s.reindex(T.day).values
    def add(f, k, v): X[k] = np.asarray(v, float); fam[k] = f
    for c in ('fvg_with', 'disp', 'mss', 'pd_disc', 'sess_disc', 'open_mid', 'open_830', 'open_930', 'gap', 'prior_day', 'prior_loc', 'smt', 'HTF', 'AL', 'AN', 'LN', 'al'): add('ICT', c, T[c])
    add('ICT', 'n_with', T.n_with); add('ICT', 'score', T.score)
    for c in ('open_val', 'px_val', 'rule80', 'migr', 'dev_poc', 'dev_va', 'ib_break', 'open_drive', 'px_poc1'): add('VP', c, T[c])
    add('OPT', 'vflow', T.vflow); add('OPT', 'gex_neg', np.where(T.gex.isna(), np.nan, (T.gex < 0)))
    for c in ('dex_al', 'net_vanna_al', 'net_charm_al', 'd_call', 'skew_25d', 'ivchg', 'pcv'): add('OPT', c, T[c])
    for c in ('volx', 'ewx', 'vr', 'ar1', 'tr20', 'gapz', 'rng_x'):
        v = ok(DAY[c]) if c != 'gapz' else ok(DAY[c])
        add('ECON', c, v * dr if c in ('tr20', 'gapz') else v)
    for c in ('rsi14', 'roc10', 'd20', 'd50', 'd200', 'tr2050', 'ret1', 'don'):
        v = ok(DAY[c]); add('TA', c, v * dr if c in ('roc10', 'd20', 'd50', 'd200', 'tr2050', 'ret1') else (v if c == 'rsi14' else np.where(dr > 0, v, 1 - v)))
    for c in ('rsi', 'adx', 'bbr', 'er', 'dol'): add('TA', c, T[c])
    cd = dd.reindex(T.day)
    for k in range(5): add('CAL', f'dow{k}', (cd.dow.values == k).astype(float))
    add('CAL', 'turn', ((cd.tdom.values <= 3) | (cd.tdoe.values <= 1)).astype(float)); add('CAL', 'opexw', cd.opexw.values.astype(float)); add('CAL', 'nfp', cd.nfp.values.astype(float)); add('CAL', 'qend', cd.qend.values.astype(float))
    add('XA', 'breadth', [breadth(d, m, r) if d >= pd.Timestamp('2018-01-05') else np.nan for d, m, r in zip(T.day, T.m, dr)])
    LEGS[nm] = (T, pd.DataFrame(X), fam)
    print(nm, len(T), 'features', len(X), flush=True)
fam = LEGS['V15a'][2]
# ---- candidate conditions (cuts from era A only, shared across legs by pooling A)
ALLT = pd.concat([LEGS[k][0].assign(leg=k) for k in LEGS], ignore_index=True); ALLX = pd.concat([LEGS[k][1] for k in LEGS], ignore_index=True)
A_ = (ALLT.era == 'A').values; B_ = (ALLT.era == 'B').values
cands = {}
for c in ALLX.columns:
    v = ALLX[c]; u = v.dropna().unique()
    if len(u) <= 6:
        for val in sorted(u): cands[f'{c}={val:g}'] = (fam[c], (v == val).astype(float).where(v.notna()))
    else:
        lo, hi = v[A_].quantile(1 / 3), v[A_].quantile(2 / 3)
        cands[f'{c}>=hi'] = (fam[c], (v >= hi).astype(float).where(v.notna())); cands[f'{c}<=lo'] = (fam[c], (v <= lo).astype(float).where(v.notna()))
CF = {k: f for k, (f, _) in cands.items()}; CM = pd.DataFrame({k: s for k, (_, s) in cands.items()}).fillna(0).values  # NaN -> not satisfied
CMn = pd.DataFrame({k: s for k, (_, s) in cands.items()}).notna().values
names = list(cands); pts = ALLT.pts.values
def wt(y, c, mask):
    a, b = y[mask & (c == 1)], y[mask & (c == 0)]
    if len(a) < 20 or len(b) < 20: return 0.0, np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)); return a.mean() - b.mean(), (a.mean() - b.mean()) / se
def select(y, thr=1.5):
    sel = []
    for j, nm_ in enumerate(names):
        d, t = wt(y, CM[:, j], A_ & CMn[:, j])
        if t == t and abs(t) >= thr: sel.append((j, 1 if d > 0 else -1))
    return sel
FAMS = ['ICT', 'VP', 'OPT', 'ECON', 'TA', 'CAL', 'XA']
def scores(sel, y):
    """sub-score per family: count of selected conditions satisfied (flipped ones count when NOT satisfied)."""
    out = {f: np.zeros(len(y)) for f in FAMS}
    for j, s in sel:
        out[CF[names[j]]] += (CM[:, j] if s > 0 else (1 - CM[:, j])) * CMn[:, j]
    return out
def mixes(sel, y):
    sc = scores(sel, y); res = []
    for r in range(1, 8):
        for S in itertools.combinations(FAMS, r):
            s = sum(sc[f] for f in S)
            if s[A_].std() == 0: continue
            lo, hi = np.quantile(s[A_], 1 / 3), np.quantile(s[A_], 2 / 3); top, bot = s >= max(hi, lo + 1e-9), s <= lo
            if top[A_].sum() < 25 or bot[A_].sum() < 25 or top[B_].sum() < 25 or bot[B_].sum() < 25: continue
            dA = y[top & A_].mean() - y[bot & A_].mean(); dB = y[top & B_].mean() - y[bot & B_].mean()
            tB = wt(y, top.astype(float), B_ & (top | bot))[1]
            res.append((S, dA, dB, tB, top, bot, s))
    return res
sel = select(pts); print(f'\nconditions selected on era A (|t|>=1.5): {len(sel)} of {len(names)}', flush=True)
print('\n--- single conditions: A-selected, B validation (oriented good=1) ---')
print(f' {"condition":28} {"fam":5} {"dirn":4} | A effect (t)      | B effect (t)   n1_B')
rows = []
for j, s in sel:
    c = CM[:, j] if s > 0 else 1 - CM[:, j]; dA, tA = wt(pts, c, A_ & CMn[:, j]); dB, tB = wt(pts, c, B_ & CMn[:, j]); rows.append((names[j], CF[names[j]], s, dA, tA, dB, tB, int(((c == 1) & B_ & CMn[:, j]).sum())))
for r in sorted(rows, key=lambda r: -r[6] if r[6] == r[6] else 9):
    print(f' {r[0]:28} {r[1]:5} {"+" if r[2] > 0 else "NOT":4} | {r[3]:+6.1f} ({r[4]:+.1f}) | {r[5]:+6.1f} ({r[6]:+.1f}) {r[7]:4d}', flush=True)
res = mixes(sel, pts)
print('\n--- family mixes: tier by era-A terciles; spread = mean pts(top third) - mean pts(bottom third) ---')
print(f' {"mix":34} spreadA  spreadB  tB')
res.sort(key=lambda r: -min(r[1], r[2]))
for S, dA, dB, tB, *_ in res[:15]: print(f' {"+".join(S):34} {dA:+6.1f}  {dB:+6.1f}  {tB:+.1f}', flush=True)
print('\n--- single-family spreads ---')
for S, dA, dB, tB, *_ in [r for r in res if len(r[0]) == 1]: print(f' {S[0]:34} {dA:+6.1f}  {dB:+6.1f}  {tB:+.1f}')
best_obs = res[0][2] if res else np.nan; best_obs_t = max(r[3] for r in res if min(r[1], r[2]) > 0) if res else np.nan
# placebo: shuffle pts within leg, rerun select+mixes
print('\n--- placebo (200 shuffles of pts within leg and era): distribution of best-mix min(spreadA, spreadB) ---', flush=True)
nul = []
for it in range(200):
    y = pts.copy()
    for lg in ('V15a', 'RA'):
        for e in ('A', 'B'):
            m_ = ((ALLT.leg == lg) & (ALLT.era == e)).values; y[m_] = rng.permutation(y[m_])
    r2 = mixes(select(y), y)
    nul.append(max([min(r[1], r[2]) for r in r2]) if r2 else 0.0)
nul = np.array(nul); obs = max(min(r[1], r[2]) for r in res)
print(f' observed best min(spreadA,spreadB) {obs:+.1f} | placebo median {np.median(nul):+.1f} 90th {np.quantile(nul, .9):+.1f} 95th {np.quantile(nul, .95):+.1f} | p = {np.mean(nul >= obs):.3f}', flush=True)
# save tiers for the top mixes for the chain run
out = {'sel': [(names[j], s) for j, s in sel], 'rows': ALLT[['leg', 'day', 'm', 'dir', 'era', 'pts']].copy()}
tops = {}
for S, dA, dB, tB, top, bot, s in res[:8] + [r for r in res if len(r[0]) in (1, 7)]:
    tops['+'.join(S)] = dict(top=top, bot=bot, s=s, spreadA=dA, spreadB=dB)
out['tops'] = tops; pickle.dump(out, open(SC + 'mix_tiers.pkl', 'wb'))
print('saved', len(tops), 'mixes')
