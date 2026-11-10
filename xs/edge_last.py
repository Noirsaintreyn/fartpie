"""Nov 10 2026: ONE LAST EDGE HUNT - density / analog methods on the current system (S6 V2 trade tables). DECLARED BEFORE RESULTS.
Target = daily points (live size sz x pts) of the RTH engines only (RA, M1, R1, RULES, V15a - they enter after 09:30). FIRST RUN LEAKED: it included ON+/ON-, which finish before 09:30 and whose P&L is the gap / overnight range used as features (IC +0.2, t > 5 - too good, caught and removed). Features known at 09:30: gap/ADR, overnight range/ADR, prior-day return, 5d return, distance from 20d mean (all /ADR20), HAR forecast vs its 1y median,
VIX level pct (1y rank), VIX 5d change, QQQ ATM IV pct, DEX pct, vanna flow. Standardised on 2014-21, clipped at +-4.
A  GMM CLUSTERS (k = 4..8 by BIC, fit on 2014-21 days only), assign 2022-26 days by predict. Mean daily points by cluster in both periods; PASS = a cluster whose sign is the same and |t| >= 2 vs the rest in both.
B  KNN ANALOGS, walk-forward: for each day from 2016, the 60 nearest PRIOR days in feature space (expanding), forecast = mean of their points. Report rank IC and top/bottom-third spread by period.
   Also the same for the sign of the day (win rate).
C  SIZING: if B has IC > 0 in both periods, size x (0.5 + forecast rank) clipped 0.5-1.5 and report points / day, daily std, worst month, by period.
Decision: nothing is adopted unless it passes in both periods; any pass goes to the Lucid sim with the room rule next."""
import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); Dn = pd.read_pickle(f'{SP}/nq_days.pkl'); Dn.index = pd.to_datetime(Dn.index).normalize(); M, F = pickle.load(open(f'{SP}/vol_greek.pkl', 'rb'))
ix = Dn.index; DAYS = ix[(ix >= '2014-02-01') & (ix <= '2026-01-30')]
y = pd.Series(0.0, index=DAYS)
for (t, u, yz), n in zip(OUT['V2'], NAMES):
    if n in ('ON+', 'ON-'): continue
    t = t.copy(); t['day'] = pd.to_datetime(t.day).dt.normalize(); t = t[(t.day >= '2014-02-01') & (t.day <= '2026-01-30') & (t.sz > 0)]
    y = y.add((t.pts * t.sz).groupby(t.day.values).sum().reindex(DAYS).fillna(0), fill_value=0)
adr = Dn.rng.shift(1).rolling(20).mean(); dc = Dn.DC.astype(float); onr = (Dn.OH - Dn.OL)
X = pd.DataFrame({'gap': Dn.gap / adr, 'onr': onr / adr, 'r1': dc.diff().shift(1) / adr, 'r5': dc.diff(5).shift(1) / adr, 'd20': (dc.shift(1) - dc.shift(1).rolling(20).mean()) / adr,
                  'har': np.log(M.HARf.shift(0)) - np.log(M.HARf).shift(1).rolling(252, min_periods=120).median()})
vx = np.exp(M.VIX); X['vixp'] = vx.rolling(252, min_periods=120).rank(pct=True); X['vix5'] = M.VIXd5; X['ivp'] = np.exp(M.IV).rolling(252, min_periods=120).rank(pct=True)
ixf = lambda col: pd.Series([F[col].iloc[i] if i >= 0 else np.nan for i in (F.index.searchsorted(ix) - 1)], index=ix)
X['dexp'] = ixf('dexp'); X['vflow'] = ixf('vflow')
X = X.reindex(DAYS).dropna(); yy = y.reindex(X.index); A = X.index < '2022-01-01'
mu, sd = X[A].mean(), X[A].std(); Z = ((X - mu) / sd).clip(-4, 4)
print(f'days {len(X)} ({X.index.min():%Y-%m-%d} - {X.index.max():%Y-%m-%d}); base points / day 14-21 {yy[A].mean():+.2f} | 22-26 {yy[~A].mean():+.2f}\n')
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
best = None
for k in range(4, 9):
    g = GaussianMixture(k, covariance_type='full', random_state=0, n_init=3).fit(Z[A]); b = g.bic(Z[A])
    if best is None or b < best[0]: best = (b, k, g)
_, k, g = best; lab = pd.Series(g.predict(Z), index=Z.index); print(f'=== A: GMM, k = {k} (BIC) ===')
print('  cluster | n 14-21 / 22-26 | pts/day 14-21 (t vs rest) | pts/day 22-26 (t vs rest) | PASS')
for c in range(k):
    m = lab == c; r = []
    for per in (A, ~A):
        a, b = yy[m & per], yy[~m & per]; r.append((a.mean(), tt(a, b), (m & per).sum()))
    ok = all(np.isfinite(x[1]) and abs(x[1]) >= 2 for x in r) and np.sign(r[0][0] - yy[A].mean()) == np.sign(r[1][0] - yy[~A].mean())
    print(f'  {c} | {r[0][2]:4d} / {r[1][2]:4d} | {r[0][0]:+7.1f} (t {r[0][1]:+5.2f}) | {r[1][0]:+7.1f} (t {r[1][1]:+5.2f}) |{" <== PASS" if ok else ""}')
print('\n=== B: kNN analogs (60 nearest prior days), forecast of the day\'s points ===', flush=True)
Zv = Z.values; yv = yy.values; fc = np.full(len(Z), np.nan); fw = np.full(len(Z), np.nan); win = (yv > 0).astype(float)
for i in range(250, len(Z)):
    d = ((Zv[:i] - Zv[i]) ** 2).sum(axis=1); nn = np.argpartition(d, 60)[:60]; fc[i] = yv[nn].mean(); fw[i] = win[nn].mean()
fcs = pd.Series(fc, index=Z.index); fws = pd.Series(fw, index=Z.index); ok_ = fcs.notna()
for lab_, f in (('points', fcs), ('win prob', fws)):
    for per_n, per in (('14-21', A), ('22-26', ~A)):
        mm = ok_ & per; ic = pd.Series(f[mm]).corr(yy[mm], method='spearman'); q = f[mm].rank(pct=True); top, bot = yy[mm][q > 2 / 3], yy[mm][q <= 1 / 3]
        print(f'  {lab_:<9} {per_n}: rank IC {ic:+.3f} | top third {top.mean():+6.1f} vs bottom third {bot.mean():+6.1f} pts/day (t {tt(top, bot):+5.2f})')
print('\n=== C: sizing by the points forecast ===')
for nm, mult in (('base', pd.Series(1.0, index=Z.index)), ('size x clip(0.5 + rank, .5, 1.5)', (0.5 + fcs.expanding(min_periods=250).rank(pct=True)).clip(.5, 1.5).fillna(1.0))):
    s = yy * mult; mo = s.groupby(s.index.to_period('M')).sum(); ap = s.index < '2022-01-01'
    print(f'  {nm:<34} pts/day all {s.mean():+6.2f} | 14-21 {s[ap].mean():+6.2f} | 22-26 {s[~ap].mean():+6.2f} | std {s.std():6.1f} | mean/std {s.mean() / s.std():.3f} | worst month {mo.min():+6.0f}')
