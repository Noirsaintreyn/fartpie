"""TERRIBLE vs GREAT days: volatility (YZ/HAR/GJR), VWAP distance, flip (VWAP-cross) speed & volume, noise-band breakout, GMM regimes. Variant Z days from tiers_days.pkl. Nov 10 2026."""
import sys, os; sys.path.insert(0, '/home/claude/fartpie/xs'); os.chdir('/home/claude/fartpie/xs')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('magnitude.py').read(); src = src[:src.index("# GEX / vxn / balance")]
src = '\n'.join(l for l in src.split('\n') if 'idea3_levels' not in l)
exec(src)
from nz_core import A, OPEN, PC, sg, NB
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
TD = pd.read_pickle(SC + 'tiers_days.pkl'); TD.index = pd.to_datetime(TD.index).normalize()
r20 = D.rng.rolling(20).mean().shift(1)
rows = {}
DEC = [j for j in range(NB) if (570 + 5 * j + 5) % 30 == 0 and 600 <= 570 + 5 * j + 5 <= 930]
for d in TD.index:
    if d not in A or not np.isfinite(r20.get(d, np.nan)): continue
    a = A[d][0]; o, h, l, c, v, vw = [np.asarray(x, float) for x in a]; R = r20[d]; op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op; s = np.asarray(sg(d), float)
    f = {}
    ok12 = ~np.isnan(c[:12]) & ~np.isnan(vw[:12])
    f['vd10_abs'] = abs(c[5] - vw[5]) / R if np.isfinite(c[5]) and np.isfinite(vw[5]) else np.nan
    f['vd_max30'] = np.nanmax(np.abs(c[:12] - vw[:12])) / R if ok12.sum() > 3 else np.nan
    sgn = np.sign(c[:24] - vw[:24]); sgn = sgn[~np.isnan(sgn)]; nz = sgn[sgn != 0]
    cr = np.flatnonzero(np.diff(nz) != 0) if len(nz) > 2 else np.array([], int)
    f['n_cross'] = len(cr); f['cross_runlen'] = len(nz) / (len(cr) + 1) if len(nz) else np.nan
    mv = np.nanmean(v[:24]) if np.isfinite(v[:24]).any() else np.nan
    cb = [i + 1 for i in cr]; idx = np.flatnonzero(~np.isnan(np.sign(c[:24] - vw[:24])) & (np.sign(c[:24] - vw[:24]) != 0))
    f['flip_vol_rel'] = np.nanmean([v[idx[i]] / mv for i in cb if i < len(idx)]) if cb and mv > 0 else np.nan
    f['morn_vol'] = np.nansum(v[:12])
    # noise breakout (30-min decisions)
    st = []; first = None
    for j in DEC:
        if j >= len(c) or np.isnan(c[j]) or j >= len(s) or not np.isfinite(s[j]): continue
        ub, lb = max(op, pc) * (1 + s[j]), min(op, pc) * (1 - s[j]); state = 1 if c[j] > ub else (-1 if c[j] < lb else 0); st.append((j, state, c[j], ub, lb))
        if first is None and state != 0: first = (j, state, c[j], ub, lb)
    f['no_break'] = float(first is None)
    if first is not None:
        j, state, cj, ub, lb = first; f['brk_time'] = 5 * j + 5; f['brk_dist'] = ((cj - ub) if state > 0 else (lb - cj)) / R
        f['brk_vol_rel'] = v[j] / np.nanmean(v[:j]) if j > 0 and np.nanmean(v[:j]) > 0 else np.nan
        nxt = [x for x in st if x[0] > j]; f['brk_hold'] = float(nxt[0][1] == state) if nxt else np.nan
        f['brk_gap_align'] = float(np.sign(op - pc) == state)
    f['n_state_chg'] = sum(1 for x, y in zip(st[:-1], st[1:]) if x[1] != y[1])
    f['yz_pct'] = yzp.get(d, np.nan); f['har_ratio'] = har_ratio.get(d, np.nan); f['gjr_ratio'] = gjr_ratio.get(d, np.nan)
    rows[d] = f
F = pd.DataFrame(rows).T.astype(float); F['morn_vol_rel'] = F.morn_vol / F.morn_vol.rolling(20, min_periods=10).mean().shift(1); F = F.drop(columns='morn_vol')
X = TD.join(F, how='inner'); X['half'] = np.where(X.index.year < 2022, 'A', 'B'); X['T'] = X.tier.str[0].astype(int)
feats = ['gap_rel', 'yz_pct', 'har_ratio', 'gjr_ratio', 'vxn_pct', 'rng_prev_rel', 'trend_eff10', 'vd10_abs', 'vd_max30', 'n_cross', 'cross_runlen', 'flip_vol_rel', 'morn_vol_rel', 'no_break', 'brk_time', 'brk_dist', 'brk_vol_rel', 'brk_hold', 'brk_gap_align', 'n_state_chg']
def tt(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 8 or len(b) < 8: return np.nan
    return (b.mean() - a.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b) + 1e-12)
print(f'days {len(X)} | terrible {int((X.T==1).sum())} great {int((X.T==5).sum())}\n')
print(f'{"feature":14} | {"TERRIBLE":>8} {"BAD":>7} {"FLAT":>7} {"GOOD":>7} {"GREAT":>7} | great-minus-terrible t: 2018-21 | 2022-26 | consistent?')
for f in feats:
    m = [X[X.T == k][f].mean() for k in (1, 2, 3, 4, 5)]; ta = tt(X[(X.T == 1) & (X.half == 'A')][f], X[(X.T == 5) & (X.half == 'A')][f]); tb = tt(X[(X.T == 1) & (X.half == 'B')][f], X[(X.T == 5) & (X.half == 'B')][f])
    flag = 'YES' if np.isfinite(ta) and np.isfinite(tb) and np.sign(ta) == np.sign(tb) and min(abs(ta), abs(tb)) >= 2 else ''
    print(f'{f:14} | ' + ' '.join(f'{x:7.2f}' for x in m) + f' | {ta:+6.1f} | {tb:+6.1f} | {flag}')
# also: do these features predict the day's P&L (correlation, per half)?
print('\ncorr(feature, day pts) by half and among "pre-open" vs "post-hoc":')
for f in feats: print(f'  {f:14} A {X[X.half=="A"][[f,"pts"]].corr().iloc[0,1]:+.2f}  B {X[X.half=="B"][[f,"pts"]].corr().iloc[0,1]:+.2f}')
# GMM regimes
from sklearn.mixture import GaussianMixture; from sklearn.preprocessing import StandardScaler
gf = ['gap_rel', 'yz_pct', 'har_ratio', 'gjr_ratio', 'vxn_pct', 'rng_prev_rel', 'trend_eff10', 'vd_max30', 'n_cross', 'brk_time', 'brk_dist']
Z_ = X[gf + ['pts', 'T', 'half']].copy(); Z_['brk_time'] = Z_.brk_time.fillna(330); Z_['brk_dist'] = Z_.brk_dist.fillna(0); Z_ = Z_.dropna()
ss = StandardScaler().fit(Z_[Z_.half == 'A'][gf]); XA = ss.transform(Z_[Z_.half == 'A'][gf]); XB = ss.transform(Z_[Z_.half == 'B'][gf]); XAll = ss.transform(Z_[gf])
bic = {k: GaussianMixture(k, covariance_type='diag', n_init=5, random_state=0).fit(XA).bic(XA) for k in range(2, 6)}; kb = min(bic, key=bic.get); print(f'\nGMM (diag, fit on 2018-21, features {gf}): BIC {dict((k, round(v)) for k, v in bic.items())} -> k={kb}')
gm = GaussianMixture(kb, covariance_type='diag', n_init=5, random_state=0).fit(XA); Z_['cl'] = gm.predict(XAll)
for cl in range(kb):
    g = Z_[Z_.cl == cl]; ga, gb = g[g.half == 'A'], g[g.half == 'B']; cen = pd.Series(ss.transform(g[gf]).mean(0), index=gf)
    hi = cen.sort_values(key=abs, ascending=False).head(4)
    print(f' cluster {cl}: n {len(g):4d} (A {len(ga)}, B {len(gb)}) | mean pts A {ga.pts.mean():+6.1f} B {gb.pts.mean():+6.1f} | terrible% A {np.mean(ga["T"]==1):4.0%} B {np.mean(gb["T"]==1):4.0%} | great% A {np.mean(ga["T"]==5):4.0%} B {np.mean(gb["T"]==5):4.0%} | defining z: ' + ', '.join(f'{k} {v:+.1f}' for k, v in hi.items()))
X.to_pickle(SC + 'tg_features.pkl')
