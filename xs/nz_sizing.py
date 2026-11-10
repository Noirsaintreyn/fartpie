"""Nov 10 2026: RISK-NORMALISED NOISE ENGINE (NZ) + GAMMA / VOLUME-PROFILE CONTEXT SIZING. DECLARED BEFORE RESULTS. ET clock, 1 NQ, cost 0.5, 2018-02..2026-01-30, periods 2018-21 | 2022-26.
Variants (points per day, daily std, mean/std, worst day, worst month, % months > 0):
 base | stop 25 | stop 15 | daily loss limit L (stop trading the day once its running P&L <= -L: 40, 60) | band-width sizing w = clip(median sigma(10:30) over 252d / today's sigma, .5, 1.5)
 | CONTEXT VOL-TARGET: w = clip(median(pred range) / pred range, .5, 1.5), pred = log trailing-14d range + ridge on [gamma pct, positive gamma, |open-POC| noise units, VA width/ADR, open-vs-VA] fitted on 2018-21 only
 | CONTEXT EDGE-TILT: negative gamma x1.25, positive gamma x0.75; open above prior VA x1.25, below x0.75 (direction chosen from noise_context TEST 2 means; weak evidence, reported as exploratory)
 | combinations: stop 25 + LDL 40 + band-width + context vol-target.
Judge by mean/std and worst month in BOTH periods, not by pts/day (sizing up cheap days raises points trivially)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs')
import noise_context as C
from nz_core import noise, DAYS, A, SIG, SI, GRID, pos_of
F = C.F.copy(); idx = F.index
sg1030 = pd.Series({d: SIG[SI[d]][pos_of[630]] for d in idx}); w_bw = (sg1030.rolling(252, min_periods=60).median().shift(1) / sg1030).clip(.5, 1.5).fillna(1.0)
from sklearn.linear_model import Ridge
X = pd.DataFrame({'gp': F.gp.fillna(.5), 'pos': F.pos.astype(float).fillna(0), 'dpoc': F.dpoc.fillna(F.dpoc.median()), 'vaw': F.vaw.fillna(F.vaw.median()), 'ab': (F['loc'] == 'above').astype(float), 'be': (F['loc'] == 'below').astype(float)})
lt = np.log(C.RR14.reindex(idx)); ok = F.y.notna() & lt.notna(); tr = ok & (idx < '2022-01-01')
rg = Ridge(alpha=5.0).fit(X[tr], F.y[tr]); pred = pd.Series(rg.predict(X), index=idx); lp = lt + pred                                    # predicted log range (as ratio of price)
w_cv = (np.exp(lp).rolling(252, min_periods=60).median().shift(1).bfill() / np.exp(lp)).clip(.5, 1.5).fillna(1.0)
w_tilt = pd.Series(1.0, index=idx) * np.where(F.pos == False, 1.25, np.where(F.pos == True, .75, 1.0)) * np.where(F['loc'] == 'above', 1.25, np.where(F['loc'] == 'below', .75, 1.0))
def series(stop=40.0, ldl=None, w=None):
    out = {}
    for d in idx:
        rs = noise(d, stop=stop); tot = 0.0; run = 0.0
        for r in sorted(rs, key=lambda r: r[1]):
            if ldl and run <= -ldl: break
            run += r[3]
        out[d] = run
    s = pd.Series(out)
    return s * (w.reindex(s.index).fillna(1.0) if w is not None else 1.0)
def rep(lab, s):
    o = []
    for nm, f in C.PER:
        x = s[f(s.index)]; mo = x.groupby(x.index.to_period('M')).sum(); o.append(f'{nm}: {x.mean():+5.2f} std {x.std():5.1f} m/s {x.mean() / x.std():.3f} worstD {x.min():+6.0f} worstM {mo.min():+6.0f} M>0 {np.mean(mo > 0):.0%}')
    print(f'  {lab:<44}' + ' | '.join(o), flush=True)
print('=== NZ risk-normalised + context sizing ===')
S = {}
S['base'] = series(); S['stop 25'] = series(25.0); S['stop 15'] = series(15.0); S['LDL 40'] = series(ldl=40); S['LDL 60'] = series(ldl=60)
S['band-width sizing'] = series(w=w_bw); S['context vol-target'] = series(w=w_cv); S['context edge-tilt (exploratory)'] = series(w=w_tilt)
S['stop25 + LDL40'] = series(25.0, 40); S['stop25 + LDL40 + band-width'] = series(25.0, 40, w_bw); S['stop25 + LDL40 + bw + ctx vol-target'] = series(25.0, 40, w_bw * w_cv)
S['stop25 + LDL40 + bw + ctx + tilt'] = series(25.0, 40, w_bw * w_cv * w_tilt)
for k, v in S.items(): rep(k, v)
pd.to_pickle(S, '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/nz_sizing.pkl')
print(f'\n  context ridge: out-of-sample corr of predicted vs actual log-range ratio 2022-26: {np.corrcoef(pred[ok & (idx >= "2022-01-01")], F.y[ok & (idx >= "2022-01-01")])[0, 1]:+.3f}')
