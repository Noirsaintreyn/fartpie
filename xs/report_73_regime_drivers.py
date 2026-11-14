"""Nov 14 2026 WHAT DRIVES THE REGIME? MONTHLY PANEL INSTEAD OF THREE ERAS (declared before results; Rey: find the cause, not just another gate). Three eras = three data points, so any gate fitted to them is overfit. Use 87 months instead. Unit = entry month; outcomes: O1 mean net return of ALL long flips entered that month, O2 HUGE (>=+100%) per 100 flips, O3 mean of high-short (DTC>=10) flips when the high-short basket leads (months with >=8 such flips), O4 mean of flips in industry groups leading by >5% (months with >=8). Predictors are measured at the LAST TRADING DAY BEFORE the month (no look-ahead), declared with mechanism = liquidity / risk appetite funds speculative leadership: X1 HY credit spread level, X2 HY spread 60d change, X3 VIX level, X4 VIX 20d change, X5 2y Treasury yield 60d change (tightening vs easing), X6 NFCI financial-conditions level, X7 universe 60d return (trend), X8 universe breadth (share in vector-bull), X9 universe 20d realized vol, X10 high-short basket 40d lead. Ten tests -> Bonferroni p<0.005. PASS for a driver = same sign in 2019-21 (36 months) and 2022-26Q1 (51 months), |Spearman rho|>=0.25 in each, and pooled p<0.005, for at least one outcome. If a driver passes, report an ex-ante rule (driver above/below its own trailing-12-month median) by era. Flips: report_64 (31.8k), 30bp, survivorship-biased."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
src = open('report_71_sector_and_si_change.py').read(); exec(src.split("def mc(S, m, base):")[0].replace("src = open('report_69_short_regime.py').read(); exec(src.split(\"print('PART A\")[0])", "src0 = open('report_69_short_regime.py').read(); exec(src0.split(\"print('PART A\")[0])"))
FR = pd.read_pickle(SC + 'fred.pkl'); FR['date'] = pd.to_datetime(FR.date); FR = FR.set_index('date').sort_index().ffill(); M = pd.read_pickle(SC + 'market_state.pkl')
ix = cl.index; unc = (1 + un.fillna(0)).cumprod(); X = pd.DataFrame(index=ix)
X['X1_HYOAS'] = FR.HY_OAS.reindex(ix, method='ffill'); X['X2_HYchg60'] = X.X1_HYOAS - X.X1_HYOAS.shift(60); X['X3_VIX'] = FR.VIX.reindex(ix, method='ffill'); X['X4_VIXchg20'] = X.X3_VIX - X.X3_VIX.shift(20)
X['X5_2ychg60'] = FR.DGS2.reindex(ix, method='ffill').diff(60); X['X6_NFCI'] = FR.NFCI.reindex(ix, method='ffill'); X['X7_trend60'] = unc / unc.shift(60) - 1; X['X8_breadth'] = M.breadth.reindex(ix); X['X9_rvol20'] = un.rolling(20).std() * np.sqrt(252); X['X10_SQlead'] = leadSQ
P['mo'] = P.t.dt.to_period('M'); P['SQL'] = P.SQ10 & P.LEAD; P['GL5'] = P.GVAL > 0.05
def agg(g): return pd.Series(dict(O1=g.rn.mean(), O2=g.HUGE.mean() * 100, O3=g[g.SQL].rn.mean() if g.SQL.sum() >= 8 else np.nan, O4=g[g.GL5].rn.mean() if g.GL5.sum() >= 8 else np.nan, n=len(g)))
Y = P.groupby('mo').apply(agg); Y = Y[Y.n >= 30]; 
def at_month_start(col, mo): 
    d = mo.to_timestamp() - pd.Timedelta(days=1); s = X[col].dropna(); s = s[s.index <= d]; return s.iloc[-1] if len(s) else np.nan
for c in X.columns: Y[c] = [at_month_start(c, m) for m in Y.index]
Y['era'] = np.where(Y.index.to_timestamp() < '2022-01-01', 'A', 'B'); print(f'months {len(Y)} (A=2019-21: {(Y.era=="A").sum()}, B=2022-26Q1: {(Y.era=="B").sum()}); months with O3 {Y.O3.notna().sum()}, O4 {Y.O4.notna().sum()}')
print('\nSpearman rho of predictor (month start) vs outcome: rho A / rho B / pooled rho (p)   [* = passes: same sign, |rho|>=0.25 both halves, pooled p<0.005]')
passed = []
for o, lab in (('O1', 'all flips mean'), ('O2', 'HUGE per 100'), ('O3', 'high-short & basket-lead mean'), ('O4', 'strong sector-lead mean')):
    print(f'-- outcome {o}: {lab}')
    for c in X.columns:
        r = []
        for e in ('A', 'B'):
            z = Y[(Y.era == e)][[o, c]].dropna(); r.append(spearmanr(z[o], z[c])[0] if len(z) >= 10 else np.nan)
        z = Y[[o, c]].dropna(); rp, pp = spearmanr(z[o], z[c]) if len(z) >= 20 else (np.nan, np.nan)
        ok = (r[0] * r[1] > 0) and abs(r[0]) >= .25 and abs(r[1]) >= .25 and pp < .005; passed += [(o, c)] if ok else []
        print(f'   {c:13s} {r[0]:+.2f} / {r[1]:+.2f} / pooled {rp:+.2f} (p {pp:.3f}, n {len(z)}) {"*" if ok else ""}')
print('\nPASSED:', passed or 'none')
Y.to_pickle(SC + 'regime_monthly.pkl')
for o, c in passed:
    med = Y[c].rolling(12, min_periods=6).median().shift(1); hi = Y[c] > med; sgn = np.sign(spearmanr(Y[[o, c]].dropna()[o], Y[[o, c]].dropna()[c])[0])
    print(f'\nex-ante rule for {o} on {c}: favourable when {c} {"above" if sgn > 0 else "below"} its prior-12m median')
    for e in ('A', 'B'):
        z = Y[Y.era == e]; fav = (hi if sgn > 0 else ~hi)[z.index] & med[z.index].notna(); print(f'   era {e}: favourable months {int(fav.sum())} mean {o} {z.loc[fav, o].mean():+.2f} | other months {int((~fav).sum())} mean {z.loc[~fav, o].mean():+.2f}')
