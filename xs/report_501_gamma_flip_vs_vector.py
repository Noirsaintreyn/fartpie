"""Report 501 (Nov 26 2026): Rey - 'does a gamma flip pair with the vector at all'. DAILY vector (xs/vector_pine run_state_machine on RTH daily bars from the 15m cache) bull flips on the large caps that have daily net_gex files (xs/ext/gex, 2018-2025). Entry = next open, exit = the vector's own bear flip (signal exit, no time hold), 4bp each way.
DECLARED BEFORE RESULTS. Gamma flip = prior-close net_gex sign change within the last 3 trading days up to the signal close: GF_PN = positive -> negative (dealers turn short gamma), GF_NP = negative -> positive. Also NEG = net_gex<0 at the signal close, POSG = >=0.
H1 CO-OCCURRENCE: vector bull flips fall within 3 days after a GF_PN more often than a random day does (rate ratio > 1) - mechanism: amplification regime starts, trend initiation more likely to register as a vector flip.
H2 QUALITY: bull flips right after a GF_PN have higher PEAK (max favourable excursion before the vector exit, % ) and net mean return than other flips; GF_NP the opposite (dampening regime ends the trend, sign -). 
Controls: ATR14% tercile + name FE, SE clustered by month. Train 2019-21 vs test 2018 + 2022-25. Pass = same sign train & test, pooled |t|>=2. Also report the plain NEG vs POSG split (is regime alone enough)."""
import os, sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
UP = '/mnt/user-data/uploads/fartpie/'; rows = []; rate = []
for f in sorted(glob.glob('xs/ext/gex/*.csv')):
    tk = os.path.basename(f)[:-4]; p = glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')
    if not p or tk == 'QQQ': continue
    H = pd.read_parquet(p[0]).rename(columns={'datetime': 'ts'}); H['dt'] = pd.to_datetime(H.ts, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].sort_values('dt')
    D = H.groupby(H.dt.dt.normalize()).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last')).dropna(); D = D[D.h > D.l]; D = D.rename(columns={'h': 'high', 'l': 'low', 'c': 'close'}).reset_index().rename(columns={'dt': 'd'})
    G = pd.read_csv(f, parse_dates=['date']).set_index('date').net_gex; G = G[~G.index.duplicated()]; g = np.sign(G.reindex(D.d).values); gp = pd.Series(g); n = len(D)
    pn = ((gp.shift(k) >= 0) & (gp.shift(k - 1) < 0) if False else None)
    s = np.sign(pd.Series(G.reindex(D.d).values)); chg_pn = (s.shift(1) >= 0) & (s < 0); chg_np = (s.shift(1) < 0) & (s >= 0)
    gfpn = chg_pn.rolling(3, min_periods=1).max().fillna(0).astype(bool).values; gfnp = chg_np.rolling(3, min_periods=1).max().fillna(0).astype(bool).values
    regs, lv, atr, fls = run_state_machine(D[['high', 'low', 'close']]); bull = [i for i, d in fls if d == 1]; bear = [i for i, d in fls if d == -1]; atrp = pd.Series(atr).values / D.close.values
    rate.append(((~np.isnan(s.values)).sum(), gfpn[~np.isnan(s.values)].sum(), sum(1 for i in bull if not np.isnan(s.values[i])), sum(1 for i in bull if gfpn[i] and not np.isnan(s.values[i]))))
    for i in bull:
        if i + 2 >= n or np.isnan(s.values[i]): continue
        j = next((b for b in bear if b > i), None); e = D.o.values[i + 1]; end = j if j is not None else n - 1
        if j is None and False: continue
        r = D.close.values[j] / e - 1 - .0008 if j is not None else np.nan; pk = D.high.values[i + 1:end + 1].max() / e - 1
        rows.append(dict(tk=tk, d=D.d.values[i], r=r, PEAK=pk, hold=end - i, GF_PN=bool(gfpn[i]), GF_NP=bool(gfnp[i]), NEG=bool(s.values[i] < 0), atrp=atrp[i]))
S = pd.DataFrame(rows).dropna(subset=['r', 'atrp']); S['d'] = pd.to_datetime(S.d); S['mo_'] = S.d.dt.to_period('M').astype(str); S['y'] = S.d.dt.year; S['tr'] = (S.y >= 2019) & (S.y <= 2021)
S['a3'] = pd.qcut(S.atrp, 3, labels=False); S['hit5'] = (S.PEAK >= .05).astype(float); S['rw'] = S.r.clip(*S.r.quantile([.01, .99]))
r_ = np.array(rate); print(f'flips n {len(S)} names {S.tk.nunique()}; H1: GF_PN in prior 3d on {r_[:,1].sum() / r_[:,0].sum():.1%} of all days vs {r_[:,3].sum() / r_[:,2].sum():.1%} of vector bull flips (rate ratio {(r_[:,3].sum() / r_[:,2].sum()) / (r_[:,1].sum() / r_[:,0].sum()):.2f}, n flips with GF_PN {int(r_[:,3].sum())})')
def eff(D, flag, y):
    if D[flag].sum() < 8: return (np.nan, np.nan, int(D[flag].sum()))
    m = smf.ols(f'{y} ~ {flag}.astype(float) + C(a3) + C(tk)', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mo_)[0]}); k = [q for q in m.params.index if q.startswith(flag)][0]; return (m.params[k], m.tvalues[k], int(D[flag].sum()))
print(f"base: mean r {S.rw.mean():+.2%} win {(S.r>0).mean():.0%} PEAK med {S.PEAK.median():+.1%} hold med {S.hold.median():.0f}d")
print(f"{'flag':8s}| n tr/te | raw mean r on vs off | raw PEAK med on vs off | ADJ PEAK effect train test all (t) | ADJ r effect train test all (t)")
for f in ('GF_PN', 'GF_NP', 'NEG'):
    te = S[~S.tr]; a = [eff(D, f, 'PEAK') for D in (S[S.tr], te, S)]; b = [eff(D, f, 'rw') for D in (S[S.tr], te, S)]
    print(f"{f:8s}| {a[0][2]:3d}/{a[1][2]:3d} | {S[S[f]].rw.mean():+.2%} vs {S[~S[f]].rw.mean():+.2%} | {S[S[f]].PEAK.median():+.1%} vs {S[~S[f]].PEAK.median():+.1%} | {a[0][0]:+.3f} {a[1][0]:+.3f} {a[2][0]:+.3f} (t {a[2][1]:+.1f}) | {b[0][0]:+.3f} {b[1][0]:+.3f} {b[2][0]:+.3f} (t {b[2][1]:+.1f})")
