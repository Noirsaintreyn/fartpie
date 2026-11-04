"""Nov 4 2026: INFORMATION SCREEN. DECLARED BEFORE RESULTS. What, known before the entry, says anything about how a live-rule vector trade will do?
Universe: the 17,447 live-rule trades (S&P, ratio >= 1.5, 54 names with bars), outcome = trade return and 'runner' (>= +15%).
~70 features in 5 groups, all from data completed BEFORE the entry day:  MACRO (VIX level/term, credit, dollar, rates, copper/gold, oil, breadth, gold, bonds, small caps,
 EM, COT positioning of asset managers / leveraged funds in S&P and Nasdaq futures, SPY trend)  SECTOR (own sector ETF: 20/63/126d return, momentum rank of 11, above 50/200d SMA, SMA 50/200
 cross, relative strength vs SPY)  STOCK (returns, distance from 52w high, SMA/EMA states, ATR %, volume surge, relative strength vs SPY and vs its sector, vol vs own history)
 ENTRY (track, time of day, weekday, month, options ratio and its rank among that day's flips, number of flips that day, earnings timing)  existing columns (gap, run-up, z20, crowd...).
Test per feature: percentile rank of the feature (-0.5..+0.5) regressed on the trade return, per period (2019-21, 2022-26), standard errors clustered by entry month.
 beta = avg-trade difference between the top and bottom of the feature's range (pts).  INFORMATIVE = same sign in both periods and |t| >= 2 in both (chance of that by luck ~ 0.1% per feature).
Second stage: a score built only from features informative in 2019-21 (sign fixed), scored on 2022-26 untouched: avg trade by score third."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
X = pd.read_parquet('/tmp/claude-0/ml/ctx_rank_X.parquet'); U = X[X.ratio >= 1.5].sort_values('entry').reset_index(drop=True); U['d'] = U.entry.dt.normalize(); U['big'] = (U.ret >= .15).astype(float)
MD = '/tmp/claude-0/ml/macro'; ED = '/mnt/user-data/uploads/fartpie/etf_daily'
def asof_feature(s, name):                          # value known before the entry day (strictly prior session)
    s = s.dropna(); s = s[~s.index.duplicated()].sort_index(); s.index = pd.DatetimeIndex(s.index).astype('datetime64[ns]')
    pos = np.searchsorted(s.index.values, U.d.values.astype('datetime64[ns]'), side='left') - 1; U[name] = np.where(pos >= 0, s.values[np.clip(pos, 0, None)], np.nan)
P = pd.read_parquet(f'{MD}/px.parquet'); P.index = pd.to_datetime(P.index).tz_localize(None)
def etf(n): d = pd.read_parquet(f'{ED}/{n}.parquet'); d.index = pd.to_datetime(d.index).tz_localize(None); return d['Close'].dropna()
r = lambda s, n: s / s.shift(n) - 1
F = {}
F['m_vix'] = P['^VIX']; F['m_vix_pct'] = P['^VIX'].rolling(252).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True); F['m_vix_ts'] = P['^VIX'] / P['^VIX3M']; F['m_vix_chg5'] = P['^VIX'] / P['^VIX'].shift(5) - 1
F['m_credit50'] = r(P.HYG / P.IEF, 50); F['m_credit5'] = r(P.HYG / P.IEF, 5); F['m_usd50'] = r(P.UUP, 50); F['m_rates50'] = P['^TNX'] - P['^TNX'].shift(50); F['m_rates5'] = P['^TNX'] - P['^TNX'].shift(5)
F['m_cuau50'] = r(P['HG=F'] / P['GC=F'], 50); F['m_oil50'] = r(P.USO, 50); F['m_breadth50'] = r(P.RSP / P.SPY, 50); F['m_gold50'] = r(P['GC=F'], 50)
F['m_spy200'] = P.SPY / P.SPY.rolling(200).mean() - 1; F['m_spy50'] = P.SPY / P.SPY.rolling(50).mean() - 1; F['m_spy5'] = r(P.SPY, 5); F['m_spy20'] = r(P.SPY, 20); F['m_spy63'] = r(P.SPY, 63)
F['m_spy_dd'] = P.SPY / P.SPY.rolling(252, min_periods=60).max() - 1
spy = P.SPY
for n, k in (('TLT', 'm_tlt50'), ('IWM', 'm_iwm_rel50'), ('EEM', 'm_eem_rel50'), ('DBC', 'm_dbc50'), ('LQD', 'm_lqd50')):
    try:
        e = etf(n); F[k] = r(e / spy.reindex(e.index).ffill(), 50) if 'rel' in k else r(e, 50)
    except Exception: pass
C = pd.read_parquet(f'{MD}/tff_all.parquet'); C['code'] = C.CFTC_Contract_Market_Code.astype(str); C['rd'] = pd.to_datetime(C['Report_Date_as_YYYY-MM-DD'])
for code, nm in (('13874A', 'es'), ('209742', 'nq')):
    x = C[C.code == code].groupby('rd').sum(numeric_only=True).sort_index(); oi = x.Open_Interest_All
    for grp, col in (('am', 'Asset_Mgr'), ('lev', 'Lev_Money')):
        net = (x[f'{col}_Positions_Long_All'] - x[f'{col}_Positions_Short_All']) / oi; idx = net.rolling(156, min_periods=52).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
        F[f'm_cot_{nm}_{grp}'] = pd.Series(idx.values, index=idx.index + pd.Timedelta(days=6))
# sector ETFs
SECE = dict(tech='XLK', semi='SMH_yf', comm='XLC', cons='XLY', stap='XLP', fin='XLF', hlth='XLV', indu='XLI', energy='XLE')
SE = {k: etf(v if v.endswith('_yf') else v + '_yf') for k, v in SECE.items()}; allx = {t: etf(f'{t}_yf') for t in ('XLK', 'XLC', 'XLY', 'XLP', 'XLF', 'XLV', 'XLI', 'XLE', 'XLU', 'XLB', 'XLRE')}
M126 = pd.DataFrame({t: r(s, 126) for t, s in allx.items()}).dropna(how='all'); RK = M126.rank(axis=1, ascending=False)
for k, v in F.items(): asof_feature(v, k)
# per-ticker: sector + stock features
U['s_tmp'] = 0; cols = {}
SPYs = spy
for tk, g in U.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = load(f[0]); R = H[(H.m >= 570) & (H.m < 960)]; dd = R.groupby(R.dt.dt.normalize()).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'), v=('volume', 'sum'))
    dd.index = dd.index.astype('datetime64[ns]'); c = dd.c; pcl = c.shift(1); atr = np.maximum(dd.h - dd.l, np.maximum((dd.h - pcl).abs(), (dd.l - pcl).abs())).ewm(alpha=1 / 14, adjust=False).mean()
    S = pd.DataFrame(index=dd.index); S['k_ret5'] = r(c, 5); S['k_ret20'] = r(c, 20); S['k_ret63'] = r(c, 63); S['k_ret126'] = r(c, 126); S['k_dist52h'] = c / c.rolling(252, min_periods=60).max() - 1
    S['k_c_sma50'] = c / c.rolling(50).mean() - 1; S['k_c_sma200'] = c / c.rolling(200).mean() - 1; S['k_sma50_200'] = (c.rolling(50).mean() > c.rolling(200).mean()).astype(float).where(c.rolling(200).mean().notna())
    S['k_ema21_55'] = (c.ewm(span=21, adjust=False).mean() > c.ewm(span=55, adjust=False).mean()).astype(float); S['k_atr_pct'] = atr / c; S['k_vol_surge'] = dd.v / dd.v.rolling(20).mean(); S['k_updays20'] = (c.pct_change() > 0).rolling(20).mean()
    sp = SPYs.reindex(dd.index, method='ffill'); S['k_rs63_spy'] = S.k_ret63 - r(sp, 63); S['k_rs20_spy'] = S.k_ret20 - r(sp, 20)
    sc = g.sec.iloc[0]; se = SE.get(sc)
    if se is not None:
        se = se.reindex(dd.index, method='ffill'); S['e_ret20'] = r(se, 20); S['e_ret63'] = r(se, 63); S['e_ret126'] = r(se, 126); S['e_above50'] = (se > se.rolling(50).mean()).astype(float); S['e_above200'] = (se > se.rolling(200).mean()).astype(float)
        S['e_sma50_200'] = (se.rolling(50).mean() > se.rolling(200).mean()).astype(float); S['e_rs63_spy'] = S.e_ret63 - r(sp, 63); S['k_rs63_sec'] = S.k_ret63 - S.e_ret63
        et = SECE[sc]; et = et if et in allx else None
        if sc != 'semi' and et in RK.columns: S['e_mom_rank'] = RK[et].reindex(dd.index, method='ffill')
    S = S.shift(1)                                                                                             # prior completed session
    idx = g.index; pos = S.index.searchsorted(g.d.values.astype('datetime64[ns]'), side='left') - 1
    for col in S.columns: cols.setdefault(col, pd.Series(np.nan, index=U.index)); cols[col].loc[idx] = np.where(pos >= 0, S[col].values[np.clip(pos, 0, None)], np.nan)   # S already shifted: row of the entry day itself
for k, v in cols.items(): U[k] = v
# sanity: S shifted by 1 and looked up at the entry day's own row = prior session
# entry features
U['x_h1'] = U.h1.astype(float); U['x_tod'] = U.entry.dt.hour * 60 + U.entry.dt.minute; U['x_dow'] = U.entry.dt.dayofweek; U['x_month'] = U.entry.dt.month; U['x_ratio'] = U.ratio
U['x_ratio_rank'] = U.groupby('d').ratio.rank(pct=True); U['x_flips_day'] = U.groupby('d').tk.transform('size'); U['x_dnext'] = U.dnext.clip(upper=90); U['x_dlast'] = U.dlast.clip(upper=90); U['x_rv'] = U.rv; U['x_rvrel'] = U.rvrel
for c0 in ('gap', 'runup', 'z20', 'prior', 'margin', 'crowd', 'whip', 'tkrec', 'qqq'):
    if c0 in U and pd.api.types.is_numeric_dtype(U[c0]): U['x_' + c0] = U[c0].astype(float)
FE = [c for c in U.columns if c[:2] in ('m_', 'e_', 'k_', 'x_')]
PERS = {'19-21': U.entry.dt.year <= 2021, '22-26': U.entry.dt.year >= 2022}; U['mo'] = U.entry.dt.strftime('%Y-%m')
def reg(y, x, cl):
    ok = ~(np.isnan(x) | np.isnan(y)); x, y, cl = x[ok], y[ok], cl[ok]
    if len(x) < 300 or np.nanstd(x) == 0: return np.nan, np.nan, len(x)
    xc = x - x.mean(); b = (xc * (y - y.mean())).sum() / (xc ** 2).sum(); e = (y - y.mean()) - b * xc; s = pd.Series(xc * e).groupby(cl).sum()
    se = np.sqrt((s ** 2).sum()) / (xc ** 2).sum(); return b, b / se, len(x)
rows = []
for f in FE:
    row = dict(f=f); rk = U[f].rank(pct=True) - .5 if U[f].nunique() > 2 else U[f].astype(float) - U[f].astype(float).mean()
    rng = 1.0 if U[f].nunique() > 2 else 1.0
    for p, m in PERS.items():
        for tgt, nm in (('ret', 'r'), ('big', 'b')):
            b, t, n = reg(U.loc[m, tgt].values * 100, rk[m].values.astype(float), U.loc[m, 'mo'].values); row[f'{nm}_b_{p}'] = b; row[f'{nm}_t_{p}'] = t; row['n_' + p] = n
    rows.append(row)
R = pd.DataFrame(rows); R['same'] = np.sign(R['r_b_19-21']) == np.sign(R['r_b_22-26']); R['mint'] = np.where(R.same, np.minimum(R['r_t_19-21'].abs(), R['r_t_22-26'].abs()), 0)
R['info'] = R.same & (R['r_t_19-21'].abs() >= 2) & (R['r_t_22-26'].abs() >= 2); R['sameb'] = np.sign(R['b_b_19-21']) == np.sign(R['b_b_22-26']); R['infob'] = R.sameb & (R['b_t_19-21'].abs() >= 2) & (R['b_t_22-26'].abs() >= 2)
R = R.sort_values('mint', ascending=False); R.to_csv('/tmp/claude-0/ml/info_screen.csv', index=False)
print(f'{len(FE)} features tested. Effect = avg trade (pts) of top vs bottom of the feature range; runner = pts of runner probability.\n')
print(f'{"feature":<16}{"avg trade: 19-21 beta (t)":>28}{"22-26 beta (t)":>20}{"runner: 19-21":>16}{"22-26":>12}  verdict')
for _, q in R.head(30).iterrows():
    print(f'{q.f:<16}{q["r_b_19-21"]:>+14.2f} ({q["r_t_19-21"]:+.1f}){q["r_b_22-26"]:>+12.2f} ({q["r_t_22-26"]:+.1f}){q["b_b_19-21"]:>+10.2f}({q["b_t_19-21"]:+.1f}){q["b_b_22-26"]:>+8.2f}({q["b_t_22-26"]:+.1f})  {"INFORMATIVE" if q["info"] else ("runner-informative" if q.infob else "")}', flush=True)
print('\ninformative (return):', list(R[R['info']].f), '\ninformative (runner):', list(R[R.infob].f), '\nexpected by chance among', len(FE), ': ~', round(len(FE) * 0.0012, 2))
# stage 2: score from 19-21-informative features (>=1.8 t in 2019-21), sign fixed, applied to 2022-26
tr = R[(R['r_t_19-21'].abs() >= 2)]; sc = np.zeros(len(U)); nuse = 0
for _, q in tr.iterrows():
    z = U[q.f].rank(pct=True).fillna(.5) - .5 if U[q.f].nunique() > 2 else U[q.f].astype(float).fillna(U[q.f].mean()) - U[q.f].mean(); sc += np.sign(q['r_b_19-21']) * z.values; nuse += 1
U['score'] = sc; print(f'\nSTAGE 2: score from {nuse} features significant (|t|>=2) in 2019-21 only')
for p, m in PERS.items():
    z = U[m]; q = pd.qcut(z.score.rank(method='first'), 3, labels=['low', 'mid', 'high'])
    print(' ', p, {k: f'{g.ret.mean()*100:+.2f}% win {g.ret.gt(0).mean():.0%} runners {g.big.mean()*100:.1f}% n{len(g)}' for k, g in z.groupby(q)})
