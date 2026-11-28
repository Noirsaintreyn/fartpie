"""Report 510 (Nov 27 2026): UNIVERSE screens for sub-$5 flips - stock-level reasons known before the flip (Moyer ch.2: inefficiency needs unequal information, forced sellers, high transaction costs). Declared BEFORE results.
 COVER (information gap): no analyst EPS estimate on the latest earnings row before the flip = NEGLECTED. Prediction: NEGLECTED = higher reach and higher r (slow information diffusion, Hong-Stein). Two-sided read.
 TAX (forced sellers): LOSER = close <= 40% of its 252d high at the signal. Prediction: LOSER flips in DECEMBER (tax-loss selling) worse, in JANUARY better (rebound). Tested as LOSER x month.
 COST (barrier): Corwin-Schultz high-low spread (20d mean) at the signal. Prediction: reach unrelated, but net r (r minus spread) falls monotonically; how much of the mean survives in each spread tercile.
Unit: flip_history.csv (sub-$5 daily flips), controls ATRP tercile, month-clustered SE, train 2019-21 / test 2022+."""
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
W = pd.read_pickle(SP + 'ext/wide_daily_ohlcv.pkl'); H = pd.read_csv('flip_history.csv', parse_dates=['d']); H['d'] = H.d.astype('datetime64[ns]')
print('flip tickers with daily bars', H.tk.isin(W).mean().round(3))
def cs_spread(b):
    h, l = np.log(b.high.values), np.log(b.low.values); bet = (h - l) ** 2; bet = bet + np.r_[np.nan, bet[:-1]]
    gam = (np.maximum(h, np.r_[np.nan, h[:-1]]) - np.minimum(l, np.r_[np.nan, l[:-1]])) ** 2
    al = (np.sqrt(2 * bet) - np.sqrt(bet)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gam / (3 - 2 * np.sqrt(2)))
    s = 2 * (np.exp(al) - 1) / (1 + np.exp(al)); return pd.Series(np.clip(s, 0, None), index=b.index).rolling(20, min_periods=10).mean()
cost, loser = np.full(len(H), np.nan), np.full(len(H), np.nan)
for tk, g in H.groupby('tk'):
    b = W.get(tk)
    if b is None or len(b) < 300: continue
    b = b[~b.index.duplicated()].sort_index(); sp = cs_spread(b); hi = b.high.rolling(252, min_periods=120).max()
    for i, d in zip(g.index, g.d):
        j = b.index.searchsorted(d, side='right') - 1
        if j >= 0: cost[i] = sp.iloc[j]; loser[i] = b.close.iloc[j] / hi.iloc[j] if hi.iloc[j] > 0 else np.nan
H['CS'] = cost; H['LOSER'] = loser <= 0.4
def neglected(tk, d):
    f = f'data/fund/yf_small/{tk}_earn.parquet'
    if not os.path.exists(f): return np.nan
    e = pd.read_parquet(f); e.index = pd.to_datetime(e.index).tz_localize(None); e = e[e.index < d - pd.Timedelta(days=1)].sort_index()
    return float(e['EPS Estimate'].iloc[-1] != e['EPS Estimate'].iloc[-1]) if len(e) else np.nan
H['NEG'] = [neglected(t, d) for t, d in zip(H.tk, H.d)]
H['mon'] = H.d.dt.strftime('%Y-%m'); H['tr'] = H.d < '2022-01-01'; H['era'] = np.where(H.tr, 'train 19-21', 'test 22-26'); H['M'] = H.d.dt.month
q = H[H.tr].ATRP.quantile([1 / 3, 2 / 3]).values; H['a3'] = np.digitize(H.ATRP, q); lo, hi = H.r.quantile([.01, .99]); H['rw'] = H.r.clip(lo, hi)
print('covered CS', H.CS.notna().mean().round(2), ' NEG known', H.NEG.notna().mean().round(2), ' NEG share', H.NEG.mean().round(2), ' LOSER share', H.LOSER.mean().round(2))
def eff(D, flag, y):
    D = D.dropna(subset=[flag])
    if D[flag].sum() < 8 or (D[flag] == 0).sum() < 8: return None
    m = smf.ols(f'{y} ~ {flag}.astype(float) + C(a3)', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mon)[0]}); k = [p for p in m.params.index if p.startswith(flag)][0]; return m.params[k] * 100, m.tvalues[k]
def show(flag, D0, label):
    print(f'== {label}')
    for era, D in list(D0.groupby('era')) + [('pooled', D0)]:
        a = D[D[flag] == 1]; b = D[D[flag] == 0]; r1 = eff(D, flag, 'hit'); r2 = eff(D, flag, 'rw')
        print(f'  {era:12} n {len(a):4}/{len(b):4} | REACH {100 * a.hit.mean():.0f}% vs {100 * b.hit.mean():.0f}% adj {r1[0]:+.1f}pp (t {r1[1]:+.2f}) | mean r {100 * a.rw.mean():+.1f}% vs {100 * b.rw.mean():+.1f}% adj {r2[0]:+.1f}pp (t {r2[1]:+.2f})' if r1 else f'  {era:12} n {len(a)} too few')
show('NEG', H, 'COVER: NEGLECTED (no analyst EPS estimate) vs covered')
H['LDEC'] = (H.LOSER & (H.M == 12)).astype(float); H['LJAN'] = (H.LOSER & (H.M == 1)).astype(float)
for fl, lab, base in (('LDEC', 'TAX: LOSER x December vs other-month flips', (H.M != 1)), ('LJAN', 'TAX: LOSER x January vs other-month flips', (H.M != 12))):
    show(fl, H[base], lab)
print('\n== COST: Corwin-Schultz spread terciles (train cuts); gross vs net of spread (full round trip = 1 spread)')
Hc = H.dropna(subset=['CS']).copy(); cq = Hc[Hc.tr].CS.quantile([1 / 3, 2 / 3]).values; Hc['c3'] = np.digitize(Hc.CS, cq); Hc['net'] = Hc.rw - Hc.CS
print(Hc.groupby(['era', 'c3']).agg(n=('hit', 'size'), spread=('CS', lambda x: round(100 * x.mean(), 2)), reach=('hit', lambda x: round(100 * x.mean())), gross=('rw', lambda x: round(100 * x.mean(), 2)), net=('net', lambda x: round(100 * x.mean(), 2)), med_net=('net', lambda x: round(100 * x.median(), 2))).to_string())
