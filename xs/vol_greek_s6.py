"""Nov 10 2026: VOL-MODEL AND GREEK TESTS ON THE CURRENT SYSTEM (S6, variant V2 = IV-jump skip + HAR relative sizing; room rule is applied later in the Lucid sim). DECLARED BEFORE RESULTS.
PART 1 - which vol measures add anything to HAR for tomorrow's RTH log range (target = log of the NQ RTH range, real NQ 5m). Fit 2014-19, judged 2020-Jan26.
  Measures (all known at the prior close): HAR (log range yesterday / week / month), YZ20 rank (QQQ daily), VIX level, VIX 5d change, QQQ ATM IV, GARCH(1,1) next-day variance on NQ close-to-close returns.
  Report: correlation matrix of the measures, R2 of HAR alone vs HAR + each, HAR + all.
PART 2 - Greeks (QQQ options at the prior close; ThetaData files): GEX pct, DEX pct, vanna sign, charm sign, vanna flow. For each engine's trades taken (sz > 0), points per trade by tercile / sign,
  2014-21 vs 2022-Jan26, t of the top-minus-bottom spread. A feature passes only if same sign and |t| >= 2 in both periods.
PART 3 - sizing at the system level: multipliers on the live size sz (clip 0.5-1.5): HAR already inside. Compare points / day (sz weighted), daily std, mean / std, worst month, both periods:
  S  base | S+VIX (median VIX / VIX) | S+IV (median ATM IV / ATM IV) | S+GARCH (median / forecast) | S+YZ (median YZ / YZ rank-scaled) | S+GEX mid-band x1.25 | S+DEX low x1.25."""
import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'; U = '/home/claude/fartpie'
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); Dn = pd.read_pickle(f'{SP}/nq_days.pkl'); Dn.index = pd.to_datetime(Dn.index).normalize()
R = (Dn.rng).astype(float); lr = np.log(R.replace(0, np.nan)); ix = Dn.index
X = pd.DataFrame({'d': lr.shift(1), 'w': np.log(R.rolling(5).mean().shift(1)), 'm': np.log(R.rolling(22).mean().shift(1))})
# --- other vol measures, shifted to be known at 09:30
import yfinance as yf
q = yf.download('QQQ', start='2013-06-01', interval='1d', progress=False, auto_adjust=False); q.columns = [c[0] if isinstance(c, tuple) else c for c in q.columns]; q.index = q.index.normalize()
o_, h_, l_, c_ = q.Open, q.High, q.Low, q.Close; n_ = 20; kz = .34 / (1.34 + (n_ + 1) / (n_ - 1))
yz = np.log(o_ / c_.shift(1)).rolling(n_).var() + kz * np.log(c_ / o_).rolling(n_).var() + (1 - kz) * (np.log(h_ / c_) * np.log(h_ / o_) + np.log(l_ / c_) * np.log(l_ / o_)).rolling(n_).mean()
YZr = yz.rolling(252, min_periods=120).rank(pct=True)
vx = pd.read_parquet(f'{U}/data/yahoo/daily_v/VIX.parquet'); vx = vx['Close'] if 'Close' in vx else vx.iloc[:, 0]; vx.index = pd.to_datetime(vx.index).normalize()
G1 = pd.read_csv(f'{U}/options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); G2 = pd.read_csv(f'{U}/options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
OWN = pd.read_csv(f'{U}/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index()
for f_ in (G1, G2): f_.index = f_.index.normalize()
OWN.index = OWN.index.normalize()
from arch import arch_model
ret = 100 * np.log(Dn.DC.astype(float)).diff().dropna(); fitr = ret[ret.index < '2020-01-01']
am = arch_model(fitr, vol='GARCH', p=1, q=1, mean='Constant', dist='normal').fit(disp='off'); pr = am.params
sg = np.empty(len(ret)); sg[0] = fitr.var(); rv_ = ret.values
for k in range(1, len(ret)): sg[k] = pr['omega'] + pr['alpha[1]'] * (rv_[k - 1] - pr['mu']) ** 2 + pr['beta[1]'] * sg[k - 1]
GAR = pd.Series(np.log(sg), index=ret.index)
def at(s, sh=1): return s.reindex(ix, method='ffill').shift(0) if False else s.shift(sh).reindex(ix, method='ffill')
M = pd.DataFrame({'HAR_d': X.d, 'HAR_w': X.w, 'HAR_m': X.m})
M['YZr'] = at(YZr, 0).values  # YZ rank at prior close: yz through t-1 -> use shift 1 below
M['YZr'] = YZr.shift(1).reindex(ix, method='ffill').values
M['VIX'] = np.log(vx.shift(1).reindex(ix, method='ffill')).values; M['VIXd5'] = np.log(vx).diff(5).shift(1).reindex(ix, method='ffill').values
M['IV'] = np.log(G1.atm_iv).shift(1).reindex(ix, method='ffill').values; M['GARCH'] = GAR.reindex(ix).values
M['y'] = lr.values
OK = M.notna().all(axis=1); fit = OK & (ix >= '2014-02-01') & (ix < '2020-01-01'); te = OK & (ix >= '2020-01-01')
HARc = ['HAR_d', 'HAR_w', 'HAR_m']
print('=== PART 1: vol measures, tomorrow RTH log range. correlation (all days) of the measures with HAR forecast ===', flush=True)
hf = LinearRegression().fit(M.loc[fit, HARc], M.loc[fit, 'y']); M['HARf'] = np.nan; M.loc[OK, 'HARf'] = hf.predict(M.loc[OK, HARc])
print(M.loc[OK, ['HARf', 'YZr', 'VIX', 'IV', 'GARCH', 'VIXd5']].corr().round(2).to_string())
def r2(cols):
    m = LinearRegression().fit(M.loc[fit, cols], M.loc[fit, 'y']); p = m.predict(M.loc[te, cols]); y = M.loc[te, 'y']
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f'\n  out-of-sample R2 2020-26: HAR {r2(HARc):.3f}')
for c in ('YZr', 'VIX', 'VIXd5', 'IV', 'GARCH'): print(f'    HAR + {c:<6} {r2(HARc + [c]):.3f}   | {c} alone {r2([c]):.3f}')
print(f'    HAR + all  {r2(HARc + ["YZr", "VIX", "VIXd5", "IV", "GARCH"]):.3f}   | HAR + VIX + IV {r2(HARc + ["VIX", "IV"]):.3f}', flush=True)
# --- trade tables
T = {}
for (t, u, y), n in zip(OUT['V2'], NAMES):
    t = t.copy(); t['day'] = pd.to_datetime(t.day).dt.normalize(); t = t[(t.day >= '2014-02-01') & (t.day <= '2026-01-30')]; T[n] = t
pct = lambda s: s.rolling(252, min_periods=120).rank(pct=True)
F = pd.DataFrame(index=G1.index); F['gex'] = OWN.reindex(F.index); F['dex'] = G1.call_dex + G1.put_dex; F['vanna'] = G2.net_vanna.reindex(F.index); F['charm'] = G2.net_charm.reindex(F.index)
F['iv5'] = G1.atm_iv.diff(5); F['vflow'] = np.sign(F.vanna) * -np.sign(F.iv5); F['gexp'] = pct(F.gex); F['dexp'] = pct(F.dex)
def pf(col, days): j = F.index.searchsorted(days) - 1; return pd.Series([F[col].iloc[i] if i >= 0 else np.nan for i in j], index=days.index if hasattr(days, 'index') else None)
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
print('\n=== PART 2: Greeks by engine. spread = top bucket minus bottom bucket pts/trade; (14-21 | 22-26), t; PASS needs same sign and |t|>=2 in both ===')
for n in NAMES:
    t = T[n][T[n].sz > 0].copy(); d = t.day.values
    for col in ('gexp', 'dexp', 'vanna', 'charm', 'vflow'):
        j = F.index.searchsorted(d) - 1; v = pd.Series([F[col].iloc[i] if i >= 0 else np.nan for i in j], index=t.index)
        if col in ('gexp', 'dexp'): lo, hi = v <= 1 / 3, v >= 2 / 3
        else: lo, hi = v < 0, v > 0
        res = []
        for lab, mm in (('14-21', t.day < '2022-01-01'), ('22-26', t.day >= '2022-01-01')):
            a, b = t.pts[mm & hi], t.pts[mm & lo]; res.append((a.mean() - b.mean(), tt(a, b), len(a), len(b)))
        ok = all(np.isfinite(r[1]) and abs(r[1]) >= 2 for r in res) and np.sign(res[0][0]) == np.sign(res[1][0])
        print(f'  {n:<6} {col:<6} 14-21 {res[0][0]:+6.1f} (t {res[0][1]:+5.2f}, n {res[0][2]}/{res[0][3]}) | 22-26 {res[1][0]:+6.1f} (t {res[1][1]:+5.2f}, n {res[1][2]}/{res[1][3]}){"   <== PASS" if ok else ""}')
print('\n=== PART 3: system sizing, weighted points / day (sz x multiplier) ===', flush=True)
DAYS = ix[(ix >= '2014-02-01') & (ix <= '2026-01-30')]
med = lambda s: s.rolling(252, min_periods=120).median()
vixl = np.exp(M.VIX); ivl = np.exp(M.IV); garl = np.exp(M.GARCH)
MUL = {'base': pd.Series(1.0, index=ix), 'S+VIX': (med(vixl) / vixl).clip(.5, 1.5), 'S+IV': (med(ivl) / ivl).clip(.5, 1.5), 'S+GARCH': (np.sqrt(med(garl) / garl)).clip(.5, 1.5),
       'S+YZ': (1.5 - M.YZr).clip(.5, 1.5)}
def run(name, mul):
    s = pd.Series(0.0, index=DAYS)
    for n in NAMES:
        t = T[n][T[n].sz > 0]; f = mul.reindex(t.day).fillna(1.0).values if n not in ('ON+', 'ON-') else mul.shift(0).reindex(t.day).fillna(1.0).values
        s = s.add((t.pts * t.sz * f).groupby(t.day.values).sum().reindex(DAYS).fillna(0.0), fill_value=0)
    return s
print('  variant       | pts/day all | 14-21 | 22-26 | daily std | mean/std | worst month | months >= 50')
for k, m in MUL.items():
    s = run(k, m); a = s.index < '2022-01-01'; mo = s.groupby(s.index.to_period('M')).sum()
    print(f'  {k:<13} | {s.mean():+7.2f} | {s[a].mean():+6.2f} | {s[~a].mean():+6.2f} | {s.std():6.2f} | {s.mean() / s.std():5.3f} | {mo.min():+7.0f} | {np.mean(mo >= 50):4.0%}', flush=True)
pickle.dump((M, F), open(f'{SP}/vol_greek.pkl', 'wb'))
