"""Sector rotation as a priority (Oct 25 2026): a sector is 'in rotation' when its ETF's 20-day return beats SPY by the most
(rank among the 11 sector ETFs, known at the prior close). Live book otherwise (6 slots, 8% breaker, live rules).
 tie-break: flips in the top-2 / top-3 rotation sectors go first (then the options ratio)
 let them in: names in a top-2 rotation sector may trade with an options ratio of 0.5-1.5 (OXY was 0.96)
 both"""
import json, numpy as np, pandas as pd, glob
exec(open('xs/batch_oct25.py').read().split("PART = os.getenv('PART', 'ALL')")[0])
SEC = json.load(open('sp500_sectors.json'))
EM = {'Information Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV', 'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP', 'Industrials': 'XLI',
      'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB', 'Energy': 'XLE', 'Communication Services': 'XLC'}
SECF = {'GOOG': 'XLC', 'GOOGL': 'XLC', 'META': 'XLC', 'NFLX': 'XLC', 'TSLA': 'XLY', 'AMZN': 'XLY', 'COIN': 'XLF', 'HOOD': 'XLF', 'PLTR': 'XLK', 'CRWD': 'XLK', 'APP': 'XLK', 'CVNA': 'XLY'}
def etf(e):
    f = f'etf_daily/{e}_yf.parquet' if os.path.exists(f'etf_daily/{e}_yf.parquet') else f'etf_daily/{e}.parquet'
    x = pd.read_parquet(f)
    if 'datetime' in x: x = x.set_index(pd.to_datetime(x.datetime).dt.tz_localize(None))
    x.index = pd.to_datetime(x.index).normalize(); c = x['adjclose'] if 'adjclose' in x else (x['Close'] if 'Close' in x else x['close'])
    return c[~c.index.duplicated()].sort_index()
PX = pd.DataFrame({e: etf(e) for e in set(EM.values()) | {'SPY'}}).sort_index().ffill()
R20 = (PX / PX.shift(20) - 1).sub(PX.SPY / PX.SPY.shift(20) - 1, axis=0).drop(columns='SPY')
RK = R20.rank(axis=1, ascending=False).shift(1)                                   # 1 = strongest sector, known at the prior close
def rank_of(tk, d):
    e = SECF.get(tk) or EM.get(SEC.get(tk, ''))
    if not e or e not in RK: return np.nan
    v = RK[e].asof(d); return v
print('energy (XLE) rotation rank now:', RK['XLE'].dropna().iloc[-1], '| 20d vs SPY', f"{R20['XLE'].dropna().iloc[-1]:+.1%}")
T['rk'] = [rank_of(t, e.normalize()) for t, e in zip(T.tk, T.entry)]
head('ROTATION as a priority')
for k in (2, 3):
    Z = T.copy(); Z['ratio'] = Z.ratio + np.where(Z.rk <= k, 100, 0); show(f'tie-break: top-{k} rotation sectors first', Z)
# names below the ratio cut, from the all-names trade files (live rules) with their own ratio history (xs/ext/slab/att_*)
ATT = {f.split('att_')[1][:-8]: pd.read_parquet(f) for f in glob.glob('xs/ext/slab/att_*.parquet')}
M = json.load(open('sp500_pit_membership.json')); mem = {k: {'META' if t == 'FB' else t for t in v} for k, v in M.items()}; months = sorted(mem)
def member(tk, d):
    k = f'{d:%Y-%m}'; k = k if k in mem else (months[-1] if k > months[-1] else months[0]); return tk in mem[k]
X = []
for tk, a in ATT.items():
    f = f'xs/ext/ahx_clean/{tk}.parquet'
    if not os.path.exists(f): continue
    x = pd.read_parquet(f); x = x[x['var'] == 'base+sticky'].copy(); x['entry'] = pd.to_datetime(x.entry); x['exit'] = pd.to_datetime(x.exit)
    x = x[(x.entry >= '2019-01-01') & ((x.entry.dt.hour * 60 + x.entry.dt.minute) < 840)]
    a = a[~a.index.duplicated()].sort_index(); x['ratio'] = a.ratio.reindex(x.entry.dt.normalize().values, method='ffill').values
    X.append(x)
X = pd.concat(X); X = X[(X.ratio >= .5) & (X.ratio < 1.5)]; X['sp'] = [member(t, e) for t, e in zip(X.tk, X.entry)]; X = X[X.sp]
X['rk'] = [rank_of(t, e.normalize()) for t, e in zip(X.tk, X.entry)]
for k in (1, 2):
    add = X[X.rk <= k]; Z = pd.concat([T, add], ignore_index=True)
    show(f'let in ratio 0.5-1.5 names in the top-{k} rotation sector ({len(add)} trades, avg {add.ret.mean():+.2%})', Z)
    Z2 = Z.copy(); Z2['ratio'] = Z2.ratio + np.where(Z2.rk <= k, 100, 0); show(f'  + those sectors first on ties', Z2)
