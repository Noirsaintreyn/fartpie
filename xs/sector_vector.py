"""Oct 25 2026: does the sector's vector matter? Daily vector (same state machine) on the stock's SPDR sector ETF, known at the prior close.
Also SPY's and QQQ's daily vector. Full-S&P screener, ratio >= 1.5, live rules, 6 slots, 8% breaker."""
from collections import deque
import json, numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('\nP = [(np.datetime64')[0])
sys.path.insert(0, '.')
try: from vector_pine import run_state_machine
except Exception: from vector_core import run_state_machine
T = T.sort_values('entry').reset_index(drop=True)
EM = {'Information Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV', 'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP',
      'Industrials': 'XLI', 'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB', 'Energy': 'XLE', 'Communication Services': 'XLC'}
SEC = json.load(open('sp500_sectors.json')); EX = {'GOOG': 'XLC', 'GOOGL': 'XLC', 'META': 'XLC', 'NFLX': 'XLC', 'TSLA': 'XLY', 'AMZN': 'XLY', 'CVNA': 'XLY', 'BRKB': 'XLF',
      'COIN': 'XLF', 'HOOD': 'XLF', 'PLTR': 'XLK', 'CRWD': 'XLK', 'APP': 'XLK'}
def dvec(t):
    x = pd.read_parquet(f'etf_daily/{t}.parquet'); x['d'] = pd.to_datetime(x.datetime).dt.tz_localize(None).dt.normalize(); x = x.drop_duplicates('d').sort_values('d').reset_index(drop=True)
    reg, *_ = run_state_machine(x[['open', 'high', 'low', 'close']].assign(datetime=x.d))
    return pd.Series(np.asarray(reg), index=x.d).shift(1)                                     # known at the prior close
V = {t: dvec(t) for t in set(EM.values()) | {'SPY', 'QQQ'}}
d = T.entry.dt.normalize()
T['etf'] = [EX.get(t) or EM.get(SEC.get(t, ''), None) for t in T.tk]
T['secv'] = [V[e].asof(x) if e else np.nan for e, x in zip(T.etf, d)]
T['spyv'] = [V['SPY'].asof(x) for x in d]; T['qqqv'] = [V['QQQ'].asof(x) for x in d]
T['per'] = np.where(T.entry.dt.year >= 2022, '22-26', '19-21')
print('share of flips with the sector ETF daily vector bull:', f'{(T.secv == 1).mean():.0%}')
g = T.groupby(['secv', 'per']).ret.agg(['mean', 'count']); g['mean'] *= 100; print('avg trade % by sector-ETF daily vector at entry:'); print(g.unstack().round(3).to_string())
g = T.groupby(['spyv', 'per']).ret.agg(['mean', 'count']); g['mean'] *= 100; print('by SPY daily vector:'); print(g.unstack().round(3).to_string())
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z, **kw): print(f'  {nm:<50}' + ''.join(f'{c:>+9.1%} /{dd:>5.0%}' for c, dd in (sim(Z, a, b, **kw) for a, b in P)), flush=True)
print(f'\n{"sector / market vector":<52}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live', T)
show('skip if the sector ETF daily vector is bear', T[T.secv != -1])
show('skip if sector bear AND SPY bear', T[~((T.secv == -1) & (T.spyv == -1))])
show('skip if SPY daily vector is bear', T[T.spyv != -1])
T['k1'] = (T.secv == 1) * 10 + T.ratio / 100; show('tie-break: sector-bull flips first', T, key='k1')
T['k2'] = (T.secv == -1) * 10 + T.ratio / 100; show('tie-break: sector-BEAR flips first (contrarian)', T, key='k2')
