"""Oct 25 2026: cap correlated bets. Max open positions per group (semiconductors, or per GICS sector), full-S&P screener, live rules.
A flip whose group is already at the cap is skipped (the slot stays free for the next flip)."""
from collections import deque
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('\nP = [(np.datetime64')[0])
T = T.sort_values('entry').reset_index(drop=True)
SEMI = {'NVDA', 'AMD', 'AVGO', 'MU', 'INTC', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'ADI', 'MCHP', 'NXPI', 'ON', 'MRVL', 'TSM', 'SMCI', 'MPWR', 'TER', 'SWKS', 'QRVO', 'ARM', 'ASML'}
SEC = json.load(open('sp500_sectors.json')); EXTRA = {'GOOG': 'Communication Services', 'META': 'Communication Services', 'NFLX': 'Communication Services', 'TSLA': 'Consumer Discretionary',
    'AMZN': 'Consumer Discretionary', 'BRKB': 'Financials', 'COIN': 'Financials', 'HOOD': 'Financials', 'PLTR': 'Information Technology', 'CRWD': 'Information Technology', 'APP': 'Information Technology', 'CVNA': 'Consumer Discretionary'}
T['sector'] = [EXTRA.get(t) or SEC.get(t, 'Other') for t in T.tk]; T['semi'] = T.tk.isin(SEMI)
print('share of trades in semis:', f'{T.semi.mean():.0%}', ' by sector:', T.sector.value_counts(normalize=True).round(2).head(5).to_dict())
def simcap(Z, a, b, group=None, cap=99, n=6, brk=.08, same_day=False):
    Z = Z[(Z.entry >= a) & (Z.entry < b)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; w = 1 / n; lastday = {}
    G = Z.semi.map({True: 'semi', False: None}).values if group == 'semi' else (Z.sector.values if group == 'sector' else [None] * len(Z))
    for e, x, r, g, tk in zip(Z.entry.values, Z.exit.values, Z.ret.values, G, Z.tk.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr * w); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o[0] > e]
        if len(op) >= n: continue
        if g is not None and sum(1 for o in op if o[1] == g) >= cap: continue
        if same_day and g is not None and lastday.get(g) == pd.Timestamp(e).normalize(): continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append((x, g)); pend.append((x, r)); lastday[g] = pd.Timestamp(e).normalize()
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr * w); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); yy = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / yy) - 1, (eq / eq.cummax() - 1).min()
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, **kw): print(f'  {nm:<46}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (simcap(T, a, b, **kw) for a, b in P)), flush=True)
print(f'\n{"correlation caps":<48}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live (no cap)')
show('max 1 semiconductor at a time', group='semi', cap=1)
show('max 2 semiconductors at a time', group='semi', cap=2)
show('semis: no 2nd semi entry the same day', group='semi', cap=99, same_day=True)
for c in (1, 2, 3): show(f'max {c} per GICS sector', group='sector', cap=c)
