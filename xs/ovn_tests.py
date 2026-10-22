"""Oct 22 2026 (after TSLA: entry 360, peak 394, gapped to ~365 on earnings): overnight exits and take-profits.
 O1  overnight-flat when in profit: if the close is >= +X% over entry, sell at that close and buy back at the next open (slot kept)
 O2  take profit at the close: first close >= +X% -> sell there (slot freed)
 O3  take profit intraday: first touch of +X% -> sell everything at that price
 O4  earnings guard: if a trade is >= +X% at the close before an earnings gap day, sell at that close
Same trade list / bars / book as xs/exit_lab2.py (full S&P, ratio >= 1.5, live rules, 6 slots, 8% breaker)."""
from collections import deque
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('P = [(np.datetime64')[0])
exec('def sim3' + open('xs/equity_models.py').read().split('def sim3')[1].split('STRAT_MED =')[0]); STRAT_MED = 1
exec(open('xs/exit_lab2.py').read().split('BARS = {}')[1].split('VARS =')[0].replace('def bars', 'BARS = {}\ndef bars', 1))
T = T.sort_values('entry').reset_index(drop=True)
TD = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet('etf_daily/SPY.parquet').datetime).dt.tz_localize(None).dt.normalize().unique()))
GAPS = {}
for f in glob.glob('xs/ext/earn/*.csv'):
    tk = f.split('/')[-1][:-4]
    if tk.startswith('_'): continue
    try: e = pd.read_csv(f); t = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    except Exception: continue
    GAPS[tk] = set(np.datetime64(d if tt.hour < 12 else TD[min(TD.searchsorted(d, 'right'), len(TD) - 1)]) for d, tt in zip(t.dt.normalize(), t))
VARS = {}
for X in (.03, .05, .08): VARS[f'O1 flat overnight when >= +{X:.0%}'] = ('o1', X)
for X in (.05, .08, .10, .15): VARS[f'O2 take profit at close >= +{X:.0%}'] = ('o2', X)
for X in (.08, .10, .15, .20): VARS[f'O3 take profit intraday at +{X:.0%}'] = ('o3', X)
for X in (.03, .05, .08, .10): VARS[f'O4 sell before earnings if >= +{X:.0%}'] = ('o4', X)
for X in (.05, .08): VARS[f'O4b sell HALF before earnings if >= +{X:.0%}'] = ('o4b', X)
if os.getenv('ONLY_O4'): VARS = {k: v for k, v in VARS.items() if v[0].startswith('o4')}
ALT = {k: (T.ret.values.copy(), T.exit.values.copy()) for k in VARS}
for tk, g in T.groupby('tk'):
    B = bars(tk)
    if B is None: continue
    dt = B.dt.values; o, h, c = B.open.values, B.high.values, B.close.values; day = B.dt.dt.normalize().values; gp = GAPS.get(tk, set())
    for i, e, x in zip(g.index, g.entry.values, g.exit.values):
        a = np.searchsorted(dt, e); b = np.searchsorted(dt, x)
        if a >= len(dt) or dt[a] != e or b >= len(dt) or b <= a: continue
        p0 = o[a]; base = o[b] / p0 - 1
        ends = [j for j in range(a, b) if day[j + 1] != day[j]]                  # last bar of each day inside the trade
        for k, v in VARS.items():
            r_, x_ = ALT[k]; X = v[1]
            if v[0] == 'o1':
                adj = sum(np.log(c[j] / o[j + 1]) for j in ends if c[j] / p0 - 1 >= X)
                if adj: r_[i] = (1 + r_[i]) * np.exp(adj) - 1
            elif v[0] == 'o2':
                hit = [j for j in ends if c[j] / p0 - 1 >= X]
                if hit: j = hit[0]; r_[i] += c[j] / p0 - 1 - base; x_[i] = dt[j]
            elif v[0] == 'o3':
                hit = np.where(h[a:b] >= p0 * (1 + X))[0]
                if len(hit): j = a + hit[0]; r_[i] += X - base; x_[i] = dt[j]
            else:
                hit = [j for j in ends if np.datetime64(day[j + 1]) in gp and c[j] / p0 - 1 >= X]
                if hit:
                    j = hit[0]
                    if v[0] == 'o4': r_[i] += c[j] / p0 - 1 - base; x_[i] = dt[j]
                    else: r_[i] += 0.5 * (c[j] / p0 - 1 - base)             # half out at that close, the rest rides the rule
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<44}' + ''.join(f'{cc:>+9.1%} /{dd:>5.0%}' for cc, dd in (sim3(Z, a, b) for a, b in P)), flush=True)
print(f'{"overnight / take-profit (S&P, ratio >= 1.5)":<46}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live', T)
for k in VARS:
    Z = T.copy(); Z['ret'], Z['exit'] = ALT[k]; show(k, Z)
on = tot = 0.0                                                                    # share of trade P&L earned in overnight gaps
for tk, g in T.groupby('tk'):
    B = bars(tk)
    if B is None: continue
    dt = B.dt.values; o, c = B.open.values, B.close.values; day = B.dt.dt.normalize().values
    for e, x in zip(g.entry.values, g.exit.values):
        a = np.searchsorted(dt, e); b = np.searchsorted(dt, x)
        if a >= len(dt) or b >= len(dt) or b <= a: continue
        tot += np.log(o[b] / o[a]); on += sum(np.log(o[j + 1] / c[j]) for j in range(a, b) if day[j + 1] != day[j])
print(f'\nshare of trade P&L earned overnight: {on / tot:.0%}')
