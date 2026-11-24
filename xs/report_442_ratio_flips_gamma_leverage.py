"""Nov 24 2026 report 442. Rey: for the perps prop, take ALL ratio-gated flips and use negative gamma as the LEVERAGE switch (not a filter).
DECLARED BEFORE RESULTS. Flips: 15m bull vector flips, G20 options-ratio gate (20d median >= 2), 25 names with real OI gex, entry next open, exit vector flip (X1), 4bp, 2018-07..2026-09. Long only (shorts failed in 440).
Sizing arms (per-trade weight w): F flat 1x | L2 1x pos, 2x neg | L3 1x pos, 3x neg | H 0.5x pos, 1.5x neg | N neg only 1x (the filter, for reference).
Mechanism: dealer-short-gamma days amplify bull follow-through (report 426 G20&NEG +0.09pp vs +0.01pp), so risk should scale with it.
Book: 6 slots, slot size = w/6 of equity at entry (leverage = w), first come first served, REALIZED-P&L basis (open drawdown not marked => max DD is a floor).
PASS: L2 beats F on total return AND on return/maxDD in BOTH train (2019-21) and test (2018-19+2022+), and by era not worse in more than one era. Report by era: 18-19 | 20-21 | 22 | 23-26."""
import re
src = open('xs/report_440_neggamma_short_and_2025.py').read(); src = src[:src.index("L = pd.DataFrame(L)")].replace("pd.Timestamp('2025-12-30')", "pd.Timestamp('2026-09-30')")
exec(src); import numpy as np, pandas as pd
L = pd.DataFrame(L); L['exit_t'] = L.entry + pd.to_timedelta(L.hold, unit='D'); L['era'] = pd.cut(L.entry.dt.year, [2017, 2019, 2021, 2022, 2100], labels=['18-19', '20-21', '22', '23-26'])
print('flips', len(L), 'NEG share', L.neg.mean().round(3), 'names', L.tk.nunique())
for lab, g in (('NEG', L[L.neg]), ('POS', L[~L.neg])): print(f"  {lab}: n {len(g)} win {(g.ret>0).mean()*100:.1f} mean {g.ret.mean()*100:.2f}% | " + ' '.join(f"{e} {g[g.era==e].ret.mean()*100:+.2f}(n{(g.era==e).sum()})" for e in ('18-19', '20-21', '22', '23-26')))
ARMS = {'F flat 1x': lambda n: 1.0, 'L2 1x/2x': lambda n: 2.0 if n else 1.0, 'L3 1x/3x': lambda n: 3.0 if n else 1.0, 'H .5x/1.5x': lambda n: 1.5 if n else .5, 'N neg only': lambda n: 1.0 if n else 0.0}
def run(w, D, slots=6):
    D = D.sort_values('entry'); open_, out = [], []
    for r in D.itertuples():
        wt = w(r.neg)
        if wt == 0: continue
        open_ = [x for x in open_ if x > r.entry]
        if len(open_) >= slots: continue
        open_.append(r.exit_t); out.append((r.entry, r.exit_t, r.ret * wt / slots, r.era))
    X = pd.DataFrame(out, columns=['entry', 'exit_t', 'pnl', 'era']).sort_values('exit_t'); s = X.groupby(X.exit_t.dt.normalize()).pnl.apply(lambda p: (1 + p).prod() - 1)
    idx = pd.bdate_range(X.entry.min().normalize(), X.exit_t.max().normalize()); r = s.reindex(idx).fillna(0); eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min(); yrs = (idx[-1] - idx[0]).days / 365.25
    return dict(n=len(X), tot=eq.iloc[-1] - 1, cagr=eq.iloc[-1] ** (1 / yrs) - 1, mdd=dd, calmar=(eq.iloc[-1] ** (1 / yrs) - 1) / abs(dd), sh=r.mean() / r.std() * np.sqrt(252), X=X, r=r)
for lab, D in (('ALL 2018-07..2026-09', L), ('TRAIN 2019-21', L[(L.entry >= '2019-01-01') & (L.entry < '2022-01-01')]), ('TEST 2018+2022+', L[(L.entry < '2019-01-01') | (L.entry >= '2022-01-01')])):
    print(f'\n=== {lab}')
    for a, w in ARMS.items():
        o = run(w, D); print(f"  {a:12s} n {o['n']:4d} total {o['tot']*100:+6.1f}% CAGR {o['cagr']*100:5.1f}% maxDD {o['mdd']*100:6.1f}% Calmar {o['calmar']:5.2f} Sharpe {o['sh']:4.2f}")
print('\n=== by era, total return of the arm on that era only (fresh book)')
for e in ('18-19', '20-21', '22', '23-26'):
    D = L[L.era == e]; print('  ' + e + ' ' + ' | '.join(f"{a.split()[0]} {run(w, D)['tot']*100:+.1f}%/{run(w, D)['mdd']*100:.1f}" for a, w in ARMS.items()))
print('\n=== slots filled (avg open positions) and trades/yr'); X = run(ARMS['F flat 1x'], L)['X']; print('  F trades/yr', round(len(X) / 8.2, 1), 'of', len(L), 'flips; skipped (slots full)', len(L) - len(X))
