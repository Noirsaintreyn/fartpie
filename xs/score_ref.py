"""Oct 26 2026: reference file for the live signal score (live_data/score_ref.json). Components picked on 2019-21 admitted trades only
(xs/edge_combo_oos.py), minus the options-volume vector (no live data for every name): asset managers' S&P COT index (-), RSP/SPY
50-day trend (+), SPY over its 200-day (-), trading days since the last earnings (+), daily vector uptrend age (-), SPY 20-day trend (-).
Each live value -> percentile vs the 2019-21 admitted trades (sign applied); score = mean; thirds cut on 2019-21. Prints the
out-of-sample check (2022-26 admitted trades by score third)."""
import json, numpy as np, pandas as pd
exec(open('xs/edge_combo_oos.py').read().split("print('picked on 2019-21 only")[0])
sel = {'es_am': -1, 'breadth': 1, 'spy200': -1, 'dse': 1, 'dage': -1, 'mkt20': -1}
P_ = {c: pct(c, s) for c, s in sel.items()}; L['score'] = sum(P_.values()) / len(P_)
R0 = L[L.inbook & (L.half == '19-21')]; t1, t2 = R0.score.quantile([1 / 3, 2 / 3])
q = np.where(L.score > t2, 'high', np.where(L.score < t1, 'low', 'mid'))
g = L[L.inbook].groupby([q[L.inbook.values], 'half']).ret.agg(['mean', 'size']).unstack(); g[[('mean', '19-21'), ('mean', '22-26')]] *= 100; print(g.round(3).to_string())
ref = {c: [float(x) for x in np.sort(REF[c].dropna().values)] for c in sel}
out = dict(signs=sel, ref=ref, cuts=[float(t1), float(t2)], stats={k: {h: round(float(g.loc[k, ('mean', h)]), 3) for h in ('19-21', '22-26')} for k in ('low', 'mid', 'high')},
           note='percentile of each input vs 2019-21 admitted trades, sign-adjusted, averaged; thirds cut on 2019-21 (xs/score_ref.py)')
json.dump(out, open('live_data/score_ref.json', 'w')); print('written', {c: len(v) for c, v in ref.items()}, 'cuts', round(t1, 3), round(t2, 3))
