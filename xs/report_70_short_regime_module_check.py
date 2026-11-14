"""Check that live/short_regime.py reproduces the research: feed it the research data (wide daily closes + FINRA history), flag the report_64 flips with module lead_series/dtc_frame, and compare SQ10&LEAD means with report_69 (2019-21 +12.98% n915, 2025-26 +11.34% n215, 2022-24 -1.62% n709)."""
import sys, numpy as np, pandas as pd; sys.path.insert(0, '/home/claude/fartpie'); import short_regime as SR
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); cl = pd.DataFrame({k: B.close[~B.index.duplicated()] for k, B in D.items()}).sort_index(); cl = cl[cl.index >= '2018-09-01']
si = pd.read_pickle(X + 'shortint.pkl'); lead, dtc, nb = SR.lead_series(cl, si)
P = pd.read_pickle(X + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003; P['d0'] = pd.to_datetime(P.d0); P['DTC'] = [dtc.at[d, tk] if (d in dtc.index and tk in dtc.columns) else np.nan for d, tk in zip(P.d0, P.tk)]; P['LEAD'] = lead.reindex(P.d0).values > 0
for en, a, b in (('2019-21', '2019', '2022'), ('2022-24', '2022', '2025'), ('2025-26Q1', '2025', '2026-04')):
    S = P[(P.t >= a) & (P.t < b) & P.DTC.notna()]; m = (S.DTC >= 10) & S.LEAD; print(f'{en}: module SQ10&LEAD n {int(m.sum())} mean {S[m].rn.mean()*100:+.2f}% | not-lead n {int(((S.DTC>=10)&~S.LEAD).sum())} mean {S[(S.DTC>=10)&~S.LEAD].rn.mean()*100:+.2f}%')
print('latest lead', float(lead.dropna().iloc[-1]), 'basket size', int(nb.iloc[-1]), 'regime on' , bool(lead.dropna().iloc[-1] > 0), lead.dropna().index[-1].date())
