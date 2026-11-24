"""Nov 24 2026 report 422. Rey: 'if negative gamma works on higher names let's try other Greeks too'.
DECLARED: live gated 15m flips (B0) on names with volume-weighted chain Greeks (xs/build_greek_flow.py: gamma_net, vega_net, vanna_net, charm_net, skew, term, atm_iv, act).  Each feature -> own-name rolling-252d percentile (PIT, shifted 1d, min 126).  Tails: HI >= .75, LO <= .25 vs the REST of the same flips.  Outcome: vector exit (live), net 4bp.  16 comparisons -> Bonferroni |t| >= 3.0 (month-clustered tail-minus-rest mean diff).  PASS: |t| >= 3.0 pooled AND same sign of (tail - rest) mean AND median in train (2019-21) and test (2018+2022-26) AND in >= 3 of 4 eras (n>=30).  Else reported as a lead only if |t| >= 2 with same sign in both halves."""
import numpy as np, pandas as pd
_t = open('xs/report_402_five_ideas.py').read(); exec(_t[:_t.index("rows = []; TK0, s0")])
G = pd.read_parquet(SP + 'greeks_daily.parquet'); G['d'] = pd.to_datetime(G.d); FE = ['gamma_net', 'vega_net', 'vanna_net', 'charm_net', 'skew', 'term', 'atm_iv', 'act']
PR = {}
for tk, g in G.groupby('tk'):
    g = g.set_index('d').sort_index(); PR[tk] = pd.DataFrame({f: g[f].rolling(252, min_periods=126).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True).shift(1) for f in FE})
B = B0[B0.tk.isin(PR)].copy().reset_index(drop=True); B['entry'] = pd.to_datetime(B.entry); B['d0'] = B.entry.dt.normalize() - pd.Timedelta(days=1)
for f in FE: B[f] = [PR[tk][f].reindex([d0], method='ffill').iloc[0] for tk, d0 in zip(B.tk, B.d0)]
B = B[B[FE].notna().all(axis=1)].copy(); B['n'] = B.ret - 4e-4; e_ = B.entry; B['half'] = np.where((e_ >= '2019-01-01') & (e_ < '2022-01-01'), 'train', 'test'); B['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); B['mo'] = e_.dt.to_period('M')
print('flips', len(B), 'names', B.tk.nunique(), '| base win %.1f med %.2f mean %.2f' % ((B.n > 0).mean() * 100, B.n.median() * 100, B.n.mean() * 100)); print('feature tail | n | win% med% mean% | vs rest pp (t) | train diff mean/med | test diff mean/med | eras sign(mean diff) 18-19 20-21 22 23-26 | verdict')
res = []
for f in FE:
    for tn, mk in (('HI', B[f] >= .75), ('LO', B[f] <= .25)):
        on = B[mk]; off = B[~mk]; m = (on.groupby('mo').n.mean() - off.groupby('mo').n.mean()).dropna(); t = m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))
        dd = {h: ((on[on.half == h].n.mean() - off[off.half == h].n.mean()) * 100, (on[on.half == h].n.median() - off[off.half == h].n.median()) * 100) for h in ('train', 'test')}
        es = ''
        for era in ('18-19', '20-21', '22', '23-26'): x = on[on.era == era]; y = off[off.era == era]; es += (' +' if x.n.mean() > y.n.mean() else ' -') if len(x) >= 30 else ' ?'
        same = all(np.sign(dd[h][0]) == np.sign(dd[h][1]) == np.sign(dd['train'][0]) for h in ('train', 'test')); neras = max(es.count('+'), es.count('-'))
        v = 'PASS' if abs(t) >= 3 and same and neras >= 3 else ('lead' if abs(t) >= 2 and np.sign(dd['train'][0]) == np.sign(dd['test'][0]) else '')
        res.append((f, tn, len(on), (on.n > 0).mean() * 100, on.n.median() * 100, on.n.mean() * 100, (on.n.mean() - off.n.mean()) * 100, t, dd, es, v))
for f, tn, n, w, md, mn, df_, t, dd, es, v in res: print(f"{f:10s}{tn} {n:5d} | {w:5.1f} {md:5.2f} {mn:5.2f} | {df_:+.2f} ({t:+.1f}) | tr {dd['train'][0]:+.2f}/{dd['train'][1]:+.2f} | te {dd['test'][0]:+.2f}/{dd['test'][1]:+.2f} |{es} | {v}")
B.to_parquet(SP + 'greek422.parquet')
