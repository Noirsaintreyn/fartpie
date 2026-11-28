"""Report 527 (Nov 27 2026): real PIPE content from SEC 8-K text (pulled via Rey's Mac browser: data.sec.gov submissions + www.sec.gov Archives, 67 filings, 27 health tickers).
DECLARED BEFORE RESULTS. Of the 91 health flips with an 8-K 3.02 in 365d, classify each 3.02 filing by full-text: TRUEPIPE = text has purchase/subscription agreement, private placement, placement agent or registered direct (excludes exchanges, M&A, licensing, notes conversions); HFPIPE = TRUEPIPE and a named specialist hedge/crossover fund (armistice, nantahala, mmcap, bvf, casdin) in the text.
Lit (Brophy 2004): hedge-fund-funded PIPEs underperform -> HFPIPE reach LOWER than TRUEPIPE-other; Consensus #5: any PIPE negative. Flip's flag = any such filing within 365d before the flip date.
Insider-in-PIPE flag NOT built: 'executive officer' matches boilerplate, regex unreliable (disclosed). Discount captured in only 5/67 filings (mostly in exhibits) -> not tested.
Trials 3 (3.02 any [rerun], TRUEPIPE, HFPIPE) x3. Outcome reach/mean/median; test 2022+; permutation 5000 vs random same-size health flips (test).
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SP + 'ext/'
P = [(1557746,'2024-11-18',''),(1771910,'2025-10-14',''),(1771910,'2025-06-12',''),(1730463,'2024-02-08','vivo'),(1332349,'2020-07-27',''),(1133869,'2020-03-26',''),(1662579,'2023-05-30',''),(1737927,'2024-05-06','mmcap'),(1737927,'2024-01-19',''),(1737927,'2024-01-12',''),(1737927,'2023-09-18',''),(1512762,'2023-10-06',''),(1024305,'2020-05-26',''),(1024305,'2020-05-12',''),(883569,'2025-08-14',''),(1162194,'2022-04-05','casdin'),(1162194,'2022-01-24','casdin'),(1162194,'2019-11-22',''),(1750284,'2024-12-02','bvf'),(944809,'2024-01-09',''),(944809,'2018-11-09',''),(1661181,'2024-11-13',''),(1299130,'2024-11-22',''),(1299130,'2024-11-07',''),(1356090,'2024-12-30',''),(1356090,'2020-01-02',''),(1782223,'2024-02-28',''),(1635088,'2022-02-14',''),(1635088,'2021-10-01',''),(1160308,'2019-12-20','bain capital'),(1806310,'2023-08-14',''),(1867096,'2022-01-03','armistice'),(1434647,'2023-08-31','nantahala')]
import json
cik2tk = {}
F = pd.read_pickle(X + 'flips517.pkl'); H = F[F.sector == 'HEALTH'].copy(); H['d'] = pd.to_datetime(H.d)
for tk in H.tk.unique():
    try: cik2tk[int(json.load(open(f'data/fund/sec_small/{tk}.json'))['cik'])] = tk
    except Exception: pass
SPEC = {'armistice', 'nantahala', 'mmcap', 'bvf', 'casdin'}
pp = pd.DataFrame([(cik2tk[c], pd.Timestamp(d), h in SPEC) for c, d, h in P if c in cik2tk], columns=['tk', 'd0', 'hf'])
def flag(r, hf=None):
    g = pp[pp.tk == r.tk]
    if hf is not None: g = g[g.hf == hf]
    return int(((r.d >= g.d0) & (r.d <= g.d0 + pd.Timedelta(days=365))).any())
H['TRUEPIPE'] = H.apply(flag, axis=1); H['HFPIPE'] = H.apply(lambda r: flag(r, True), axis=1); H['OTHERPIPE'] = H.apply(lambda r: flag(r, False), axis=1)
H['yr'] = H.d.dt.year; te = H.yr >= 2022; tr = H.yr.between(2019, 2021); rng = np.random.default_rng(5); hit = H[te].hit.astype(float).values
print('health flips', len(H), '| any 3.02 (earlier) 91 | TRUEPIPE', H.TRUEPIPE.sum(), 'HFPIPE', H.HFPIPE.sum(), 'OTHERPIPE(non-HF)', H.OTHERPIPE.sum()); print('base test reach', round(H[te].hit.mean() * 100), 'mean', round(H[te].r.mean() * 100, 1), 'med', round(H[te].r.median() * 100, 1), 'n', te.sum())
for v in ('TRUEPIPE', 'HFPIPE', 'OTHERPIPE'):
    s = H[H[v] == 1]; t = s[te.loc[s.index]]; a = s[tr.loc[s.index]]
    if len(t) < 5: print(v, 'n', len(s), 'test n', len(t), 'too few'); continue
    d = rng.choice(len(hit), (5000, len(t))); lo = (hit[d].mean(1) <= t.hit.mean()).mean(); hi = (hit[d].mean(1) >= t.hit.mean()).mean()
    print(f'{v:10} n={len(s):3d} reach {s.hit.mean()*100:3.0f}% mean {s.r.mean()*100:5.1f} med {s.r.median()*100:5.1f} | train n{len(a)} reach {a.hit.mean()*100 if len(a) else float("nan"):.0f}% | test n{len(t)} reach {t.hit.mean()*100:.0f}% mean {t.r.mean()*100:.1f} med {t.r.median()*100:.1f} | p(lower) {lo:.3f} p(higher) {hi:.3f}  x3 {min(1,min(lo,hi)*3):.3f}')
