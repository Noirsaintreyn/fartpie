"""Nov 18 2026 report 315. Literature-backed leads from the paper search + the screener page (Harvest Premia weekly prep).
(A) Gao-Han-Li-Zhou 2018 JFE 'Market intraday momentum': first-half-hour return predicts last-half-hour return (mechanism: dealer gamma/leveraged-ETF/rebalancing flow into the close; stronger on high-vol and big-news days). Fixed rule, declared before running, NQ RTH 5m: r_first = close(10:00)/open(09:30)-1, r_last = close(16:00)/close(15:30)-1; trade last half-hour in sign(r_first); also with the overnight+first-half-hour composite. Cost 1.0 pt per contract round trip. Train 2019-21 | test 2018+2022+. Placebo: sign of r_first from the PREVIOUS day. Why it would stop: 0DTE changed closing-flow structure; crowded.
(B) Screener-page features, price-only, point-in-time from b302 market-state columns: breadth (share of S&P above 50d/200d), advancing share; test flip return (ret) by tercile cut on train, same-sign + |t|>=2 admission rule from report 303."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); N = N[N.rth & (N.m >= 570) & (N.m <= 955)].copy(); N['day'] = pd.to_datetime(N.day).dt.normalize(); rows = []
for d, x in N.groupby('day'):
    x = x.sort_values('m')
    if len(x) < 70 or x.m.iloc[0] != 570: continue
    f = x[x.m <= 595]; l = x[x.m >= 925]
    if len(f) < 5 or len(l) < 5: continue
    rows.append(dict(day=d, o=x.open.iloc[0], c1000=f.close.iloc[-1], c1530=x[x.m <= 920].close.iloc[-1], c16=x.close.iloc[-1], cl=x.close.iloc[-1], rng=x.high.max() - x.low.min()))
E = pd.DataFrame(rows).set_index('day'); E['adr'] = E.rng.rolling(14).mean().shift(1); E['pc'] = E.cl.shift(1)
E['rf'] = E.c1000 / E.o - 1; E['rl_pts'] = E.c16 - E.c1530; E['on'] = E.o / E.pc - 1; E['comp'] = E.rf + E.on; E['tr'] = (E.index >= '2019-01-01') & (E.index < '2022-01-01'); E = E.dropna()
def rep(sig, lab):
    out = []
    for nm, m in (('train', E.tr), ('test', ~E.tr)):
        z = E[m]; p = np.sign(sig[m]) * z.rl_pts - 1.0; out.append(f'{nm} {p.mean():+5.2f} pts/trade (t{p.mean() / (p.std() / np.sqrt(len(p))):+.1f}, win {(p > 0).mean():.0%}, n{len(p)})')
    print(f'{lab:34s}', ' | '.join(out))
print('(A) last-half-hour trade (09:30-10:00 first half-hour; 15:30-16:00 last), pts per contract net of 1pt')
rep(E.rf, 'sign first half-hour'); rep(E.comp, 'sign overnight + first half-hour'); rep(E.rf.shift(1).fillna(0), 'placebo: yesterday first half-hour'); rep(E.on, 'sign overnight only')
big = E.rf.abs() > E.rf[E.tr].abs().median(); print(' by size of first half-hour (train median cut):')
for nm, m in (('train', E.tr), ('test', ~E.tr)):
    for b in (True, False): z = E[m & (big == b)]; p = np.sign(z.rf) * z.rl_pts - 1; print(f'   {nm} big={b}: {p.mean():+5.2f} (t{p.mean() / (p.std() / np.sqrt(len(p))):+.1f}, n{len(p)})')
print(' by year (first half-hour rule):', ((np.sign(E.rf) * E.rl_pts - 1).groupby(E.index.year).mean()).round(2).to_dict())
B = pd.read_parquet(SP + 'b302.parquet'); print('\n(B) b302 columns:', [c for c in B.columns if any(k in c.lower() for k in ('bread', 'adv', '50', 'idx', 'spy', 'corr'))])
