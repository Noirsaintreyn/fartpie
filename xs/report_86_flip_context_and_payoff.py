"""Nov 14 2026 WHAT MAKES A FLIP BIG: pre-flip volatility/extension, supply proxies, and payoff structure. Declared before results. Universe/flips as report_85 (daily vector long flips 2019-Mar 2026, entry next open, exit open after opposite flip, 30bp). Features known at the flip-bar close: VR = ATR10/ATR60 (range compression if low); EXT = close / 60-day low - 1 (extension off the base); R60 = 60-day return; BLOW = 20d realized vol / 120d realized vol (volatility blow-off if high); PRICE tier (<2, 2-5, 5-20, 20+); DV tier by 20d median dollar volume (<1M, 1-5M, 5-25M, 25M+). H1: low VR (compressed) -> better mean and huge rate. H2: high EXT / R60 / BLOW (late, after the blow-off) -> worse. H3: low price and low dollar volume -> higher huge-rate but lower median (supply/liquidity effect, capacity-limited). Continuous features are ranked in terciles WITHIN each month's flips (removes regime drift). PASS for H1/H2 = top-minus-bottom tercile mean differs in the predicted direction in BOTH eras (2019-21 / 2022-26Q1) with pooled month-clustered t >= 2. Part B (payoff structure): bootstrap K flips per year (equal weight, K = 25, 50, 100, 250, 500) sampled with replacement from each era; report P(year profitable), P(year contains no >=+100% flip), median and 10th percentile year return. Not tested: shares outstanding / market cap / float (no data), options, intraday."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']
    if len(B) < 300: continue
    o = B.open.values; h = B.high.values; l = B.low.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; n = len(B)
    reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))
    a10 = pd.Series(tr).rolling(10).mean().values; a60 = pd.Series(tr).rolling(60).mean().values; lr = pd.Series(np.log(c)).diff(); v20 = lr.rolling(20).std().values; v120 = lr.rolling(120).std().values
    lo60 = pd.Series(l).rolling(60).min().values; dv = pd.Series(c * v).rolling(20).median().values; vm = pd.Series(v).rolling(20).mean().shift(1).values; i = 130
    while i < n - 2:
        if reg[i] == 1 and reg[i - 1] != 1:
            j = i + 1
            while j < n and reg[j] != -1: j += 1
            if j + 1 < n and '2019-01-01' <= str(idx[i])[:10] < '2026-03-01' and c[i] >= 1 and dv[i] >= 5e5:
                rows.append((tk, idx[i], o[j + 1] / o[i + 1] - 1 - 0.003, a10[i] / a60[i], c[i] / lo60[i] - 1, c[i] / c[i - 60] - 1, v20[i] / v120[i], c[i], dv[i]))
            i = j
        i += 1
F = pd.DataFrame(rows, columns=['tk', 'd', 'r', 'VR', 'EXT', 'R60', 'BLOW', 'px', 'dv']); F = F.replace([np.inf, -np.inf], np.nan).dropna(); F['era'] = np.where(F.d < '2022-01-01', '2019-21', '2022-26Q1'); F['mo'] = F.d.dt.to_period('M'); print('flips', len(F))
def st(x): return f'n {len(x):5d} mean {x.mean()*100:+6.1f}% med {x.median()*100:+5.1f}% huge {(x>=1).mean()*100:4.1f}% ex-top3 {x.sort_values().iloc[:-3].mean()*100 if len(x)>5 else np.nan:+5.1f}%'
for f in ('VR', 'EXT', 'R60', 'BLOW'): F[f + '_t'] = F.groupby('mo')[f].transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False) if len(s) >= 6 else np.nan)
def ct(z, f):
    a = z[z[f + '_t'] == 2].groupby('mo').r.mean(); b = z[z[f + '_t'] == 0].groupby('mo').r.mean(); d = (a - b).dropna(); return d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 5 else np.nan, (z[z[f + '_t'] == 2].r.mean() - z[z[f + '_t'] == 0].r.mean()) * 100
print('\nPART A: terciles within month (0 = lowest feature, 2 = highest)'); out = {}
for f in ('VR', 'EXT', 'R60', 'BLOW'):
    print(f'\n {f}')
    for en in ('2019-21', '2022-26Q1', 'POOLED'):
        z = F if en == 'POOLED' else F[F.era == en]; t, d = ct(z, f); out[(f, en)] = (d, t)
        print(f'  {en:9s} ' + ' || '.join(f'T{k} {st(z[z[f+"_t"]==k].r)}' for k in (0, 2)) + f' || top-bottom {d:+.1f}pp t {t:+.1f}')
for f, sign in (('VR', -1), ('EXT', -1), ('R60', -1), ('BLOW', -1)): print(f'  {f}: H predicts top tercile WORSE' if f != 'VR' else '  VR: H predicts LOW (compressed) BETTER', 'PASS' if all(out[(f, e)][0] * sign > 0 for e in ('2019-21', '2022-26Q1')) and out[(f, 'POOLED')][1] * sign >= 2 else 'no pass')
F['pxt'] = pd.cut(F.px, [0, 2, 5, 20, 1e9], labels=['<2', '2-5', '5-20', '20+']); F['dvt'] = pd.cut(F.dv, [0, 1e6, 5e6, 25e6, 1e12], labels=['<1M', '1-5M', '5-25M', '25M+'])
print('\nH3 supply proxies')
for col in ('pxt', 'dvt'):
    for en in ('2019-21', '2022-26Q1'):
        z = F[F.era == en]; print(f' {col} {en}'); [print(f'    {k:6s} {st(g.r)}') for k, g in z.groupby(col)]
print('\nPART B: payoff structure; K random flips per year, equal weight, bootstrap 4000 years')
rng = np.random.default_rng(7)
for en in ('2019-21', '2022-26Q1'):
    r = F[F.era == en].r.values; print(f' {en}: flips {len(r)} mean {r.mean()*100:+.1f}% median {np.median(r)*100:+.1f}% huge {(r>=1).mean()*100:.1f}%')
    for K in (25, 50, 100, 250, 500):
        s = rng.choice(r, (4000, K)); y = s.mean(axis=1); nh = (s >= 1).sum(axis=1)
        print(f'   K={K:3d}: P(year>0) {(y>0).mean()*100:4.0f}% | P(no huge flip) {(nh==0).mean()*100:4.0f}% | year return median {np.median(y)*100:+6.1f}% p10 {np.percentile(y,10)*100:+6.1f}% | median WITHOUT top-1 trade {np.median((s.sum(axis=1)-s.max(axis=1))/K)*100:+6.1f}%')
F.to_pickle(X + 'flip_context.pkl')
