"""Nov 15 2026 RETURNS OF THE OUTLIER MODEL (Rey: 'what will the returns be with turnover, daily ATR as a share of price, extension off the 60-day low'). Declared before results. MODEL OS = sum of three TRAIN-era percentile ranks (empirical CDF built on 2019-21 flips only, then applied to every flip, so no test-era information in the scale): ATRP (daily ATR / price), EXT (close / 60d low - 1), TURN (shares-based turnover, from report_99's shares file); all three directions + (declared from report_117, whose consistency screen used test AUC, so the test numbers below are optimistic). Cutoff = 80th percentile of OS on the TRAIN flips, frozen. Flips lacking any of the three (missing shares data) are not eligible; coverage is reported. Universes: ALL names and price < $5. Per-trade: n, mean/median net at 30bp and 100bp, win10, huge, ex-top-3, by year. BOOK (same simulator as report_100: additive, entry next open, exit open after the bear flip, TEST 2022-01..2026-02): slots N = 10 and 20, sizing 5% of starting capital per position (FIXED5) and 1/N (FULL); selection first-come by date with OS tie-break within a day; random-order draws averaged over 20 seeds for the unranked reference. Compared against: ALL flips, price < $5, the frozen daily score top 20% (report_100) and the same with price < $5. Metrics: total return over the window, annualised, max drawdown in points of starting capital, Sharpe, trades, average positions, worst month, per-year book returns. SURVIVORSHIP-BIASED and thin-fill costs unmodeled: the sub-$5 results are the most flattered."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('report_100_portfolio_dd.py').read(); exec(src[:src.index("S = {'CURRENT-LIKE")])
Of = pd.read_pickle(X + 'outlier_features.pkl'); Of['LTURN'] = np.log(Of.TURN.clip(lower=1e-9)); tr = Of[~Of.test]; ok = Of.dropna(subset=['ATRP', 'EXT', 'LTURN']).copy()
def ecdf(col): s = np.sort(tr[col].dropna().values); return lambda x: np.searchsorted(s, x, side='right') / len(s)
fa, fe, ft = ecdf('ATRP'), ecdf('EXT'), ecdf('LTURN'); ok['OS'] = fa(ok.ATRP.values) + fe(ok.EXT.values) + ft(ok.LTURN.values); cutOS = ok[~ok.test].OS.quantile(.8); print('coverage', len(ok), 'of', len(Of), '| train 80th pct OS cut', round(cutOS, 3))
ok['n30'] = ok.r - .003; ok['n100'] = ok.r - .01; ok['y'] = ok.d.dt.year
def ex3(s): s = s.sort_values(ascending=False); return s.iloc[3:].mean() if len(s) > 6 else np.nan
print('\nPER-TRADE (daily flips, hold to bear flip)')
for lab, mk in (('ALL flips', ok.px > 0), ('px<5', ok.px < 5), ('OS top20', ok.OS >= cutOS), ('OS top20 & px<5', (ok.OS >= cutOS) & (ok.px < 5)), ('frozen score top20 & px<5', (ok.lr >= cut) & (ok.px < 5)), ('frozen score top20', ok.lr >= cut)):
    for era, e in (('TRAIN', ~ok.test), ('TEST ', ok.test)):
        z = ok[mk & e]; 
        if len(z) < 10: continue
        yrs = (z.d.max() - z.d.min()).days / 365; print(f' {era} {lab:26s} n {len(z):5d} ({len(z)/yrs:4.0f}/yr) mean30 {z.n30.mean()*100:+7.2f}% mean100 {z.n100.mean()*100:+7.2f}% med {z.n30.median()*100:+6.2f}% win10 {(z.n30>=.1).mean()*100:5.1f}% huge {(z.n30>=1).mean()*100:4.2f}% ex3 {ex3(z.n30)*100:+6.2f}%' + (' | ' + ' '.join(f'{y}:{g.n30.mean()*100:+.1f}' for y, g in z.groupby('y')) if era.startswith('TEST') else ''))
D_ = {tk: q.index for tk, q in D.items()}; key = {}
for tk, g in ok.groupby('tk'):
    ix = D_.get(tk)
    if ix is None: continue
    for d, os_ in zip(g.d, g.OS):
        k = ix.searchsorted(d) 
        if k + 1 < len(ix): key[(tk, ix[k + 1])] = os_
paths2 = [(p[0], p[1], p[2], key[(p[7], p[0])], p[4], p[5], p[6], p[7]) for p in paths if (p[7], p[0]) in key]; print('\npaths with OS', len(paths2), 'of', len(paths))
a, b = pd.Timestamp('2022-01-01'), pd.Timestamp('2026-03-01')
def book(PS, sel, order, N, f, cost, seeds=1):
    global paths
    old = paths; paths = PS
    try: R = [sim(sel, a, b, np.random.default_rng(sd), order, f=f, N=N, cost=cost) for sd in range(seeds)]
    finally: paths = old
    g = lambda k: np.array([r[k] for r in R]); return g('total').mean(), g('annual').mean(), g('mdd').mean(), np.nanmean(g('sharpe')), g('trades').mean(), g('avgpos').mean(), g('worstmo').mean(), R[0].get('years')
S = {'ALL flips (random)': (paths, lambda p: True, 'random', 20), 'px<5 (random)': (paths, lambda p: p[4] < 5, 'random', 20), 'frozen score top20 (score)': (paths, lambda p: p[3] >= cut, 'score', 1), 'frozen score top20 & px<5 (score)': (paths, lambda p: p[3] >= cut and p[4] < 5, 'score', 1), 'OS top20 (OS)': (paths2, lambda p: p[3] >= cutOS, 'score', 1), 'OS top20 & px<5 (OS)': (paths2, lambda p: p[3] >= cutOS and p[4] < 5, 'score', 1), 'OS top10 & px<5 (OS)': (paths2, lambda p: p[3] >= ok[~ok.test].OS.quantile(.9) and p[4] < 5, 'score', 1)}
for cost in (.003, .01):
    for rule, fn in (('FIXED5 (5% each)', lambda N: .05), ('FULL (1/N each)', lambda N: 1.0 / N)):
        print(f'\n== BOOK TEST 2022-01..2026-02, {rule}, cost {cost*1e4:.0f}bp ==')
        for nm, (PS, sel, order, seeds) in S.items():
            for N in (10, 20):
                t, ann, mdd, sh, tr_, ap, wm, yrs = book(PS, sel, order, N, fn(N), cost, seeds)
                print(f' {nm:36s} N={N:2d}: total {t:+8.1f}% ann {ann:+7.1f}% maxDD {mdd:+7.1f}pts sharpe {sh:+.2f} trades {tr_:4.0f} avgpos {ap:4.1f} worstmo {wm:+6.1f}%' + (f' | years {yrs}' if (yrs is not None and cost == .003 and rule.startswith('FIXED') and N == 10) else ''))
