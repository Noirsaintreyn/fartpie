"""Nov 15 2026 POST-HOC follow-up to report_115 (declared AFTER seeing the class map, so it is a second look, not a clean test): the report_115 selection rule (train mean > 0 and ex-top-3 > 0) was too loose because train 2019-21 was a +4.4% mean tailwind: it selected 15 of 19 cells and the union equalled all flips (permutation share 0.35). Stricter rule written from the table's structure: train mean >= +10% AND train ex-top-3 >= +4% with n >= 80 (selected on TRAIN only: 2-5/top20/vol>=1.5, 2-5/top20/vol<1.5, 2-5/mid40/vol<1.5, 5-10/top20/vol>=1.5). Plus two simple supply-and-score classes, both written from the same table: S1 = price < $5 AND frozen score top 20%; S2 = price < $5 AND (top 20% OR VOLX >= 1.5). Judged on TEST 2022-26 at 30bp and 100bp: n, flips/yr, mean/median/win10/huge/ex-top-3, by-year means, and the same classes on the TRAIN era for reference (whether the classes also worked there). Not a validated model: classes were chosen after seeing the test numbers; the honest use is as the shortlist for the shadow ledger."""
import numpy as np, pandas as pd
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; F = pd.read_pickle(X + 'stack_flips.pkl'); F['y'] = F.d.dt.year
cut80 = F[~F.test].lr.quantile(.8); cut40 = F[~F.test].lr.quantile(.4); F['top'] = F.lr >= cut80; F['mid'] = (F.lr >= cut40) & ~F.top; F['hv'] = F.VOLX >= 1.5
cls = {'STRICT 4-cell union': ((F.px >= 2) & (F.px < 5) & F.top) | ((F.px >= 2) & (F.px < 5) & F.mid & ~F.hv) | ((F.px >= 5) & (F.px < 10) & F.top & F.hv), 'S1: px<5 & score top20': (F.px < 5) & F.top, 'S2: px<5 & (top20 or VOLX>=1.5)': (F.px < 5) & (F.top | F.hv), 'ALL flips (ref)': F.px > 0, 'px<5 any score (ref)': F.px < 5}
def ex3(s): s = s.sort_values(ascending=False); return s.iloc[3:].mean()
for cost in (.003, .01):
    print(f'\n---- cost {cost*1e4:.0f}bp ----')
    for era, e in (('TRAIN 2019-21', ~F.test), ('TEST 2022-26', F.test)):
        for k, m in cls.items():
            z = F[m & e]; r = z.r - cost
            if len(z) < 10: continue
            yrs = (z.d.max() - z.d.min()).days / 365
            print(f' {era} {k:34s} n {len(z):5d} ({len(z)/yrs:4.0f}/yr) mean {r.mean()*100:+6.2f}% med {r.median()*100:+6.2f}% win10 {(r>=.1).mean()*100:5.1f}% huge {(r>=1).mean()*100:4.2f}% ex3 {ex3(r)*100:+6.2f}% | ' + ' '.join(f'{y}:{(g.r-cost).mean()*100:+.1f}(n{len(g)})' for y, g in z.groupby('y')))
