"""Nov 15 2026 HOW MANY POSITIONS? (Rey: 20 seems like a lot). Same simulator as report_100 (flips, frozen score cutoff, entry next open, exit open after bear flip, 30bp, TEST 2022-01..2026-02, additive, marked daily). Vary the slot cap N in (3, 5, 8, 10, 15, 20) with two sizing rules: FULL = each position 1/N of starting capital (always fully invested when slots fill), and FIXED5 = 5% each (exposure = 5% x N). Scenarios: unfiltered flips, flip-bar volume>=1.5x, frozen SCORE top 20%, $1-10 only. Random-order books averaged over 20 tie-break draws (min..max total shown). Metrics: total, maxDD (pts of starting capital), Sharpe, calmar, trades taken, average open positions. Declared expectation: fewer slots raise variance and drawdown per unit of capital (fewer names, more waiting), and slot-limited selection by FIRST COME wastes the score's ranking, so the score's advantage should shrink at small N; with FIXED5 the drawdown scales with exposure but Sharpe should stay about the same."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('report_100_portfolio_dd.py').read(); exec(src[:src.index("S = {'CURRENT-LIKE")])
S = {'unfiltered flips': (lambda p: True, 'random'), 'flip-bar vol>=1.5x': (lambda p: p[5] >= 1.5, 'random'), 'SCORE top 20%': (lambda p: p[3] >= cut, 'score'), '$1-10 only': (lambda p: p[4] < 10, 'random')}
a, b = pd.Timestamp('2022-01-01'), pd.Timestamp('2026-03-01')
for rule in ('FULL (1/N each)', 'FIXED5 (5% each)'):
    print(f'\n== SIZING {rule} == TEST 2022-01..2026-02')
    for nm, (sel, order) in S.items():
        print(f' {nm}')
        for N in (3, 5, 8, 10, 15, 20):
            f = 1.0 / N if rule.startswith('FULL') else .05; seeds = range(20) if order == 'random' else range(1); R = [sim(sel, a, b, np.random.default_rng(sd), order, f=f, N=N) for sd in seeds]; g = lambda k: np.array([r[k] for r in R]); t = g('total')
            print(f'   N={N:2d}: total {t.mean():+7.1f}%' + (f' ({t.min():+.0f}..{t.max():+.0f})' if len(t) > 1 else '') + f' maxDD {g("mdd").mean():+6.1f}pts sharpe {np.nanmean(g("sharpe")):+.2f} calmar {g("annual").mean()/abs(g("mdd").mean()):+.2f} trades {g("trades").mean():4.0f} avgpos {g("avgpos").mean():4.1f} worst month {g("worstmo").mean():+.1f}%')
