"""Nov 15 2026 'MARCH-JULY, OUT IN FEBRUARY' (Rey). Interpretation declared before results: W1 = take only flips whose flip date is in March-July (exit rule unchanged, hold to the bear flip, positions opened in the window may run past July); W2 = skip flips in February only; W1 is the 'in March-July, out the rest incl. February' reading. HINDSIGHT WARNING: this window came from the month table in report_112 (February is the worst month in both eras), so it is data-snooped by construction; the only fair read is whether it holds in each calendar year. Daily flips (report_90 file, 30,323 flips 2019-Mar 2026, entry next open, hold to bear flip, 30bp), ALL names and LARGE-CAP (dv >= $100M, px >= $10). Metrics: mean/median net, win10, huge share, flips per year, and YEAR-paired difference (in-window mean minus out-of-window mean per calendar year; clustering by year because every calendar month is entirely in or out, which made report_112's month-clustered t undefined): share of years positive and t over years, TRAIN 2019-21 and TEST 2022-26 separately, plus a placebo: the same test for every other 5-consecutive-month window, ranking Mar-Jul among all 12 rotations by TEST year-paired difference."""
import numpy as np, pandas as pd
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; F = pd.read_pickle(X + 'stack_flips.pkl'); COST = .003; F['n'] = F.r - COST; F['m'] = F.d.dt.month; F['y'] = F.d.dt.year; F['lc'] = (F.dv >= 1e8) & (F.px >= 10)
def stt(z): return f'n {len(z):5d} mean {z.n.mean()*100:+6.2f}% med {z.n.median()*100:+6.2f}% win10 {(z.n>=.1).mean()*100:5.1f}% huge {(z.n>=1).mean()*100:4.2f}%'
def yp(z, mask):
    g = z.assign(i=mask).groupby(['y', 'i']).n.mean().unstack(); g = g.dropna(); d = (g[True] - g[False]); return d, (d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan)
W = lambda ms: (lambda m: m.isin(ms))
for lab, U in (('ALL', F), ('LARGE-CAP', F[F.lc])):
    print(f'\n######## {lab} ########')
    for era, z in (('TRAIN 2019-21', U[U.y <= 2021]), ('TEST 2022-26', U[U.y >= 2022])):
        print(f'\n {era}'); mj = z.m.isin([3, 4, 5, 6, 7]); fb = z.m == 2
        print('  all flips            ', stt(z)); print('  Mar-Jul only (W1)    ', stt(z[mj])); print('  outside Mar-Jul      ', stt(z[~mj])); print('  February             ', stt(z[fb])); print('  all but February (W2)', stt(z[~fb]))
        for lab2, mk in (('W1 Mar-Jul minus rest', mj), ('W2 non-Feb minus Feb', ~fb)):
            d, t = yp(z, mk); print(f'  year-paired {lab2}: ' + ' '.join(f'{y}:{v*100:+.1f}' for y, v in d.items()) + f' | years positive {(d>0).sum()}/{len(d)} mean {d.mean()*100:+.2f}pp t {t:+.2f}')
    print('\n PLACEBO: all 12 five-month rotations, TEST year-paired difference (in window minus rest), Mar-Jul marked')
    rs = []
    for s in range(1, 13):
        ms = [((s - 1 + k) % 12) + 1 for k in range(5)]; z = U[U.y >= 2022]; d, t = yp(z, z.m.isin(ms)); z0 = U[U.y <= 2021]; d0, t0 = yp(z0, z0.m.isin(ms)); rs.append((s, ms, d.mean() * 100, t, d0.mean() * 100, t0))
    for s, ms, dm, t, dm0, t0 in sorted(rs, key=lambda r: -r[2]): print(f'  start {s:2d} months {ms}: TEST {dm:+6.2f}pp t {t:+5.2f} | TRAIN {dm0:+6.2f}pp t {t0:+5.2f}' + ('   <-- Mar-Jul' if s == 3 else ''))
