"""Nov 15 2026 DECEMBER / DEC-JAN (Rey meant December, not February: an earlier model found 12-1 worked, mechanism = year-end unloading of books). Declared before results; hypothesis from the mechanism: flips fired in December (selling pressure from fund unloading / tax-loss selling) do WORSE than the rest of the year, with the January reversal; so the testable claim is 'skip or de-size December flips' (and 'Dec-Jan' as the 12-1 window), sign: December flips below the rest on BOTH eras. Daily flips (report_90 file, 30,323 flips 2019-Mar 2026, entry next open, hold to bear flip, 30bp), ALL and LARGE-CAP (dv >= $100M, px >= $10). Windows: DEC; JAN; DEC-JAN (months 12 and 1); DEC 1-15 vs DEC 16-31 (descriptive). Metrics: n, mean/median net, win10, huge share; year-paired difference (window flips mean minus same-year other-month mean; calendar-year clustering because each month is wholly in or out); per-year table; TRAIN 2019-21 / TEST 2022-25 (2026 has no December yet). Placebo: rank of December among the 12 single months by test year-paired difference (low rank = worst month). A second pass notes the flips opened in December that exit in January or later (hold through the turn)."""
import numpy as np, pandas as pd
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; F = pd.read_pickle(X + 'stack_flips.pkl'); COST = .003; F['n'] = F.r - COST; F['m'] = F.d.dt.month; F['y'] = F.d.dt.year; F['lc'] = (F.dv >= 1e8) & (F.px >= 10); F['dd'] = F.d.dt.day
def stt(z): return f'n {len(z):5d} mean {z.n.mean()*100:+6.2f}% med {z.n.median()*100:+6.2f}% win10 {(z.n>=.1).mean()*100:5.1f}% huge {(z.n>=1).mean()*100:4.2f}%'
def yp(z, mask):
    g = z.assign(i=mask).groupby(['y', 'i']).n.mean().unstack().dropna(); d = g[True] - g[False]; return d, (d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan)
for lab, U in (('ALL', F), ('LARGE-CAP', F[F.lc])):
    print(f'\n######## {lab} ########')
    for era, z in (('TRAIN 2019-21', U[U.y <= 2021]), ('TEST 2022-25', U[(U.y >= 2022) & (U.y <= 2025)])):
        print(f'\n {era}'); print('  all flips      ', stt(z))
        for nm, ms in (('DEC', [12]), ('JAN', [1]), ('DEC-JAN', [12, 1])):
            m = z.m.isin(ms); d, t = yp(z, m); print(f'  {nm:7s} flips  ', stt(z[m]), '| rest', stt(z[~m])[:62], f'| year-paired ' + ' '.join(f'{y}:{v*100:+.1f}' for y, v in d.items()) + f' mean {d.mean()*100:+.2f}pp t {t:+.2f}')
        zd = z[z.m == 12]; print('  DEC 1-15       ', stt(zd[zd.dd <= 15])); print('  DEC 16-31      ', stt(zd[zd.dd > 15]))
    z = U[U.y >= 2022]; z = z[z.y <= 2025]; r = []
    for m in range(1, 13): d, t = yp(z, z.m == m); r.append((m, d.mean() * 100, t))
    print('\n  PLACEBO single-month year-paired difference, TEST 2022-25 (sorted worst first): ' + ' '.join(f'm{m}:{v:+.1f}(t{t:+.1f})' for m, v, t in sorted(r, key=lambda x: x[1])))
    r0 = []; z0 = U[U.y <= 2021]
    for m in range(1, 13): d, t = yp(z0, z0.m == m); r0.append((m, d.mean() * 100, t))
    print('  PLACEBO single-month year-paired difference, TRAIN 2019-21 (sorted worst first): ' + ' '.join(f'm{m}:{v:+.1f}(t{t:+.1f})' for m, v, t in sorted(r0, key=lambda x: x[1])))
F['exm'] = (F.d + pd.to_timedelta(F.hold if 'hold' in F else 0, unit='D')) if False else 0
