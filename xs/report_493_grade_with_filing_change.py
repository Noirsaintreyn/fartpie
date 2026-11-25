"""Report 493 (Nov 25 2026): Rey - 'filing words is grading'. Add the filing-text change (report 491) to the report-492 grade and see whether it helps. Sub-$5 live group, TEST 2022+ flips that have a 10-K/10-Q text pair (n ~239; cached similarities from report 491).
DECLARED BEFORE RESULTS. Variants of the 492 points: BASE (492 rule), PAPER = BASE - 1 when the filing changed a lot (lowest tercile of cosine similarity; Cohen-Malloy-Nathan direction), OPPOSITE = BASE + 1 when changed (the direction report 491 measured). Read grade A/B/C/D hits/n, mean, median for each variant and whether the ordering is monotone. The paper direction is the declared one; the opposite is shown because 491 found it, and is NOT adopted unless it also passes a fresh split (it cannot here: same sample)."""
import io, contextlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(open('xs/report_492_setup_grade.py').read().replace("for cap in (5, 10):", "for cap in ():"))
    S = O[O.px < 5].copy(); S['s_'] = score(S); cut5 = np.quantile(S[S.tr].s_.dropna(), .7); G5 = S[S.s_ >= cut5].copy(); ac = np.quantile(G5[G5.tr].ATRP, 2 / 3); G5['HI'] = G5.ATRP >= ac; G5['pts'] = G5.HI.astype(int) + G5.FIRST.astype(int) + G5.REC.astype(int) - G5.EARLY30.astype(int) - 2 * G5.NEGBAD.astype(int)
    src491 = open('xs/report_491_lazy_prices.py').read(); exec(src491[:src491.index('t0 = time.time()')])
    sims = []
    for tk, d in zip(G.tk, G.d):
        try: p = pair(tk, d)
        except Exception: p = None
        sims.append(p[0] if p else np.nan)
    SM = pd.DataFrame({'tk': G.tk.values, 'd': G.d.values, 'sim': sims}).dropna()
T = G5[~G5.tr].merge(SM, on=['tk', 'd'], how='inner'); lo = T.sim.quantile(1 / 3); T['CH'] = T.sim <= lo
for nm, pts in (('BASE', T.pts), ('PAPER (changed -1)', T.pts - T.CH.astype(int)), ('OPPOSITE (changed +1)', T.pts + T.CH.astype(int))):
    gr = np.select([pts >= 2, pts == 1, pts == 0], ['A', 'B', 'C'], 'D'); print(f'\n{nm}: test flips with a text pair {len(T)}')
    for g in 'ABCD':
        x = T[gr == g]; print(f"  grade {g}: {int(x.hit.sum()):3d}/{len(x):3d} = {x.hit.mean() if len(x) else float('nan'):.0%} reach | mean {x.r.mean() if len(x) else float('nan'):+6.1%} med {x.r.median() if len(x) else float('nan'):+.1%}")
print(f"\nCHANGED (sim <= {lo:.3f}) n {int(T.CH.sum())}: reach {T[T.CH].hit.mean():.0%} mean {T[T.CH].r.mean():+.1%} | SAME-ish n {int((~T.CH).sum())}: reach {T[~T.CH].hit.mean():.0%} mean {T[~T.CH].r.mean():+.1%}")
