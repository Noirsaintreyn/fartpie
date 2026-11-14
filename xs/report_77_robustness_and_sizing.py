"""Nov 14 2026 TWO CHECKS BEFORE WE MOVE ON (declared before results). (1) TOP-WINNER REMOVAL: for the screener rule (volume >=1.5x), insider+volume, squeeze-hot (DTC>=10 & basket lead), and insider+volume skipping TIGHTENING months: per era mean net, mean after removing the top 1/3/5 winners of that era, 2%-trimmed mean (drop top 2% only), median, count of HUGE. A lead survives if it still beats the volume rule after removing the same number of winners from both. (2) SIZING IN TIGHTENING MONTHS: weight 1.0 in EASING/MID/unclassified months and w in {1.0 (now), 0.5, 0.0 (skip)} in TIGHTENING months; report per era the return per ORIGINAL flip (capital-normalised: sum(w*r)/N) and per dollar deployed (sum(w*r)/sum(w)). 2021-22 are classed TIGHTENING (yields rising), so the half-size variant shows what skipping threw away. Class = expanding terciles of the 60d 2y-yield change, known at month start (report_73b)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('report_74_return_now_vs_with.py').read(); exec(src.split("B = P[P.BV >= 1.5].copy()")[0].split("def book(")[0])
B = P[P.BV >= 1.5].copy(); SETS = {'volume>=1.5x (now)': B, 'insider + volume': B[B.NI >= 1], 'squeeze-hot (DTC>=10 & lead)': B[B.HOT], 'insider+vol skip TIGHTENING': B[(B.NI >= 1) & (B.cls != 'TIGHTENING')], 'volume skip TIGHTENING': B[B.cls != 'TIGHTENING']}
print('PART 1: robustness to the biggest winners (net %, per-flip mean)'); print(f'{"set":30s} {"era":9s} {"n":>5s} {"mean":>7s} {"-top1":>7s} {"-top3":>7s} {"-top5":>7s} {"trim2%":>7s} {"median":>7s} {"HUGE":>5s} {"top3 share of sum":>17s}')
for k, S in SETS.items():
    for en, z in (('2019-21', S[S.t < '2022']), ('2022-26Q1', S[S.t >= '2022'])):
        r = np.sort(z.rn.values)[::-1]
        if len(r) < 20: print(f'{k:30s} {en:9s} n {len(r)}'); continue
        mm = lambda a: r[a:].mean() * 100; tr = np.sort(r)[:int(len(r) * .98)].mean() * 100
        print(f'{k:30s} {en:9s} {len(r):5d} {r.mean()*100:+7.2f} {mm(1):+7.2f} {mm(3):+7.2f} {mm(5):+7.2f} {tr:+7.2f} {np.median(r)*100:+7.2f} {int(z.HUGE.sum()):5d} {r[:3].sum()/max(r.sum(),1e-9)*100:16.0f}%')
print('\nPART 2: sizing in TIGHTENING months (weight w there, 1.0 elsewhere)'); print(f'{"set":26s} {"era":9s} ' + ' | '.join(f'w={w}: per-orig-flip / per-$' for w in (1.0, 0.5, 0.0)) + f'  (tightening share of flips)')
for k in ('volume>=1.5x (now)', 'insider + volume', 'squeeze-hot (DTC>=10 & lead)'):
    S = SETS[k]
    for en, z in (('2019-21', S[S.t < '2022']), ('2022-26Q1', S[S.t >= '2022']), ('ALL', S)):
        tight = (z.cls == 'TIGHTENING').values; out = []
        for w in (1.0, 0.5, 0.0):
            wt = np.where(tight, w, 1.0); out.append(f'{(wt*z.rn.values).sum()/len(z)*100:+6.2f}% / {(wt*z.rn.values).sum()/max(wt.sum(),1e-9)*100:+6.2f}%')
        print(f'{k:26s} {en:9s} ' + ' | '.join(f'{o:>26s}' for o in out) + f'  ({tight.mean()*100:.0f}%)')
print('\nTIGHTENING months by year (share of flips classed): ' + ', '.join(f'{y}: {(P[P.yr==y].cls=="TIGHTENING").mean()*100:.0f}%' for y in range(2019, 2027)))
