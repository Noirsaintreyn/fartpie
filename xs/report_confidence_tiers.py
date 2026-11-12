"""Does a simple count of favourable flags order E+V sub-$5 trades? Flags (all at entry): F1 vr>=2.5 | F2 svr20<=0.532 (train cut) | F3 next earnings reaction >10 sessions away | F4 vwx<=0.5 | F5 IWM above 50d (m1). Report by flag-count tiers, TRAIN and TEST, to see whether a 'confidence' label is honest."""
exec(open('xs/report_sub5_shortvol.py').read().split("FS = ['svr_dev'")[0])
b = T[(T.vr >= 1.5) & ~(T.dte <= 3) & ~(T.vwx > 1) & T.lv].copy()   # C+E+V (the screener's set)
b['f1'] = b.vr >= 2.5; b['f2'] = b.svr20 <= .532; b['f3'] = b.dte > 10; b['f4'] = b.vwx <= .5; b['f5'] = b.m1 == 1; b['score'] = b[['f1', 'f2', 'f3', 'f4', 'f5']].sum(axis=1)
print('C+E+V trades', len(b)); 
for f in ('f1', 'f2', 'f3', 'f4', 'f5'):
    print(f'  {f}: share true {b[f].mean():.2f} | mean true {(b[b[f]].ret - COST).mean() * 100:+.2f}% (n{b[f].sum()}) vs false {(b[~b[f]].ret - COST).mean() * 100:+.2f}% | win true {((b[b[f]].ret - COST) > 0).mean() * 100:.0f}% vs {((b[~b[f]].ret - COST) > 0).mean() * 100:.0f}%')
for nm, d in (('ALL', b), ('TRAIN', b[b.et < '2025-07-01']), ('TEST', b[b.et >= '2025-07-01'])):
    print(f'\n{nm}'); 
    for lab, m in (('score 0-2', d.score <= 2), ('score 3', d.score == 3), ('score 4-5', d.score >= 4)):
        x = d[m]; n = x.ret - COST; print(f'   {lab:10s} n {len(x):4d} mean {n.mean() * 100:+6.2f}% median {n.median() * 100:+5.2f}% win {(n > 0).mean() * 100:3.0f}% P>10% {(n > .1).mean() * 100:4.1f}% P<-8% {(n < -.08).mean() * 100:4.1f}%')
