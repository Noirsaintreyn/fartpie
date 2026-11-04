"""Nov 4 2026 follow-up (declared before running): AMT open-location alignment as an eval rule for the NQ AM trade. ALIGNED = long after an open above the prior value area
or short after an open below it. Rules: A1 aligned only (filter) | A2 aligned 1 NQ, others 0.5 NQ | A3 skip 'against' (long after an open below value / short after an open above), others 1 NQ.
Fixed 75/50 and AMT-level TP; realistic months 2022 - Aug 2026 + 2026; coin-flip direction for A1 / A3 (10 draws)."""
_s = open('xs/nq_size_tp.py').read(); exec(_s[:_s.index("print('\\nby context")])
rng = np.random.default_rng(71)
def go(nm, Cc, tcol, mult=None, coin=False):
    R = run_t(Cc.assign(ctx=Cc.aligned.map({True: 'al', False: 'no'})), tcol); m = {'al': 1, 'no': 1} if mult is None else mult
    show(nm, R, m)
    if coin:
        pr = []
        for _ in range(10):
            Z = Cc.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); Rz = run_t(Z.assign(ctx='al'), tcol); M_ = months((Rz.net * 20).groupby(Rz.day).sum()); pr.append(M_.res.eq('passed').mean())
        print(f'      coin-flip direction, same entries: months passed {np.median(pr):.0%}', flush=True)
for tcol, tn in (('T0', 'fixed 75'), ('T4', 'AMT prior POC/VAH/VAL')):
    print(f'\n== TP {tn}', flush=True)
    go('base (all signals, 1 NQ)', C, tcol, coin=True)
    go('A1 aligned only', C[C.aligned], tcol, coin=True)
    go('A2 aligned 1 / others 0.5', C, tcol, {'al': 1, 'no': .5})
    go('A3 skip against', C[~C.against], tcol, coin=True)
