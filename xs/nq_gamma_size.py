"""Nov 4 2026, POST-HOC: gamma SIZING on top of CUT-A (nq_structure_size.py: skip thin 60-day path / hard break / monthly LVN, then 1 / 0.5 / 0.3 by HARH + VWAPX).
POS = QQQ net gamma >= 0 at the prior close. Sizing in eval AND funded, Topstep + Lucid Flex sims (real-time drawdown).
  CUT-A | CUT-A x0.5 on positive gamma | CUT-A x1.5 on negative / x0.5 on positive | CUT-A with positive gamma as a third risk flag | S0 x1.5 negative / x0.5 positive"""
_ss = open('xs/nq_structure_size.py').read(); pre = _ss[:_ss.rindex("if __name__ == '__main__':")]
pre = pre.replace("FL = ['THIN', 'HARH', 'VWAPX', 'HARD', 'STRONG', 'LVN', 'HVN', 'MID20', 'EXTD', 'WEAK']", "X['POS'] = ~X.neg.astype(bool); FL = ['THIN', 'HARH', 'VWAPX', 'HARD', 'STRONG', 'LVN', 'HVN', 'MID20', 'EXTD', 'WEAK', 'POS']")
assert "X['POS']" in pre; exec(pre)
post_ = Z[Z.day >= P0]; print('per trade by gamma, all trades: ' + ' | '.join(f'{p} neg {Z[(Z.per == p) & ~Z.POS].net.mean():+.1f} / pos {Z[(Z.per == p) & Z.POS].net.mean():+.1f}' for p in ('18-21', '22-23', '24-26')), flush=True)
VARS = {'CUT-A': ('cutA', CUT), 'CUT-A x0.5 positive gamma': ('cutA', lambda th: CUT(th) * (.5 if th[10] else 1.0)),
        'CUT-A x1.5 neg / x0.5 pos': ('cutA', lambda th: CUT(th) * (.5 if th[10] else 1.5)),
        'CUT-A, pos gamma as risk flag': ('cutA', lambda th: (1.0, .5, .3)[min(th[1] + th[2] + th[10], 2)]),
        'S0 x1.5 neg / x0.5 pos': ('all', lambda th: S0(th) * (.5 if th[10] else 1.5))}
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VARS for fn in ('TOPSTEP', 'LUCID FLEX')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job3, jobs)
    print('\nFIRM SIMS, sizing in eval and funded', flush=True)
    for r in RES:
        print(f'  {r["var"]:<30} {r["firm"]:<11} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m paid {r["k_paid"]:4.0%} median {r["k_med"]:+6,.0f} 10th {r["k_p10"]:+6,.0f}', flush=True)
