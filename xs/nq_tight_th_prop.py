"""Nov 8 2026: the TIGHTER-STOP TREND-HOLD inside Lucid. DECLARED BEFORE RESULTS. Trade lists from xs/nq_tight_th_trades.py (2022 - Oct 2026), engine and book as
xs/nq_gex_prop.py (5 accounts on GREEN days, eval x0.3 / funded x0.7 cold streaks), 3 micros per contract-unit. Compared with the current prop system (S1)."""
_g = open('xs/nq_gex_prop.py').read(); exec(_g[:_g.rindex("if __name__ == '__main__':")])
_b = open('xs/nq_buy_policy.py').read(); exec(_b[_b.index('def to_D2'):_b.index('one = pd.Series')]); exec(_b[_b.index('def make_get'):_b.rindex("if __name__ == '__main__':")])
TT = pd.read_parquet('/tmp/claude-0/ml/nq_tight_th_trades.parquet')
V2 = {'TH stop 50 (B0)': lists('B0', 3), 'TH stop 30, x1.67 contracts (B2)': lists('B2', 3), 'TH stop 35, x1.43 (B3)': lists('B3', 3), 'TH stop 30, same contracts (B5)': lists('B5', 3)}
if __name__ == '__main__':
    for nm, (de, df) in V2.items():
        DE, DF = de, df; print(f'\n  {nm}', flush=True)
        for a in ('50K', '150K'):
            A = ACC[a]; get = make_get(A); o = [get(s) for s in range(s0, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), 5)]
            pe = np.mean([x[0] == 'pass' for x in o]); pp = np.mean([len(x[3]) > 0 for x in o]); r, py, med, worst, mm = line(get, A, GREEN, 'ALL5')
            print(f'    {a}: single eval pass {pe:.0%}, evals paid {pp:.0%}', flush=True); show('ALL5-GREEN', a, r, py, med, worst, mm)
