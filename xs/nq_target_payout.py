"""Nov 8 2026: TARGET SIZE vs PAYOUT FREQUENCY. DECLARED BEFORE RESULTS. Lucid payouts are capped (150K average payout is already ~129 of a 135-point max) and need 5 winning
days of $250 / $150, so payouts are limited by how many winning days come, not by points per trade. A closer target trades points per trade for more winning days.
S1 system (Rule A, flat 12:00, <= 2 a day, POC-to-VAH skip, ladder, cold-streak cuts) with target 20 / 30 / 40 / 50 / 75 (current), stop 50; and target 30 with stop 30.
Report: per trade (1 contract) win rate and average, 2022-26; Lucid 50K / 150K: eval pass, paid, payouts per funded account and trading days between them, 5-account payouts."""
_s = open('xs/nq_signal_size2.py').read(); exec(_s[:_s.rindex("if __name__ == '__main__':")])
del day
_as = open('xs/nq_account_sizes.py').read(); exec(_as[_as.index('def day('):_as.index('def ev(')])
def seq_t(T, S, allow=None):
    out = {}; sub = XA.loc[order] if allow is None else XA.loc[order][allow.loc[order].values]
    for day_, g in sub.groupby('day', sort=False):
        busy = -1; nn = 0
        for idx, r in g.iterrows():
            if nn >= 2 or r.j <= busy: continue
            res = sim2(int(r.j), int(r.dir), r.k, float(T), float(S))
            if res is None: continue
            out[idx] = res; busy = res[1]; nn += 1
    return out
if __name__ == '__main__':
    for T, S in ((20, 50), (30, 50), (40, 50), (50, 50), (75, 50), (30, 30)):
        fs = seq_t(T, S); es = seq_t(T, S, ~AG['GAP']); idx = [i for i in fs if pd.Timestamp(XA.at[i, 'day']) >= pd.Timestamp('2022-01-01') and skip2.loc[i] > 0]
        n_ = np.array([fs[i][0] - 1.0 for i in idx])
        print(f'\n  target {T} / stop {S}: per trade 2022-26 win {np.mean(n_ > 0):.0%}, avg {n_.mean():+.1f} pts ({len(n_)} trades)', flush=True)
        DE = to_D2(es, eqm(.3) * skip2, 'YZV'); DF = to_D2(fs, eqm(.7) * skip2, 'YZV')
        for a in ('50K', '150K'):
            A = ACC[a]; get = make_get(A); o = [get(s) for s in range(s0, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), 5)]
            pe = np.mean([x[0] == 'pass' for x in o]); pp = np.mean([len(x[3]) > 0 for x in o]); fund = [x for x in o if x[0] == 'pass']
            npay = np.mean([len(x[3]) for x in fund]); gaps = [b[0] - a_[0] for x in fund for a_, b in zip(x[3][:-1], x[3][1:])]; avgp = np.mean([amt for x in fund for _, amt in x[3]]) / 20
            r, py, med, worst, mm = line(get, A, GREEN, 'ALL5')
            print(f'    {a}: eval pass {pe:.0%}, evals paid {pp:.0%}, funded: {npay:.1f} payouts of {avgp:.0f} pts, {np.median(gaps):.0f} trading days apart', flush=True); show('ALL5-GREEN', a, r, py, med, worst, mm)
