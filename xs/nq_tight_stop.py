"""Nov 8 2026: LEVERAGE AS A TIGHTER STOP (more contracts for the same dollar risk) on high-quality trades. DECLARED BEFORE RESULTS.
Quality Q = the 4 pillars (energy, participation, location, timing; xs/nq_quality_prop.py). PROP trades: Rule A, 75 target, flat 12:00, <= 2 a day, re-simulated with each stop.
Part 1, per trade (1 contract, after 1 pt cost): stop 25 / 30 / 35 / 40 / 50 for Q 0-2 and Q 3-4: average, win rate, and R = average / stop
  ('points at equal risk' = average x 50 / stop = what the same dollar risk earns with more contracts), 2018-21 and 2022-26.
Part 2, Lucid (5 accounts on GREEN days, Yang-Zhang + VWAP ladder, x0.3 / x0.7 cold streaks, POC-to-VAH skip, size capped at 1.75x so 150K stays under 40 micros):
  S1 stop 50 (current)            T1 Q3+ stop 30, same contracts          T2 Q3+ stop 30, x1.67 contracts (same $ risk)       T3 Q3+ stop 25, x1.75
  T4 every trade stop 30, x1.67   T5 Q3+ x1.5 contracts, stop 50          T6 Q0-2 stop 30 same contracts, Q3+ stop 50         T7 T6 + Q3+ x1.5"""
_q = open('xs/nq_quality_prop.py').read(); exec(_q[:_q.rindex("if __name__ == '__main__':")])
def seq_s(S, allow=None):
    out = {}; sub = XA.loc[order] if allow is None else XA.loc[order][allow.loc[order].values]
    for day, g in sub.groupby('day', sort=False):
        busy = -1; nn = 0
        for idx, r in g.iterrows():
            if nn >= 2 or r.j <= busy: continue
            res = sim2(int(r.j), int(r.dir), r.k, 75.0, float(S.loc[idx]))
            if res is None: continue
            out[idx] = res; busy = res[1]; nn += 1
    return out
hiQ = (Q >= 3); cs = lambda v: pd.Series(float(v), index=XA.index)
def stops(lo, hi_): return pd.Series(np.where(hiQ, hi_, lo), index=XA.index, dtype=float)
def mults(lo, hi_): return pd.Series(np.minimum(np.where(hiQ, hi_, lo), 1.75), index=XA.index) * skip2
V = {'S1 stop 50 (current)': (cs(50), mults(1, 1)), 'T1 Q3+ stop 30, same contracts': (stops(50, 30), mults(1, 1)), 'T2 Q3+ stop 30, x1.67 (same $ risk)': (stops(50, 30), mults(1, 50 / 30)),
     'T3 Q3+ stop 25, x1.75': (stops(50, 25), mults(1, 1.75)), 'T4 every trade stop 30, x1.67': (cs(30), mults(50 / 30, 50 / 30)), 'T5 Q3+ x1.5 contracts, stop 50': (cs(50), mults(1, 1.5)),
     'T6 Q0-2 stop 30, Q3+ stop 50': (stops(30, 50), mults(1, 1)), 'T7 T6 + Q3+ x1.5': (stops(30, 50), mults(1, 1.5))}
if __name__ == '__main__':
    print('PART 1: prop trades, 1 contract, per trade (n, avg, win, R = avg/stop, pts at equal risk = avg x 50/stop)', flush=True)
    for S in (25, 30, 35, 40, 50):
        res = seq_s(cs(S)); idx = list(res.keys()); net = pd.Series([res[i][0] - 1.0 for i in idx], index=idx); yy = pd.DatetimeIndex(XA.loc[idx, 'day']).year; qq = Q.loc[idx].values
        out = []
        for lab, mq in (('Q0-2', qq <= 2), ('Q3-4', qq >= 3)):
            for plab, mp in (('18-21', yy < 2022), ('22-26', yy >= 2022)):
                g = net.values[mq & mp]; out.append(f'{lab} {plab}: n{len(g):3d} {g.mean():+6.1f} win {np.mean(g > 0):3.0%} R {g.mean() / S:+.2f} eq {g.mean() * 50 / S:+6.1f}')
        print(f'  stop {S}: ' + ' | '.join(out), flush=True)
    for nm, (S, mult) in V.items():
        ev_s = seq_s(S, ~AG['GAP']); full_s = seq_s(S); DE = to_D2(ev_s, eqm(.3) * mult, 'YZV'); DF = to_D2(full_s, eqm(.7) * mult, 'YZV'); print(f'\n  {nm}', flush=True)
        for a in ('50K', '150K'):
            A = ACC[a]; get = make_get(A); o = [get(s) for s in range(s0, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), 5)]
            pe = np.mean([x[0] == 'pass' for x in o]); pp = np.mean([len(x[3]) > 0 for x in o]); r, py, med, worst, mm = line(get, A, GREEN, 'ALL5')
            print(f'    {a}: single eval pass {pe:.0%}, evals paid {pp:.0%}', flush=True); show('ALL5-GREEN', a, r, py, med, worst, mm)
