"""Nov 9 2026: ADAPTIVE ENGINES - does an engine's last 5 / 10 / 20 / 50 / 100 / 200 trades predict its next one? (Rey). DECLARED BEFORE RESULTS.
Setup: V2 + skip slow flips (the current 50K setup), each engine's own trade record in time order (all signals it would take, kept as a shadow record even when the
gate is off, so the gate can turn back on). Warm-up uses the engines' trades before 2018 where they exist; judged Feb 2018 on, 2018-21 / 2022-25.
1) Does recent form predict? per engine and lookback N: points per next trade when the last-N average is > 0 vs <= 0; rank correlation of the last-N average with the
   next trade.
2) Rules, same N for every engine:  G1 ON / OFF - take the trade only if the engine's last-N average > 0 | G2 SIZE - x1.5 when > 0, x0.5 when <= 0.
   Weighted points a year (pts x size) and the worst drawdown of the combined daily P&L, 2018-21 / 2022-25, for every N.
3) N for G1 and G2 picked on 2018-21 weighted points -> Lucid Flex 50K: per eval 2018-25 and the fleets (from 2018; 24 starts 2022-23 = out of sample)."""
_fc = open('/home/claude/fartpie/xs/cut_bottom.py').read(); exec(_fc[:_fc.rindex("if __name__ == '__main__':")])
from scipy.stats import spearmanr
NS_ = (5, 10, 20, 50, 100, 200)
if __name__ == '__main__':
    V = run_sig2('s_vector'); Vg = V[~hig(V)].copy(); RA_ = RAs[(RAs.sz > 0)].copy()
    REF = V[V.day >= '2018-02-01'].join(feats(V[V.day >= '2018-02-01'])); cut = REF[pd.to_datetime(REF.day).dt.year < 2022].mv30.quantile(1 / 3); SL = {}
    for key, T_ in (('V15a', Vg), ('RA', RA_)):
        F = T_.join(feats(T_)); SL[key] = set(zip(pd.to_datetime(F[F.mv30 < cut].day), F[F.mv30 < cut].m))
    base = []
    for en, (T_, u, yz) in zip(NAMES, variant(True, relm)):
        T_ = T_.copy()
        if en in SL: T_.loc[np.array([(pd.Timestamp(d), mm_) in SL[en] for d, mm_ in zip(T_.day, T_.m)]), 'sz'] = 0.0
        base.append((T_, u, yz))
    print('trade history per engine: ' + ', '.join(f'{en} from {pd.to_datetime(T_[T_.sz > 0].day).min().date()} ({int((T_.sz > 0).sum())})' for en, (T_, u, yz) in zip(NAMES, base)), flush=True)
    ROLL = {}
    for en, (T_, u, yz) in zip(NAMES, base):
        X = T_[T_.sz > 0].copy(); X['_d'] = pd.to_datetime(X.day); X = X.sort_values(['_d', 'm'])
        ROLL[en] = {N: X.pts.shift(1).rolling(N, min_periods=N).mean() for N in NS_}; ROLL[en]['X'] = X
    print('\n1) next-trade points when the engine\'s last-N average is > 0 vs <= 0 (n) | rank corr | 2018-21 ; 2022-25', flush=True)
    for en in NAMES:
        X = ROLL[en]['X']; per = np.where(X._d.dt.year >= 2022, 'b', np.where(X._d >= '2018-02-01', 'a', '-')); cells = []
        for N in NS_:
            r_ = ROLL[en][N]; parts = []
            for p_ in 'ab':
                mk = (per == p_) & r_.notna().values; up, dn = X.pts[mk & (r_ > 0).values], X.pts[mk & (r_ <= 0).values]
                rc = spearmanr(r_[mk], X.pts[mk])[0] if mk.sum() > 20 else np.nan
                parts.append(f'{up.mean():+5.1f}({len(up)})/{dn.mean():+5.1f}({len(dn)}) r{rc:+.2f}')
            cells.append(f'N{N}: ' + ' ; '.join(parts))
        print(f'  {en}:', flush=True)
        for cl_ in cells: print(f'    {cl_}', flush=True)
    def gated(N, kind):
        out = []
        for en, (T_, u, yz) in zip(NAMES, base):
            T_ = T_.copy(); r_ = ROLL[en][N].reindex(T_.index); on = ~(r_ <= 0)
            if kind == 'G1': T_.loc[~on.values, 'sz'] = 0.0
            else: T_['sz'] = T_.sz * np.where(on.values, 1.5, .5)
            out.append((T_, u, yz))
        return out
    def book(sp):
        rows = []
        for en, (T_, u, yz) in zip(NAMES, sp):
            X = T_[(T_.sz > 0) & (pd.to_datetime(T_.day) >= '2018-02-01')]; rows.append(pd.DataFrame({'d': pd.to_datetime(X.day), 'w': X.pts * u * X.sz}))
        Dp = pd.concat(rows).groupby('d').w.sum(); res = []
        for a_, b_ in (('2018-02-01', '2022-01-01'), ('2022-01-01', '2026-02-01')):
            s_ = Dp[(Dp.index >= a_) & (Dp.index < b_)]; eq = s_.cumsum(); res += [s_.sum() / ((pd.Timestamp(b_) - pd.Timestamp(a_)).days / 365.25), (eq - eq.cummax()).min()]
        return res
    print('\n2) weighted points a year (pts x size) | worst drawdown of the combined daily P&L | 2018-21 ; 2022-25', flush=True)
    b0 = book(base); print(f'  now (no adaptive)        {b0[0]:+6.0f} / DD {b0[1]:6.0f} ; {b0[2]:+6.0f} / DD {b0[3]:6.0f}', flush=True); SC = {}
    for kind in ('G1', 'G2'):
        for N in NS_:
            r = book(gated(N, kind)); SC[(kind, N)] = r
            print(f'  {kind} {"on/off" if kind == "G1" else "size  "} N{N:<4}          {r[0]:+6.0f} / DD {r[1]:6.0f} ; {r[2]:+6.0f} / DD {r[3]:6.0f}', flush=True)
    pick = {k: max(NS_, key=lambda N: SC[(k, N)][0]) for k in ('G1', 'G2')}; print(f'  picked on 2018-21: G1 N{pick["G1"]}, G2 N{pick["G2"]}', flush=True)
    LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']
    print('\n3) Lucid Flex 50K', flush=True)
    for lab, sp in (('V2 + skip slow flips (now)', base), (f'G1 on/off, last {pick["G1"]} trades', gated(pick['G1'], 'G1')), (f'G2 size, last {pick["G2"]} trades', gated(pick['G2'], 'G2'))):
        LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = eval_x(LE_, A, k, NS, 1000)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, None); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
            return TRJ[k]
        tr_ = [traj(s) for s in range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)]; ppe = np.mean([len(t[2]) for t in tr_]); paid = np.mean([len(t[2]) > 0 for t in tr_])
        s = 0; chain = []
        while s < NS - 1:
            r, end, pays = traj(s); chain.append(pays); s = end + 1
        run = mx = 0
        for p_ in chain: run = run + 1 if not p_ else 0; mx = max(mx, run)
        hot = (pd.Series([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in LE_]).rolling(20).sum().shift(1) > 0).values
        cash = 40 * fee; act = []; pq = []; P = {}; Fe = {}; NP = {}; months = pd.period_range(D_[0].to_period('M'), D_[NS - 2].to_period('M'), freq='M')
        for k in range(0, NS):
            mo = D_[k].to_period('M')
            while pq and pq[0][0] == k:
                kk_, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x; NP[mo] = NP.get(mo, 0) + 1
            act = [e for e in act if e >= k]
            if len(act) < 5 and cash >= fee and hot[k]:
                r, end, pays = traj(k); cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                for kk_, x in pays: heapq.heappush(pq, (kk_, x))
        N_ = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months}); C = pd.Series({mo: NP.get(mo, 0) for mo in months}); fl = multi_x({'x': sp}, '50K')
        print(f'  {lab:<30} pass {np.mean([t[0] == "pass" for t in tr_]):3.0%}, paid {paid:3.0%}, evals/payout {1 / ppe:4.2f}, dry run {mx} | fleet 2018+: {C.mean():.2f} payouts/mo, '
              f'losing {np.mean(N_ < 0):3.0%}, ${N_.mean():,.0f}/mo, worst ${N_.min():,.0f} | 24 starts 2022-23: avg ${fl["avg"]:,.0f} / median ${fl["med"]:,.0f}, losing {fl["lose"]:.0%}', flush=True)
