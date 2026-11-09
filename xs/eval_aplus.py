"""Nov 9 2026: FIX THE EVAL - TIGHTER STOPS AND MORE SIZE ON A+ TRADES, EVAL PHASE ONLY (Rey). DECLARED BEFORE RESULTS.
Funded phase untouched (the current 50K setup: V2 + skip slow flips). Lucid Flex 50K eval: +$3,000 target, $2,000 EOD trailing drawdown, best day <= 50% of profit.
A+ = Rule A and V15a flips with a FAST move into the flip (30-min move >= the top-third cut of 2018-21, ~0.44 ADR) - the slice that paid in both periods (xs/flip_char.py).
Eval variants:
  E1 A+ x1.5 | E2 A+ x2 (per-trade cap 1.75 NQ stays) | E3 tighter stops: Rule A 35 (was 50, target 75 kept), V15a 0.10 ADR (was 0.15) |
  E4 E3 + A+ x1.5 | E5 E3 + A+ x2 (about the same $ risk per A+ trade as now) | E6 eval takes ONLY A+ Rule A / V15a trades, x1.5 (other engines as now)
Report: per-trade points with the tighter stops; eval pass rate and median days to pass; then the whole account life: paid at least once, evals per payout, longest run
without a payout (one at a time), 5-account fleet from 2018 and the 24-start fleet 2022-23."""
_w = open('/home/claude/fartpie/xs/winners.py').read(); exec(_w[:_w.rindex("if __name__ == '__main__':")])
import re
_fl = open('/home/claude/fartpie/xs/fleet_levers.py').read(); _mx = _fl[_fl.index('def multi_x('):]; _mx = _mx[:_mx.index('\ndef ', 5) if '\ndef ' in _mx[5:] else _mx.index("if __name__")]
exec(_mx.replace('def multi_x(', 'def multi_x2(').replace("LL = {t: (lists(sp, 'e'), lists(sp, 'f')) for t, sp in types.items()}", "LL = {t: (lists(sp[0], 'e'), lists(sp[1], 'f')) for t, sp in types.items()}"))
if __name__ == '__main__':
    V = run_sig2('s_vector'); Vg = V[~hig(V)].copy(); RA_ = RAs[(RAs.sz > 0)].copy()
    REF = V[V.day >= '2018-02-01'].join(feats(V[V.day >= '2018-02-01'])); ra_ = REF[pd.to_datetime(REF.day).dt.year < 2022].mv30
    cut, top = ra_.quantile(1 / 3), ra_.quantile(2 / 3); SL = {}; AP = {}
    for key, T_ in (('V15a', Vg), ('RA', RA_)):
        F = T_.join(feats(T_)); SL[key] = set(zip(pd.to_datetime(F[F.mv30 < cut].day), F[F.mv30 < cut].m)); AP[key] = set(zip(pd.to_datetime(F[F.mv30 >= top].day), F[F.mv30 >= top].m))
    base = []
    for en, (T_, u, yz) in zip(NAMES, variant(True, relm)):
        T_ = T_.copy()
        if en in SL: T_.loc[np.array([(pd.Timestamp(d), mm_) in SL[en] for d, mm_ in zip(T_.day, T_.m)]), 'sz'] = 0.0
        base.append((T_, u, yz))
    TIGHT = {}
    print(f'A+ cut: 30-min move into the flip >= {top:.3f} ADR', flush=True)
    for en in ('RA', 'V15a'):
        T_ = dict(zip(NAMES, base))[en][0]; out = []
        for r_ in T_.itertuples():
            d = pd.Timestamp(r_.day); ix = gi.get(d); w = np.where(m[ix] == r_.m)[0] if ix is not None else []
            if r_.sz <= 0 or not len(w): out.append((r_.pts, r_.mae)); continue
            a_ = DY.adr[d]; kw = dict(S0=35, T=75, until=720, tf=None) if en == 'RA' else dict(S0=.10 * a_, tf=15)
            pt, mae, *_ = path(d, ix[w[0]], r_.dir, **kw); out.append((pt, mae))
        TIGHT[en] = np.array(out); X = T_.assign(p2=TIGHT[en][:, 0]); X = X[(X.sz > 0) & (pd.to_datetime(X.day) >= '2018-02-01')]; per = pd.to_datetime(X.day).dt.year >= 2022
        isap = np.array([(pd.Timestamp(d), mm_) in AP[en] for d, mm_ in zip(X.day, X.m)])
        print(f'  {en}: now {X[~per].pts.mean():+6.2f} / {X[per].pts.mean():+6.2f} | tighter stop {X[~per].p2.mean():+6.2f} / {X[per].p2.mean():+6.2f} pts per trade (2018-21 / 2022-25) | '
              f'A+ trades: {isap.mean():.0%}, now {X[isap & ~per].pts.mean():+.2f} / {X[isap & per].pts.mean():+.2f}, tighter {X[isap & ~per].p2.mean():+.2f} / {X[isap & per].p2.mean():+.2f}', flush=True)
    def ev_spec(tight=False, up=1.0, only=False):
        out = []
        for en, (T_, u, yz) in zip(NAMES, base):
            T_ = T_.copy()
            if en in ('RA', 'V15a'):
                if tight: T_['pts'], T_['mae'] = TIGHT[en][:, 0], TIGHT[en][:, 1]
                isap = np.array([(pd.Timestamp(d), mm_) in AP[en] for d, mm_ in zip(T_.day, T_.m)])
                T_['ce'] = T_.ce * np.where(isap, up, 0.0 if only else 1.0)
            out.append((T_, u, yz))
        return out
    VAR = {'E0 now': ev_spec(), 'E1 A+ x1.5': ev_spec(up=1.5), 'E2 A+ x2': ev_spec(up=2.0), 'E3 tighter stops': ev_spec(tight=True), 'E4 tighter + A+ x1.5': ev_spec(True, 1.5),
           'E5 tighter + A+ x2': ev_spec(True, 2.0), 'E6 only A+ (RA / V15a), x1.5': ev_spec(False, 1.5, True)}
    LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']; LF_ = lists(base, 'f')
    print('\n=== Lucid Flex 50K: eval changed, funded as now ===', flush=True)
    for lab, spe in VAR.items():
        LE_ = lists(spe, 'e'); TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = eval_x(LE_, A, k, NS, 1000)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [], None)
                else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, None); TRJ[k] = (r, NS if fr == 'open' else k3, pays, k2)
            return TRJ[k]
        ST = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)); tr_ = [(s, traj(s)) for s in ST]
        pas = np.mean([t[0] == 'pass' for _, t in tr_]); dp = np.median([t[3] - s + 1 for s, t in tr_ if t[0] == 'pass']); paid = np.mean([len(t[2]) > 0 for _, t in tr_]); ppe = np.mean([len(t[2]) for _, t in tr_])
        pa = {y: np.mean([t[0] == 'pass' for s, t in tr_ if D_[s].year == y]) for y in range(2018, 2026)}
        s = 0; chain = []
        while s < NS - 1:
            r, end, pays, _ = traj(s); chain.append(pays); s = end + 1
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
                r, end, pays, _ = traj(k); cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                for kk_, x in pays: heapq.heappush(pq, (kk_, x))
        N_ = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months}); C = pd.Series({mo: NP.get(mo, 0) for mo in months}); fl = multi_x2({'x': (spe, base)}, '50K')
        print(f'  {lab:<30} EVAL pass {pas:3.0%} (median {dp:.0f} days) | paid {paid:3.0%}, evals/payout {1 / ppe:4.2f}, dry {mx} | fleet 2018+: {C.mean():.2f} pay/mo, losing {np.mean(N_ < 0):3.0%}, '
              f'${N_.mean():,.0f}/mo | 24 starts 2022-23: avg ${fl["avg"]:,.0f} / med ${fl["med"]:,.0f}, losing {fl["lose"]:.0%}', flush=True)
        print('      pass rate by start year: ' + ' '.join(f'{y % 100:02d}:{v:.0%}' for y, v in pa.items()), flush=True)
