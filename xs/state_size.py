"""Nov 9 2026: STATE-DEPENDENT SIZING - SIZE EACH DAY BY THE ACCOUNT'S ROOM, NOT ONLY THE SETUP (the account policy as the strategy). DECLARED BEFORE RESULTS.
System: the current 50K setup (V2 + skip slow flips), Lucid Flex 50K ($88; +$3,000 target, $2,000 EOD trailing floor that locks at +$100, best day <= 50% of profit in the
eval; funded: 5 days >= $150, payout 50% of balance capped at $2,000, floor at +$100 after the first payout). Every trade's size x a multiplier set at the start of the day
from the state: cushion c = (balance - floor) / $2,000 (1.0 at the start), progress p = balance / $3,000.
EVAL policies:  E0 x1 | E1 de-risk when hurt: x clip(c, 0.4, 1) | E2 harder: x clip(c^2, 0.3, 1) | E3 press when ahead: x1.3 once p >= 0.5 | E4 protect near the target:
                x0.6 once p >= 0.7 | E5 E1 + E3
FUNDED policies: F0 x1 | F1 x clip(c, 0.4, 1) | F2 x clip(c, 0.5, 1.5) (also bigger with room) | F3 x0.5 after a payout until c >= 0.75 (the earlier 'half' lever) |
                 F4 x clip(c, 0.3, 1.25)
CHOSEN ON evals started 2018-21 (best E by payouts per eval with F0; best F by $ per eval with the chosen E), JUDGED on evals started 2022-25 (Jul) and the 5-account
fleet (24 monthly starts 2022-23; and one path from 2018). Integer micros, at least 1."""
_fc = open('/home/claude/fartpie/xs/flip_char.py').read(); exec(_fc[:_fc.rindex("if __name__ == '__main__':")])
def eval_p(L, A, s, end, pol):
    st = dict(bal=0.0, floor=-A['mll'], peak=0.0); best = 0.0
    for k in range(s, end):
        cu = (st['bal'] - st['floor']) / A['mll']; pr = st['bal'] / A['target']
        pnl, br = run_day_x(L, k, st, A['mll'], 100.0, A['scale'], pol(cu, pr), 1000)
        if br: return 'fail', k
        st['bal'] += pnl; best = max(best, pnl)
        if st['bal'] >= A['target'] and best <= .5 * st['bal']: return 'pass', k
        if st['floor'] < 100: st['peak'] = max(st['peak'], st['bal']); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
    return 'open', end
def fund_p(L, A, s, end, pol):
    st = dict(bal=0.0, floor=-A['mll'], peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0
    for k in range(s, end):
        cu = (st['bal'] - st['floor']) / A['mll']
        pnl, br = run_day_x(L, k, st, A['mll'], 100.0, A['scale'], pol(cu, len(pays)), 1000)
        if br: return 'lost', k, pays
        st['bal'] += pnl; since += pnl; wins += pnl >= A['qday']; b = st['bal']
        if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
        if wins >= 5 and b > 0 and (not pays or since > 0):
            amt = min(.5 * b, A['cap'])
            if amt >= 500:
                pays.append((k, .9 * amt)); st['bal'] -= amt; st['floor'] = 100.0; fixed = True; wins = 0; since = 0.0
                if len(pays) >= 5: return 'graduated', k, pays
    return 'open', end, pays
EP = {'E0 x1': lambda c_, p_: 1.0, 'E1 de-risk when hurt': lambda c_, p_: float(np.clip(c_, .4, 1)), 'E2 harder de-risk': lambda c_, p_: float(np.clip(c_ ** 2, .3, 1)),
      'E3 press when ahead': lambda c_, p_: 1.3 if p_ >= .5 else 1.0, 'E4 protect near target': lambda c_, p_: .6 if p_ >= .7 else 1.0,
      'E5 E1 + E3': lambda c_, p_: float(np.clip(c_, .4, 1)) * (1.3 if p_ >= .5 else 1.0)}
_half = {}
FP = {'F0 x1': lambda c_, n_: 1.0, 'F1 x clip(c, .4, 1)': lambda c_, n_: float(np.clip(c_, .4, 1)), 'F2 x clip(c, .5, 1.5)': lambda c_, n_: float(np.clip(c_, .5, 1.5)),
      'F3 half after a payout until c >= .75': lambda c_, n_: .5 if (n_ > 0 and c_ < .75) else 1.0, 'F4 x clip(c, .3, 1.25)': lambda c_, n_: float(np.clip(c_, .3, 1.25))}
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
    LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']; LE_, LF_ = lists(base, 'e'), lists(base, 'f')
    hot = (pd.Series([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in LE_]).rolling(20).sum().shift(1) > 0).values
    k22, k25 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))), int(np.searchsorted(D_, pd.Timestamp('2025-07-01')))
    SA, SB = list(range(0, k22, 2)), list(range(k22, k25, 2))
    def make(ep, fp):
        TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = eval_p(LE_, A, k, NS, ep)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                else: fr, k3, pays = fund_p(LF_, A, k2 + 1, NS, fp); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
            return TRJ[k]
        return traj
    def per_eval(traj, S):
        tr_ = [traj(s) for s in S]; ppe = np.mean([len(t[2]) for t in tr_])
        return dict(pas=np.mean([t[0] == 'pass' for t in tr_]), paid=np.mean([len(t[2]) > 0 for t in tr_]), ppe=ppe, usd=np.mean([sum(x for _, x in t[2]) for t in tr_]) - fee)
    def fleets(traj):
        out = []
        for k0 in [int(np.searchsorted(D_, p.start_time)) for p in pd.period_range('2022-01', '2023-12', freq='M')] + [0]:
            cash = 40 * fee; act = []; pq = []; P = {}; Fe = {}
            for k in range(k0, NS):
                mo = D_[k].to_period('M')
                while pq and pq[0][0] == k:
                    kk_, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x
                act = [e for e in act if e >= k]
                if len(act) < 5 and cash >= fee and hot[k]:
                    r, end, pays = traj(k); cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                    for kk_, x in pays: heapq.heappush(pq, (kk_, x))
            months = pd.period_range(D_[k0].to_period('M'), D_[NS - 2].to_period('M'), freq='M'); out.append(pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months}))
        allv = np.concatenate([s_.values for s_ in out[:-1]]); f18 = out[-1]
        return dict(avg=allv.mean(), med=np.median(allv), lose=np.mean(allv < 0), f18=f18.mean(), f18l=np.mean(f18 < 0), f18w=f18.min())
    def chain_dry(traj):
        s = 0; run = mx = 0
        while s < NS - 1:
            r, end, pays = traj(s); run = run + 1 if not pays else 0; mx = max(mx, run); s = end + 1
        return mx
    print('=== EVAL policies (funded x1) | evals started 2018-21 (CHOOSE): pass, paid, payouts per eval, net $ per eval | started 2022-25 (JUDGE) ===', flush=True)
    RE = {}
    for en_, ep in EP.items():
        tj = make(ep, FP['F0 x1']); a, b = per_eval(tj, SA), per_eval(tj, SB); RE[en_] = (a, b, tj)
        print(f'  {en_:<26} {a["pas"]:3.0%}, {a["paid"]:3.0%}, {a["ppe"]:.2f}, ${a["usd"]:5,.0f} | {b["pas"]:3.0%}, {b["paid"]:3.0%}, {b["ppe"]:.2f}, ${b["usd"]:5,.0f}', flush=True)
    be = max(EP, key=lambda k_: RE[k_][0]['ppe']); print(f'  chosen on 2018-21: {be}', flush=True)
    print(f'\n=== FUNDED policies (eval = {be}) | 2018-21 starts: paid, payouts per eval, net $ per eval | 2022-25 starts ===', flush=True)
    RF = {}
    for fn_, fp in FP.items():
        tj = make(EP[be], fp); a, b = per_eval(tj, SA), per_eval(tj, SB); RF[fn_] = (a, b, tj)
        print(f'  {fn_:<38} {a["paid"]:3.0%}, {a["ppe"]:.2f}, ${a["usd"]:5,.0f} | {b["paid"]:3.0%}, {b["ppe"]:.2f}, ${b["usd"]:5,.0f}', flush=True)
    bf = max(FP, key=lambda k_: RF[k_][0]['usd']); print(f'  chosen on 2018-21: {bf}', flush=True)
    print('\n=== judged: 5-account fleets | 24 starts 2022-23: avg / median a month, losing months | one path from 2018: avg a month, losing, worst | longest run w/o payout ===', flush=True)
    for lab, tj in (('now (E0 + F0)', RE['E0 x1'][2]), (f'{be} + F0', RE[be][2]), (f'E0 + {bf}', make(EP['E0 x1'], FP[bf])), (f'chosen: {be} + {bf}', RF[bf][2])):
        f_ = fleets(tj); pe = per_eval(tj, SB)
        print(f'  {lab:<52} ${f_["avg"]:,.0f} / ${f_["med"]:,.0f}, losing {f_["lose"]:.0%} | ${f_["f18"]:,.0f}, losing {f_["f18l"]:.0%}, worst ${f_["f18w"]:,.0f} | dry {chain_dry(tj)} | '
              f'2022-25 starts: pass {pe["pas"]:.0%}, paid {pe["paid"]:.0%}, ${pe["usd"]:,.0f}/eval', flush=True)
