"""Nov 9 2026: THREE ACCOUNT-LEVEL LEVERS. DECLARED BEFORE RESULTS. Base: S6 (profit algo + rules C + TPO + V15a), Lucid Flex 5-account fleet, 24 monthly starts
Jan 2022 - Dec 2023, budget 40 fees, buy one eval a day while under 5 and the trailing 20-day P&L is positive (xs/fleet_compare2.py).
1) PAYOUT CUSHION: request 50% of the balance up to the cap (now) | 25% | the $500 minimum; each also with HALF SIZE after a payout until balance - floor is back to
   0.75 x max loss (eval unchanged).
2) DECORRELATED ACCOUNTS: profit engines split by account, rotating as evals are bought: A = Rule A + both overnight legs, B = noise momentum, C = V15a + 80% rule;
   profit engines at 1.5x their S6 weight on their account; the rules algo on EVERY account at 1x (it makes the qualifying days). Also: split with the rules algo only on B.
3) SHOCK DAYS: S6's worst 5% of days (book P&L, 50K). Pre-open candidates, flagged at their top decile on 2018-21: |gap| / ADR, overnight range / ADR, 5-day change in
   QQQ ATM IV, 1-day Yang-Zhang pct, prior-day range / ADR; plus FOMC and big 8:30 days. The candidate with the highest worst-day lift on 2018-21 (>= 20 flagged days)
   becomes the guard: RTH engines (Rule A, noise, 80% rule, rules algo, V15a) at half size / skipped on flagged days (overnight legs are already on). 2022+ lift reported.
   Also a tighter combined daily stop: -$600 x size instead of -$1,000."""
_fc2 = open('/home/claude/fartpie/xs/fleet_compare2.py').read(); exec(_fc2[:_fc2.rindex("if __name__ == '__main__':")])
def run_day_x(L, k, st, mll, lock, scale, mult, dstop):
    cum = 0.0; bal = st['bal']
    for th, net, mae, p in L[k]:
        if cum <= -dstop * scale: break
        mm = 2.0 * max(1, int(round(10 * th[0] * scale * mult)))
        if bal + cum + mae * mm <= st['floor']: return st['floor'] - bal, True
        cum += net * mm
    return cum, False
def eval_x(L, A, s, end, dstop):
    st = dict(bal=0.0, floor=-A['mll'], peak=0.0); best = 0.0
    for k in range(s, end):
        pnl, br = run_day_x(L, k, st, A['mll'], 100.0, A['scale'], 1.0, dstop)
        if br: return 'fail', k
        st['bal'] += pnl; best = max(best, pnl)
        if st['bal'] >= A['target'] and best <= .5 * st['bal']: return 'pass', k
        if st['floor'] < 100: st['peak'] = max(st['peak'], st['bal']); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
    return 'open', end
def fund_x(L, A, s, end, frac, half, dstop):
    st = dict(bal=0.0, floor=-A['mll'], peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0; mult = 1.0
    for k in range(s, end):
        pnl, br = run_day_x(L, k, st, A['mll'], 100.0, A['scale'], mult, dstop)
        if br: return 'lost', k, pays
        st['bal'] += pnl; since += pnl; wins += pnl >= A['qday']; b = st['bal']
        if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
        if half and mult < 1 and b - st['floor'] >= .75 * A['mll']: mult = 1.0
        if wins >= 5 and b > 0 and (not pays or since > 0):
            amt = min(frac * b if frac else 500.0, A['cap'])
            if amt >= 500 and b >= amt:
                pays.append((k, .9 * amt)); st['bal'] -= amt; st['floor'] = 100.0; fixed = True; wins = 0; since = 0.0
                if half: mult = .5
                if len(pays) >= 5: return 'graduated', k, pays
    return 'open', end, pays
ACCX = {'50K': dict(mll=2000, target=3000, scale=1.0, qday=150, cap=2000, fee=88), '150K': dict(mll=4500, target=9000, scale=2.25, qday=250, cap=3000, fee=244)}
def multi_x(types, zn, frac=.5, half=False, dstop=1000):
    A = ACCX[zn]; fee = A['fee']; LL = {t: (lists(sp, 'e'), lists(sp, 'f')) for t, sp in types.items()}; tk = list(types)
    pnl = sum(pd.Series([sum(net * 2 * max(1, round(10 * th[0] * A['scale'])) for th, net, mae, p in zz) for zz in LL[t][0]]) for t in tk) / len(tk)
    hot = (pnl.rolling(20).sum().shift(1) > 0).values; TRJ = {}
    def traj(k, t):
        if (k, t) not in TRJ:
            LE_, LF_ = LL[t]; r, k2 = eval_x(LE_, A, k, NS, dstop)
            if r != 'pass': TRJ[(k, t)] = (k2 if r == 'fail' else NS, [])
            else: fr, k3, pays = fund_x(LF_, A, k2 + 1, NS, frac, half, dstop); TRJ[(k, t)] = (NS if fr == 'open' else k3, pays)
        return TRJ[(k, t)]
    starts = [int(np.searchsorted(D_, p.start_time)) for p in pd.period_range('2022-01', '2023-12', freq='M')]; allN = []; dry = []; ev_n = pay_n = 0
    for k0 in starts:
        cash = 40 * fee; act = []; pq = []; P = {}; Fe = {}; Cn = {}; nb = 0
        for k in range(k0, NS):
            mo = D_[k].to_period('M')
            while pq and pq[0][0] == k:
                kk, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x; Cn[mo] = Cn.get(mo, 0) + 1
            act = [e for e in act if e >= k]
            if len(act) < 5 and cash >= fee and hot[k]:
                end, pays = traj(k, tk[nb % len(tk)]); nb += 1; cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                for kk, x in pays: heapq.heappush(pq, (kk, x))
        months = pd.period_range(D_[k0].to_period('M'), D_[NS - 2].to_period('M'), freq='M')
        N = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months}); C_ = pd.Series({mo: Cn.get(mo, 0) for mo in months}); allN.append(N.values)
        run = mx = 0
        for v_ in C_.values: run = run + 1 if v_ == 0 else 0; mx = max(mx, run)
        dry.append(mx)
    Av = np.concatenate(allN)
    fs = [traj(k, t) for (k, t) in TRJ]; ndone = len(fs); paid = [x for e, p_ in fs for _, x in p_]
    return dict(avg=Av.mean(), med=np.median(Av), ge25=np.mean(Av >= 2500), ge5=np.mean(Av >= 5000), lose=np.mean(Av < 0), dry=np.median(dry), avgpay=np.mean(paid) if paid else 0)
def line(nm, r): return (f'  {nm:<44} avg ${r["avg"]:6,.0f} / median ${r["med"]:6,.0f} | >= $2.5k {r["ge25"]:4.0%}, >= $5k {r["ge5"]:4.0%}, losing {r["lose"]:4.0%}, '
                         f'dry {r["dry"]:.1f} mo | avg payout ${r["avgpay"]:,.0f}')
if __name__ == '__main__':
    LOCK['on'] = False; S6 = SET['S6 S3 + V15a']
    for zn in ('150K', '50K'):
        print(f'\n=== {zn} x 5 Lucid Flex ===', flush=True)
        print('-- 1) payout cushion', flush=True)
        for frac, half, lab in ((.5, False, 'take 50% (now)'), (.25, False, 'take 25%'), (None, False, 'take the $500 minimum'),
                                (.5, True, 'take 50% + half size until rebuilt'), (.25, True, 'take 25% + half size until rebuilt'), (None, True, 'take $500 + half size until rebuilt')):
            print(line(lab, multi_x({'S6': S6}, zn, frac, half)), flush=True)
        print('-- 2) decorrelated accounts (rotate A / B / C as evals are bought)', flush=True)
        Aw = [(RAs, 1.5, RULE['RA']), (OBpos, .6, RULE['ON+']), (OBneg, .15, RULE['ON-'])]; Bw = [(M1gg, .45, RULE['M1'])]; Cw = [(V15a, .45, None), (R1g, .75, RULE['R1'])]; Rw = [(RC, 1.0, None)]
        print(line('split, rules algo on every account', multi_x({'A': Aw + Rw, 'B': Bw + Rw, 'C': Cw + Rw}, zn)), flush=True)
        print(line('split, rules algo only on B', multi_x({'A': Aw, 'B': Bw + Rw, 'C': Cw}, zn)), flush=True)
    # ---- 3) shock days ----
    print('\n=== 3) shock days: S6 book, one 50K account ===', flush=True)
    names = ['RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES', 'V15a']; book = sum(pd.Series([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in lists([s_], 'f')], index=D_) for s_ in S6)
    a_ = D_ < pd.Timestamp('2022-01-01'); worst = book <= book[a_].quantile(.05); worst_b = book <= book[~a_].quantile(.05)
    feat = pd.DataFrame(index=D_)
    feat['gap'] = [abs(DY.o[d] - DY.c.shift(1)[d]) / DY.adr[d] for d in D_]; feat['onr'] = [NSO['NR'].get(d, np.nan) / DY.adr[d] for d in D_]
    feat['iv5'] = [fget(d, 'iv5') for d in D_]; feat['yz1'] = [DY.yz1p.get(d, np.nan) for d in D_]; feat['prng'] = [((DY.h - DY.l).shift(1)[d]) / DY.adr[d] for d in D_]
    flags = {c_: (feat[c_] >= feat[c_][a_].quantile(.9)).values for c_ in feat}
    flags['FOMC'] = np.array([d in FOMC for d in D_]); flags['big 8:30'] = np.array([NW.r830.get(d, 0) >= 3 for d in D_])
    best = None
    for c_, f_ in flags.items():
        la = worst[a_ & f_].mean() / worst[a_].mean() if (a_ & f_).sum() >= 20 else np.nan; lb = worst_b[~a_ & f_].mean() / worst_b[~a_].mean() if (~a_ & f_).sum() else np.nan
        print(f'  {c_:<9} flagged {f_[a_].mean():4.0%} / {f_[~a_].mean():4.0%} of days | worst-day lift {la:4.2f} (2018-21) / {lb:4.2f} (2022+) | book on flagged days '
              f'${book[a_ & f_].mean():+5.0f} / ${book[~a_ & f_].mean():+5.0f} (other days ${book[a_ & ~f_].mean():+4.0f} / ${book[~a_ & ~f_].mean():+4.0f})', flush=True)
        if np.isfinite(la) and (best is None or la > best[1]): best = (c_, la)
    gc = best[0]; gflag = pd.Series(flags[gc], index=D_); print(f'  GUARD = {gc} (2018-21 lift {best[1]:.2f})', flush=True)
    def guard(T_, mult):
        T_ = T_.copy()
        if 'sz' not in T_: T_['sz'], T_['ce'], T_['cf'] = 1.0, 1.0, 1.0
        msk = T_.day.map(gflag).fillna(False).values.astype(bool); T_.loc[msk, 'sz'] = T_.loc[msk, 'sz'] * mult; return T_
    for zn in ('150K', '50K'):
        for mult, lab in ((.5, f'half size RTH on {gc} days'), (0.0, f'skip RTH on {gc} days')):
            sp = [(guard(T_, mult) if nm in ('RA', 'M1', 'R1', 'RULES', 'V15a') else T_, u, yz) for nm, (T_, u, yz) in zip(names, S6)]
            print(line(f'{zn} {lab}', multi_x({'S6g': sp}, zn)), flush=True)
        print(line(f'{zn} daily stop -$600 x size', multi_x({'S6': S6}, zn, dstop=600)), flush=True)
        print(line(f'{zn} S6 as is', multi_x({'S6': S6}, zn)), flush=True)
