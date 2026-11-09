"""Nov 9 2026: LUCID FLEX 50K - EVALS PER PAYOUT, WORST STRETCHES, WORST MONTHS, AND WHETHER HAR + YANG-ZHANG SIZING HELPS. DECLARED BEFORE RESULTS.
Variants (all on S6 = profit algo + rules C + TPO + V15a; Yang-Zhang tercile sizing per engine is already inside S6):
  V0 S6 as is | V1 S6 + skip RTH engines on IV-jump days (QQQ ATM IV 5-day change in its top decile, 2018-21 cut; xs/fleet_levers.py)
  V2 V1 + HAR sizing, RELATIVE: HAR forecast of tomorrow's RTH range (log range on log of yesterday's, last week's and last month's average range; fit Feb 2014 - Dec 2019),
     size x clip(median forecast of the trailing year / forecast, 0.5, 1.5) - smaller when NQ is about to be wilder than usual
  V3 V1 + HAR sizing, ABSOLUTE dollar risk: size x clip(K / forecast, 0.4, 1.5), K = the 2018-21 median forecast (so later, wilder years trade smaller in contracts)
  V4 V1 + funded LOCK: once a funded day is >= $150 (a qualifying day), no more RTH trades that day     V5 V2 + LOCK     V6 V3 + LOCK
Measured, Lucid Flex 50K ($88): per eval started every 2nd trading day Feb 2018 - Jun 2025 - pass rate, share of evals that reach a payout, payouts per eval, EVALS PER PAYOUT,
$ per eval; ONE ACCOUNT AT A TIME Feb 2018 - Jan 2026 (next eval the day after one ends) - evals per payout, LONGEST RUN OF EVALS WITHOUT A PAYOUT, months with no payout,
worst 12 months; 5-ACCOUNT FLEET (24 monthly starts 2022-23, 40 fees) - average / median / losing months; and the fleet from Feb 2018: worst month and payouts by year.
HAR check: how well HAR vs ADR20 vs Yang-Zhang 20d predicts tomorrow's log range (R2), fit 2014-19, judged 2020-25."""
_fl = open('/home/claude/fartpie/xs/fleet_levers.py').read(); exec(_fl[:_fl.rindex("if __name__ == '__main__':")])
from sklearn.linear_model import LinearRegression
R_ = (DY.h - DY.l).astype(float); lr = np.log(R_.replace(0, np.nan))
HX = pd.DataFrame({'d': lr.shift(1), 'w': np.log(R_.rolling(5).mean().shift(1)), 'm': np.log(R_.rolling(22).mean().shift(1))}); ok_ = HX.notna().all(axis=1) & lr.notna()
fit = (DY.index >= '2014-02-01') & (DY.index < '2020-01-01') & ok_; har = LinearRegression().fit(HX[fit], lr[fit])
FC = pd.Series(np.nan, index=DY.index); FC[ok_] = np.exp(har.predict(HX[ok_]))
relm = (FC.rolling(252, min_periods=120).median() / FC).clip(.5, 1.5); K = FC[(FC.index >= '2018-01-01') & (FC.index < '2022-01-01')].median(); absm = (K / FC).clip(.4, 1.5)
G_IV = (pd.Series([fget(d, 'iv5') for d in DY.index], index=DY.index)); cut_iv = G_IV[(G_IV.index >= '2018-02-01') & (G_IV.index < '2022-01-01')].quantile(.9); IVJ = G_IV >= cut_iv
RTH = {'RA', 'M1', 'R1', 'RULES', 'V15a'}; NAMES = ['RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES', 'V15a']
def variant(skip_iv=True, hmult=None):
    out = []
    for nm, (T_, u, yz) in zip(NAMES, SET['S6 S3 + V15a']):
        T_ = T_.copy()
        if 'sz' not in T_: T_['sz'], T_['ce'], T_['cf'] = 1.0, 1.0, 1.0
        dser = pd.to_datetime(T_.day)
        if hmult is not None: T_['sz'] = T_['sz'] * dser.map(hmult).fillna(1.0).values
        if skip_iv and nm in RTH: T_.loc[dser.map(IVJ).fillna(False).values.astype(bool), 'sz'] = 0.0
        if nm in RTH: T_['m'] = T_['m'].clip(lower=0)
        out.append((T_, u, yz))
    return out
def run_day_l(L, k, st, mll, scale, mult, dstop, lock):
    cum = 0.0; bal = st['bal']
    for th, net, mae, p in L[k]:
        if cum <= -dstop * scale or (lock is not None and cum >= lock * scale): break
        mm = 2.0 * max(1, int(round(10 * th[0] * scale * mult)))
        if bal + cum + mae * mm <= st['floor']: return st['floor'] - bal, True
        cum += net * mm
    return cum, False
def fund_l(L, A, s, end, lock):
    st = dict(bal=0.0, floor=-A['mll'], peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0
    for k in range(s, end):
        pnl, br = run_day_l(L, k, st, A['mll'], A['scale'], 1.0, 1000, lock)
        if br: return 'lost', k, pays
        st['bal'] += pnl; since += pnl; wins += pnl >= A['qday']; b = st['bal']
        if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
        if wins >= 5 and b > 0 and (not pays or since > 0):
            amt = min(.5 * b, A['cap'])
            if amt >= 500:
                pays.append((k, .9 * amt)); st['bal'] -= amt; st['floor'] = 100.0; fixed = True; wins = 0; since = 0.0
                if len(pays) >= 5: return 'graduated', k, pays
    return 'open', end, pays
if __name__ == '__main__':
    LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']
    for lab, X_ in (('HAR forecast', np.log(FC)), ('ADR20', np.log(DY.adr)), ('YZ 20d pct (rank)', DY.yz20p)):
        mk = (DY.index >= '2020-01-01') & X_.notna() & lr.notna(); r2 = np.corrcoef(X_[mk], lr[mk])[0, 1] ** 2
        print(f'  predicting tomorrow\'s log RTH range 2020-25: {lab:<18} R2 {r2:.2f}', flush=True)
    VAR = {'V0 S6 as is': (variant(False), None), 'V1 + IV-jump skip': (variant(True), None), 'V2 V1 + HAR relative': (variant(True, relm), None),
           'V3 V1 + HAR absolute $': (variant(True, absm), None), 'V4 V1 + funded lock $150': (variant(True), 150), 'V5 V2 + lock': (variant(True, relm), 150),
           'V6 V3 + lock': (variant(True, absm), 150)}
    print(f'\n  {"variant":<26} | per eval (2018-25 starts): pass, paid, payouts/eval, EVALS PER PAYOUT, $/eval | one at a time 2018-26: evals/payout, LONGEST NO-PAYOUT RUN (evals), '
          f'months w/o payout, worst 12 mo | fleet 2022-23 starts: avg / median, losing | fleet from 2018: worst month, by year', flush=True)
    for nm, (sp, lock) in VAR.items():
        LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = eval_x(LE_, A, k, NS, 1000)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, lock); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
            return TRJ[k]
        ST = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)); tr_ = [traj(s) for s in ST]
        pas = np.mean([t[0] == 'pass' for t in tr_]); paid = np.mean([len(t[2]) > 0 for t in tr_]); ppe = np.mean([len(t[2]) for t in tr_]); usd = np.mean([sum(x for _, x in t[2]) for t in tr_])
        s = 0; chain = [];
        while s < NS - 1:
            r, end, pays = traj(s); chain.append((s, pays)); s = end + 1
        npay = sum(len(p_) for _, p_ in chain); run = mx = 0
        for _, p_ in chain: run = run + 1 if not p_ else 0; mx = max(mx, run)
        months = pd.period_range(D_[0].to_period('M'), D_[NS - 2].to_period('M'), freq='M'); PM = pd.Series(0.0, index=months)
        for _, p_ in chain:
            for k_, x in p_: PM[D_[k_].to_period('M')] += x
        FM = pd.Series(0.0, index=months)
        for s_, _ in chain: FM[D_[s_].to_period('M')] += fee
        net12 = (PM - FM).rolling(12).sum().min()
        globals()['SET']['_tmp'] = sp
        fl = multi_x({'x': sp}, '50K') if lock is None else None
        # fleet from Feb 2018, single path, with lock support
        hot = (pd.Series([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in LE_]).rolling(20).sum().shift(1) > 0).values
        cash = 40 * fee; act = []; pq = []; P = {}; Fe = {}
        for k in range(0, NS):
            mo = D_[k].to_period('M')
            while pq and pq[0][0] == k:
                kk, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x
            act = [e for e in act if e >= k]
            if len(act) < 5 and cash >= fee and hot[k]:
                r, end, pays = traj(k); cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                for kk, x in pays: heapq.heappush(pq, (kk, x))
        N = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months})
        yrs = ' '.join(f'{y % 100:02d}:{N[N.index.year == y].mean() / 1000:.1f}k' for y in sorted(set(months.year)) if y < 2026)
        flt = f'${fl["avg"]:5,.0f} / ${fl["med"]:5,.0f}, losing {fl["lose"]:3.0%}' if fl else 'see from-2018 path'
        print(f'  {nm:<26} | {pas:3.0%}, {paid:3.0%}, {ppe:.2f}, {1 / ppe if ppe else np.inf:4.1f}, ${usd:5,.0f} | {len(chain) / max(npay, 1):4.1f}, {mx:2d}, {np.mean(PM == 0):3.0%}, '
              f'${net12:7,.0f} | {flt} | worst month ${N.min():6,.0f}, avg ${N.mean():5,.0f}/mo: {yrs}', flush=True)
