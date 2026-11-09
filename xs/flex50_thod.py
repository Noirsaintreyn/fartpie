"""Nov 9 2026: REY'S H/L-ANCHORED (THOD/TLOD) VOLATILITY vs HAR FOR SIZING - Lucid Flex 50K. DECLARED BEFORE RESULTS.
THOD sigma, ported from backend.compute_session_volatility + calculate_hurst_exponent + fractional_brownian_adjustment (thod_bot_match.bot_sigma): Garman-Klass variance
averaged over the last 60 completed bars (1h, or 4h = clock-aligned 4h bars), sigma = sqrt x price, times the Hurst multiplier (H > 0.6: up to 1.3x; H < 0.4: down to 0.8x;
Hurst from the last 102 closes, max lag 20). Built from real NQ futures 24h bars (ETH + RTH). As-of times: 09:30 for RTH engines, 18:00 the evening before for the
overnight legs (only completed bars). Same two sizing rules as HAR (xs/flex50.py): RELATIVE size x clip(trailing-year median / sigma, 0.5, 1.5) and ABSOLUTE size x
clip(K / sigma, 0.4, 1.5), K = 2018-21 median. All on V1 (S6 + IV-jump skip). Also: how well each predicts the day's RTH log range (2020-25), as of 09:30."""
_f5 = open('/home/claude/fartpie/xs/flex50.py').read(); exec(_f5[:_f5.rindex("if __name__ == '__main__':")])
X1 = X.set_index('dt')[['open', 'high', 'low', 'close']].resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
X4 = X1.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
def bot_sig(Bk, t):
    k = Bk.index.searchsorted(t, 'left'); b = Bk.iloc[max(0, k - 102):k]
    if len(b) < 40: return np.nan
    r = b.iloc[-60:]; o_, h_, l_, c_ = r.open.values, r.high.values, r.low.values, r.close.values
    var = np.maximum(.5 * np.log(h_ / l_) ** 2 - (2 * np.log(2) - 1) * np.log(c_ / o_) ** 2, 1e-9); s = np.sqrt(var.mean()) * c_[-1]
    cl = b.close.values; lags = range(2, 20); tau = [np.std(cl[g:] - cl[:-g]) for g in lags]; H_ = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    mult = 1.0 + .3 * (H_ - .5) / .5 if H_ > .6 else (1.0 - .2 * (.5 - H_) / .5 if H_ < .4 else 1.0); return s * mult if abs(mult - 1) * 2 > .1 else s
SIG = {}
for k_, d in enumerate(DY.index):
    if d < pd.Timestamp('2014-03-01'): continue
    t930 = d + pd.Timedelta(hours=9, minutes=30); prev = DY.index[k_ - 1] if k_ else d; t18 = prev + pd.Timedelta(hours=18)
    SIG[d] = (bot_sig(X1, t930), bot_sig(X4, t930), bot_sig(X1, t18), bot_sig(X4, t18))
SG = pd.DataFrame(SIG, index=['h1_rth', 'h4_rth', 'h1_on', 'h4_on']).T.sort_index()
def mults(s):
    rel = (s.rolling(252, min_periods=120).median() / s).clip(.5, 1.5); K_ = s[(s.index >= '2018-01-01') & (s.index < '2022-01-01')].median(); return rel, (K_ / s).clip(.4, 1.5)
def variant2(rth_m, on_m):
    out = []
    for nm, (T_, u, yz) in zip(NAMES, SET['S6 S3 + V15a']):
        T_ = T_.copy()
        if 'sz' not in T_: T_['sz'], T_['ce'], T_['cf'] = 1.0, 1.0, 1.0
        dser = pd.to_datetime(T_.day); mm = on_m if nm in ('ON+', 'ON-') else rth_m
        T_['sz'] = T_['sz'] * dser.map(mm).fillna(1.0).values
        if nm in RTH: T_.loc[dser.map(IVJ).fillna(False).values.astype(bool), 'sz'] = 0.0
        out.append((T_, u, yz))
    return out
if __name__ == '__main__':
    LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']
    for lab, X_ in (('HAR (prior close)', np.log(FC)), ('ADR20', np.log(DY.adr)), ('THOD 1h sigma @09:30', np.log(SG.h1_rth.reindex(DY.index))), ('THOD 4h sigma @09:30', np.log(SG.h4_rth.reindex(DY.index)))):
        mk = (DY.index >= '2020-01-01') & X_.notna() & lr.notna(); print(f'  predicting the day\'s log RTH range 2020-25: {lab:<22} R2 {np.corrcoef(X_[mk], lr[mk])[0, 1] ** 2:.2f}', flush=True)
    hr, ha = relm, absm; t1r, t1a = mults(SG.h1_rth); t4r, t4a = mults(SG.h4_rth); o1r, o1a = mults(SG.h1_on); o4r, o4a = mults(SG.h4_on)
    VAR = {'V1 (S6 + IV skip), no vol sizing': variant2(pd.Series(dtype=float), pd.Series(dtype=float)), 'HAR relative': variant2(hr, hr), 'HAR absolute $': variant2(ha, ha),
           'THOD 1h relative': variant2(t1r, o1r), 'THOD 1h absolute $': variant2(t1a, o1a), 'THOD 4h relative': variant2(t4r, o4r), 'THOD 4h absolute $': variant2(t4a, o4a)}
    print(f'\n  {"variant":<34} | per eval: pass, paid, EVALS PER PAYOUT, $/eval | one at a time 2018-26: evals/payout, LONGEST NO-PAYOUT RUN | fleet 2022-23: avg / median, losing, dry | from 2018: worst month, avg', flush=True)
    for nm, sp in VAR.items():
        LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = eval_x(LE_, A, k, NS, 1000)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, None); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
            return TRJ[k]
        ST = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)); tr_ = [traj(s) for s in ST]
        pas = np.mean([t[0] == 'pass' for t in tr_]); paid = np.mean([len(t[2]) > 0 for t in tr_]); ppe = np.mean([len(t[2]) for t in tr_]); usd = np.mean([sum(x for _, x in t[2]) for t in tr_])
        s = 0; chain = []
        while s < NS - 1:
            r, end, pays = traj(s); chain.append(pays); s = end + 1
        npay = sum(len(p_) for p_ in chain); run = mx = 0
        for p_ in chain: run = run + 1 if not p_ else 0; mx = max(mx, run)
        fl = multi_x({'x': sp}, '50K')
        hot = (pd.Series([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in LE_]).rolling(20).sum().shift(1) > 0).values
        cash = 40 * fee; act = []; pq = []; P = {}; Fe = {}; months = pd.period_range(D_[0].to_period('M'), D_[NS - 2].to_period('M'), freq='M')
        for k in range(0, NS):
            mo = D_[k].to_period('M')
            while pq and pq[0][0] == k:
                kk, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x
            act = [e for e in act if e >= k]
            if len(act) < 5 and cash >= fee and hot[k]:
                r, end, pays = traj(k); cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                for kk, x in pays: heapq.heappush(pq, (kk, x))
        N = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months})
        print(f'  {nm:<34} | {pas:3.0%}, {paid:3.0%}, {1 / ppe if ppe else np.inf:4.1f}, ${usd:5,.0f} | {len(chain) / max(npay, 1):4.1f}, {mx:2d} | ${fl["avg"]:5,.0f} / ${fl["med"]:5,.0f}, '
              f'{fl["lose"]:3.0%}, {fl["dry"]:.1f} mo | ${N.min():6,.0f}, ${N.mean():5,.0f}/mo', flush=True)
