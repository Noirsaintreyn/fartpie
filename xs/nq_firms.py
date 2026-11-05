"""Nov 4 2026: which prop firm fits the NQ flip system? Eval -> funded -> payouts or blown, firm by firm (50K accounts). DECLARED BEFORE RESULTS.
Trade: base NQ flip (QQQ 15m vector flip 09:30-11:30, both ways, <= 2 a day, 75/50, flat 12:00, 1 pt cost), QQQ proxy -> NQ points, 5m path of every trade kept so the
  drawdown is enforced IN REAL TIME (open P&L counts; bar order assumed against us: high before low, target bars low first, stop bars high first).
  Own rule everywhere: no new trade once the day is down $1,000. Firm daily loss limits liquidate at the limit.
FIRMS (rules from help pages / rule guides, Sep-Nov 2026; prices are list prices before codes):
  TOPSTEP     eval $49 / 21 sessions, +$3,000, $2,000 EOD trailing (stops at start), $1,000 DLL, best day <= 50%;  funded $149 activation, $2,000 EOD trailing stops at $0,
              payout after 5 days >= $150 (and profit since last after the first): 50% of balance up to $2,000, 90% - $30, max loss locks at $0 after a payout.
  TOPSTEP-NAF the No Activation Fee path: eval $95 / 21 sessions, $0 activation, same rules.
  LUCID FLEX  eval $88 one-time (list $146; Rey uses code VAULT, ~$85-90), +$3,000, $2,000 EOD trailing (locks at +$100 once the close reaches +$2,100), no DLL, best day <= 50%;  funded $0 activation, same drawdown,
              no consistency, payout after 5 days >= $150 and profit in the cycle: 50% of profit up to $2,000 (min $500), 90%; max loss moves to +$100 after a payout.
  MFFU RAPID EOD  eval $157 one-time, +$3,000, $2,000 EOD trailing (locks at +$100), no DLL, best day <= 30%, >= 4 trading days;  funded $0 activation, EOD trailing locks at +$100,
              no consistency, daily payouts: everything above a $2,100 buffer (min $500; after the first, $500 profit since the last), 90%.
  MFFU BUILDER    eval $153 one-time, +$3,000, $2,000 EOD trailing (locks at +$100), no consistency;  funded $0 activation, $1,000 soft DLL, payout per cycle (>= 2 trading days,
              best day <= 50% of the cycle's profit): above the $2,100 buffer, min $500, max $2,000, 80%.
  TPT         test $170 / 21 sessions, +$3,000, $2,000 EOD trailing (stops at start), no DLL, best day <= 50%, >= 5 trading days;  PRO $130 activation, $2,000 INTRADAY trailing
              (stops at start), withdraw anything above a $2,000 buffer (only when > $250 to skip the fee), 80%.  (PRO+ upgrade not modelled.)
SIZING  eval E1 = 1 NQ or E0.5 = 5 MNQ;  funded F1 = 1 NQ | FT = 0.5 NQ when the 60-day profile is thin on the path to target, else 1 NQ | F0.5 = 5 MNQ.
OUTPUT  per eval bought (start on every session 2022 - 2025): passed; funded: blown before the 1st payout, paid; evals per first payout; net $ per eval bought;
        12-month careers (start each month 2022-01 .. 2025-10, rebuy after every fail / blow): % paid, median and 10th-percentile net;  one career Jan 2022 -> Oct 2026."""
_s = open('xs/nq_vp_nodes.py').read(); exec(_s[:_s.index("R = run_l(C); Z = ")])
import multiprocessing as mp
def run_p(Cc, T=75, S=50, per_day=2, flat_m=720, cost=1.0):
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for idx, j, d, kk in zip(g.index, g.j.values, g.dir.values, g.k.values):
            if n >= per_day or j <= busy: continue
            e = o5[j]; path = []; q_end = None
            for q in range(j, len(H5)):
                if d5[q] != d5[j] or m5[q] >= flat_m: path.append((c5[q - 1] - e) * d * kk); q_end = q - 1; break
                lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
                if lo_ <= -S: path += [min(hi_, T), -S]; q_end = q; break
                if hi_ >= T: path += [lo_, T]; q_end = q; break
                path += [hi_, lo_]
            if q_end is None: continue
            p = np.array(path) - cost; out.append((day, idx, p[-1], p.min(), p)); busy = q_end; n += 1
    return out
TR = run_p(C); thin = (C['AHEAD_60D'] == 'LVN path ahead')
D_ = days[days >= P0]; dpos = {d: i for i, d in enumerate(D_)}; DAY = [[] for _ in D_]
for day, idx, net, mae, p in TR:
    if day in dpos: DAY[dpos[day]].append((bool(thin.loc[idx]), net, mae, p))
NS = len(D_); print(f'{len(TR)} trades, 2022-26 sessions {NS}, trading days {sum(1 for x in DAY if x)}', flush=True)
F = {
 'TOPSTEP':        dict(fee=49, monthly=True, e_lock=0, e_dll=1000, cons=.5, min_days=0, act=149, f_lock=0, f_dll=1000, intraday=False, pay='topstep'),
 'TOPSTEP-NAF':    dict(fee=95, monthly=True, e_lock=0, e_dll=1000, cons=.5, min_days=0, act=0, f_lock=0, f_dll=1000, intraday=False, pay='topstep'),
 'LUCID FLEX':     dict(fee=88, monthly=False, e_lock=100, e_dll=None, cons=.5, min_days=0, act=0, f_lock=100, f_dll=None, intraday=False, pay='lucid'),
 'MFFU RAPID EOD': dict(fee=157, monthly=False, e_lock=100, e_dll=None, cons=.3, min_days=4, act=0, f_lock=100, f_dll=None, intraday=False, pay='mffu_rapid'),
 'MFFU BUILDER':   dict(fee=153, monthly=False, e_lock=100, e_dll=None, cons=None, min_days=0, act=0, f_lock=100, f_dll=1000, intraday=False, pay='mffu_builder'),
 'TPT':            dict(fee=170, monthly=True, e_lock=0, e_dll=None, cons=.5, min_days=5, act=130, f_lock=0, f_dll=None, intraday=True, pay='tpt'),
}
def day_run(k, st, size, dll, intraday, mll=2000):
    """one session; st = dict(bal, floor, peak, lock). returns (pnl, breached)"""
    cum = 0.0; bal = st['bal']
    for th, net, mae, p in DAY[k]:
        if cum <= -1000: break
        m = 20.0 * size(th)
        if intraday:
            for x in p:
                eq = bal + cum + x * m
                if not st['fixed']: st['ipeak'] = max(st['ipeak'], eq); st['floor'] = min(max(st['floor'], st['ipeak'] - mll), st['lock'])
                if eq <= st['floor']: return cum + x * m, True
                if dll and cum + x * m <= -dll: return -dll, False
        else:
            if bal + cum + mae * m <= st['floor']: return st['floor'] - bal, True
            if dll and cum + mae * m <= -dll: cum = -float(dll); break
        cum += net * m
    return cum, False
def new_state(lock): return dict(bal=0.0, floor=-2000.0, peak=0.0, ipeak=0.0, lock=lock, fixed=False)
def eval_phase(f, s, end, size):
    st = new_state(f['e_lock']); fees = f['fee']; best = 0.0; td = 0
    for k in range(s, end):
        if f['monthly'] and k > s and (k - s) % 21 == 0: fees += f['fee']
        pnl, br = day_run(k, st, size, f['e_dll'], False); td += bool(DAY[k])
        if br: return 'fail', k, fees
        st['bal'] += pnl; best = max(best, pnl)
        if st['bal'] >= 3000 and td >= f['min_days'] and (f['cons'] is None or best <= f['cons'] * st['bal']): return 'pass', k, fees
        st['peak'] = max(st['peak'], st['bal']); st['floor'] = min(max(st['floor'], st['peak'] - 2000), st['lock'])
    return 'open', end, fees
def funded(f, s, end, size):
    st = new_state(f['f_lock']); pays = []; wins = td = 0; since = best = 0.0
    for k in range(s, end):
        pnl, br = day_run(k, st, size, f['f_dll'], f['intraday'])
        if br: return 'lost', k, pays
        st['bal'] += pnl; since += pnl; best = max(best, pnl); wins += pnl >= 150; td += bool(DAY[k]); b = st['bal']
        if not f['intraday'] and not st['fixed']: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - 2000), st['lock'])
        amt = 0.0; P = f['pay']
        if P == 'topstep' and wins >= 5 and b > 0 and (not pays or since > 0): amt = min(.5 * b, 2000.0); amt = amt if amt >= 125 else 0.0; split, fee_, newfloor = .9, 30, 0.0
        elif P == 'lucid' and wins >= 5 and since > 0: amt = min(.5 * b, 2000.0); amt = amt if amt >= 500 else 0.0; split, fee_, newfloor = .9, 0, 100.0
        elif P == 'mffu_rapid' and b - 2100 >= 500 and (not pays or since >= 500): amt = b - 2100; split, fee_, newfloor = .9, 0, None
        elif P == 'mffu_builder' and td >= 2 and since > 0 and best <= .5 * since and b - 2100 >= 500: amt = min(b - 2100, 2000.0); split, fee_, newfloor = .8, 0, None
        elif P == 'tpt' and b - 2000 > 250: amt = b - 2000; split, fee_, newfloor = .8, 0, None
        if amt > 0:
            pays.append((k, split * amt - fee_)); st['bal'] -= amt; wins = td = 0; since = best = 0.0
            if newfloor is not None: st['floor'] = newfloor; st['fixed'] = True
    return 'open', end, pays
def career(f, s, end, es, fs):
    k = s; fees = got = 0.0; evals = blown = 0; ev_first = None
    while k < end:
        evals += 1; res, k2, fe = eval_phase(f, k, end, es); fees += fe
        if res == 'fail': k = k2 + 1; continue
        if res == 'open': break
        fees += f['act']; fr, k3, pays = funded(f, k2 + 1, end, fs)
        for _, a in pays:
            got += a
            if ev_first is None: ev_first = evals
        if fr == 'lost': blown += 1; k = k3 + 1; continue
        break
    return dict(net=got - fees, paid=ev_first is not None, ev_first=ev_first, evals=evals, blown=blown, got=got, fees=fees)
SZE = {'E1': lambda th: 1.0, 'E0.5': lambda th: .5}
SZF = {'F1': lambda th: 1.0, 'FT': lambda th: .5 if th else 1.0, 'F0.5': lambda th: .5}
def job(arg):
    fn, en, fsn = arg; f = F[fn]; es, fs = SZE[en], SZF[fsn]; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); A = []
    for s in range(last):
        res, k, fees = eval_phase(f, s, NS, es); r = dict(res=res, fees=fees, npay=0, got=0.0, fres=None)
        if res == 'pass':
            r['fees'] += f['act']; fr, k2, pays = funded(f, k + 1, NS, fs); r.update(fres=fr, npay=len(pays), got=sum(a for _, a in pays))
        A.append(r)
    A = pd.DataFrame(A); Fd = A[A.res == 'pass']
    starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
    K = pd.DataFrame([career(f, s, min(s + 252, NS), es, fs) for s in starts]); full = career(f, 0, NS, es, fs); p = (A.npay > 0).mean()
    return dict(firm=fn, eval=en, funded=fsn, passed=(A.res == 'pass').mean(), blown0=((Fd.fres == 'lost') & (Fd.npay == 0)).mean(), paid=(Fd.npay > 0).mean(),
                per_pay=1 / p if p else np.inf, net_eval=(A.got - A.fees).mean(), fee_eval=A.fees.mean(), k_paid=K.paid.mean(), k_med=K.net.median(), k_p10=K.net.quantile(.1),
                k_mean=K.net.mean(), k_evfirst=K.ev_first.median(), full=full['net'], full_evals=full['evals'], full_blown=full['blown'])
if __name__ == '__main__':
    jobs = [(fn, en, fsn) for fn in F for en in SZE for fsn in SZF]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job, jobs)
    T = pd.DataFrame(RES); T.to_csv('/tmp/claude-0/ml/nq_firms.csv', index=False)
    for _, r in T.iterrows():
        print(f'  {r.firm:<15} {r.eval:<4} {r.funded:<4} | eval passed {r.passed:4.0%} avg fees {r.fee_eval:4.0f} | funded blown before 1st payout {r.blown0:4.0%} paid {r.paid:4.0%} | evals per 1st payout {r.per_pay:5.1f} | net/eval {r.net_eval:+6,.0f} | 12m paid {r.k_paid:4.0%} median {r.k_med:+6,.0f} 10th {r.k_p10:+6,.0f} mean {r.k_mean:+6,.0f} | Jan22-Oct26 net {r.full:+7,.0f} ({r.full_evals} evals, {r.full_blown} blown)', flush=True)
