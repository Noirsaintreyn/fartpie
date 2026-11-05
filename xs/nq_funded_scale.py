"""Nov 4 2026: bigger FUNDED days (Rey: aim for ~$400 days, not $150). Eval fixed (final: Rule A, gap-gated, Yang-Zhang + VWAP ladder, x0.3 cold streaks). DECLARED BEFORE RESULTS.
Funded variants (Lucid 50K: 2 minis allowed from the start, 4 once profit >= $2,000; max loss $2,000 EOD trailing locking at +$100, moves to +$100 after a payout):
  F0 x0.7 cold-streak ladder (current) | F1 SCALE size x min(2, max(1, cushion / $2,000)), cushion = balance - floor at the day's start
  F2 STEP x1.5 when cushion >= $3,000, x2 when >= $4,500 | F3 x1.5 always | F4 stop for the day after a winning trade | F5 F2 + F4
Report: funded paid, payouts per funded, average payout, $ per funded account, avg $ on trading days, share of trading days >= $150 / >= $400, blown before 1st payout, net per eval, 12-month median."""
_rt = open('xs/nq_reach_tp.py').read(); exec(_rt[:_rt.rindex("if __name__ == '__main__':")])
MODE = dict(scale=None, stopwin=False, flat=1.0)
def dayf(k, st, f):
    cum = 0.0; bal = st['bal']; cush = max(bal - st['floor'], 0.0)
    if MODE['scale'] == 'scale': cf = min(2.0, max(1.0, cush / 2000.0))
    elif MODE['scale'] == 'step': cf = 2.0 if cush >= 4500 else (1.5 if cush >= 3000 else 1.0)
    else: cf = 1.0
    cf *= MODE['flat']
    for th, net, mae, p in DAY[k]:
        if cum <= -f['pstop']: break
        mic = min(40, max(1, int(round(10 * th[0] * cf)))); m = 2.0 * mic
        if bal + cum + mae * m <= st['floor']: return st['floor'] - bal, True
        if f['dll'] and cum + mae * m <= -f['dll']: cum = -float(f['dll']); break
        cum += net * m
        if MODE['stopwin'] and net > 0: break
    return cum, False
def fu2(f, s, end, D):
    global DAY
    DAY = D; st = dict(bal=0.0, floor=-2000.0, peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0; dlog = []
    for k in range(s, end):
        pnl, br = dayf(k, st, f)
        if DAY[k]: dlog.append(pnl)
        if br: return 'lost', k, pays, dlog
        st['bal'] += pnl; since += pnl; wins += pnl >= f['win_thr']; b = st['bal']
        if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - 2000), f['lock'])
        if wins >= 5 and b > 0 and (not pays or since > 0):
            amt = min(.5 * b, f['cap'])
            if amt >= f['minpay']: pays.append((k, f['split'] * amt - f['pfee'])); st['bal'] -= amt; st['floor'] = float(f['post_floor']); fixed = True; wins = 0; since = 0.0
    return 'open', end, pays, dlog
def run_var(fn):
    f = TOP if fn == 'TOPSTEP' else LUC; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); A = []; days_ = []
    for s in range(last):
        res, k, fees = evS(f, s, NS, DS['e_EQ20 x0.3']); r = dict(res=res, fees=fees + (f['act'] if res == 'pass' else 0), npay=0, got=0.0, fres=None)
        if res == 'pass':
            fr, k2, pays, dl = fu2(f, k + 1, NS, DS['f_T75']); r.update(fres=fr, npay=len(pays), got=sum(a for _, a in pays), blown0=(fr == 'lost' and not pays)); days_ += dl
        A.append(r)
    A = pd.DataFrame(A); Fd = A[A.res == 'pass']; dd = np.array(days_)
    starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]; K = []
    for s0 in starts:
        k = s0; end = min(s0 + 252, NS); fees = got = 0.0
        while k < end:
            res, k2, fe = evS(f, k, end, DS['e_EQ20 x0.3']); fees += fe
            if res == 'fail': k = k2 + 1; continue
            if res == 'open': break
            fees += f['act']; fr, k3, pays, _ = fu2(f, k2 + 1, end, DS['f_T75']); got += sum(a for _, a in pays)
            if fr == 'lost': k = k3 + 1; continue
            break
        K.append(got - fees)
    K = np.array(K)
    return dict(paid=(Fd.npay > 0).mean(), npay=Fd.npay.mean(), avgpay=Fd.got.sum() / max(Fd.npay.sum(), 1), perf=Fd.got.mean(), blown0=Fd.blown0.mean(), dayavg=dd.mean(), d150=np.mean(dd >= 150), d400=np.mean(dd >= 400),
                net_eval=(A.got - A.fees).mean(), k_med=np.median(K), k_pos=np.mean(K > 0))
VAR = {'F0 x0.7 ladder (current)': dict(scale=None, stopwin=False, flat=1.0), 'F1 scale with cushion (to 2x)': dict(scale='scale', stopwin=False, flat=1.0),
       'F2 step 1.5x / 2x at $3k / $4.5k cushion': dict(scale='step', stopwin=False, flat=1.0), 'F3 1.5x always': dict(scale=None, stopwin=False, flat=1.5),
       'F4 stop the day after a winner': dict(scale=None, stopwin=True, flat=1.0), 'F5 step + stop after a winner': dict(scale='step', stopwin=True, flat=1.0)}
def job16(a):
    global MODE
    nm, fn = a; MODE = VAR[nm]; return dict(run_var(fn), nm=nm, fn=fn)
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job16, jobs)
    print('FUNDED SIZING (eval fixed: final system with x0.3 cold-streak cut)', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<40} {r["fn"]:<9} | funded paid {r["paid"]:4.0%} | blown before 1st payout {r["blown0"]:4.0%} | payouts/funded {r["npay"]:3.1f} | avg payout ${r["avgpay"]:,.0f} | $ per funded {r["perf"]:+6,.0f} '
              f'| trading days: avg {r["dayavg"]:+5.0f}, >= $150 {r["d150"]:4.0%}, >= $400 {r["d400"]:4.0%} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%}', flush=True)
