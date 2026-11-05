"""Nov 4 2026: VOLATILITY stops and targets (target at a farther band than the stop), with micro sizing that holds the dollar risk. DECLARED BEFORE RESULTS.
Trades: Rule A (15m flip 09:30-11:30, skip thin 60-day path / hard break / monthly LVN), flat 12:00, 1 pt cost per NQ-equivalent, path on 5m bars (stop first inside a bar).
RISK: micros = floor(budget / (stop pts x $2)), 1..20 MNQ, budget $1,000 x the Rule A ladder (1 / 0.5 / 0.3 by HAR high + VWAP stretched).  Own day stop -$1,000.
EXITS (stop / target, all known at entry):
  F    fixed 50 / 75 pts, 10 / 5 / 3 MNQ (current)
  P1   0.30% / 0.45% of price          P2   0.25% / 0.50%
  A1   0.20 / 0.35 x daily ATR14       A2   0.15 / 0.30 x ATR   (RTH ATR at the prior close, NQ points)
  VW   session VWAP bands (VWAP +- n x volume-weighted SD since 09:30, n = -3..3): stop at the first band behind the entry, target two bands ahead of it
  ORB  opening range (09:30-10:00, or 09:30 to entry if earlier): stop at its midpoint, target one range-width beyond the range in the trade's direction
  VW and ORB distances are clipped: stop 0.12% - 0.45% of price, target at least 1.2 x the stop.
Report: per trade by year 2022 - 2026 (win, avg $, avg R, target / stop / noon), stop and target in points; Lucid 50K ($88) and Topstep sims (real-time drawdown)."""
_lz = open('xs/nq_lucid25k.py').read(); exec(_lz[:_lz.rindex("if __name__ == '__main__':")])
XA = X[~cutA].copy()
RTD = pd.DataFrame({'o': o5, 'h': h5, 'l': l5, 'c': c5, 'd': d5})[rthm].groupby('d').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last')); RTD.index = pd.DatetimeIndex(RTD.index)
pcl = RTD.c.shift(1); trd = np.maximum(RTD.h - RTD.l, np.maximum((RTD.h - pcl).abs(), (RTD.l - pcl).abs())); ATRq = trd.ewm(alpha=1 / 14, adjust=False).mean().shift(1)
rows = []
for idx, r in XA.iterrows():
    j, d, k = int(r.j), int(r.dir), r.k; e = o5[j]; px = e * k; i0 = j
    while i0 > 0 and d5[i0 - 1] == d5[j] and m5[i0 - 1] >= 570: i0 -= 1
    w = vol5[i0:j]; tpx = tp[i0:j]; vw_ = (tpx * w).sum() / w.sum() if j > i0 and w.sum() > 0 else e; sd_ = np.sqrt((w * (tpx - vw_) ** 2).sum() / w.sum()) if j > i0 and w.sum() > 0 else 0.0
    orr = slice(i0, j) if m5[j] <= 600 else slice(i0, i0 + 6); oh, ol = h5[orr].max(), l5[orr].min()
    atr_ = ATRq.get(pd.Timestamp(r.day), np.nan) * k
    bands = np.array([vw_ + n * sd_ for n in range(-3, 4)]); dist = (bands - e) * d * k
    behind = dist[dist < -0.5]; ahead = np.sort(dist[dist > 0.5])
    vws = -behind.max() if len(behind) else np.nan; vwt = ahead[1] if len(ahead) > 1 else (ahead[0] if len(ahead) else np.nan)
    mid = (oh + ol) / 2; orbs = (e - mid) * d * k; orbt = ((oh + (oh - ol)) - e) * k if d > 0 else (e - (ol - (oh - ol))) * k
    rows.append(dict(ix=idx, px=px, atr=atr_, vws=vws, vwt=vwt, orbs=orbs, orbt=orbt))
V = pd.DataFrame(rows).set_index('ix'); XA = XA.join(V)
def clipst(s, t, px):
    s = np.clip(s, .0012 * px, .0045 * px) if np.isfinite(s) else .003 * px; t = max(t, 1.2 * s) if np.isfinite(t) else 1.5 * s; return s, t
def st_for(nm, r):
    if nm == 'F': return 50.0, 75.0
    if nm == 'P1': return .003 * r.px, .0045 * r.px
    if nm == 'P2': return .0025 * r.px, .005 * r.px
    if nm == 'A1': return .20 * r.atr, .35 * r.atr
    if nm == 'A2': return .15 * r.atr, .30 * r.atr
    if nm == 'VW': return clipst(r.vws, r.vwt, r.px)
    if nm == 'ORB': return clipst(r.orbs, r.orbt, r.px)
def sim2(j, d, kk, T, S):
    e = o5[j]; path = []
    for q in range(j, len(H5)):
        if d5[q] != d5[j] or m5[q] >= 720: path.append((c5[q - 1] - e) * d * kk); return path[-1], q - 1, path, 'noon'
        lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
        if lo_ <= -S: path += [min(hi_, T), -S]; return -S, q, path, 'stop'
        if hi_ >= T: path += [lo_, T]; return T, q, path, 'target'
        path += [hi_, lo_]
    return None
SYSV = {}; STATS = {}
for nm in ('F', 'P1', 'P2', 'A1', 'A2', 'VW', 'ORB'):
    out = []; D = [[] for _ in D_]
    for day, g in XA.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for idx, r in g.iterrows():
            if n >= 2 or r.j <= busy: continue
            S, T = st_for(nm, r)
            if not (np.isfinite(S) and np.isfinite(T)) or S <= 0: continue
            res = sim2(int(r.j), int(r.dir), r.k, T, S)
            if res is None: continue
            net, qe, p, how = res; th = tuple(bool(X.at[idx, c]) for c in FL); mult = (1.0, .5, .3)[min(th[1] + th[2], 2)]
            mic = 10 * mult if nm == 'F' else int(np.clip(np.floor(1000 * mult / (S * 2)), 1, 20)); size = mic / 10
            p = np.array(p) - 1.0; out.append((day, net - 1.0, (net - 1.0) / S, how, S, T, size)); busy = qe; n += 1
            if day in dpos: D[dpos[day]].append(((size,), net - 1.0, p.min(), p))
    SYSV[nm] = D; STATS[nm] = pd.DataFrame(out, columns=['day', 'net', 'R', 'how', 'S', 'T', 'size'])
print('PER TRADE BY YEAR (Rule A)', flush=True)
for nm, T_ in STATS.items():
    T_['usd'] = T_.net * 20 * T_['size']; T_['y'] = T_.day.dt.year
    print(f'\n  {nm}: stop median {T_.S.median():.0f} pts (2022 {T_[T_.y == 2022].S.median():.0f} / 2026 {T_[T_.y == 2026].S.median():.0f}), target median {T_["T"].median():.0f}, avg size {T_["size"].mean() * 10:.1f} MNQ', flush=True)
    for y, g in T_[T_.y >= 2022].groupby('y'):
        print(f'    {y}: n {len(g):>3} win {g.net.gt(0).mean():.0%} avg ${g.usd.mean():+5.0f} avg R {g.R.mean():+.2f} | target {np.mean(g.how == "target"):.0%} stop {np.mean(g.how == "stop"):.0%} noon {np.mean(g.how == "noon"):.0%} | year ${g.usd.sum():+,.0f}', flush=True)
TOP = dict(fee=49, monthly=True, act=149, target=3000, mll=2000, lock=0, cons=.5, min_days=0, pstop=1000, dll=1000, win_thr=150, cap=2000, minpay=125, split=.9, pfee=30, post_floor=0)
LUC = dict(fee=88, monthly=False, act=0, target=3000, mll=2000, lock=100, cons=.5, min_days=0, pstop=1000, dll=None, win_thr=150, cap=2000, minpay=500, split=.9, pfee=0, post_floor=100)
def dayq(k, st, f):
    cum = 0.0; bal = st['bal']
    for th, net, mae, p in DAY[k]:
        if cum <= -f['pstop']: break
        m = 20.0 * th[0]
        if bal + cum + mae * m <= st['floor']: return st['floor'] - bal, True
        if f['dll'] and cum + mae * m <= -f['dll']: cum = -float(f['dll']); break
        cum += net * m
    return cum, False
def evq(f, s, end):
    st = dict(bal=0.0, floor=-2000.0, peak=0.0); best = 0.0; td = 0; fees = f['fee']
    for k in range(s, end):
        if f['monthly'] and k > s and (k - s) % 21 == 0: fees += f['fee']
        pnl, br = dayq(k, st, f); td += bool(DAY[k])
        if br: return 'fail', k, fees
        st['bal'] += pnl; best = max(best, pnl)
        if st['bal'] >= 3000 and td >= f['min_days'] and best <= .5 * st['bal']: return 'pass', k, fees
        st['peak'] = max(st['peak'], st['bal']); st['floor'] = min(max(st['floor'], st['peak'] - 2000), f['lock'])
    return 'open', end, fees
def fuq(f, s, end):
    st = dict(bal=0.0, floor=-2000.0, peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0
    for k in range(s, end):
        pnl, br = dayq(k, st, f)
        if br: return 'lost', k, pays
        st['bal'] += pnl; since += pnl; wins += pnl >= f['win_thr']; b = st['bal']
        if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - 2000), f['lock'])
        if wins >= 5 and b > 0 and (not pays or since > 0):
            amt = min(.5 * b, f['cap'])
            if amt >= f['minpay']: pays.append((k, f['split'] * amt - f['pfee'])); st['bal'] -= amt; st['floor'] = float(f['post_floor']); fixed = True; wins = 0; since = 0.0
    return 'open', end, pays
def carq(f, s, end):
    k = s; fees = got = 0.0; ev = npay = 0
    while k < end:
        ev += 1; res, k2, fe = evq(f, k, end); fees += fe
        if res == 'fail': k = k2 + 1; continue
        if res == 'open': break
        fees += f['act']; fr, k3, pays = fuq(f, k2 + 1, end); got += sum(a for _, a in pays); npay += len(pays)
        if fr == 'lost': k = k3 + 1; continue
        break
    return dict(fees=fees, got=got, net=got - fees, evals=ev, payouts=npay)
def job7(a):
    global DAY
    nm, fn = a; DAY = SYSV[nm]; f = TOP if fn == 'TOPSTEP' else LUC; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); A = []
    for s in range(last):
        res, k, fees = evq(f, s, NS); r = dict(res=res, fees=fees + (f['act'] if res == 'pass' else 0), npay=0, got=0.0, fres=None)
        if res == 'pass': fr, k2, pays = fuq(f, k + 1, NS); r.update(fres=fr, npay=len(pays), got=sum(a_ for _, a_ in pays))
        A.append(r)
    A = pd.DataFrame(A); Fd = A[A.res == 'pass']; p = (A.npay > 0).mean()
    starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
    K = pd.DataFrame([carq(f, s, min(s + 252, NS)) for s in starts])
    return dict(nm=nm, fn=fn, passed=(A.res == 'pass').mean(), blown0=((Fd.fres == 'lost') & (Fd.npay == 0)).mean(), per_pay=1 / p if p else np.inf, net_eval=(A.got - A.fees).mean(),
                k_med=K.net.median(), k_pos=(K.net > 0).mean(), full=carq(f, 0, NS), y26=carq(f, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), NS))
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in SYSV for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job7, jobs)
    print('\nFIRM SIMS', flush=True)
    for r in RES:
        fu, y = r['full'], r['y26']
        print(f'  {r["nm"]:<4} {r["fn"]:<9} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} '
              f'| 2022-Oct26 net {fu["net"]:+7,.0f} ({fu["evals"]} evals, {fu["payouts"]} payouts) | 2026 net {y["net"]:+6,.0f} ({y["evals"]} evals, {y["payouts"]} payouts)', flush=True)
