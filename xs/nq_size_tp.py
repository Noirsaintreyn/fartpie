"""Nov 4 2026: context SIZING (less in bad environments, more in good) and LIQUIDITY take-profits (prior-day / premarket / session highs-lows, VWAP) for the NQ NY AM trade. DECLARED BEFORE RESULTS.
Trades: 15m QQQ vector flip, signal bars closing 09:30-11:30, any day, up to 2 a day, flat 12:00, 1 pt cost, QQQ proxy -> NQ pts. Base size 1 NQ, stop 50 pts.
CONTEXT SCORE at the signal (only pieces that held up in both halves earlier): +1 each: HAR forecast in its low third | VXN within 10% of its 20d average | QQQ 20d return up |
   negative QQQ gamma | SPY 15m vector agrees;   -1 each: QQQ 20d z > 1.5 (stretched up) | VXN > 10% above its 20d average | SPY 15m vector disagrees.
   bad = score <= 0, normal = 1-2, good >= 3.
SIZING  K0 flat 1 NQ | K1 bad 0.5 / normal 1 / good 1 | K2 bad 0.5 / normal 1 / good 1.5 | K3 bad 0 (skip) / normal 1 / good 1.5 | K4 bad 0.5 / normal 1 / good 2
TAKE PROFIT (stop 50 throughout)  T0 fixed 75 | T1 nearest liquidity level in the trade's direction 25-150 pts away (prior-day high/low, premarket high/low, today's high/low so far,
   session VWAP), else 75 | T2 same but 5 pts before the level (front-run) | T3 the SECOND level out (25-200 pts), else 75
AMT (asked Nov 4, declared with the rest): prior day's RTH volume profile from QQQ 5m bars (volume spread evenly over each bar's range, 0.05% bins): POC and 70% value area.
   OPEN location vs prior value (above VAH / inside / below VAL); ALIGNED = long when the open is above VAH or short when below VAL (initiative, out-of-value open);
   IB = the 09:30-10:30 range; IBX = signal after 10:30 in the direction price already broke the IB.  T4 take-profit = nearest of prior POC / VAH / VAL 25-150 pts ahead, else 75.
Reported as REALISTIC MONTHS (fresh eval each calendar month 2022 - Aug 2026: -$1,000 day floor, $2,000 trailing = lost, +$3,000 with best day <= 50% = passed) and 2026."""
_s = open('xs/nq_month.py').read(); exec(_s[:_s.index("SETS = {")])
_q = open('xs/nq_qqq_ctx.py').read(); exec(_q[_q.index("from arch import arch_model"):_q.index("VOL = pd.DataFrame")])
HP = har.rolling(252, min_periods=120).rank(pct=True)
Y2 = yf.download(['QQQ', '^VXN'], start='2016-01-01', interval='1d', progress=False, auto_adjust=True)['Close'].ffill(); Y2.index = Y2.index.normalize()
q_, vx = Y2['QQQ'], Y2['^VXN']; XQ = pd.DataFrame({'qz': (q_ - q_.rolling(20).mean()) / q_.rolling(20).std(), 'qr20': q_ / q_.shift(20) - 1, 'vrel': vx / vx.rolling(20).mean() - 1})
def prior(s, dd): return s.reindex(pd.DatetimeIndex(dd) - pd.Timedelta(days=1), method='ffill').values
C = AL.copy().reset_index(drop=True); st = t5[C.j.values]
C['har_lo'] = prior(HP, C.day) < 1 / 3; xq = XQ.reindex(pd.DatetimeIndex(C.day) - pd.Timedelta(days=1), method='ffill').reset_index(drop=True)
C['spy'] = st_at(SPY15.end.values, SPY15.reg.values, st)
C['score'] = (C.har_lo.astype(int) + (xq.vrel.abs() <= .10).astype(int).values + (xq.qr20 > 0).astype(int).values + C.neg.astype(int) + (C.spy == C.dir).astype(int)
              - (xq.qz > 1.5).astype(int).values - (xq.vrel > .10).astype(int).values - ((C.spy != C.dir) & (C.spy != 0)).astype(int))
C['ctx'] = np.where(C.score <= 0, 'bad', np.where(C.score >= 3, 'good', 'normal'))
# ---- liquidity levels at the signal (QQQ prices, from bars before the entry bar)
vol5 = H5.volume.values.astype(float); rthm = (m5 >= 570) & (m5 < 960)
D = pd.DataFrame({'h': h5, 'l': l5, 'd': d5})[rthm].groupby('d').agg(h=('h', 'max'), l=('l', 'min')); PD = D.shift(1)
PM = pd.DataFrame({'h': h5, 'l': l5, 'd': d5})[m5 < 570].groupby('d').agg(h=('h', 'max'), l=('l', 'min'))
tp = (h5 + l5 + c5) / 3
def levels(j):
    d = d5[j]; i0 = j
    while i0 > 0 and d5[i0 - 1] == d and m5[i0 - 1] >= 570: i0 -= 1
    L = []
    if d in PD.index and PD.loc[d].notna().all(): L += [PD.loc[d].h, PD.loc[d].l]
    if d in PM.index: L += [PM.loc[d].h, PM.loc[d].l]
    if j > i0:
        L += [h5[i0:j].max(), l5[i0:j].min()]; w = vol5[i0:j]
        if w.sum() > 0: L.append((tp[i0:j] * w).sum() / w.sum())
    return np.array(L, float)
T1, T2, T3 = [], [], []
for j, d_, k in zip(C.j.values, C.dir.values, C.k.values):
    e = o5[j]; L = levels(j); dist = np.sort(((L - e) * d_ * k)[np.isfinite(L)]); ahead = dist[(dist >= 25) & (dist <= 150)]; ahead2 = dist[(dist >= 25) & (dist <= 200)]
    T1.append(ahead[0] if len(ahead) else 75.0); T2.append(ahead[0] - 5 if len(ahead) else 75.0); T3.append(ahead2[1] if len(ahead2) > 1 else 75.0)
C['T0'] = 75.0; C['T1'] = T1; C['T2'] = T2; C['T3'] = T3
# ---- AMT: prior-day volume profile, open location, initial balance
VA = {}
for d, idx in pd.Series(np.arange(len(H5))[rthm]).groupby(d5[rthm]):
    ii = idx.values; lo, hi = l5[ii].min(), h5[ii].max(); step = c5[ii[-1]] * .0005; nb = max(int((hi - lo) / step) + 1, 1); prof = np.zeros(nb)
    for i in ii:
        a, b = int((l5[i] - lo) / step), int((h5[i] - lo) / step); prof[a:b + 1] += vol5[i] / (b - a + 1)
    p = int(prof.argmax()); tot = prof.sum(); a = b = p; acc = prof[p]
    while acc < .7 * tot and (a > 0 or b < nb - 1):
        up = prof[b + 1] if b < nb - 1 else -1; dn = prof[a - 1] if a > 0 else -1
        if up >= dn: b += 1; acc += prof[b]
        else: a -= 1; acc += prof[a]
    VA[pd.Timestamp(d)] = (lo + (p + .5) * step, lo + (b + 1) * step, lo + a * step)        # POC, VAH, VAL
VAs = pd.DataFrame(VA, index=['poc', 'vah', 'val']).T.sort_index(); PV = VAs.shift(1)
op930 = pd.Series({pd.Timestamp(d): o5[idx.values[0]] for d, idx in pd.Series(np.arange(len(H5))[rthm]).groupby(d5[rthm])})
ib = pd.DataFrame({'h': h5, 'l': l5, 'd': d5})[(m5 >= 570) & (m5 < 630)].groupby('d').agg(h=('h', 'max'), l=('l', 'min'))
pv = PV.reindex(pd.DatetimeIndex(C.day)).reset_index(drop=True); opn = op930.reindex(pd.DatetimeIndex(C.day)).values
C['open_loc'] = np.where(opn > pv.vah.values, 'open above value', np.where(opn < pv.val.values, 'open below value', 'open inside value'))
C['aligned'] = ((C.open_loc == 'open above value') & (C.dir > 0)) | ((C.open_loc == 'open below value') & (C.dir < 0))
C['against'] = ((C.open_loc == 'open above value') & (C.dir < 0)) | ((C.open_loc == 'open below value') & (C.dir > 0))
ibr = ib.reindex(pd.DatetimeIndex(C.day)).reset_index(drop=True); after = m5[C.j.values] >= 630
pre_hi = np.array([h5[max(j - 12, 0):j].max() for j in C.j.values]); pre_lo = np.array([l5[max(j - 12, 0):j].min() for j in C.j.values])
C['ibx'] = after & (((C.dir > 0) & (pre_hi > ibr.h.values)) | ((C.dir < 0) & (pre_lo < ibr.l.values)))
T4 = []
for j, d_, k, row in zip(C.j.values, C.dir.values, C.k.values, pv.itertuples(index=False)):
    e = o5[j]; L = np.array([row.poc, row.vah, row.val], float); dist = np.sort(((L - e) * d_ * k)[np.isfinite(L)]); ah = dist[(dist >= 25) & (dist <= 150)]; T4.append(ah[0] if len(ah) else 75.0)
C['T4'] = T4
print(f'context: bad {np.mean(C.ctx == "bad"):.0%} normal {np.mean(C.ctx == "normal"):.0%} good {np.mean(C.ctx == "good"):.0%} of signals; liquidity target found on {np.mean(np.array(T1) != 75):.0%}, median {np.median([t for t in T1 if t != 75]):.0f} pts', flush=True)
def run_t(Cc, tcol, S=50, per_day=2, flat_m=720, cost=1.0):
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for _, c in g.iterrows():
            if n >= per_day or c.j <= busy: continue
            e = o5[c.j]; d = c.dir; kk = c.k; T = c[tcol]; res = None
            for q in range(c.j, len(H5)):
                if d5[q] != d5[c.j] or m5[q] >= flat_m: res = ((c5[q - 1] - e) * d * kk, q - 1); break
                lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
                if lo_ <= -S: res = (-S, q); break
                if hi_ >= T: res = (T, q); break
            if res: out.append((day, res[0] - cost, c.ctx)); busy = res[1]; n += 1
    return pd.DataFrame(out, columns=['day', 'net', 'ctx'])
def months(dv, a='2022-01-01', target=3000, mll=2000, dll=1000):
    v = dv.reindex(days).fillna(0.0); v = v[v.index >= a]; out = []
    for mo, x in v.groupby(v.index.to_period('M')):
        bal, peak, floor, best, res = 0.0, 0.0, -mll, 0.0, 'neither'
        for day in x.values:
            day = max(day, -dll); bal += day; best = max(best, day)
            if bal <= floor: res = 'lost'; break
            if bal >= target and best <= .5 * bal: res = 'passed'; break
            peak = max(peak, bal); floor = min(max(floor, peak - mll), 0.0)
        out.append((str(mo), res, bal))
    return pd.DataFrame(out, columns=['month', 'res', 'pnl'])
def show(nm, R, mult):
    w = R.ctx.map(mult).fillna(1.0); dv = (R.net * 20 * w).groupby(R.day).sum(); M_ = months(dv); M6 = M_[M_.month >= '2026-01']; post = R[R.day >= P0]
    print(f'  {nm:<34} trades/mo {len(post[w[R.day >= P0] > 0]) / len(M_):4.1f} win {post.net.gt(0).mean():.0%} avg {post.net.mean():+5.1f} pts | months passed {M_.res.eq("passed").mean():.0%} lost {M_.res.eq("lost").mean():.0%} neither {M_.res.eq("neither").mean():.0%} | month P&L mean {M_.pnl.mean():+,.0f} worst {M_.pnl.min():+,.0f} | 2026 passed {M6.res.eq("passed").sum()}/{len(M6)} lost {M6.res.eq("lost").sum()}', flush=True)
K = {'K0 flat': {'bad': 1, 'normal': 1, 'good': 1}, 'K1 bad 0.5': {'bad': .5, 'normal': 1, 'good': 1}, 'K2 bad 0.5 / good 1.5': {'bad': .5, 'normal': 1, 'good': 1.5},
     'K3 skip bad / good 1.5': {'bad': 0, 'normal': 1, 'good': 1.5}, 'K4 bad 0.5 / good 2': {'bad': .5, 'normal': 1, 'good': 2}}
print('\nby context, fixed 75/50, 2022-26: avg pts per trade', flush=True)
R0 = run_t(C, 'T0'); p0 = R0[R0.day >= P0]
for c_, g in p0.groupby('ctx'): print(f'  {c_:<7} n{len(g):>4} win {g.net.gt(0).mean():.0%} avg {g.net.mean():+.1f} (22-23 {g[g.day < "2024"].net.mean():+.1f} / 24-26 {g[g.day >= "2024"].net.mean():+.1f})', flush=True)
print('\nAMT states, fixed 75/50, 2022-26', flush=True)
R0j = run_t(C.assign(tag=C.open_loc), 'T0')
for col in ('open_loc', 'aligned', 'against', 'ibx'):
    tmp = C.copy(); tmp['ctx'] = tmp[col].astype(str); Rr = run_t(tmp, 'T0'); pp = Rr[Rr.day >= P0]
    for v_, g in pp.groupby('ctx'): print(f'  {col:<9} {v_:<18} n{len(g):>4} win {g.net.gt(0).mean():.0%} avg {g.net.mean():+.1f} (22-23 {g[g.day < "2024"].net.mean():+.1f} / 24-26 {g[g.day >= "2024"].net.mean():+.1f})', flush=True)
for tcol, tn in (('T0', 'fixed 75'), ('T1', 'nearest liquidity'), ('T2', 'nearest liquidity - 5'), ('T3', 'second liquidity level'), ('T4', 'AMT prior POC/VAH/VAL')):
    R = run_t(C, tcol); print(f'\n== TP {tn}', flush=True)
    for kn, mult in K.items(): show(kn, R, mult)
