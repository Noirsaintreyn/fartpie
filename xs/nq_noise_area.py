"""Nov 4 2026: Zarattini-Aziz-Barbon (2024, SSRN) 'Beat the Market' NOISE-AREA intraday momentum on NQ (QQQ 5m proxy -> NQ pts), and whether Rey's vector improves it. DECLARED BEFORE RESULTS.
NOISE AREA: sigma(t) = mean over the prior 14 days of |price at time t / 09:30 open - 1|; UB = max(open, prior close) x (1 + sigma), LB = min(open, prior close) x (1 - sigma).
  Decisions only at bars closing HH:00 / HH:30 from 10:00, executed at the next 5m open. Flat inside the area; long above UB, short below LB; trailing stop max(UB, VWAP) for longs /
  min(LB, VWAP) for shorts (checked at the same times); reverse on the opposite break; flat at the 16:00 close. 1 NQ, 1 pt cost per round trip.
VARIANTS:  NA (paper rules, full day) | NA-AM (entries 10:00-11:30, flat 12:00) | NA+V (entry only if the 15m vector agrees) | NA-VX (exit when the 15m vector turns against,
  instead of the VWAP / band stop) | NA+V-VX (both) | FLIP-in-NA (our 15m flip, 75 / 50, AM, only when the flip's entry is outside the noise area in its direction; on all flips and on Rule A)
  | EMA112 (arXiv 2504.10914: one ~112-day EMA of volatility-normalised daily returns) as a direction gate on Rule A.
Report: trades, win, avg NQ pts per trade 2018-21 / 2022-23 / 2024-25 / 2026, random-direction avg, annualised Sharpe of daily P&L."""
_vx = open('xs/nq_vol_exits.py').read(); exec(_vx[:_vx.rindex("if __name__ == '__main__':")])
rt = (m5 >= 570) & (m5 < 960); dd_ = pd.DatetimeIndex(d5)
s15 = st_at(H15.end.values, H15.reg.values, t5 + np.timedelta64(5, 'm'))         # 15m vector state known at each 5m bar's close
days_rt = pd.DatetimeIndex(sorted(set(dd_[rt])))
bars = {d: np.where(rt & (d5 == np.datetime64(d)))[0] for d in days_rt}
mv = {}                                                                         # |close / open - 1| by bar end minute, per day
for d, ii in bars.items():
    if len(ii) < 70: continue
    o0 = o5[ii[0]]; mv[d] = pd.Series(np.abs(c5[ii] / o0 - 1), index=m5[ii] + 5)
MV = pd.DataFrame(mv).T.sort_index(); SIG = MV.rolling(14, min_periods=14).mean().shift(1)
pcl = {d: c5[bars[d][-1]] for d in bars}; dl = list(bars)
def lvl(d): return NQs.reindex([d - pd.Timedelta(days=1)], method='ffill').values[0]
def run_na(entry_lo=600, entry_hi=960, flat=960, vec_entry=False, vec_exit=False, band_exit=True):
    out = []
    for n_, d in enumerate(dl):
        if d not in SIG.index or n_ == 0: continue
        ii = bars[d]; o0 = o5[ii[0]]; pc = pcl[dl[n_ - 1]]; sg = SIG.loc[d]
        if sg.isna().all(): continue
        k = lvl(d) / o0; ub0, lb0 = max(o0, pc), min(o0, pc); pos = 0; ent = None; cv = cvv = 0.0
        for q in ii:
            cv += tp[q] * vol5[q]; cvv += vol5[q]; vw = cv / cvv if cvv > 0 else c5[q]; me = m5[q] + 5
            if me >= flat or q == ii[-1]:
                if pos: out.append((d, pos, ent, c5[q], k))
                break
            if me % 30 or me < 600 or q + 1 > ii[-1]: continue
            s_ = sg.get(me, np.nan)
            if not np.isfinite(s_): continue
            ub, lb, px, nx = ub0 * (1 + s_), lb0 * (1 - s_), c5[q], o5[q + 1]
            if pos == 1:
                stop = (px < max(ub, vw)) if band_exit else False
                if vec_exit and s15[q] == -1: stop = True
                if stop: out.append((d, 1, ent, nx, k)); pos = 0
            elif pos == -1:
                stop = (px > min(lb, vw)) if band_exit else False
                if vec_exit and s15[q] == 1: stop = True
                if stop: out.append((d, -1, ent, nx, k)); pos = 0
            if pos == 0 and entry_lo <= me <= entry_hi:
                want = 1 if px > ub else (-1 if px < lb else 0)
                if want and (not vec_entry or s15[q] == want): pos, ent = want, nx
    T = pd.DataFrame(out, columns=['day', 'dir', 'e', 'x', 'k']); T['raw'] = (T.x - T.e) * T.dir * T.k; T['net'] = T.raw - 1.0; return T
PER = lambda d: np.select([d < P0, d < pd.Timestamp('2024-01-01'), d < pd.Timestamp('2026-01-01')], ['18-21', '22-23', '24-25'], '2026')
def show(nm, T):
    T = T.copy(); T['per'] = PER(T.day); dv = T.groupby('day').net.sum().reindex(days_rt).fillna(0); sh = dv.mean() / dv.std() * np.sqrt(252) if dv.std() > 0 else np.nan
    rnd = (-T.raw.abs().mean() * 0 + (T.raw * np.random.default_rng(3).choice([-1, 1], len(T))).mean()) - 1.0
    print(f'  {nm:<34} trades {len(T):>5} ({len(T) / len(days_rt):.2f}/day) win {T.net.gt(0).mean():.0%} | ' + ' / '.join(f'{p} {T[T.per == p].net.mean():+5.1f}' for p in ('18-21', '22-23', '24-25', '2026'))
          + f' | random-direction {rnd:+.1f} | Sharpe {sh:+.2f} | total {T.net.sum():+,.0f} pts', flush=True)
print('NOISE-AREA MODEL on NQ (QQQ proxy), 2018 - Oct 2026, avg NQ pts per trade (1 pt cost)', flush=True)
show('NA paper rules (full day)', run_na()); show('NA-AM (entries to 11:30, flat 12:00)', run_na(600, 690, 720))
show('NA+V (15m vector must agree)', run_na(vec_entry=True)); show('NA-VX (vector exit, no VWAP/band stop)', run_na(vec_exit=True, band_exit=False))
show('NA+V-VX (both)', run_na(vec_entry=True, vec_exit=True, band_exit=False)); show('NA+V, VWAP stop + vector exit', run_na(vec_entry=True, vec_exit=True))
show('NA-AM+V', run_na(600, 690, 720, vec_entry=True))
print('\nOUR 15m FLIP inside / outside the noise area (75 / 50, AM)', flush=True)
def outside(Cc):
    res = []
    for d, j, dr in zip(Cc.day, Cc.j, Cc.dir):
        d = pd.Timestamp(d)
        if d not in SIG.index or dl.index(d) == 0: res.append(np.nan); continue
        ii = bars[d]; o0 = o5[ii[0]]; pc = pcl[dl[dl.index(d) - 1]]; me = m5[j]; s_ = SIG.loc[d].get(me, np.nan)
        if not np.isfinite(s_): res.append(np.nan); continue
        e = o5[j]; res.append(1.0 if (dr > 0 and e > max(o0, pc) * (1 + s_)) or (dr < 0 and e < min(o0, pc) * (1 - s_)) else 0.0)
    return np.array(res)
for nm, Cc in (('every flip', X), ('Rule A', XA)):
    oa = outside(Cc); R = run_l(Cc); R = R.join(pd.Series(oa, index=Cc.index, name='out')); R['per'] = PER(R.day)
    for lab, m in (('outside the area', R.out == 1), ('inside the area', R.out == 0)):
        print(f'  {nm:<10} {lab:<17} n{m.sum():>4} | ' + ' / '.join(f'{p} {R[m & (R.per == p)].net.mean():+5.1f} (n{(m & (R.per == p)).sum()})' for p in ('18-21', '22-23', '24-25', '2026')), flush=True)
print('\nEMA112 trend gate on Rule A (arXiv 2504.10914: EMA of volatility-normalised daily returns, ~112-day)', flush=True)
r_ = np.log(RTD.c).diff(); nr = r_ / r_.rolling(60).std().shift(1); ema = nr.ewm(span=112, adjust=False).mean().shift(1)
g = np.sign(pd.Series(ema.reindex(pd.DatetimeIndex(XA.day)).values, index=XA.index)); R = run_l(XA).join(XA[['dir']]); R['g'] = g; R['per'] = PER(R.day)
for lab, m in (('with the EMA trend', R.g == R.dir), ('against it', R.g == -R.dir)):
    print(f'  {lab:<20} n{m.sum():>4} | ' + ' / '.join(f'{p} {R[m & (R.per == p)].net.mean():+5.1f} (n{(m & (R.per == p)).sum()})' for p in ('18-21', '22-23', '24-25', '2026')), flush=True)
