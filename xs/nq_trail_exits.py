"""Nov 4 2026: TRAIL-OUTS and CONTEXT EARLY STOPS on the final system. DECLARED BEFORE RESULTS.
Final system: Rule A trades (15m flip 09:30-11:30; skip thin 60-day path / hard break / monthly LVN), Yang-Zhang + VWAP ladder in whole micros; eval gap-gated with x0.3 in cold
streaks, funded x0.7 (cold streak = last 20 trades of THAT exit variant net negative). Base exit 75 / 50, flat 12:00, 1 pt cost; 5m path, stop checked first inside a bar; changes apply from the next bar.
  BASE | BE40 stop to entry once open profit >= 40 | BE25 same at 25 | TRAIL best - 35 once open profit >= 40 | LOCK stop to +20 once open profit >= 50
  TIME45 exit at the next open if open profit < +15 at 45 minutes | VWAP exit when a 5m bar closes back through session VWAP against the trade
  FAIL exit when a 5m bar closes beyond the 15m flip bar's opposite extreme (failed break)
Report: per trade (1 lot) by period, exit mix; Lucid 50K ($88) + Topstep: eval pass, funded paid, evals -> payout, net per eval, 12-month median, 2026."""
_rt = open('xs/nq_reach_tp.py').read(); exec(_rt[:_rt.rindex("if __name__ == '__main__':")])
rthm2 = (m5 >= 570) & (m5 < 960); newday = np.r_[True, d5[1:] != d5[:-1]]
ctp = np.zeros(len(H5)); cvv = np.zeros(len(H5)); a = b = 0.0
for i in range(len(H5)):
    if newday[i]: a = b = 0.0
    if rthm2[i]: a += tp[i] * vol5[i]; b += vol5[i]
    ctp[i], cvv[i] = a, b
vwap5 = np.where(cvv > 0, ctp / np.maximum(cvv, 1e-9), c5)
kfb = end_i.reindex(pd.DatetimeIndex(t5[XA.j.values])).values.astype(int) - 1; FBH = pd.Series(H15.high.values[kfb], index=XA.index); FBL = pd.Series(H15.low.values[kfb], index=XA.index)
def simx(j, d, kk, mode, fbh, fbl, T=75.0, S=50.0):
    e = o5[j]; path = []; stop = -S; best = 0.0
    for q in range(j, len(H5)):
        if d5[q] != d5[j] or m5[q] >= 720: path.append((c5[q - 1] - e) * d * kk); return path[-1], q - 1, path, 'noon'
        lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk; cl_ = (c5[q] - e) * d * kk
        if lo_ <= stop: path += [min(hi_, T), stop]; return stop, q, path, ('stop' if stop <= -S else 'trail')
        if hi_ >= T: path += [lo_, T]; return T, q, path, 'target'
        path += [hi_, lo_]; best = max(best, hi_)
        if mode == 'BE40' and best >= 40: stop = max(stop, 0.0)
        if mode == 'BE25' and best >= 25: stop = max(stop, 0.0)
        if mode == 'TRAIL' and best >= 40: stop = max(stop, best - 35)
        if mode == 'LOCK' and best >= 50: stop = max(stop, 20.0)
        nxt = q + 1 < len(H5) and d5[q + 1] == d5[j] and m5[q + 1] < 720
        if not nxt: continue
        ex = False
        if mode == 'TIME45' and (q - j + 1) * 5 >= 45 and (q - j + 1) * 5 < 50 and cl_ < 15: ex = True
        if mode == 'VWAP' and ((d > 0 and c5[q] < vwap5[q]) or (d < 0 and c5[q] > vwap5[q])): ex = True
        if mode == 'FAIL' and ((d > 0 and c5[q] < fbl) or (d < 0 and c5[q] > fbh)): ex = True
        if ex: x = (o5[q + 1] - e) * d * kk; path.append(x); return x, q, path, 'early'
    return None
def seqx(mode, allow=None):
    out = {}; sub = XA.loc[order] if allow is None else XA.loc[order][allow.loc[order].values]
    for day, g in sub.groupby('day', sort=False):
        busy = -1; nn = 0
        for idx, r in g.iterrows():
            if nn >= 2 or r.j <= busy: continue
            res = simx(int(r.j), int(r.dir), r.k, mode, FBH.loc[idx], FBL.loc[idx])
            if res is None: continue
            out[idx] = res; busy = res[1]; nn += 1
    return out
def eqm(nets, cut):
    m = pd.Series(1.0, index=XA.index); hist = []
    for idx in order:
        if idx in nets.index: m.loc[idx] = cut if len(hist) >= 20 and sum(hist[-20:]) < 0 else 1.0; hist.append(nets.loc[idx])
    return m
VAR = {}; print('Rule A trades by exit rule (1 lot, NQ pts)', flush=True)
for mode in ('BASE', 'BE40', 'BE25', 'TRAIL', 'LOCK', 'TIME45', 'VWAP', 'FAIL'):
    full = seqx(mode); nets = pd.Series({i: v[0] - 1.0 for i, v in full.items()}); hows = pd.Series({i: v[3] for i, v in full.items()}); per = PER(XA.loc[nets.index].day)
    print(f'  {mode:<7} n{len(nets)} win {nets.gt(0).mean():.0%} avg loss {nets[nets <= 0].mean():+6.1f} | exits ' + ', '.join(f'{k} {v:.0%}' for k, v in hows.value_counts(normalize=True).items()) + ' | ' +
          ' / '.join(f'{p} {nets[per == p].mean():+5.1f}' for p in ('18-21', '22-23', '24-25', '2026')), flush=True)
    DS[f'e_{mode}'] = to_D(seqx(mode, ~AG['GAP']), eqm(nets, .3)); DS[f'f_{mode}'] = to_D(full, eqm(nets, .7)); VAR[mode] = (f'e_{mode}', f'f_{mode}', f'f_{mode}')
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job10, jobs)
    print('\nFINAL SYSTEM WITH EACH EXIT RULE', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<7} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
