"""Nov 4 2026: ANCHORED high / low volatility take-profit (Rey's RTH excursion idea: PUV = high - open, PDV = open - low, in ATR14 units, percentiles), NQ via QQQ. DECLARED BEFORE RESULTS.
Levels each day, from the 09:30 open: up_p = open + P_p(PUV / ATR) x ATR, down_p = open - P_p(PDV / ATR) x ATR, p in 50 / 75 / 90, percentiles over the PRIOR 252 sessions
  (min 120), ATR14 (Wilder) of RTH sessions at the prior close.  Long target = an up level ahead of entry, short = a down level ahead.
TAKE PROFITS (stop stays 50, 1 pt cost, flat 12:00):
  T0    fixed 75 (current)
  AV50  the first of P50 / P75 / P90 that is >= 25 pts ahead (else 75)       AV75  first of P75 / P90 >= 25 pts ahead (else 75)
  AV90  P90 if >= 25 pts ahead (else 75)  (runner)                             AVCAP AV50 capped at 150
Run on the RULE-A trades (skip thin 60-day path / hard break / monthly LVN; size 1 / 0.5 / 0.3 by HAR high + VWAP stretched) in eval and funded:
  per trade avg pts by period + win + median target; Topstep and Lucid Flex sims (real-time drawdown)."""
_ss = open('xs/nq_structure_size.py').read(); exec(_ss[:_ss.rindex("if __name__ == '__main__':")])
RTD = pd.DataFrame({'o': o5, 'h': h5, 'l': l5, 'c': c5, 'd': d5})[rthm].groupby('d').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last')); RTD.index = pd.DatetimeIndex(RTD.index)
pc = RTD.c.shift(1); tr = np.maximum(RTD.h - RTD.l, np.maximum((RTD.h - pc).abs(), (RTD.l - pc).abs())); atr = tr.ewm(alpha=1 / 14, adjust=False).mean().shift(1)
RTD['puv'] = (RTD.h - RTD.o) / atr; RTD['pdv'] = (RTD.o - RTD.l) / atr; RTD['atr'] = atr
for p in (50, 75, 90):
    RTD[f'u{p}'] = RTD.o + RTD.puv.rolling(252, min_periods=120).quantile(p / 100).shift(1) * atr
    RTD[f'd{p}'] = RTD.o - RTD.pdv.rolling(252, min_periods=120).quantile(p / 100).shift(1) * atr
def tgt(row, order, cap=None):
    lv = RTD.loc[row.day] if row.day in RTD.index else None
    if lv is None: return 75.0
    e = o5[row.j]
    for p in order:
        L_ = lv[f'u{p}'] if row.dir > 0 else lv[f'd{p}']
        if not np.isfinite(L_): continue
        dist = (L_ - e) * row.dir * row.k
        if dist >= 25: return float(min(dist, cap) if cap else dist)
    return 75.0
TPS = {'T0 fixed 75': None, 'AV50 first P50/75/90 >= 25 ahead': (50, 75, 90), 'AV75 first P75/90': (75, 90), 'AV90 P90 runner': (90,), 'AVCAP AV50 capped 150': ((50, 75, 90), 150)}
XA = X[~cutA].copy()
for nm, spec in TPS.items():
    if spec is None: XA[nm] = 75.0
    elif isinstance(spec[-1], int) and not isinstance(spec[0], tuple): XA[nm] = [tgt(r, spec) for r in XA.itertuples()]
    else: XA[nm] = [tgt(r, spec[0], spec[1]) for r in XA.itertuples()]
def run_pT(Cc, tcol, S=50, per_day=2, flat_m=720, cost=1.0):
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for idx, j, d, kk, T in zip(g.index, g.j.values, g.dir.values, g.k.values, g[tcol].values):
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
print('RULE-A trades, by take profit (1 NQ, avg pts per trade)', flush=True)
for nm in TPS:
    T_ = run_pT(XA, nm); R_ = pd.DataFrame([(d, i, n) for d, i, n, _, _ in T_], columns=['day', 'idx', 'net']).set_index('idx')
    per = np.where(R_.day < P0, '18-21', np.where(R_.day < '2024-01-01', '22-23', '24-26')); tg = XA.loc[R_.index, nm]
    print(f'  {nm:<34} trades {len(R_)} win {R_.net.gt(0).mean():.0%} median target {tg.median():.0f} pts (75 fallback {np.mean(tg == 75):.0%}) | ' +
          ' / '.join(f'{p} {R_.net[per == p].mean():+5.1f}' for p in ('18-21', '22-23', '24-26')), flush=True)
    D = [[] for _ in D_]
    for day, idx, net, mae, p in T_:
        if day in dpos: D[dpos[day]].append((tuple(bool(X.at[idx, c]) for c in FL), net, mae, p))
    SYSX[nm] = D
VARS = {nm: (nm, CUT) for nm in TPS}
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VARS for fn in ('TOPSTEP', 'LUCID FLEX')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job3, jobs)
    print('\nFIRM SIMS, Rule A in eval and funded', flush=True)
    for r in RES:
        print(f'  {r["var"]:<34} {r["firm"]:<11} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m paid {r["k_paid"]:4.0%} median {r["k_med"]:+6,.0f}', flush=True)
