"""Nov 4 2026: EXITS for the Rule-A NQ trade - tighter stops, smaller targets, vector exits (stop capped at 50), scale-outs. DECLARED BEFORE RESULTS.
Trades: Rule A (15m flip 09:30-11:30, skip thin 60-day path / hard break / monthly LVN; size 1 / 0.5 / 0.3 NQ by HAR high + VWAP stretched, eval and funded), flat 12:00, 1 pt cost.
Path on 5m bars, stop checked before targets inside a bar.
  FIXED  T75/S50 (current) | T75/S35 | T75/S25 | T50/S50 | T50/S35 | T50/S25 | T37.5/S25 | T100/S50
  VECTOR VX15 T75/S50 + exit at the next open once the 15m vector flips back | VX15-noT no target, S50, exit on the 15m flip back or 12:00 |
         VX5 T75/S50 + exit once the 5m vector turns against the trade
  SCALE  SC3 thirds at 25 / 50 / 75, S50 | SC3-BE same, stop to breakeven after the first third | SC2-BE halves at 37.5 / 75, breakeven after the first |
         RUN-BE half at 75, half at the anchored P90 level (nq_anchored_tp.py; 150 if it is not >= 100 ahead), breakeven after the first
Per variant: win (net > 0), avg pts per 1-lot trade 2018-21 / 2022-23 / 2024-26, avg loss; Topstep + Lucid Flex sims (real-time drawdown on the path)."""
_at = open('xs/nq_anchored_tp.py').read(); exec(_at[:_at.rindex("if __name__ == '__main__':")])
s15_at = st_at(H15.end.values, H15.reg.values, t5)                                   # 15m state known at each 5m bar's open
reg5 = np.asarray(vec(H5[['open', 'high', 'low', 'close']].reset_index(drop=True)))
def sim(j, d, kk, Ts, fr, S, be=False, vx=None, flat_m=720, cost=1.0):
    e = o5[j]; r = 0.0; f = 1.0; stop = -S; ti = 0; path = []
    for q in range(j, len(H5)):
        if d5[q] != d5[j] or m5[q] >= flat_m: r += f * (c5[q - 1] - e) * d * kk; path.append(r); return r - cost, q - 1, path
        if q > j and ((vx == '15' and s15_at[q] == -d) or (vx == '5' and reg5[q - 1] == -d)): r += f * (o5[q] - e) * d * kk; path.append(r); return r - cost, q - 1, path
        lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
        if lo_ <= stop: path += [r + f * min(hi_, Ts[ti] if ti < len(Ts) else hi_), r + f * stop]; r += f * stop; return r - cost, q, path
        path.append(r + f * lo_); hit = False
        while ti < len(Ts) and hi_ >= Ts[ti]: r += fr[ti] * Ts[ti]; f -= fr[ti]; ti += 1; hit = True
        if hit and be: stop = max(stop, 0.0)
        path.append(r + f * hi_)
        if f <= 1e-9: return r - cost, q, path
    return None
def runx(Cc, spec):
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for idx, row in g.iterrows():
            if n >= 2 or row.j <= busy: continue
            Ts, fr, S, be, vx = spec(row); res = sim(int(row.j), int(row.dir), row.k, Ts, fr, S, be, vx)
            if res is None: continue
            net, qe, p = res; p = np.array(p) - 1.0; out.append((day, idx, net, p.min(), p)); busy = qe; n += 1
    return out
def fx(T, S): return lambda r: ([T], [1.0], S, False, None)
def p90(r): v = r['AV90 P90 runner']; return v if v >= 100 else 150.0
EX = {'T75/S50 (current)': fx(75, 50), 'T75/S35': fx(75, 35), 'T75/S25': fx(75, 25), 'T50/S50': fx(50, 50), 'T50/S35': fx(50, 35), 'T50/S25': fx(50, 25),
      'T37.5/S25': fx(37.5, 25), 'T100/S50': fx(100, 50),
      'VX15 T75/S50 + 15m flip-back': lambda r: ([75], [1.0], 50, False, '15'), 'VX15-noT S50 + 15m flip-back': lambda r: ([1e9], [1.0], 50, False, '15'),
      'VX5 T75/S50 + 5m against': lambda r: ([75], [1.0], 50, False, '5'),
      'SC3 thirds 25/50/75': lambda r: ([25, 50, 75], [1 / 3, 1 / 3, 1 / 3], 50, False, None), 'SC3-BE thirds + breakeven': lambda r: ([25, 50, 75], [1 / 3, 1 / 3, 1 / 3], 50, True, None),
      'SC2-BE halves 37.5/75 + BE': lambda r: ([37.5, 75], [.5, .5], 50, True, None), 'RUN-BE half 75 / half P90 + BE': lambda r: ([75, p90(r)], [.5, .5], 50, True, None)}
print('RULE-A trades by exit (1 lot, pts)', flush=True)
for nm, spec in EX.items():
    T_ = runx(XA, spec); R_ = pd.DataFrame([(d, i, n) for d, i, n, _, _ in T_], columns=['day', 'idx', 'net'])
    per = np.where(R_.day < P0, '18-21', np.where(R_.day < '2024-01-01', '22-23', '24-26'))
    print(f'  {nm:<32} trades {len(R_)} win {R_.net.gt(0).mean():.0%} avg loss {R_.net[R_.net <= 0].mean():+6.1f} | ' + ' / '.join(f'{p} {R_.net[per == p].mean():+5.1f}' for p in ('18-21', '22-23', '24-26')), flush=True)
    D = [[] for _ in D_]
    for day, idx, net, mae, p in T_:
        if day in dpos: D[dpos[day]].append((tuple(bool(X.at[idx, c]) for c in FL), net, mae, p))
    SYSX[nm] = D
VARS = {nm: (nm, CUT) for nm in EX}
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VARS for fn in ('TOPSTEP', 'LUCID FLEX')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job3, jobs)
    print('\nFIRM SIMS, Rule A sizing in eval and funded', flush=True)
    for r in RES:
        print(f'  {r["var"]:<32} {r["firm"]:<11} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m paid {r["k_paid"]:4.0%} median {r["k_med"]:+6,.0f}', flush=True)
