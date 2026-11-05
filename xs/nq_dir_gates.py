"""Nov 4 2026: DIRECTION GATES - a reason to be long or short - on the Rule A NQ trade (fixed 50 / 75, Rule A sizing 10 / 5 / 3 MNQ). DECLARED BEFORE RESULTS.
Gate direction (+1 long allowed / -1 short allowed; missing data = both allowed), all known before the entry:
  SMA20 / SMA50 / SMA200  QQQ prior RTH close above (+1) / below (-1) its daily SMA
  Z-MR  -sign(QQQ 20d z at the prior close): buy below the mean, sell above it      Z-TR  +sign(z20): with the stretch
  VWAP  entry above / below today's session VWAP (09:30 to entry)                    WVWAP entry vs the week's anchored VWAP (from the week's first RTH bar)
  GAP   09:30 open above / below the prior RTH close                                H1 / H4  QQQ 1h / 4h vector state at entry
  VXN   VXN falling vs its 20d average -> longs, rising -> shorts
Per gate: avg pts (1 lot) of trades WITH vs AGAINST the gate in 2018-21 / 2022-23 / 2024-25 / 2026 (consistent = with > against in all four);
then the gated system (skip trades against the gate) through Lucid 50K ($88) and Topstep: eval passed, blown before 1st payout, net per eval, 12-month median, 2022-26 net, 2026 net."""
_vx = open('xs/nq_vol_exits.py').read(); exec(_vx[:_vx.rindex("if __name__ == '__main__':")])
cl = RTD.c; G_ = pd.DataFrame(index=RTD.index)
for n in (20, 50, 200): G_[f'SMA{n}'] = np.sign(cl - cl.rolling(n).mean()).shift(1)
zq = pd.Series(prior(XQ.qz, RTD.index), index=RTD.index); G_['Z-MR'] = -np.sign(zq); G_['Z-TR'] = np.sign(zq)
G_['GAP'] = np.sign(RTD.o - cl.shift(1)); G_['VXN'] = -np.sign(pd.Series(prior(XQ.vrel, RTD.index), index=RTD.index))
rt_tpv = np.where(rthm, tp * vol5, 0.0).cumsum(); rt_v = np.where(rthm, vol5, 0.0).cumsum()
wk = pd.DatetimeIndex(d5).to_period('W'); first_of_week = pd.Series(np.arange(len(H5))).groupby(np.asarray(wk)).transform('min').values
gates = {g: [] for g in ('SMA20', 'SMA50', 'SMA200', 'Z-MR', 'Z-TR', 'VWAP', 'WVWAP', 'GAP', 'H1', 'H4', 'VXN')}
for idx, r in XA.iterrows():
    j = int(r.j); day = pd.Timestamp(r.day); e = o5[j]; i0 = j
    while i0 > 0 and d5[i0 - 1] == d5[j] and m5[i0 - 1] >= 570: i0 -= 1
    w = vol5[i0:j]; vw_ = (tp[i0:j] * w).sum() / w.sum() if j > i0 and w.sum() > 0 else np.nan
    a = first_of_week[j]; den = rt_v[j - 1] - (rt_v[a - 1] if a > 0 else 0); wv_ = (rt_tpv[j - 1] - (rt_tpv[a - 1] if a > 0 else 0)) / den if den > 0 else np.nan
    for g in ('SMA20', 'SMA50', 'SMA200', 'Z-MR', 'Z-TR', 'GAP', 'VXN'): gates[g].append(G_.at[day, g] if day in G_.index else np.nan)
    gates['VWAP'].append(np.sign(e - vw_)); gates['WVWAP'].append(np.sign(e - wv_))
    gates['H1'].append(st_at(Q1.end.values, Q1.reg.values, [t5[j]])[0]); gates['H4'].append(st_at(Q4.end.values, Q4.reg.values, [t5[j]])[0])
for g, v in gates.items(): XA[f'g_{g}'] = np.asarray(v, float)
def agree(g): v = XA[f'g_{g}']; return (v.isna() | (v == 0) | (v == XA.dir)).values
R0 = run_l(XA); R0 = R0.join(XA[[c for c in XA.columns if c.startswith('g_')] + ['dir']])
R0['per'] = np.select([R0.day < P0, R0.day < '2024-01-01', R0.day < '2026-01-01'], ['18-21', '22-23', '24-25'], '2026')
PERS = ('18-21', '22-23', '24-25', '2026')
print(f'Rule A trades {len(R0)}; avg ' + ' / '.join(f'{p} {R0[R0.per == p].net.mean():+.1f}' for p in PERS), flush=True)
print('\nWITH vs AGAINST each gate (avg pts, 1 lot)', flush=True)
for g in gates:
    v = R0[f'g_{g}']; wth = v.isna() | (v == 0) | (v == R0.dir); ag = ~wth; ok = all(R0[wth & (R0.per == p)].net.mean() > R0[ag & (R0.per == p)].net.mean() for p in PERS)
    print(f'  {g:<6} with {wth.mean():4.0%} | ' + ' | '.join(f'{p} with {R0[wth & (R0.per == p)].net.mean():+6.1f} (n{(wth & (R0.per == p)).sum():>3}) vs against {R0[ag & (R0.per == p)].net.mean():+6.1f} (n{(ag & (R0.per == p)).sum():>3})' for p in PERS) + ('   <- with > against in all four' if ok else ''), flush=True)
def build_gated(mask):
    D = [[] for _ in D_]; sub = XA[mask]
    for day, g in sub.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for idx, r in g.iterrows():
            if n >= 2 or r.j <= busy: continue
            res = sim2(int(r.j), int(r.dir), r.k, 75.0, 50.0)
            if res is None: continue
            net, qe, p, how = res; th = tuple(bool(X.at[idx, c]) for c in FL); size = (1.0, .5, .3)[min(th[1] + th[2], 2)]
            p = np.array(p) - 1.0; busy = qe; n += 1
            if day in dpos: D[dpos[day]].append(((size,), net - 1.0, p.min(), p))
    return D
SYSV = {'none (Rule A)': SYSV['F']}
for g in gates: SYSV[g] = build_gated(agree(g))
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in SYSV for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job7, jobs)
    print('\nGATED SYSTEMS (skip trades against the gate), firm sims', flush=True)
    for r in RES:
        fu, y = r['full'], r['y26']
        print(f'  {r["nm"]:<14} {r["fn"]:<9} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} '
              f'| 2022-Oct26 net {fu["net"]:+7,.0f} ({fu["payouts"]} payouts) | 2026 net {y["net"]:+6,.0f} ({y["payouts"]} payouts)', flush=True)
