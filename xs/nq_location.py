"""Nov 8 2026: WHAT HAPPENED IN 2022 AND 2024, and WHERE THE FLIP HAPPENS (VWAP, value area, POC, discount / premium). DECLARED BEFORE RESULTS.
Part 1, the years: the prop system's Rule A trades (1 contract, 75 / 50, flat 12:00, after 1 pt cost) by year; the Lucid book's events by year (evals bought, passed, payouts,
  accounts graduated after 5 payouts); the market by year (QQQ return, NQ daily range, VXN, flips a day, share of flips that pass Rule A).
Part 2, location of the flip (all known at entry; mirrored for shorts so 'discount' = long below / short above):
  VWAP z   (entry - session VWAP) / session VWAP standard deviation, in the trade's direction: < -1 | -1..0 (discount side) | 0..1 | 1..2 | > 2
  VALUE    prior RTH day's volume profile (5m bars spread over 0.05% bins, value area 70%): beyond value on the trade side (long above VAH) | in value, trade side of POC |
           in value, far side of POC | beyond value on the far side (long below VAL = discount)
  BROKE    what the 15m flip bar moved through from open to close in the trade's direction: trade-side value edge (long VAH) | POC | far value edge (long VAL, a reclaim
           from discount) | none
  RANGE    entry in the prior day's range, mirrored: beyond the far extreme (long below PDL) | discount half | premium half | beyond the trade-side extreme (long above PDH)
  Z        daily z-scores already saved (21 / 63 / 252 day, in the trade's direction), thirds
Outcomes, NQ points per trade: prop trades (Rule A, 75 / 50 noon) and every RTH flip held to the vector exit (TH) and 75 / 50 (F75). By period 18-21 / 2022 / 2023 / 2024 / 2025 / 2026.
A label counts when its best bucket beats its worst in every period with 8+ trades."""
_as = open('xs/nq_account_sizes.py').read(); exec(_as[:_as.rindex("if __name__ == '__main__':")])
Hq = load(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/QQQ_5m_*.parquet')[0]); Hq = Hq[(Hq.m >= 570) & (Hq.m < 960)].reset_index(drop=True); Hq['day'] = Hq.dt.dt.normalize()
qo, qh, ql, qc, qv, qd = Hq.open.values, Hq.high.values, Hq.low.values, Hq.close.values, Hq.volume.values.astype(float), Hq.day.values
tp = (qh + ql + qc) / 3; cv = Hq.volume.groupby(Hq.day).cumsum().values; vw = (pd.Series(tp * qv).groupby(Hq.day.values).cumsum().values) / np.maximum(cv, 1)
vsd = np.sqrt(np.maximum(pd.Series(tp * tp * qv).groupby(Hq.day.values).cumsum().values / np.maximum(cv, 1) - vw * vw, 0))
qpos = {np.datetime64(t, 'ns'): i for i, t in enumerate(Hq.dt.values)}
# ---- prior-day volume profile
LV = {}; days = pd.DatetimeIndex(sorted(Hq.day.unique())); grp = Hq.groupby('day').indices
for d in days:
    ix = grp[d]; lo, hi, vv = ql[ix], qh[ix], qv[ix]; bs = qo[ix[0]] * .0005; b0 = np.floor(lo.min() / bs); nb = int(np.floor(hi.max() / bs) - b0) + 1; prof = np.zeros(nb)
    for a, b, v in zip(np.floor(lo / bs) - b0, np.floor(hi / bs) - b0, vv): a, b = int(a), int(b); prof[a:b + 1] += v / (b - a + 1)
    p = int(prof.argmax()); u = dn = p; tot = prof[p]; tgt = .7 * prof.sum()
    while tot < tgt and (u < nb - 1 or dn > 0):
        up = prof[u + 1] if u < nb - 1 else -1; down = prof[dn - 1] if dn > 0 else -1
        if up >= down: u += 1; tot += up
        else: dn -= 1; tot += down
    LV[d] = dict(poc=(b0 + p + .5) * bs, vah=(b0 + u + 1) * bs, val=(b0 + dn) * bs, pdh=hi.max(), pdl=lo.min())
PRIOR = {days[i]: LV[days[i - 1]] for i in range(1, len(days))}
def loc(i, d):
    """features for an entry at Hq index i (next 5m open after the 15m flip bar), direction d"""
    day = pd.Timestamp(qd[i]); L = PRIOR.get(day)
    if L is None or i < 3 or qd[i - 3] != qd[i]: return None
    e = qo[i]; z = (e - vw[i - 1]) / vsd[i - 1] * d if vsd[i - 1] > 0 else np.nan
    TE, FE = (L['vah'], L['val']) if d > 0 else (L['val'], L['vah']); y = (e - L['poc']) * d; yT = (TE - L['poc']) * d; yF = (FE - L['poc']) * d
    value = 'beyond value, trade side' if y > yT else ('value, trade side of POC' if y > 0 else ('value, far side of POC' if y >= yF else 'beyond value, far side'))
    fo, fc = qo[i - 3], qc[i - 1]; cross = lambda Lv: (fo - Lv) * d < 0 <= (fc - Lv) * d
    broke = 'trade-side edge (VAH long)' if cross(TE) else ('POC' if cross(L['poc']) else ('far edge reclaim (VAL long)' if cross(FE) else 'none'))
    rg = L['pdh'] - L['pdl']; pr = (e - L['pdl']) / rg if rg > 0 else np.nan; pr = pr if d > 0 else 1 - pr
    rng = 'beyond far extreme' if pr < 0 else ('discount half' if pr < .5 else ('premium half' if pr <= 1 else 'beyond trade-side extreme'))
    return dict(vz=z, value=value, broke=broke, rng=rng)
PERS = ['18-21', '2022', '2023', '2024', '2025', '2026']
perof = lambda days_: np.where(pd.DatetimeIndex(days_) < '2022-01-01', '18-21', pd.DatetimeIndex(days_).year.astype(str))
def table(Z, name, lab, col):
    out = []
    for v in [x for x in lab.dropna().unique()]:
        g = Z[lab == v]; out.append((v, len(g), g[col].mean(), [g[g.per == p][col].mean() if (g.per == p).sum() >= 8 else np.nan for p in PERS]))
    out.sort(key=lambda o: -o[2])
    if len(out) < 2: return None
    best, worst = out[0], out[-1]; cons = all(b > w for b, w in zip(best[3], worst[3]) if np.isfinite(b) and np.isfinite(w))
    print(f'  -- {name} [{col}] spread {best[2] - worst[2]:+.1f}{"  * every period" if cons else ""}', flush=True)
    for v, n, m, pp in out: print(f'    {str(v):<30} n{n:>4} {m:+6.1f} | ' + ' '.join(f'{x:+6.1f}' if np.isfinite(x) else '     .' for x in pp), flush=True)
    return (name, col, best[2] - worst[2], cons, str(best[0]), str(worst[0]))
VZB = lambda s: pd.cut(s, [-np.inf, -1, 0, 1, 2, np.inf], labels=['z < -1', 'z -1..0 (discount side)', 'z 0..1', 'z 1..2', 'z > 2']).astype(str).where(s.notna())
T3 = lambda s: pd.qcut(s.rank(method='first'), 3, labels=['low third', 'mid third', 'high third']).astype(str).where(s.notna())
if __name__ == '__main__':
    # ---------------- PART 1: the years
    T = XA.loc[nets.index, ['day', 'j', 'dir']].copy(); T['pts'] = nets.values; T['yr'] = pd.DatetimeIndex(T.day).year
    print('PART 1. Prop trades (Rule A, 75 / 50, flat 12:00), 1 contract, NQ points after cost', flush=True)
    print('  year  trades  win   avg   total  avg win  avg loss  full stops', flush=True)
    for y, g in T.groupby('yr'):
        print(f'  {y}  {len(g):5d}  {(g.pts > 0).mean():4.0%} {g.pts.mean():+6.1f} {g.pts.sum():+7.0f} {g.pts[g.pts > 0].mean():+8.1f} {g.pts[g.pts <= 0].mean():+9.1f} {(g.pts <= -50.5).mean():9.0%}', flush=True)
    s0 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01')))
    for a in ('50K', '150K'):
        A = ACC[a]; E, r = book(A, s0, NS, 5); E = E.assign(yr=[D_[int(k)].year for k in E.k])
        print(f'\n  Lucid {a}, 5 accounts from Jan 2022 (events by year):', flush=True)
        for y, g in E.groupby('yr'):
            print(f'    {y}: evals bought {(g.t == "buy").sum()}, passed {(g.t == "pass").sum()}, payouts {(g.t == "pay").sum()} ({g[g.t == "pay"].amt.sum() / 20:,.0f} pts), '
                  f'accounts done after 5 payouts {(g.t == "graduated").sum()}, fees {-g[g.t == "buy"].amt.sum():,.0f}', flush=True)
    dq = Hq.groupby('day').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last')); rng_nq = (dq.h - dq.l) / dq.o * 100; vxn = Y2['^VXN'].reindex(dq.index).ffill()
    XF = pd.read_parquet('/tmp/claude-0/ml/nq_flip_features_opt.parquet'); XF['yr'] = pd.DatetimeIndex(XF.day).year
    print('\n  Market by year: QQQ return, QQQ daily range (%), VXN average, every-flip 15m count a day, share passing Rule A, every flip held (TH) avg pts', flush=True)
    for y in range(2018, 2027):
        g = dq[dq.index.year == y]; f = XF[XF.yr == y]
        print(f'    {y}: QQQ {g.c.iloc[-1] / g.o.iloc[0] - 1:+6.1%} | daily range {rng_nq[rng_nq.index.year == y].mean():4.2f}% | VXN {vxn[vxn.index.year == y].mean():4.1f} | '
              f'flips/day {len(f) / len(g):.2f} | Rule A {f.ruleA.mean():4.0%} | TH {f.th.mean():+5.1f} F75 {f.f75.mean():+5.1f}', flush=True)
    # ---------------- PART 2: location
    rows = []
    for r_ in XF.itertuples():
        f = loc(int(r_.j), int(r_.dir)); rows.append(f if f else dict(vz=np.nan, value=None, broke=None, rng=None))
    XF = pd.concat([XF.reset_index(drop=True), pd.DataFrame(rows)], axis=1); XF['per'] = perof(XF.day)
    rows = []
    for r_ in T.itertuples():
        i = qpos.get(np.datetime64(t5[int(r_.j)], 'ns')); f = loc(i, int(r_.dir)) if i is not None else None
        rows.append(f if f else dict(vz=np.nan, value=None, broke=None, rng=None))
    T = pd.concat([T.reset_index(drop=True), pd.DataFrame(rows)], axis=1); T['per'] = perof(T.day)
    T = T.merge(XF[['t', 'dir', 'z21', 'z63', 'z252', 'gex']].assign(t=lambda d_: pd.to_datetime(d_.t)), how='left', left_on=[pd.to_datetime(t5[T.j.values.astype(int)]), 'dir'], right_on=['t', 'dir'])
    print(f'\nPART 2. Location. Prop trades with features: {T.vz.notna().sum()} of {len(T)}; flips: {XF.vz.notna().sum()} of {len(XF)}', flush=True)
    print(f'  periods: ' + ' '.join(f'{p:>6}' for p in PERS), flush=True)
    RES = []
    for Z, nm, cols in ((T, 'PROP TRADES (Rule A, 75/50 noon)', ['pts']), (XF, 'EVERY RTH FLIP', ['th', 'f75'])):
        print(f'\n{nm}', flush=True)
        for col in cols:
            RES += [table(Z, 'VWAP z (trade direction)', VZB(Z.vz), col), table(Z, 'prior-day value area', Z['value'], col), table(Z, 'flip bar broke', Z.broke, col),
                    table(Z, 'prior-day range position', Z.rng, col), table(Z, 'z21 daily', T3(Z.z21), col), table(Z, 'z63 daily', T3(Z.z63), col), table(Z, 'z252 daily', T3(Z.z252), col)]
    print('\nSUMMARY (* = best beats worst in every period with 8+ trades)', flush=True)
    for r_ in [x for x in RES if x]: print(f'  {"*" if r_[3] else " "} {r_[0]:<26} {r_[1]:<4} spread {r_[2]:+6.1f}  best {r_[4]:<30} worst {r_[5]}', flush=True)
    T.to_parquet('/tmp/claude-0/ml/nq_location_prop.parquet'); XF.to_parquet('/tmp/claude-0/ml/nq_location_flips.parquet')
