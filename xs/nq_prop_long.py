"""Nov 4 2026: long-history check (2018 - Sep 2026) of S0 from xs/nq_prop_of.py without order flow: QQQ 15m vector flip, bar closing 09:30-11:30, negative gamma (prior close),
first signal of the day, entry at the next 5m open, fixed NQ-point target / stop (converted with the prior NQ close), path on 5m highs/lows (stop first if both in one bar),
flat 12:00, 1 pt cost. Coin-flip placebo (30 draws). Prop eval as in nq_prop_of.py (with consistency rule)."""
_s = open('xs/nq_gamma_prop.py').read(); exec(_s[:_s.rindex("if __name__ == ")])
_p = open('xs/nq_prop_of.py').read(); exec(_p[_p.index('def prop('):_p.rindex("if __name__ == ")])
H5 = bars('QQQ'); H15 = agg15(H5); H15['reg'] = np.asarray(vec(H15[['open', 'high', 'low', 'close']])); H15['end'] = H15.dt + pd.Timedelta('15min'); H15['me'] = H15.end.dt.hour * 60 + H15.end.dt.minute
o5, h5, l5, c5, m5 = H5.open.values, H5.high.values, H5.low.values, H5.close.values, H5.m.values; d5 = H5.dt.dt.normalize().values; t5 = H5.dt.values
NQs = NQd.sort_index(); days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(d5[m5 >= 570]))))
C = []; r = H15.reg.values
for k in range(2, len(H15)):
    if r[k] == r[k - 1] or r[k - 1] == 0 or not (570 < H15.me.iloc[k] <= 690): continue
    day = H15.dt.iloc[k].normalize(); g = G[G.index < day]
    if not len(g) or g.iloc[-1] >= 0: continue
    j = np.searchsorted(t5, np.datetime64(H15.end.iloc[k]))
    if j < len(H5) and d5[j] == np.datetime64(day): C.append((day, int(j), int(r[k])))
C = pd.DataFrame(C, columns=['day', 'j', 'dir']).groupby('day').head(1).reset_index(drop=True)
lv = NQs.reindex(pd.DatetimeIndex(C.day) - pd.Timedelta(days=1), method='ffill').values; C['k'] = lv / o5[C.j.values]      # NQ points per QQQ dollar
def run_q(C, T, S):
    out = []
    for day, j, d, k in C[['day', 'j', 'dir', 'k']].itertuples(index=False):
        e = o5[j]; res = None
        for q in range(j, len(H5)):
            if d5[q] != d5[j] or m5[q] >= 720: res = (c5[q - 1] - e) * d * k; break
            lo, hi = (l5[q] - e) * d * k, (h5[q] - e) * d * k
            if d < 0: lo, hi = (h5[q] - e) * d * k, (l5[q] - e) * d * k
            if lo <= -S: res = -S; break
            if hi >= T: res = T; break
        if res is not None: out.append((day, res - 1.0))
    return pd.DataFrame(out, columns=['day', 'pts'])
rng = np.random.default_rng(0); print(f'S0 signals {len(C)} over {len(days)} sessions ({len(C) / len(days) * 5:.1f}/wk)', flush=True)
for T, S in (() if __import__("os").getenv("ONLY_PASS") else ((75, 40), (75, 50), (100, 50), (150, 75))):
    R = run_q(C, T, S); pl = []
    for _ in range(30): Z = C.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); pl.append(run_q(Z, T, S).pts.mean())
    R['yr'] = R.day.dt.year; pr, mdn, na = prop(R, days)
    print(f'  {T}/{S}: n{len(R)} win {R.pts.gt(0).mean():.0%} avg {R.pts.mean():+.1f} pts (coin-flip {np.median(pl):+.1f}, 90% {np.percentile(pl, 5):+.1f}..{np.percentile(pl, 95):+.1f}) | 2018-21 {R[R.day < "2022"].pts.sum():+.0f} / 2022-26 {R[R.day >= "2022"].pts.sum():+.0f} pts | eval pass {pr:.0%} median {mdn:.0f} sessions | by year {R.groupby("yr").pts.sum().round(0).astype(int).to_dict()}', flush=True)
print('\neval pass rate: vector vs coin-flip direction (10 draws)', flush=True)
for T, S in ((100, 50), (150, 75)):
    R = run_q(C, T, S); pv = prop(R, days)[0]; cf = []
    for _ in range(10): Z = C.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); cf.append(prop(run_q(Z, T, S), days)[0])
    print(f'  {T}/{S}: vector {pv:.0%}  coin-flip median {np.median(cf):.0%} (range {min(cf):.0%}-{max(cf):.0%})', flush=True)
