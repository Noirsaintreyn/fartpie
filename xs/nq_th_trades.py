"""Nov 8 2026, step 1 of the trend-hold prop test: the actual trade lists (<= 4 a day, no overlap) with each trade's worst open loss (MAE) for the Lucid engine.
Trend-hold = hold until the 15m vector flips back, stop 50, flat 15:55, NQ points after a 1 pt cost. Sizes (in units; the prop script turns units into micros):
  RA_TH    Rule A entries, prop ladder: 1 / 0.5 / 0.3 by Yang-Zhang 20d top third + VWAP stretched (the current prop sizing)
  WF_A     every RTH flip, walk-forward alignment count (nq_context_models M2: 0.5 / 1 / 2, fitted on prior years only)
  FIX_A    every RTH flip, fixed A+ count (POST-HOC cut points from the full sample): skip 0-2 conditions, 1 at 3, 2 at 4-5
Saves /tmp/claude-0/ml/nq_th_trades.parquet."""
_cm = open('xs/nq_context_models.py').read(); exec(_cm[:_cm.index("ALLD = pd.DatetimeIndex")])
def sim_mae(j, d):
    e = o5[j]; kk = k5[j]; mae = 0.0
    for q in range(j, len(H)):
        if d5[q] != d5[j] or m5[q] >= 955: return (c5[q - 1] - e) * d * kk, q - 1, mae
        lo = ((l5[q] if d > 0 else h5[q]) - e) * d * kk
        if lo <= -50: return -50.0, q, -50.0
        mae = min(mae, lo)
        if m5[q] % 15 == 10:
            st = reg15.get(t5[q] + np.timedelta64(5, 'm'))
            if st is not None and st == -d and q + 1 < len(H) and d5[q + 1] == d5[j]: return (o5[q + 1] - e) * d * kk, q, min(mae, ((o5[q + 1] - e) * d * kk))
    return None
q3 = lambda c, hi=True: (X[c] >= X[c].quantile(2 / 3)) if hi else (X[c] <= X[c].quantile(2 / 3))
X['aplus'] = (q3('rvol').astype(int) + q3('rv5').astype(int) + q3('vwd', False).astype(int) + (X.me <= 615).astype(int) + q3('comp').astype(int))
cut_vw = X[X.day < '2022-01-01'].vwd.quantile(2 / 3)
X['ladder'] = np.array([(1.0, .5, .3)[min(int(a) + int(b), 2)] for a, b in zip(X.yz20.fillna(0) >= 2 / 3, X.vwd > cut_vw)])
SZ = {'RA_TH': np.where(X.ruleA == 1, X.ladder, 0.0), 'WF_A': SIZE['M2'].values, 'FIX_A': np.where(X.aplus >= 4, 2.0, np.where(X.aplus == 3, 1.0, 0.0))}
out = []
for nm, sz in SZ.items():
    for day, g in X.assign(sz=sz).groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            if n >= 4 or r.j <= busy or not np.isfinite(r.sz) or r.sz <= 0: continue
            res = sim_mae(int(r.j), int(r.dir))
            if res is None: continue
            out.append(dict(var=nm, day=day, j=int(r.j), dir=int(r.dir), me=int(r.me), size=float(r.sz), net=res[0] - 1.0, mae=res[2] - 1.0, aplus=int(r.aplus)))
            busy = res[1]; n += 1
T = pd.DataFrame(out); T.to_parquet('/tmp/claude-0/ml/nq_th_trades.parquet')
for nm, g in T.groupby('var'):
    g2 = g[g.day >= '2022-01-01']; days = pd.DatetimeIndex(sorted(H.day.unique())); nd = (days >= '2022-01-01').sum()
    print(f'{nm:<6} trades {len(g)} (2022+: {len(g2)}) | 2022+ sized points {(g2.net * g2["size"]).sum():+,.0f} = {(g2.net * g2["size"]).sum() / nd:+.2f} a day | avg size {g2["size"].mean():.2f} | worst MAE {g2.mae.min():+.0f}', flush=True)
