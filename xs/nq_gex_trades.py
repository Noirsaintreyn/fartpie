"""Nov 8 2026: trade lists with worst open loss for the A6 trend-hold bots (nq_gex_points sizing, cut points from 2018-21) -> /tmp/claude-0/ml/nq_gex_trades.parquet."""
_gp = open('xs/nq_gex_points.py').read(); exec(_gp[:_gp.index("ALLD = pd.DatetimeIndex")])
def sim_mae(j, d):
    e = o5[j]; kk = k5[j]; mae = 0.0
    for qq in range(j, len(H)):
        if d5[qq] != d5[j] or m5[qq] >= 955: return (c5[qq - 1] - e) * d * kk, qq - 1, mae
        lo = ((l5[qq] if d > 0 else h5[qq]) - e) * d * kk
        if lo <= -50: return -50.0, qq, -50.0
        mae = min(mae, lo)
        if m5[qq] % 15 == 10:
            st = reg15.get(t5[qq] + np.timedelta64(5, 'm'))
            if st is not None and st == -d and qq + 1 < len(H) and d5[qq + 1] == d5[j]: return (o5[qq + 1] - e) * d * kk, qq, min(mae, (o5[qq + 1] - e) * d * kk)
    return None
out = []
for nm in ('A6 skip 0-2 / 1 / 2 (with GEX)', 'A6x3 skip 0-2 / 1 / 2 / 3'):
    sz = SZ[nm]
    for day, g in X.assign(sz=sz).groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            if n >= 4 or r.j <= busy or r.sz <= 0: continue
            res = sim_mae(int(r.j), int(r.dir))
            if res is None: continue
            out.append(dict(var=nm.split()[0], day=day, j=int(r.j), size=float(r.sz), net=res[0] - 1.0, mae=res[2] - 1.0)); busy = res[1]; n += 1
T = pd.DataFrame(out); T.to_parquet('/tmp/claude-0/ml/nq_gex_trades.parquet'); print(T.groupby('var').size().to_dict(), flush=True)
