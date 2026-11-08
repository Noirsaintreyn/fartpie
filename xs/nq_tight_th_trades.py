"""Nov 8 2026: trade lists (with max adverse excursion) for the tighter-stop trend-hold, for the Lucid test in xs/nq_tight_th_prop.py. Same book as xs/nq_tight_stop_th.py
(every RTH flip, vector exit, <= 4 a day, no overlap, A6x3 contracts), 2022 - Oct 2026: B0 stop 50 | B2 stop 30 x1.67 | B3 stop 35 x1.43 | B5 stop 30 same contracts."""
_t = open('xs/nq_tight_stop_th.py').read(); exec(_t[:_t.rindex("if __name__ == '__main__':")])
V = {'B0': (np.full(len(X), 50.0), base), 'B2': (np.full(len(X), 30.0), base * 50 / 30), 'B3': (np.full(len(X), 35.0), base * 50 / 35), 'B5': (np.full(len(X), 30.0), base)}
rows = []
for nm, (sv, sz) in V.items():
    S = float(sv[0]); p_, q_ = run1(S); msk = (X.day >= '2022-01-01').values
    for day, g in X[msk].assign(sz=sz[msk], pts=p_[msk], qx=q_[msk]).groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            if n >= 4 or r.j <= busy or r.sz <= 0 or not np.isfinite(r.pts): continue
            j, d, q = int(r.j), int(r.dir), int(r.qx); e = o5[j]; kk = k5[j]
            adv = ((l5[j:q + 1] if d > 0 else h5[j:q + 1]) - e) * d * kk; mae = max(float(adv.min()), -S) - 1.0
            rows.append((nm, day, j, r.sz, r.pts, mae)); busy = q; n += 1
pd.DataFrame(rows, columns=['var', 'day', 'j', 'size', 'net', 'mae']).to_parquet('/tmp/claude-0/ml/nq_tight_th_trades.parquet')
print(pd.DataFrame(rows, columns=['var', 'day', 'j', 'size', 'net', 'mae']).groupby('var').agg(n=('net', 'size'), net=('net', 'mean'), mae=('mae', 'mean')))
