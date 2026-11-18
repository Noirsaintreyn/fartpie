"""Nov 18 2026 report 291: RULE A with ADR-scaled exits (Rey: ATR instead of fixed). Same entries/directions (live_data/nq_rule_a_trades.csv, 357 trades 2018-Sep 2026), flat 12:00, 1 pt cost.
Fixed baseline 75/50. ADR variants declared before results: target/stop in ADR20 units; centre = train-median-equivalent (75/50 / median ADR 2019-21), plus a 3x3 neighbourhood. Choose on TRAIN 2019-21 by pts per trade AND per ADR, judge TEST (2018, 2022-Oct 2026). Report pts/trade, win rate, pts in ADR units, by year."""
import pandas as pd, numpy as np
_rp = pd.read_parquet; pd.read_parquet = lambda p, *a, **k: _rp(str(p).replace('nq5_real.parquet', 'nq5_real_ext.parquet'), *a, **k)
_p = open('/home/claude/fartpie/xs/dte_prop.py').read(); exec(_p[:_p.rindex("if __name__ == '__main__':")])
R0 = pd.read_csv('/home/claude/fartpie/live_data/nq_rule_a_trades.csv'); ent = []
for r_ in R0.itertuples():
    i = tpos.get(np.datetime64(pd.Timestamp(r_.entry), 'ns'))
    if i is None: continue
    d = pd.Timestamp(dd[i]); ix = gi[d]; j = int(np.where(ix == i)[0][0]); ent.append((d, ix, j, o[i], int(r_.dir), DY.adr.get(d, np.nan)))
ent = [e for e in ent if np.isfinite(e[5]) and e[0] <= pd.Timestamp('2026-10-31')]
def sim(tm, sm):
    rows = []
    for d, ix, j, e, dr, adr in ent:
        T_, S_ = (75.0, 50.0) if tm is None else (tm * adr, sm * adr)
        pts, mae = walk2(ix, j, dr, e, e + T_ * dr, e - S_ * dr, until_m=720); rows.append((d, pts - 1.0, mae - 1.0, (pts - 1.0) / adr, T_, S_))
    return pd.DataFrame(rows, columns=['d', 'pts', 'mae', 'adrunits', 'T', 'S'])
tr_ = lambda s: s[(s.d >= '2019-01-01') & (s.d <= '2021-12-31')]; te_ = lambda s: s[(s.d < '2019-01-01') | (s.d >= '2022-01-01')]
a19 = np.median([e[5] for e in ent if '2019' <= str(e[0].year) <= '2021']); print('median ADR 2019-21', round(a19, 0), '-> 75/50 = ', round(75 / a19, 3), '/', round(50 / a19, 3))
cT, cS = 75 / a19, 50 / a19; res = {}
res['fixed 75/50'] = sim(None, None)
for kt in (.75, 1.0, 1.25):
    for ks in (.75, 1.0, 1.25): res[f'ADR {cT * kt:.2f}/{cS * ks:.2f}'] = sim(cT * kt, cS * ks)
print(f"{'variant':20s} train n/pts/win/adr | test n/pts/win/adr | test maxDD(pts) worst-yr")
for nm, s in res.items():
    a, z = tr_(s), te_(s); zz = z.sort_values('d'); cum = zz.pts.cumsum(); yr = z.groupby(z.d.dt.year).pts.mean()
    print(f'{nm:20s} {len(a):3d} {a.pts.mean():+6.1f} {np.mean(a.pts > 0):.0%} {a.adrunits.mean():+.3f} | {len(z):3d} {z.pts.mean():+6.1f} {np.mean(z.pts > 0):.0%} {z.adrunits.mean():+.3f} | {(cum - cum.cummax()).min():+6.0f} {yr.min():+.1f}')
b = res['fixed 75/50']; print('\nby year pts/trade: fixed vs best-train ADR variant')
best = max([k for k in res if k.startswith('ADR')], key=lambda k: tr_(res[k]).adrunits.mean()); print('best on train (ADR units):', best)
print(pd.DataFrame({'fixed': b.groupby(b.d.dt.year).pts.mean(), best: res[best].groupby(res[best].d.dt.year).pts.mean(), 'fixed_adru': b.groupby(b.d.dt.year).adrunits.mean(), 'adr_adru': res[best].groupby(res[best].d.dt.year).adrunits.mean()}).round(2).to_string())
