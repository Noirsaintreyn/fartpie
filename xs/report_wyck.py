"""WYCKOFF + CLEAN AMD SCREEN (Nov 11 2026, declared before results). Same rule as every screen: PASS = same sign, |t|>=2, n>=30 in both eras (A 2014-21, B 2022+), thresholds from era A only.
Wyckoff (all 'with' = in trade direction): spring/upthrust (pierced 20d range extreme in last 5 days and closed back inside), selling/buying climax (prior-day volume z>2, range>1.5 ATR, close back off the extreme), SOS/SOW (price beyond prior-day range at entry on rvol>1.2), effort-vs-result absorption (rvol high, range-so-far low), volume dry-up (5d vs 20d volume), range compression (10d range / ATR), location in 20d range.
AMD clean: Accumulation = tight Asia range (19:00-03:00 ET, bottom third of its 20-day history); Manipulation = London/pre-NY (03:00-09:25) wick beyond the Asia extreme that closes back inside; Distribution = trade opposite the sweep. Features: asia_tight, sweep_dir, amd_with (trade opposite the sweep), amd_clean (tight & swept & with), plus sweep against."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
q = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); q['day'] = pd.to_datetime(q.day); print(q.columns.tolist(), flush=True)
vcol = [c for c in q.columns if 'vol' in c.lower()][0]
D = pd.read_pickle(SC + 'nq_days.pkl'); D.index = pd.to_datetime(D.index)
dh, dl, dc = D.DH, D.DL, D.DC  # full-session daily high/low/close
rng = D.rng; atr = rng.rolling(14).mean().shift(1)
hi20, lo20 = dh.rolling(20).max().shift(1), dl.rolling(20).min().shift(1)
rv = q[q.rth].groupby('day')[vcol].sum(); rv = rv.reindex(D.index); vz = ((rv - rv.rolling(20).mean()) / rv.rolling(20).std()).shift(1)
vr5 = (rv.rolling(5).mean() / rv.rolling(20).mean()).shift(1)
G = {d: g for d, g in q.groupby('day')}
rth_avg_cum = {}
rows = []
for nm in ('V15a', 'RA'):
    N = pd.read_pickle(SC + f'narr_{nm}.pkl').drop_duplicates(['day', 'm']); N['day'] = pd.to_datetime(N.day)
    for _, t in N.iterrows():
        d = t.day; m = int(t.m); dr = int(t.dir)
        if d not in G or d not in D.index or pd.isna(atr.get(d)): continue
        g = G[d]; ent = d + pd.Timedelta(minutes=m); g = g[g.dt < ent]
        asia = g[(g.dt >= d - pd.Timedelta(hours=5)) & (g.dt < d + pd.Timedelta(hours=3))]; lon = g[(g.dt >= d + pd.Timedelta(hours=3)) & (g.dt < d + pd.Timedelta(hours=9, minutes=25))]
        rth = g[g.rth] if 'rth' in g else g[g.dt >= d + pd.Timedelta(hours=9, minutes=30)]
        f = dict(leg=nm, day=d, m=m, dir=dr, pts=t.pts, era=t.era)
        # Wyckoff
        i = D.index.get_loc(d); pr = D.iloc[max(0, i - 5):i]
        sp_dn = ((pr.DL < lo20.iloc[i - 5:i].values) & (pr.DC > lo20.iloc[i - 5:i].values)).any() if i > 25 else np.nan   # spring
        up_up = ((pr.DH > hi20.iloc[i - 5:i].values) & (pr.DC < hi20.iloc[i - 5:i].values)).any() if i > 25 else np.nan   # upthrust
        f['spring_with'] = float(sp_dn if dr > 0 else up_up) if i > 25 else np.nan
        f['spring_against'] = float(up_up if dr > 0 else sp_dn) if i > 25 else np.nan
        pdd = D.iloc[i - 1] if i > 0 else None
        if pdd is not None:
            wide = (pdd.rng > 1.5 * atr.iloc[i]) and vz.iloc[i] > 2; pos = (pdd.DC - pdd.DL) / max(pdd.DH - pdd.DL, 1e-9)
            f['sc_with'] = float(wide and ((pos > .66 and dr > 0) or (pos < .34 and dr < 0)))   # climax that closed back off its extreme, trade the reversal
            f['sc_against'] = float(wide and ((pos > .66 and dr < 0) or (pos < .34 and dr > 0)))
            last = rth.close.iloc[-1] if len(rth) else np.nan
            f['sos_with'] = float(len(rth) > 0 and ((dr > 0 and last > pdd.DH) or (dr < 0 and last < pdd.DL)))
        f['volz'] = vz.iloc[i]; f['vr5'] = vr5.iloc[i]; f['comp10'] = (D.DH.iloc[i - 10:i].max() - D.DL.iloc[i - 10:i].min()) / atr.iloc[i] if i > 10 else np.nan
        f['loc20'] = ((D.DC.iloc[i - 1] - lo20.iloc[i]) / (hi20.iloc[i] - lo20.iloc[i]) if hi20.iloc[i] > lo20.iloc[i] else np.nan); f['loc20'] = f['loc20'] if dr > 0 else 1 - f['loc20']
        if len(rth) > 2:
            f['rng_sofar'] = (rth.high.max() - rth.low.min()) / atr.iloc[i]; f['rvol_sofar'] = rth[vcol].sum()
        # AMD
        if len(asia) > 20 and len(lon) > 3:
            ah, al = asia.high.max(), asia.low.min(); ar = ah - al
            f['asia_rng'] = ar / atr.iloc[i]
            sw_hi = (lon.high.max() > ah) and (lon.close.iloc[-1] < ah); sw_lo = (lon.low.min() < al) and (lon.close.iloc[-1] > al)
            sd = 1 if sw_lo and not sw_hi else (-1 if sw_hi and not sw_lo else 0)   # +1 = swept low (manipulation down) -> distribution up
            f['sweep_dir'] = sd; f['amd_with'] = float(sd == dr); f['amd_against'] = float(sd == -dr)
        rows.append(f)
F = pd.DataFrame(rows); print(len(F), 'trades with features', flush=True)
A_ = F.era == 'A'
F['asia_tight'] = np.where(F.asia_rng.isna(), np.nan, (F.asia_rng <= F.asia_rng[A_].quantile(1 / 3)).astype(float))
F['amd_clean'] = np.where(F.asia_tight.isna(), np.nan, ((F.asia_tight == 1) & (F.amd_with == 1)).astype(float))
for c in ('volz', 'vr5', 'comp10', 'loc20', 'rng_sofar'):
    lo, hi = F[c][A_].quantile(1 / 3), F[c][A_].quantile(2 / 3); F[c + '_hi'] = np.where(F[c].isna(), np.nan, (F[c] >= hi).astype(float)); F[c + '_lo'] = np.where(F[c].isna(), np.nan, (F[c] <= lo).astype(float))
F['absorb'] = np.where(F.rng_sofar.isna(), np.nan, ((F.rng_sofar_lo == 1) & (F.rvol_sofar >= F.rvol_sofar[A_].quantile(2 / 3))).astype(float))
def tst(x): x = np.asarray(x, float); return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
feats = ['spring_with', 'spring_against', 'sc_with', 'sc_against', 'sos_with', 'volz_hi', 'volz_lo', 'vr5_hi', 'vr5_lo', 'comp10_hi', 'comp10_lo', 'loc20_hi', 'loc20_lo', 'absorb', 'asia_tight', 'amd_with', 'amd_against', 'amd_clean']
print(f'\n{"feature":16} {"fam":8} | with: n  mean(t) A / B    | without: mean A / B | PASS')
for c in feats:
    fam = 'AMD' if c.startswith(('asia', 'amd')) else 'Wyckoff'
    out = []; ok = True
    for e in ('A', 'B'):
        m = (F.era == e) & F[c].notna(); a = F.pts[m & (F[c] == 1)]; b = F.pts[m & (F[c] == 0)]
        if len(a) < 30 or len(b) < 30: ok = False; out.append((len(a), np.nan, np.nan, np.nan)); continue
        se = np.sqrt(a.var() / len(a) + b.var() / len(b)); out.append((len(a), a.mean() - b.mean(), (a.mean() - b.mean()) / se, b.mean()))
    (na, da, ta, ba), (nb, db, tb, bb) = out
    if ok and not (np.sign(da) == np.sign(db) and abs(ta) >= 2 and abs(tb) >= 2): ok = False
    print(f'{c:16} {fam:8} | A n{na:4d} {da:+6.1f} ({ta:+.1f}) | B n{nb:4d} {db:+6.1f} ({tb:+.1f}) | {"PASS" if ok else ""}', flush=True)
F.to_pickle(SC + 'wyck_amd.pkl')
