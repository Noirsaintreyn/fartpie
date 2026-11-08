"""Nov 8 2026: SIGNAL-DRIVEN SIZE AND STOP on the prop trades - more size (or a smaller stop) when the signal is strong, less when it disagrees. DECLARED BEFORE RESULTS.
Base: S1 prop trades (Rule A, 75 / 50, flat 12:00, <= 2 a day, POC-to-VAH skip, Yang-Zhang + VWAP ladder, cold-streak cuts). Three ways to read 'signal strength':
  Q     the 4-pillar quality score (energy, participation, location, timing), cut points from 2018-21
  M20   20-DAY LOOKBACK: is the vector flip working lately? sum of every RTH flip's 75 / 50 result over the last 20 trading days (all closed before today) > 0 = HOT
  ML    WALK-FORWARD MODEL: logistic regression trained every month on all earlier RTH flips (2018 on, 75 / 50 win = target), features known at entry: Rule A, minute,
        RVOL, 5-day vol, ATR expansion, VWAP z, VWAP distance, GEX sign, prior-day value / range location, 21-day z, 50-day trend, prior-day Yang-Zhang; first used 2020.
        Strong = top third of the model's predictions on earlier Rule A flips, weak = bottom third.
Per trade (1 contract): average by group, 2018-21 vs 2022-26 (ML: 2020-21 vs 2022-26).
Sizing (multiplier on the ladder, capped 1.75x; same in eval and funded):
  S1 as is | Q-UD 0.5x at Q 0-1, 1.5x at Q 3-4 | M20-UD 1.5x HOT, 0.5x COLD | ML-UD 1.5x strong, 0.5x weak | ML-SKIP skip weak | ML-STOP strong: 1.5x, weak: stop 30 at 0.5x
  ALL-UD  multiply the three (each 0.75 / 1 / 1.25), capped 0.25-1.75 | ALL-UP only size up: 1.5x when 2+ of the 3 say strong, else 1x
Report: points per trade and a day at 1 contract x multiplier (no prop rules), and Lucid 50K / 150K (eval pass, paid, payouts a year, 24-month median)."""
_t = open('xs/nq_tight_stop.py').read(); exec(_t[:_t.rindex("if __name__ == '__main__':")])
from sklearn.linear_model import LogisticRegression
XL = pd.read_parquet('/tmp/claude-0/ml/nq_location_flips.parquet').sort_values('j').reset_index(drop=True); XL['day'] = pd.DatetimeIndex(XL.day).normalize()
XL['a50'] = np.sign(a50.reindex(XL.day).values) * XL.dir.values
def feats(Z):
    f = pd.DataFrame({'ruleA': Z.ruleA.astype(float), 'me': Z.me, 'rvol': np.log(Z.rvol.clip(.05)), 'rv5': Z.rv5, 'comp': np.log(Z.comp.clip(.05)), 'vz': Z.vz.clip(-4, 6), 'vwd': Z.vwd.clip(-4, 6),
                      'gexneg': (Z.gex < 0).astype(float), 'z21': Z.z21.clip(-4, 4), 'a50': Z.a50, 'yz1': Z.yz1}, index=Z.index)
    for v in ('value, trade side of POC', 'value, far side of POC', 'beyond value, far side'): f['v_' + v[:12]] = (Z['value'] == v).astype(float)
    for v in ('discount half', 'premium half', 'beyond far extreme'): f['r_' + v[:8]] = (Z.rng == v).astype(float)
    return f
FX = feats(XL); YW = (XL.f75 > 0).astype(int).values
# 20-day lookback
dsum = XL.groupby('day').f75.sum().reindex(D_).fillna(0.0); m20 = dsum.rolling(20, min_periods=20).sum().shift(1)
HOT = pd.Series(np.sign(m20.reindex(pd.DatetimeIndex(XA.day).normalize()).values), index=XA.index)
# walk-forward model, retrained monthly
KQL = {(np.datetime64(t, 'ns'), int(d)): i for i, (t, d) in enumerate(zip(pd.to_datetime(XL.t).values, XL.dir.values))}
ixa = pd.Series([KQL.get((np.datetime64(t5[int(j)], 'ns'), int(d))) for j, d in zip(XA.j.values, XA.dir.values)], index=XA.index)
PRED = pd.Series(np.nan, index=XA.index); GRADE = pd.Series(0, index=XA.index)
mons = pd.period_range('2020-01', pd.Timestamp(D_[NS - 1]).strftime('%Y-%m'), freq='M'); xam = pd.DatetimeIndex(XA.day).to_period('M')
for mo in mons:
    tr = (XL.day < mo.start_time).values; ok = tr & FX.notna().all(axis=1).values
    if ok.sum() < 200: continue
    mu, sd = FX[ok].mean(), FX[ok].std().replace(0, 1); clf = LogisticRegression(C=.3, max_iter=2000).fit(((FX[ok] - mu) / sd).values, YW[ok])
    ptr = clf.predict_proba(((FX[ok & (XL.ruleA == 1).values] - mu) / sd).values)[:, 1]; lo_, hi_ = np.quantile(ptr, [1 / 3, 2 / 3])
    for idx in XA.index[(xam == mo)]:
        i = ixa.loc[idx]
        if i is None or (isinstance(i, float) and np.isnan(i)) or not np.isfinite(FX.iloc[int(i)].values).all(): continue
        p = clf.predict_proba(((FX.iloc[[int(i)]] - mu) / sd).values)[0, 1]; PRED.loc[idx] = p; GRADE.loc[idx] = 1 if p >= hi_ else (-1 if p <= lo_ else 0)
qg = pd.Series(np.select([Q <= 1, Q >= 3], [-1, 1], 0), index=XA.index); hg = HOT.fillna(0).astype(int); mg = GRADE
cap = lambda s: s.clip(upper=1.75)
def ud(g, up=1.5, dn=.5): return pd.Series(np.select([g > 0, g < 0], [up, dn], 1.0), index=XA.index)
allm = (ud(qg, 1.25, .75) * ud(hg, 1.25, .75) * ud(mg, 1.25, .75)).clip(.25, 1.75); nstrong = (qg > 0).astype(int) + (hg > 0).astype(int) + (mg > 0).astype(int)
V = {'S1 as is': (cs(50), skip2), 'Q-UD 0.5x Q0-1, 1.5x Q3-4': (cs(50), cap(skip2 * ud(qg))), 'M20-UD 1.5x hot, 0.5x cold': (cs(50), cap(skip2 * ud(hg))),
     'ML-UD 1.5x strong, 0.5x weak': (cs(50), cap(skip2 * ud(mg))), 'ML-SKIP skip weak': (cs(50), skip2 * (mg >= 0)),
     'ML-STOP strong 1.5x, weak stop 30 at 0.5x': (pd.Series(np.where(mg < 0, 30.0, 50.0), index=XA.index), cap(skip2 * ud(mg))),
     'ALL-UD three signals multiplied': (cs(50), cap(skip2 * allm)), 'ALL-UP 1.5x when 2+ strong': (cs(50), cap(skip2 * pd.Series(np.where(nstrong >= 2, 1.5, 1.0), index=XA.index)))}
if __name__ == '__main__':
    n1 = nets; yy = pd.DatetimeIndex(XA.loc[n1.index, 'day']).year.values; kept = (skip2.loc[n1.index] > 0).values
    print(f'model coverage of prop trades 2020+: {PRED.loc[n1.index][yy >= 2020].notna().mean():.0%}', flush=True)
    print('\nPER TRADE, 1 contract (S1 trades): n / avg by group | early period | 2022-26 | by year 2022-2026', flush=True)
    for nm, g, early in (('Q (4 pillars)', qg, yy < 2022), ('M20 (20-day lookback)', hg, yy < 2022), ('ML (walk-forward)', mg, (yy >= 2020) & (yy < 2022))):
        gv = g.loc[n1.index].values
        for lab, v in (('strong', 1), ('middle', 0), ('weak', -1)):
            m = (gv == v) & kept; a = m & early; b = m & (yy >= 2022)
            print(f'  {nm:<22} {lab:<7} | {n1.values[a].mean():+6.1f} (n{a.sum():3d}) | {n1.values[b].mean():+6.1f} (n{b.sum():3d}) | ' +
                  ' '.join(f'{n1.values[b & (yy == y)].mean():+6.1f}' if (b & (yy == y)).sum() >= 4 else '     .' for y in range(2022, 2027)), flush=True)
    print('\nSIZING. 1 contract x multiplier, no prop rules, 2022 - Oct 2026 | Lucid 5 accounts on GREEN days', flush=True)
    AD = pd.DatetimeIndex(D_[s0:NS])
    for nm, (S, mult) in V.items():
        full_s = seq_s(S) if (S != 50).any() else full; ev_s = seq_s(S, ~AG['GAP']) if (S != 50).any() else ev_res
        idx = [i for i in full_s if pd.Timestamp(XA.at[i, 'day']) >= pd.Timestamp('2022-01-01')]; pts = pd.Series([(full_s[i][0] - 1.0) * mult.loc[i] for i in idx], index=pd.DatetimeIndex(XA.loc[idx, 'day']).normalize())
        tr = [mult.loc[i] for i in idx if mult.loc[i] > 0]; pday = pts.groupby(level=0).sum().reindex(AD).fillna(0.0)
        print(f'\n  {nm}: {pts[pts != 0].mean():+.1f} pts per trade taken ({len(tr)} trades, avg size {np.mean(tr):.2f}), {pday.mean():+.2f} pts a day, worst month {pday.groupby(pday.index.to_period("M")).sum().min():+.0f}', flush=True)
        DE = to_D2(ev_s, eqm(.3) * mult, 'YZV'); DF = to_D2(full_s, eqm(.7) * mult, 'YZV')
        for a in ('50K', '150K'):
            A = ACC[a]; get = make_get(A); o = [get(s) for s in range(s0, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), 5)]
            pe = np.mean([x[0] == 'pass' for x in o]); pp = np.mean([len(x[3]) > 0 for x in o]); r, py, med, worst, mm = line(get, A, GREEN, 'ALL5')
            print(f'    {a}: single eval pass {pe:.0%}, evals paid {pp:.0%}', flush=True); show('ALL5-GREEN', a, r, py, med, worst, mm)
