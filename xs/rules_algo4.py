"""Nov 9 2026: DO PRICE ACTION OR NEWS (FOMC, CPI / JOBS 8:30 RELEASES) ADD TO THE RULES ALGO - OR DOES VOLATILITY ALREADY MODEL THEM? DECLARED BEFORE RESULTS.
Base = rules algo C + TPO (xs/rules_algo3.py: best so far, 150K +58.7 pts per account-month). Same protocol: real NQ futures, train 2014-19, cut on 2020-21, test 2022 - Jan 2026.
PA (price action, all known at the decision): distance to the prior-day high / low and the overnight high / low ahead and behind the trade (ADR), opening-range (09:30-10:00)
    place and size, last 15m bar: body / range and where it closed (signed by side), wick against the trade, inside bar, engulfing in the trade's direction, swing structure
    (higher highs minus lower lows over the last 8 15m bars, signed), breakout of the last 8 bars' high / low in the trade's direction.
NEWS: FOMC statement day (Fed calendar: federalreserve.gov 2021-2026; 2014-2020 from the Fed's statement dates; checked below against the 14:00 volume spike),
    'after 13:45 on FOMC day', FOMC eve; 8:30 release size, known before the open: 8:30 bar volume / its 20-day median, |8:30 - 8:45 move| / ADR, and the premarket move after
    8:25 signed by the trade side (CPI, jobs, PPI, retail sales... all print at 8:30 - the bar shows how big the news was).
Tests: (1) does volatility already price news days - RTH range / ADR on FOMC and big-8:30 days vs other days, and whether the news flags still explain the day's range
after the vol inputs (Yang-Zhang 1/5/20, IV pct, overnight range); (2) base vs + PA / + NEWS / + both out of sample; (3) the base rules algo's own 2022+ trades on news days vs
other days; Lucid only if a group improves the out-of-sample trades."""
_r2 = open('/home/claude/fartpie/xs/rules_algo2.py').read(); exec(_r2[:_r2.rindex("if __name__ == '__main__':")])
from sklearn.linear_model import LinearRegression
FOMC = """2014-01-29 2014-03-19 2014-04-30 2014-06-18 2014-07-30 2014-09-17 2014-10-29 2014-12-17 2015-01-28 2015-03-18 2015-04-29 2015-06-17 2015-07-29 2015-09-17 2015-10-28
2015-12-16 2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14 2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01
2017-12-13 2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19 2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30
2019-12-11 2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16 2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-28 2026-09-16 2026-10-28 2026-12-09""".split()
FOMC = set(pd.to_datetime(FOMC)); Xv = X.volume.values
def at(d, hh, mm):
    k = np.searchsorted(Xdt, np.datetime64(d + pd.Timedelta(hours=hh, minutes=mm), 'ns')); return k if k < len(Xdt) and Xdt[k] == np.datetime64(d + pd.Timedelta(hours=hh, minutes=mm), 'ns') else None
DAYN = {}
for k, d in enumerate(days):
    j830, j825, j840, j1400 = at(d, 8, 30), at(d, 8, 25), at(d, 8, 40), at(d, 14, 0); a = int(np.searchsorted(Xdt, np.datetime64(days[k - 1] + pd.Timedelta(hours=18), 'ns'))) if k else 0
    b = int(np.searchsorted(Xdt, np.datetime64(d + pd.Timedelta(hours=9, minutes=30), 'ns')))
    DAYN[d] = dict(v830=Xv[j830] if j830 is not None else np.nan, mv830=abs(Xc[j840] - Xo[j830]) if j830 is not None and j840 is not None else np.nan,
                   c825=Xc[j825] if j825 is not None else np.nan, v1400=Xv[j1400] if j1400 is not None else np.nan,
                   onh=Xh[a:b].max() if b > a else np.nan, onl=Xl[a:b].min() if b > a else np.nan)
NW = pd.DataFrame(DAYN).T.astype(float); NW['r830'] = NW.v830 / NW.v830.rolling(20, min_periods=10).median().shift(1); NW['r1400'] = NW.v1400 / NW.v1400.rolling(20, min_periods=10).median().shift(1)
NW['fomc'] = [float(d in FOMC) for d in NW.index]; NW['fomc_eve'] = NW.fomc.shift(-1).fillna(0)
def pa_news():
    rows = []; b15 = B15.reset_index(); bpos = {t: q for q, t in enumerate(B15.index.values)}; BO, BH, BL, BC = (B15[k].values for k in ('open', 'high', 'low', 'close'))
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d]: continue
        ix = gi[d]; mm_ = m[ix]; endp = np.where(mm_ < 955)[0]
        if not len(endp): continue
        endp = endp[-1]; adr = DY.adr[d]; pix = gi[days[k - 1]]; pdh, pdl = h[pix].max(), l[pix].min(); nw = NW.loc[d]; o0 = o[ix[0]]
        orh = orl = np.nan
        for p in range(len(ix)):
            i = ix[p]; t = m[i] + 5
            if t == 600: orh, orl = h[ix[:p + 1]].max(), l[ix[:p + 1]].min()
            if t % 15 or t < 585 or t > 900 or p + 1 > endp: continue
            px = c[i]; q = bpos.get(np.datetime64(H.dt.values[i] - np.timedelta64(10, 'm')))
            if q is None or q < 9: continue
            rg = BH[q] - BL[q]; body = (BC[q] - BO[q]) / rg if rg > 0 else 0.0; cl_loc = (BC[q] - BL[q]) / rg if rg > 0 else .5
            ins = float(BH[q] <= BH[q - 1] and BL[q] >= BL[q - 1]); hh8, ll8 = BH[q - 8:q].max(), BL[q - 8:q].min()
            hh = np.sum(np.diff(BH[q - 8:q + 1]) > 0); ll_ = np.sum(np.diff(BL[q - 8:q + 1]) < 0)
            for dr in (1, -1):
                eng = float(BH[q] >= BH[q - 1] and BL[q] <= BL[q - 1] and (BC[q] - BO[q]) * dr > 0)
                rows.append(dict(day=d, i=p, dir=dr, pd_ahead=((pdh - px) if dr > 0 else (px - pdl)) / adr, pd_behind=((px - pdl) if dr > 0 else (pdh - px)) / adr,
                                 on_ahead=((nw.onh - px) if dr > 0 else (px - nw.onl)) / adr, on_behind=((px - nw.onl) if dr > 0 else (nw.onh - px)) / adr,
                                 or_pos=((px - (orh + orl) / 2) / (orh - orl) * dr) if t >= 600 and orh > orl else np.nan, or_rng=(orh - orl) / adr if t >= 600 else np.nan,
                                 body=body * dr, close_loc=cl_loc if dr > 0 else 1 - cl_loc, wick_against=((BH[q] - BC[q]) if dr > 0 else (BC[q] - BL[q])) / rg if rg > 0 else 0.0,
                                 inside=ins, engulf=eng, swing=(hh - ll_) * dr, brk8=float((px > hh8) if dr > 0 else (px < ll8)),
                                 fomc=nw.fomc, fomc_after=float(nw.fomc and t >= 825), fomc_eve=nw.fomc_eve, r830=nw.r830, mv830=nw.mv830 / adr,
                                 pre_dir=((o0 - nw.c825) / adr * dr) if np.isfinite(nw.c825) else np.nan))
    return pd.DataFrame(rows)
GR2 = {'PA': ['pd_ahead', 'pd_behind', 'on_ahead', 'on_behind', 'or_pos', 'or_rng', 'body', 'close_loc', 'wick_against', 'inside', 'engulf', 'swing', 'brk8'],
       'NEWS': ['fomc', 'fomc_after', 'fomc_eve', 'r830', 'mv830', 'pre_dir']}
if __name__ == '__main__':
    f14 = NW[NW.fomc == 1].r1400; o14 = NW[(NW.fomc == 0) & (NW.index.dayofweek == 2)].r1400
    print(f'FOMC calendar check: 14:00 bar volume / 20-day median - FOMC days median {f14.median():.1f}x ({np.mean(f14 >= 3):.0%} >= 3x, n{f14.notna().sum()}); '
          f'other Wednesdays median {o14.median():.1f}x ({np.mean(o14 >= 3):.0%} >= 3x)', flush=True)
    P3 = '/tmp/claude-0/rules_pa.parquet'
    if os.path.exists(P3): PX = pd.read_parquet(P3)
    else: PX = pa_news(); PX.to_parquet(P3)
    DS = pd.read_parquet('/tmp/claude-0/rules_ds2.parquet').merge(pd.read_parquet('/tmp/claude-0/rules_ex.parquet'), on=['day', 'i', 'dir'], how='left').merge(PX, on=['day', 'i', 'dir'], how='left')
    DS['yr'] = DS.day.dt.year; print(f'rows {len(DS):,}; PA/NEWS filled {DS.body.notna().mean():.0%} / {DS.r830.notna().mean():.0%}', flush=True)
    # (1) does volatility already price the news?
    Dd = pd.DataFrame({'rng': (DY.h - DY.l) / DY.adr, 'yz1': DY.yz1p, 'yz5': DY.yz5p, 'yz20': DY.yz20p, 'onr': pd.Series({d: NSO['NR'].get(d, np.nan) for d in DY.index}) / DY.adr,
                       'ivp': pd.Series({d: (DAYF[d]['ivp'] if DAYF.get(d) is not None else np.nan) for d in DY.index})}).join(NW[['fomc', 'r830', 'mv830']])
    Dd['mv830'] = Dd.mv830 / DY.adr; Dd['big830'] = (Dd.r830 >= 3).astype(float); Dd = Dd[Dd.index >= '2014-06-01'].dropna()
    print(f'\n(1) RTH range / ADR: FOMC days {Dd[Dd.fomc == 1].rng.mean():.2f} (n{int(Dd.fomc.sum())}), big 8:30 release days (8:30 volume >= 3x) {Dd[Dd.big830 == 1].rng.mean():.2f} '
          f'(n{int(Dd.big830.sum())}), other days {Dd[(Dd.fomc == 0) & (Dd.big830 == 0)].rng.mean():.2f}', flush=True)
    for per, msk in (('2014-21', Dd.index < '2022-01-01'), ('2022-26', Dd.index >= '2022-01-01')):
        q = Dd[msk]; Xv_ = ['yz1', 'yz5', 'yz20', 'onr', 'ivp']; m0 = LinearRegression().fit(q[Xv_], q.rng); res = q.rng - m0.predict(q[Xv_])
        print(f'    {per}: after the vol inputs, leftover range on FOMC days {res[q.fomc == 1].mean():+.2f} ADR, big 8:30 days {res[q.big830 == 1].mean():+.2f} ADR, '
              f'8:30 move size corr with leftover {np.corrcoef(q.mv830, res)[0, 1]:+.2f} (vol inputs explain R2 {m0.score(q[Xv_], q.rng):.2f})', flush=True)
    # (2) model groups
    mir = DS.set_index(['day', 'i', 'dir']); tr = DS.yr <= 2019; va = DS.yr.between(2020, 2021); te = DS.yr >= 2022; n_te = DS[te].day.nunique(); nm = 'C'
    print(f'\n(2) rules algo C + TPO, test 2022 - Jan 2026', flush=True)
    for sn, add in (('base (C + TPO)', []), ('+ PA', GR2['PA']), ('+ NEWS', GR2['NEWS']), ('+ PA + NEWS', GR2['PA'] + GR2['NEWS'])):
        cols = [x if x != 'tadr' else 'tadrC' for x in FEAT] + GR['TPO'] + add
        mdl = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=.05, min_samples_leaf=300, l2_regularization=1.0, random_state=0).fit(DS.loc[tr, cols], DS.loc[tr, 'yC'])
        DS['_p'] = mdl.predict_proba(DS[cols])[:, 1]; q = DS[te]; auc = roc_auc_score(q.yC, q._p)
        def daily(msk, cut):
            z = DS[msk & (DS._p >= cut)]
            return z.sort_values(['day', 'tmin', '_p'], ascending=[True, True, False]).drop_duplicates('day')
        res = [(cut, daily(va, cut).usdC.mean(), len(daily(va, cut))) for cut in np.round(np.arange(.40, .97, .01), 2)]; res = [r for r in res if r[2] >= 40]
        good = [r for r in res if r[1] >= 10]; cut = min(good)[0] if good else max(res, key=lambda r: r[1])[0]
        z = daily(te, cut); oth = np.array([mir.yC.get((a, b, -c_), np.nan) for a, b, c_ in zip(z.day, z.i, z.dir)], dtype=float)
        print(f'  {sn:<15} AUC {auc:.3f} | {len(z) / n_te * 21:4.1f} trades/mo, hit {z.yC.mean():.0%} (other side {np.nanmean(oth):.0%}), ${z.usdC.mean():+5.1f} a trade = {z.ptsC.mean():+.2f} pts, '
              f'wins {z.yC.sum() / n_te * 21:4.1f}/mo | by year ' + ' '.join(f'{y % 100:02d}:${g_.usdC.mean():+.0f}' for y, g_ in z.groupby('yr')), flush=True)
        if sn.startswith('base'):
            print(f'\n(3) base rules algo trades 2022+ by day type:', flush=True)
            for lab, mk in (('FOMC day', z.fomc == 1), ('FOMC day, after 13:45', z.fomc_after == 1), ('big 8:30 release day', z.r830 >= 3), ('other days', (z.fomc == 0) & ~(z.r830 >= 3))):
                g_ = z[mk]; print(f'    {lab:<24} n{len(g_):4d} hit {g_.yC.mean():.0%} ${g_.usdC.mean():+6.1f} a trade', flush=True)
            print('', flush=True)
