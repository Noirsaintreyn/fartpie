"""Nov 9 2026: DO S/R LEVELS, TPO, HIGH/LOW-OF-DAY TIMING OR GEX WALLS ADD TO THE RULES ALGO? DECLARED BEFORE RESULTS.
Already known from Rey's own point-in-time tests (xs/algo_levels_eval.py, xs/ext/thod_by_stock_app.parquet, Oct 2024 - Oct 2026): touched algo levels reverse
no more than placebo levels shifted +-0.5 sigma (1h 83.4% vs 82.8%, 4h 69.0% vs 69.2%; QQQ 75.5% vs 77.6%); THOD/TLOD lines 1h 50.8% vs 49.6%, 4h 61.8% vs 61.2%
(QQQ below its placebo). Here, on the rules algo (xs/rules_algo.py versions C and D, real NQ futures, train 2014-19, cut on 2020-21, test 2022 - Jan 2026), new groups:
  TPO   prior-day 30-minute TPO profile: TPO POC distance, inside / above / below TPO value, nearest single prints ahead and behind, poor high / low ahead;
        today: initial balance (09:30 - 10:30) range / ADR and place vs the IB (from 10:30), acceptance (periods at this price / periods so far), developing TPO POC distance
  HL    high/low-of-day timing: minutes since the extreme behind the trade and the one ahead, 'the extreme behind looks in' (set >= 30 min earlier, price >= 0.5 daily
        sigma off it), distance to the extreme ahead in daily sigma
  WALLS QQQ call wall / put wall mapped to NQ at the prior close: distance to the wall ahead and behind (ADR)
Compared: base / + TPO / + HL / + WALLS / + all - out-of-sample ranking (AUC, hit rate top vs bottom predicted decile) and the rules algo's own trades; Lucid for C base vs C + all.
LEVELS ON NQ (Oct 2024 - Jan 2026, all the point-in-time history there is): Rey's QQQ 1h / 4h algo levels and THOD/TLOD lines mapped to NQ at the open (x NQ open /
QQQ price). Decision rows within 0.03 ADR of a level, trading the bounce (long just above support, short just below resistance): C / D hit rate and $ a trade,
real levels vs the same levels shifted U(-0.5, 0.5) THOD sigma (placebo)."""
_ra = open('/home/claude/fartpie/xs/rules_algo.py').read(); exec(_ra[:_ra.rindex("if __name__ == '__main__':")])
import json
from sklearn.metrics import roc_auc_score
U = '/mnt/user-data/uploads/fartpie'
WL = pd.read_csv(f'{U}/gex_walls_nq.csv', parse_dates=['date']).set_index('date').sort_index()
DSIG = np.log(DY.c).diff().rolling(20).std().shift(1)
def tpo_counts(ix):
    bs = o[ix[0]] * .0005; lo0 = np.floor(l[ix].min() / bs); nb = int(np.floor(h[ix].max() / bs) - lo0) + 1; seen = [set() for _ in range(nb)]
    for hh, ll, pp in zip(h[ix], l[ix], (m[ix] - 570) // 30):
        for k in range(int(np.floor(ll / bs) - lo0), int(np.floor(hh / bs) - lo0) + 1): seen[k].add(pp)
    return bs, lo0, np.array([len(s) for s in seen])
def prior_stats(ix):
    bs, lo0, cnt = tpo_counts(ix); nb = len(cnt); pr = lambda k: (lo0 + k + .5) * bs
    poc = int(np.argmax(cnt)); tot = cnt.sum(); u = dn = poc; acc = cnt[poc]
    while acc < .7 * tot and (u < nb - 1 or dn > 0):
        up = cnt[u + 1] if u < nb - 1 else -1; dw = cnt[dn - 1] if dn > 0 else -1
        if up >= dw: u += 1; acc += up
        else: dn -= 1; acc += dw
    body = np.where(cnt >= 2)[0]; sp = [pr(k) for k in range(body.min(), body.max() + 1) if cnt[k] == 1] if len(body) else []
    return dict(tpoc=pr(poc), tval=(lo0 + dn) * bs, tvah=(lo0 + u + 1) * bs, sp=np.array(sp), poor_hi=cnt[-1] >= 2, poor_lo=cnt[0] >= 2, hi=h[ix].max(), lo=l[ix].min())
def extra():
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d]: continue
        ix = gi[d]; mm_ = m[ix]; endp = np.where(mm_ < 955)[0]
        if not len(endp): continue
        endp = endp[-1]; adr = DY.adr[d]; PS = prior_stats(gi[days[k - 1]]); sig = DSIG.get(d, np.nan) * c[gi[days[k - 1]][-1]]
        wl = WL[WL.index < d]; cw, pw = (wl.call_wall_nq.iloc[-1], wl.put_wall_nq.iloc[-1]) if len(wl) and (d - wl.index[-1]).days < 6 else (np.nan, np.nan)
        bs = o[ix[0]] * .0005; seen = {}; hi_s, lo_s, t_hi, t_lo = -np.inf, np.inf, 570, 570; ibh = ibl = np.nan
        for p in range(len(ix)):
            i = ix[p]; t = m[i] + 5; pp = (m[i] - 570) // 30
            if h[i] > hi_s: hi_s, t_hi = h[i], t
            if l[i] < lo_s: lo_s, t_lo = l[i], t
            for kk in range(int(np.floor(l[i] / bs)), int(np.floor(h[i] / bs)) + 1): seen.setdefault(kk, set()).add(pp)
            if t == 630: ibh, ibl = hi_s, lo_s
            if t % 15 or t < 585 or t > 900 or p + 1 > endp: continue
            px = c[i]; cur = int(np.floor(px / bs)); dpoc = (max(seen, key=lambda z: (len(seen[z]), -abs(z - cur))) + .5) * bs
            spp = PS['sp']; sp_up = min(((spp[spp > px] - px) / adr).min(), 1.0) if (spp > px).any() else 1.0; sp_dn = min(((px - spp[spp < px]) / adr).min(), 1.0) if (spp < px).any() else 1.0
            ph = (PS['hi'] - px) / adr if PS['poor_hi'] and PS['hi'] > px else 1.0; pl = (px - PS['lo']) / adr if PS['poor_lo'] and PS['lo'] < px else 1.0
            tloc = 1 if px > PS['tvah'] else (-1 if px < PS['tval'] else 0); ibr = (ibh - ibl) / adr if t >= 630 and ibh > ibl else np.nan
            ibp = (px - (ibh + ibl) / 2) / (ibh - ibl) if t >= 630 and ibh > ibl else np.nan; acc = len(seen.get(cur, ())) / (pp + 1)
            lo_in = float(t - t_lo >= 30 and px - lo_s >= .5 * sig); hi_in = float(t - t_hi >= 30 and hi_s - px >= .5 * sig)
            for dr in (1, -1):
                rows.append(dict(day=d, i=p, dir=dr, tpoc_dd=(px - PS['tpoc']) / adr * dr, tloc_d=tloc * dr, sp_ahead=sp_up if dr > 0 else sp_dn, sp_behind=sp_dn if dr > 0 else sp_up,
                                 poor_ahead=ph if dr > 0 else pl, ib_rng=ibr, ib_pos_d=ibp * dr, accept=acc, dpoc_dd=(px - dpoc) / adr * dr,
                                 since_behind=t - (t_lo if dr > 0 else t_hi), since_ahead=t - (t_hi if dr > 0 else t_lo), behind_in=lo_in if dr > 0 else hi_in,
                                 ahead_sig=((hi_s - px) if dr > 0 else (px - lo_s)) / sig, wall_ahead=((cw - px) if dr > 0 else (px - pw)) / adr,
                                 wall_behind=((px - pw) if dr > 0 else (cw - px)) / adr, px=px))
    return pd.DataFrame(rows)
GR = {'TPO': ['tpoc_dd', 'tloc_d', 'sp_ahead', 'sp_behind', 'poor_ahead', 'ib_rng', 'ib_pos_d', 'accept', 'dpoc_dd'],
      'HL': ['since_behind', 'since_ahead', 'behind_in', 'ahead_sig'], 'WALLS': ['wall_ahead', 'wall_behind']}
if __name__ == '__main__':
    P2 = '/tmp/claude-0/rules_ex.parquet'
    if os.path.exists(P2): EX = pd.read_parquet(P2)
    else: EX = extra(); EX.to_parquet(P2)
    DS = pd.read_parquet('/tmp/claude-0/rules_ds2.parquet').merge(EX, on=['day', 'i', 'dir'], how='left'); DS['yr'] = DS.day.dt.year
    print(f'rows {len(DS):,}; new features filled: ' + ', '.join(f'{c_} {DS[c_].notna().mean():.0%}' for g_ in GR.values() for c_ in g_[:2]), flush=True)
    mir = DS.set_index(['day', 'i', 'dir']); SETS = {'base': [], '+TPO': GR['TPO'], '+HL': GR['HL'], '+WALLS': GR['WALLS'], '+all': GR['TPO'] + GR['HL'] + GR['WALLS']}; TAKE = {}
    tr = DS.yr <= 2019; va = DS.yr.between(2020, 2021); te = DS.yr >= 2022; n_te = DS[te].day.nunique()
    for nm in ('C', 'D'):
        print(f'\n=== rules algo {nm} ===  test 2022 - Jan 2026', flush=True)
        for sn, add in SETS.items():
            cols = [x if x != 'tadr' else f'tadr{nm}' for x in FEAT] + add
            mdl = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=.05, min_samples_leaf=300, l2_regularization=1.0, random_state=0).fit(DS.loc[tr, cols], DS.loc[tr, f'y{nm}'])
            pr_ = mdl.predict_proba(DS[cols])[:, 1]; DS['_p'] = pr_; q = DS[te]; auc = roc_auc_score(q[f'y{nm}'], q._p); dec = pd.qcut(q._p, 10, labels=False, duplicates='drop')
            hr = q.groupby(dec)[f'y{nm}'].mean()
            def daily(msk, cut):
                z = DS[msk & (DS._p >= cut)]
                return z.sort_values(['day', 'tmin', '_p'], ascending=[True, True, False]).drop_duplicates('day')
            res = [(cut, daily(va, cut)[f'usd{nm}'].mean(), len(daily(va, cut))) for cut in np.round(np.arange(.40, .97, .01), 2)]; res = [r for r in res if r[2] >= 40]
            good = [r for r in res if r[1] >= 10]; cut = min(good)[0] if good else max(res, key=lambda r: r[1])[0]
            z = daily(te, cut); oth = np.array([mir[f'y{nm}'].get((a, b, -c_), np.nan) for a, b, c_ in zip(z.day, z.i, z.dir)], dtype=float)
            print(f'  {sn:<7} AUC {auc:.3f} | hit bottom / top decile {hr.iloc[0]:.0%} / {hr.iloc[-1]:.0%} | cut {cut:.2f}: {len(z) / n_te * 21:4.1f} trades/mo, hit {z[f"y{nm}"].mean():.0%} '
                  f'(other side {np.nanmean(oth):.0%}), ${z[f"usd{nm}"].mean():+5.1f} a trade = {z[f"pts{nm}"].mean():+.2f} pts, wins {z[f"y{nm}"].sum() / n_te * 21:4.1f}/mo | by year '
                  + ' '.join(f'{y % 100:02d}:${g_[f"usd{nm}"].mean():+.0f}' for y, g_ in z.groupby('yr')), flush=True)
            if sn in ('base', '+all'):
                w_ = daily(DS.yr >= 2014, cut); TAKE[(nm, sn)] = DS.loc[w_.index].assign(m=lambda q_: q_.entry_m, pts=lambda q_: q_[f'pts{nm}'], mae=lambda q_: q_[f'mae{nm}'],
                                                                                         sz=lambda q_: q_[f'mic{nm}'] / 10, ce=1.0, cf=1.0)[['day', 'm', 'pts', 'mae', 'sz', 'ce', 'cf']]
    # ---- Rey's levels and THOD lines on NQ, Oct 2024 - Jan 2026 ----
    print('\n=== Rey\'s QQQ algo levels and THOD/TLOD lines mapped to NQ (Oct 2024 - Jan 2026): decision rows within 0.03 ADR, trading the bounce ===', flush=True)
    rng = np.random.default_rng(7); Q = DS[DS.day >= '2024-10-14'].copy(); out = []
    for tf in ('1h', '4h'):
        TH = pd.read_csv(f'{U}/xs/ext/thod_hist/QQQ_{tf}.csv', parse_dates=['day']).drop_duplicates('day').set_index('day')
        LV = {pd.Timestamp(json.loads(x)['day']): json.loads(x) for x in open(f'{U}/xs/ext/levels_hist2/QQQ_{tf}.jsonl')}
        for kind in ('algo levels', 'THOD/TLOD'):
            for plc in (0, 1):
                hits = []
                for d, g_ in Q.groupby('day'):
                    if d not in TH.index or (kind == 'algo levels' and d not in LV): continue
                    r_ = TH.loc[d]; ratio = o[gi[d][0]] / r_.price; sg = (r_.hod1 - r_.price) * ratio
                    L = np.array([x['price'] for x in LV[d]['levels']]) * ratio if kind == 'algo levels' else np.array([r_[c_] for c_ in ('hod1', 'hod2', 'hod3', 'lod1', 'lod2', 'lod3')]) * ratio
                    if plc: L = L + rng.uniform(-.5, .5, len(L)) * sg
                    adr = DY.adr[d]
                    for rr in g_.itertuples():
                        gapv = (rr.px - L) * rr.dir
                        if ((gapv >= 0) & (gapv <= .03 * adr)).any(): hits.append((rr.yC, rr.usdC, rr.yD, rr.usdD))
                H_ = np.array(hits); out.append((tf, kind, plc, len(H_), *(H_.mean(0) if len(H_) else [np.nan] * 4)))
    for tf, kind, plc, n, yc, uc, yd, ud in out:
        print(f'  {tf} {kind:<12} {"PLACEBO" if plc else "real   "} n{n:5d} | C hit {yc:.1%} ${uc:+5.1f} | D hit {yd:.1%} ${ud:+5.1f}', flush=True)
    a = Q[['yC', 'usdC', 'yD', 'usdD']].mean(); print(f'  all rows in the window       n{len(Q):5d} | C hit {a.yC:.1%} ${a.usdC:+5.1f} | D hit {a.yD:.1%} ${a.usdD:+5.1f}', flush=True)
    # ---- Lucid: profit algo + rules C, base features vs + all ----
    gx = (DY.gex < 0); M1t, R1t = m1_trades(), r1_trades(); R1g = R1t[R1t.day.map(gx) == False]; M1g = M1t[M1t.day.map(gx) == True]
    M1gg = M1g[[not (fget(d, 'dexp') >= 2 / 3) for d in M1g.day]]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    OB = on_list(0, 600, .35, 'all'); RAs = ra_full2(True)
    OBpos = OB[[bool(DY.gex.get(d, np.nan) > 0) for d in OB.day]]; OBneg = OB[[bool(DY.gex.get(d, np.nan) < 0) for d in OB.day]]
    RULE = {'RA': (5, 'low', 'mid'), 'M1': (5, 'high', 'low'), 'R1': (1, 'low', 'mid'), 'ON+': (5, 'high', 'mid'), 'ON-': (1, 'high', 'low')}
    def lists(spec, phase):
        L = [[] for _ in range(NS)]
        for T_, u, yzrule in spec:
            for r_ in T_.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * (r_.sz * (r_.ce if phase == 'e' else r_.cf) if hasattr(r_, 'sz') else 1.0)
                if yzrule is not None:
                    n, best, worst = yzrule; v = DY[f'yz{n}p'].get(pd.Timestamp(r_.day), np.nan); b3 = t3(np.array([v]))[0]; uu *= 1.5 if b3 == best else (.5 if b3 == worst else 1.0)
                if uu > 0: L[k].append((r_.m, ((min(uu, 1.75),), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    base = [(RAs, 1.0, RULE['RA']), (M1gg, .3, RULE['M1']), (R1g, .5, RULE['R1']), (OBpos, .4, RULE['ON+']), (OBneg, .1, RULE['ON-'])]
    VAR = {'+ rules C, base features': base + [(TAKE[('C', 'base')], 1.0, None)], '+ rules C, + TPO / HL / walls': base + [(TAKE[('C', '+all')], 1.0, None)],
           '+ rules D, + TPO / HL / walls': base + [(TAKE[('D', '+all')], 1.0, None)]}
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); s22 = [s for s in starts if D_[s] >= pd.Timestamp('2022-01-01')]; LOCK['on'] = False
    print(f'\n  {"version":<32} {"acct":<5} qual/mo pass  paid  payouts x size  blown | pts per acct-month | 1st payout  every  payouts/mo funded   (accounts started 2022+)', flush=True)
    for nm, spec in VAR.items():
        DE = lists(spec, 'e'); DF = lists(spec, 'f')
        for a_ in ('50K', '150K'):
            A = ACC[a_]; usd = np.array([sum(net * 2 * max(1, round(10 * th[0] * A['scale'])) for th, net, mae, p in z) for z in DF])
            qm = np.mean(usd[[D_[k] >= pd.Timestamp('2022-01-01') for k in range(NS)]] >= A['qday']) * 21; Rr = []
            for s in s22:
                res, k2 = ev(A, s, NS)
                if res != 'pass': Rr.append((res, k2, [], 'none', k2 if res == 'fail' else NS)); continue
                fr, k3, pays = fu(A, k2 + 1, NS, 5); Rr.append((res, k2, pays, fr, NS if fr == 'open' else k3))
            net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in Rr]); life = np.array([(r[4] - s + 1) / 21 for r, s in zip(Rr, s22)])
            fund = [r for r in Rr if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a__[0] for r in fund for a__, b in zip(r[2][:-1], r[2][1:])]
            fp = [(D_[min(r[2][0][0], NS - 1)] - D_[s]).days / 30.44 for r, s in zip(Rr, s22) if r[2]]; flife = sum((r[4] - r[1]) / 21 for r in fund)
            print(f'  {nm:<32} {a_:<5} {qm:7.1f} {np.mean([r[0] == "pass" for r in Rr]):4.0%} {np.mean([len(r[2]) > 0 for r in Rr]):5.0%}  '
                  f'{np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x ${np.mean(pays) if pays else 0:5.0f} {np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | '
                  f'{net.sum() / life.sum():+18.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {len(pays) / flife if flife else 0:10.2f}', flush=True)
