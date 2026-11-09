"""Nov 9 2026: THE RULES ALGO - a separate model whose only job is Lucid qualifying days, next to the profit algo (dealer-regime model). DECLARED BEFORE RESULTS.
Goal: 2+ payouts a month of > $1,350 each = ~75 NQ points of profit + 5 qualifying days (+$150 = +7.5 pts at 10 micros) per payout.
Decision points: every 15m bar close 09:45 - 15:00 on real NQ futures 5m bars (2014 - Jan 2026), long AND short; entry at the next 5m open; flat 15:55.
Labels (first touch, a bar touching both: bullish bar = open-low-high-close, bearish = open-high-low-close; 1 pt cost):
  A  +8.5 gross (+7.5 net) before -20, 10 micros         B  +16 gross (+15 net) before -40, 5 micros
  RUN 1 RESULT (kept for the record): fixed-point labels do not carry across eras - +15 before -40 hit 39% in 2014-19 and 71% in 2022-26, so the model learned
  'volatility is high', not 'this spot is good'; A out of sample hit 68% vs 74% the other way. Added after run 1, before any C / D result:
  C  +(0.06 ADR + 1) gross before -0.12 ADR      D  +(0.10 ADR + 1) gross before -0.15 ADR      (ADR = 20-day average RTH range, known at the open)
     sized so a win pays >= $150: micros = ceil(150 / (2 x net target)) - today ~18-30 pt targets at 3-5 micros; scale-free, so 2014-19 training means something.
Features (all known at the decision; directional ones signed by the trade side):
  dealer: QQQ gamma pct, GEX sign (2018+), DEX pct, ATM IV pct, skew pct, put/call volume pct, vanna and charm signs | vol: Yang-Zhang 1/5/20-day pct, ADR, target/ADR,
  last-hour 5m range/ADR, overnight range/ADR | location: price vs prior POC (ADR), above VAH / inside / below VAL, open's place vs prior value, VWAP distance (sd),
  move from open (ADR), noise-band z (move from open / 14-day average move at this time), move from prior close, place in today's range, range so far / ADR, gap |
  momentum: 30- and 60-minute move, 15m vector state and bars since its flip | time of day.
Model: gradient boosting (depth 4, 300 trees, lr 0.05, min leaf 300). Train 2014-2019; threshold chosen on 2020-21 = the LOWEST probability cut whose validation
average is >= +$10 a trade at its size (most trading days at a positive edge); judged on 2022 - Jan 2026, never touched before.
Rules algo: each day the first decision whose better side clears the cut; one trade a day. Mirror check: the same trades taken the other way.
Then in Lucid with the profit algo (U3 + YZ): + rules B / C / D, both phases and funded only (eval has no qualifying days); vs the old qualifier (first noise
signal, +7.5 / 40 stop). Accounts started 2022+ (out of sample for the model) and all starts."""
_q = open('/home/claude/fartpie/xs/dte_qual.py').read(); exec(_q[:_q.rindex("if __name__ == '__main__':")])
import os
from sklearn.ensemble import HistGradientBoostingClassifier
CFG = {'A': ('pts', 8.5, 20.0, 10), 'B': ('pts', 16.0, 40.0, 5), 'C': ('adr', .06, .12, None), 'D': ('adr', .10, .15, None)}
ddi = pd.DatetimeIndex(dd); o_day = DY.o.reindex(ddi).values
PV = pd.DataFrame({'day': dd, 'm': m, 'mv': np.abs(c / o_day - 1)}).pivot_table(index='day', columns='m', values='mv'); SIGN = PV.rolling(14, min_periods=10).mean().shift(1)
grp = (B15.reg != B15.reg.shift()).cumsum(); B15['since'] = B15.groupby(grp).cumcount(); REG = dict(zip(B15.index.values, zip(B15.reg.values, B15.since.values)))
FEAT = ['dir', 'tmin', 'g2p', 'gexs', 'dexp', 'ivp', 'skewp', 'pcvp', 'vanna', 'charm', 'yz1', 'yz5', 'yz20', 'adr', 'tadr', 'rv1h', 'onr', 'poc_d', 'va_loc', 'open_loc',
        'vwz', 'from_open', 'noise_z', 'from_pc', 'rng_pos', 'rng_adr', 'gap', 'mom30', 'mom60', 'vec', 'since']
def first_touch(ix, j, end, dr, T, S):
    e = o[ix[j]]; hi, lo, op, cl = h[ix[j:end + 1]], l[ix[j:end + 1]], o[ix[j:end + 1]], c[ix[j:end + 1]]
    adv = (lo <= e - S) if dr > 0 else (hi >= e + S); fav = (hi >= e + T) if dr > 0 else (lo <= e - T)
    ia = int(np.argmax(adv)) if adv.any() else 10 ** 9; ib = int(np.argmax(fav)) if fav.any() else 10 ** 9
    exc = ((lo - e) if dr > 0 else (e - hi))
    if ia == 10 ** 9 and ib == 10 ** 9: return 0, (cl[-1] - e) * dr - 1.0, exc.min() - 1.0
    stop_first = ((cl[ia] >= op[ia]) == (dr > 0)) if ia == ib else ia < ib
    if stop_first: return 0, -S - 1.0, -S - 1.0
    return 1, T - 1.0, max(exc[:ib + 1].min(), -S) - 1.0
def build():
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d]: continue
        ix = gi[d]; mm_ = m[ix]; endp = np.where(mm_ < 955)[0]
        if not len(endp): continue
        endp = endp[-1]; adr = DY.adr[d]; o0 = o[ix[0]]; pc = c[gi[days[k - 1]][-1]]; val, poc, vah = VA[days[k - 1]]; f = DAYF.get(d)
        g = lambda key: (f[key] if f is not None and key in f and pd.notna(f[key]) else np.nan)
        gx = DY.gex.get(d, np.nan); onr = NSO['NR'].get(d, np.nan)
        dayf = dict(g2p=DY.g2p.get(d, np.nan), gexs=np.sign(gx) if np.isfinite(gx) else 0.0, dexp=g('dexp'), ivp=g('ivp'), skewp=g('skewp'), pcvp=g('pcvp'),
                    vanna=np.sign(g('net_vanna')), charm=np.sign(g('net_charm')), yz1=DY.yz1p.get(d, np.nan), yz5=DY.yz5p.get(d, np.nan), yz20=DY.yz20p.get(d, np.nan), adr=adr,
                    onr=onr / adr)
        TS = {nm: ((a, b) if kind == 'pts' else (a * adr + 1.0, b * adr)) for nm, (kind, a, b, mic) in CFG.items()}
        oloc = 1 if o0 > vah else (-1 if o0 < val else 0); hi_s = -np.inf; lo_s = np.inf
        for p in range(len(ix)):
            i = ix[p]; hi_s = max(hi_s, h[i]); lo_s = min(lo_s, l[i]); t = m[i] + 5
            if t % 15 or t < 585 or t > 900 or p + 1 > endp: continue
            px = c[i]; sg = SIGN.at[d, m[i]] if (d in SIGN.index and m[i] in SIGN.columns) else np.nan
            rg, sn = REG.get(np.datetime64(H.dt.values[i] - np.timedelta64(10, 'm')), (np.nan, np.nan))
            rv = (h[ix[max(0, p - 11):p + 1]] - l[ix[max(0, p - 11):p + 1]]).mean() / adr
            base = dict(day=d, i=p, tmin=t, rv1h=rv, rng_adr=(hi_s - lo_s) / adr, **dayf)
            for dr in (1, -1):
                row = dict(base, dir=dr, poc_d=(px - poc) / adr * dr, va_loc=(1 if px > vah else (-1 if px < val else 0)) * dr, open_loc=oloc * dr,
                           vwz=((px - vw[i]) / vsd[i] if vsd[i] > 0 else np.nan) * dr, from_open=(px - o0) / adr * dr,
                           noise_z=((px / o0 - 1) / sg if np.isfinite(sg) and sg > 0 else np.nan) * dr, from_pc=(px - pc) / adr * dr,
                           rng_pos=((px - lo_s) / (hi_s - lo_s) if hi_s > lo_s else .5) if dr > 0 else ((hi_s - px) / (hi_s - lo_s) if hi_s > lo_s else .5),
                           gap=(o0 - pc) / adr * dr, mom30=((px - c[ix[p - 6]]) / adr * dr) if p >= 6 else np.nan, mom60=((px - c[ix[p - 12]]) / adr * dr) if p >= 12 else np.nan,
                           vec=rg * dr if np.isfinite(rg) else np.nan, since=sn, entry_m=m[ix[p + 1]])
                for nm, (T, S) in TS.items():
                    y, pts, mae = first_touch(ix, p + 1, endp, dr, T, S); mic = CFG[nm][3] or int(np.ceil(150 / (2 * (T - 1.0))))
                    row[f'y{nm}'] = y; row[f'pts{nm}'] = pts; row[f'mae{nm}'] = mae; row[f'tadr{nm}'] = T / adr; row[f'mic{nm}'] = mic; row[f'usd{nm}'] = pts * 2 * mic
                rows.append(row)
    return pd.DataFrame(rows)
if __name__ == '__main__':
    P = '/tmp/claude-0/rules_ds2.parquet'
    if os.path.exists(P): DS = pd.read_parquet(P)
    else: DS = build(); DS.to_parquet(P)
    DS['yr'] = DS.day.dt.year; print(f'decision rows {len(DS):,} on {DS.day.nunique()} days ({DS.day.min().date()} - {DS.day.max().date()})', flush=True)
    mir = DS.set_index(['day', 'i', 'dir'])
    for nm, (kind, a, b, mic0) in CFG.items():
        cols = [x if x != 'tadr' else f'tadr{nm}' for x in FEAT]; tr = DS.yr <= 2019; va = DS.yr.between(2020, 2021); te = DS.yr >= 2022
        mdl = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=.05, min_samples_leaf=300, l2_regularization=1.0, random_state=0).fit(DS.loc[tr, cols], DS.loc[tr, f'y{nm}'])
        DS[f'p{nm}'] = mdl.predict_proba(DS[cols])[:, 1]; nl = b / (a + b) if kind == 'adr' else b / (a + b)
        lab_ = f'+{a - 1:g} pts net before -{b:g}' if kind == 'pts' else f'+{a:g} ADR (+1) before -{b:g} ADR'
        print(f'\n=== {nm}: {lab_} (no-edge line ~{nl:.0%}); size {"%d micros" % mic0 if mic0 else "micros = ceil(150 / (2 x target))"} ===', flush=True)
        for lab, msk in (('train 14-19', tr), ('valid 20-21', va), ('test 22-26', te)):
            q = DS[msk]; dec = pd.qcut(q[f'p{nm}'], 10, labels=False, duplicates='drop'); hr = q.groupby(dec)[f'y{nm}'].mean()
            print(f'  {lab}: base hit {q[f"y{nm}"].mean():.0%} | hit by predicted decile (low -> high): ' + ' '.join(f'{v:.0%}' for v in hr.values), flush=True)
        def daily(msk, cut):
            q = DS[msk & (DS[f'p{nm}'] >= cut)].copy()
            if not len(q): return q
            return q.sort_values(['day', 'tmin', f'p{nm}'], ascending=[True, True, False]).drop_duplicates('day')
        res = []
        for cut in np.round(np.arange(.40, .97, .01), 2):
            q = daily(va, cut)
            if len(q) >= 40: res.append((cut, q[f'usd{nm}'].mean(), len(q)))
        good = [r for r in res if r[1] >= 10]; cut = min(good)[0] if good else max(res, key=lambda r: r[1])[0]
        print(f'  validation 2020-21 ($ a trade at size, trades): ' + ' '.join(f'{c_:.2f}:{e:+.0f}({n})' for c_, e, n in res[::4]) + f'  -> chosen cut {cut:.2f}', flush=True)
        for lab, msk in (('train 14-19', tr), ('valid 20-21', va), ('test 22-26', te)):
            q = daily(msk, cut); n_ = DS[msk].day.nunique()
            oth = np.array([mir[f'y{nm}'].get((d_, i_, -dr_), np.nan) for d_, i_, dr_ in zip(q.day, q.i, q.dir)], dtype=float)
            oth_u = np.array([mir[f'usd{nm}'].get((d_, i_, -dr_), np.nan) for d_, i_, dr_ in zip(q.day, q.i, q.dir)], dtype=float)
            print(f'  {lab}: {len(q)} trades ({len(q) / n_ * 21:.1f} a month), hit {q[f"y{nm}"].mean():.0%} (other side {np.nanmean(oth):.0%}), avg ${q[f"usd{nm}"].mean():+.0f} '
                  f'= {q[f"pts{nm}"].mean():+.2f} pts at {q[f"mic{nm}"].mean():.1f} micros (other side ${np.nanmean(oth_u):+.0f}), wins (qualifying) {q[f"y{nm}"].sum() / n_ * 21:.1f} a month | '
                  + ' '.join(f'{y % 100:02d}:${g_[f"usd{nm}"].mean():+.0f}/{g_[f"y{nm}"].mean():.0%}/{len(g_)}' for y, g_ in q.groupby('yr')), flush=True)
        DS[f'take{nm}'] = False; q = daily(DS.yr >= 2014, cut); DS.loc[q.index, f'take{nm}'] = True
    DS.to_parquet('/tmp/claude-0/rules_ds2_scored.parquet')
    # ---- Lucid: profit algo + rules algo ----
    gx = (DY.gex < 0); M1t, R1t = m1_trades(), r1_trades(); R1g = R1t[R1t.day.map(gx) == False]; M1g = M1t[M1t.day.map(gx) == True]
    M1gg = M1g[[not (fget(d, 'dexp') >= 2 / 3) for d in M1g.day]]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    OB = on_list(0, 600, .35, 'all'); RAs = ra_full2(True)
    OBpos = OB[[bool(DY.gex.get(d, np.nan) > 0) for d in OB.day]]; OBneg = OB[[bool(DY.gex.get(d, np.nan) < 0) for d in OB.day]]
    E_ = entries(); hm = {}
    for d in days:
        for i in gi[d]: hm[h2x[i]] = m[i]
    def touch2(j, jend, dr, S_):
        pts, jx = touch(j, jend, dr, S_); e = Xo[j]; seg = Xl[j:jx + 1] if dr > 0 else Xh[j:jx + 1]
        return pts, (max(min(((seg - e) * dr).min(), 0.0), -S_) if pts > -S_ - 1 else -S_) - 1.0
    Q40 = pd.DataFrame([(d, hm[j]) + touch2(j, end_x[d], dr, 40) for d, j, dr in E_['NOISE']], columns=['day', 'm', 'pts', 'mae'])
    RL = {nm: DS[DS[f'take{nm}']].assign(m=lambda z: z.entry_m, pts=lambda z: z[f'pts{nm}'], mae=lambda z: z[f'mae{nm}'], sz=lambda z: z[f'mic{nm}'] / 10, ce=1.0, cf=1.0)[
          ['day', 'm', 'pts', 'mae', 'sz', 'ce', 'cf']] for nm in CFG}
    RULE = {'RA': (5, 'low', 'mid'), 'M1': (5, 'high', 'low'), 'R1': (1, 'low', 'mid'), 'ON+': (5, 'high', 'mid'), 'ON-': (1, 'high', 'low')}
    def lists(spec, phase):
        L = [[] for _ in range(NS)]
        for T_, u, yzrule, ph in spec:
            if ph is not None and ph != phase: continue
            for r_ in T_.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * (r_.sz * (r_.ce if phase == 'e' else r_.cf) if hasattr(r_, 'sz') else 1.0)
                if yzrule is not None:
                    n, best, worst = yzrule; v = DY[f'yz{n}p'].get(pd.Timestamp(r_.day), np.nan); b3 = t3(np.array([v]))[0]; uu *= 1.5 if b3 == best else (.5 if b3 == worst else 1.0)
                if uu > 0: L[k].append((r_.m, ((min(uu, 1.75),), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    base = [(RAs, 1.0, RULE['RA'], None), (M1gg, .3, RULE['M1'], None), (R1g, .5, RULE['R1'], None), (OBpos, .4, RULE['ON+'], None), (OBneg, .1, RULE['ON-'], None)]
    VAR = {'profit algo only (U3 + YZ)': base, '+ old qualifier (noise +7.5/40)': base + [(Q40, 1.0, None, None)],
           '+ rules B (+15/40, 5 mic)': base + [(RL['B'], 1.0, None, None)], '+ rules C (ADR-scaled)': base + [(RL['C'], 1.0, None, None)],
           '+ rules C funded only': base + [(RL['C'], 1.0, None, 'f')], '+ rules D (ADR-scaled)': base + [(RL['D'], 1.0, None, None)],
           '+ rules D funded only': base + [(RL['D'], 1.0, None, 'f')]}
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); s22 = [s for s in starts if D_[s] >= pd.Timestamp('2022-01-01')]; LOCK['on'] = False
    print(f'\n  {"version":<32} {"acct":<5} {"starts":<6} qual/mo pass  paid  payouts x size  blown | per acct-month | 1st payout  every  payouts/mo funded', flush=True)
    for nm, spec in VAR.items():
        DE = lists(spec, 'e'); DF = lists(spec, 'f')
        for a_ in ('50K', '150K'):
            A = ACC[a_]; usd = np.array([sum(net * 2 * max(1, round(10 * th[0] * A['scale'])) for th, net, mae, p in z) for z in DF])
            qm = np.mean(usd[[D_[k] >= pd.Timestamp('2022-01-01') for k in range(NS)]] >= A['qday']) * 21
            for slab, ST in (('2022+', s22), ('all', starts)):
                Rr = []
                for s in ST:
                    res, k2 = ev(A, s, NS)
                    if res != 'pass': Rr.append((res, k2, [], 'none', k2 if res == 'fail' else NS)); continue
                    fr, k3, pays = fu(A, k2 + 1, NS, 5); Rr.append((res, k2, pays, fr, NS if fr == 'open' else k3))
                net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in Rr]); life = np.array([(r[4] - s + 1) / 21 for r, s in zip(Rr, ST)])
                fund = [r for r in Rr if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a__[0] for r in fund for a__, b in zip(r[2][:-1], r[2][1:])]
                fp = [(D_[min(r[2][0][0], NS - 1)] - D_[s]).days / 30.44 for r, s in zip(Rr, ST) if r[2]]; flife = sum((r[4] - r[1]) / 21 for r in fund)
                print(f'  {nm:<32} {a_:<5} {slab:<6} {qm:7.1f} {np.mean([r[0] == "pass" for r in Rr]):4.0%} {np.mean([len(r[2]) > 0 for r in Rr]):5.0%}  '
                      f'{np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x ${np.mean(pays) if pays else 0:5.0f} {np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | '
                      f'{net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {len(pays) / flife if flife else 0:10.2f}', flush=True)
