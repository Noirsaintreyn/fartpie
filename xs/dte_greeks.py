"""Nov 8 2026: ALL THE GREEKS on the new engines (real NQ futures 5m, 2014 - Jan 2026). DECLARED BEFORE RESULTS.
Daily QQQ options aggregates at the PRIOR close (ThetaData files): net GEX (own, 2018+; and call - put GEX from the greeks file, 2014+), net DEX (call + put dealer delta),
net vanna, net charm, ATM IV level and 5-day change, 25-delta skew, put/call volume and open interest, call / put walls (NQ points).
Engines: M1 noise momentum (both directions, stop 0.15 ADR) | ON overnight long 18:00 - 04:00, stop 0.35 x overnight range | R1 80% rule | RA Rule A (75 / 50 noon).
Directional features are read in the trade's direction ('agrees' = the Greek leans the way the trade goes): DEX sign, vanna flow (sign(vanna) x -sign(5-day IV change)),
charm sign, room to the wall the trade runs into (call wall for longs, put wall for shorts, in ADR).
A feature counts when its best bucket beats its worst in every period with 15+ trades (2014-17 / 2018-21 / 2022-25)."""
_e = open('/home/claude/fartpie/xs/dte_engines.py').read(); exec(_e[:_e.rindex("if __name__ == '__main__':")])
NSO = {}; _s = open('/home/claude/fartpie/xs/dte_overnight_stops.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")], NSO)
U = '/mnt/user-data/uploads/fartpie'
G1 = pd.read_csv(f'{U}/options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); G2 = pd.read_csv(f'{U}/options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
W = pd.read_csv(f'{U}/gex_walls_nq.csv', parse_dates=['date']).set_index('date').sort_index(); OWN = pd.read_csv(f'{U}/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index()
for f_ in (G1, G2, W): f_.index = f_.index.normalize()
OWN.index = OWN.index.normalize()
pct = lambda s: s.rolling(252, min_periods=120).rank(pct=True)
F = pd.DataFrame(index=G1.index); F['gex2'] = G1.call_gex - G1.put_gex; F['dex'] = G1.call_dex + G1.put_dex; F['iv'] = G1.atm_iv; F['skew25'] = G1.skew_25d
F['pcv'] = G1.put_volume / G1.call_volume; F['pco'] = G1.put_oi / G1.call_oi; F = F.join(G2[['net_vanna', 'net_charm']]).join(W[['call_wall_nq', 'put_wall_nq']]).join(OWN.rename('gex'), how='outer')
F['iv5'] = F.iv.diff(5); F['gexp'] = pct(F.gex); F['gex2p'] = pct(F.gex2); F['dexp'] = pct(F.dex); F['ivp'] = pct(F.iv); F['skewp'] = pct(F.skew25); F['pcvp'] = pct(F.pcv); F['pcop'] = pct(F.pco)
F['vflow'] = np.sign(F.net_vanna) * -np.sign(F.iv5)
def prior(d):
    i = F.index.searchsorted(d) - 1
    return F.iloc[i] if i >= 0 else None
DAYF = {d: prior(d) for d in DY.index}
def m1_dir():
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or d not in SIG.index: continue
        ix = gi[d]; o0 = o[ix[0]]; pc = c[gi[days[k - 1]][-1]]; S = .15 * DY.adr[d]; pos = 0; e = None; pend = None
        for p, i in enumerate(ix):
            if pend is not None: pos, e = pend, o[i]; pend = None
            if pos != 0 and ((l[i] <= e - S) if pos > 0 else (h[i] >= e + S)): rows.append((d, pos, -S - 1.0, e)); pos = 0; continue
            if p == len(ix) - 1: break
            if (m[i] + 5) % 30 != 0 or m[i] + 5 < 600 or m[i] + 5 > 930: continue
            s = SIG.loc[d].get(p, np.nan)
            if not np.isfinite(s): continue
            ub, lb = max(o0, pc) * (1 + s), min(o0, pc) * (1 - s); nxt = o[ix[p + 1]]
            if pos == 1 and c[i] < max(ub, vw[i]): rows.append((d, 1, nxt - e - 1.0, e)); pos = 0
            elif pos == -1 and c[i] > min(lb, vw[i]): rows.append((d, -1, e - nxt - 1.0, e)); pos = 0
            if pos == 0 and c[i] > ub: pend = 1
            elif pos == 0 and c[i] < lb: pend = -1
        if pos != 0: rows.append((d, pos, (c[ix[-1]] - e) * pos - 1.0, e))
    return pd.DataFrame(rows, columns=['day', 'dir', 'pts', 'e'])
def r1_dir():
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d]: continue
        val, poc, vah = VA[days[k - 1]]; ix = gi[d]; o0 = o[ix[0]]; side = 1 if o0 > vah else (-1 if o0 < val else 0)
        if side == 0: continue
        run = 0
        for p, i in enumerate(ix[:-1]):
            if m[i] >= 780: break
            run = run + 1 if val <= c[i] <= vah else 0
            if run >= 6:
                d_ = -side; e = o[ix[p + 1]]; T = val if d_ < 0 else vah; S = (vah + .08 * DY.adr[d]) if d_ < 0 else (val - .08 * DY.adr[d])
                if (T - e) * d_ <= 0 or (e - S) * d_ <= 0: break
                rows.append((d, d_, walk(ix, p + 1, d_, e, T, S) - 1.0, e)); break
    return pd.DataFrame(rows, columns=['day', 'dir', 'pts', 'e'])
def on_dir():
    rows = []
    for d in NSO['R'].index:
        if d not in NSO['gi'] or not NSO['symok'].get(d, False): continue
        r_ = NSO['night'](d, 0, 600, .35)
        if r_ is not None: rows.append((d, 1, r_[0], np.nan))
    return pd.DataFrame(rows, columns=['day', 'dir', 'pts', 'e'])
def ra_dir():
    R0 = pd.read_csv('/home/claude/fartpie/live_data/nq_rule_a_trades.csv'); tp_ = {t: i for i, t in enumerate(H.dt.values)}; rows = []
    for r_ in R0.itertuples():
        i = tp_.get(np.datetime64(pd.Timestamp(r_.entry), 'ns'))
        if i is None: continue
        d = pd.Timestamp(dd[i]); ix = gi[d]; j = int(np.where(ix == i)[0][0]); e = o[i]; dr = int(r_.dir)
        rows.append((d, dr, walk(ix, j, dr, e, e + 75 * dr, e - 50 * dr, until_m=720) - 1.0, e))
    return pd.DataFrame(rows, columns=['day', 'dir', 'pts', 'e'])
def label(T):
    L = {}
    fz = [DAYF.get(d) for d in T.day]; get = lambda col: np.array([f[col] if f is not None and col in f.index else np.nan for f in fz], dtype=float)
    dr = T.dir.values.astype(float); adr = DY.adr.reindex(T.day).values
    sg = lambda x, a='neg', b='pos': np.where(np.isnan(x), None, np.where(x < 0, a, b))
    t3 = lambda x: np.where(np.isnan(x), None, np.where(x <= 1 / 3, 'low', np.where(x >= 2 / 3, 'high', 'mid')))
    L['GEX sign (own, 2018+)'] = sg(get('gex')); L['GEX sign (greeks file, 2014+)'] = sg(get('gex2')); L['GEX percentile'] = t3(get('gex2p'))
    L['DEX percentile'] = t3(get('dexp')); L['DEX agrees with trade'] = sg(np.sign(get('dex')) * dr, 'against', 'agrees')
    L['vanna sign'] = sg(get('net_vanna')); L['vanna flow agrees'] = sg(get('vflow') * dr, 'against', 'agrees'); L['charm sign'] = sg(get('net_charm'))
    L['charm agrees with trade'] = sg(np.sign(get('net_charm')) * dr, 'against', 'agrees'); L['ATM IV percentile'] = t3(get('ivp'))
    L['IV 5-day change'] = sg(get('iv5'), 'falling', 'rising'); L['skew percentile'] = t3(get('skewp')); L['put/call volume pct'] = t3(get('pcvp')); L['put/call OI pct'] = t3(get('pcop'))
    if np.isfinite(T.e.values.astype(float)).any():
        wall = np.where(dr > 0, get('call_wall_nq'), get('put_wall_nq')); room = (wall - T.e.values) * dr / adr
        L['room to wall (ADR)'] = np.where(np.isnan(room), None, np.where(room < 0, 'beyond wall', np.where(room < .5, '< 0.5', np.where(room < 1.5, '0.5-1.5', '> 1.5'))))
    return L
PER = [('14-17', 2014, 2017), ('18-21', 2018, 2021), ('22-25', 2022, 2026)]
if __name__ == '__main__':
    ENG = {'M1 noise': m1_dir(), 'ON overnight 18-04': on_dir(), 'R1 80% rule': r1_dir(), 'RA Rule A': ra_dir()}
    SUM = []
    for en, T in ENG.items():
        T = T[T.day.isin(DY.index)].reset_index(drop=True); yr = T.day.dt.year.values; L = label(T); print(f'\n=== {en}: {len(T)} trades, avg {T.pts.mean():+.2f}', flush=True)
        for fn, lab in L.items():
            lab = pd.Series(lab); out = []
            for v in [x for x in lab.dropna().unique()]:
                msk = (lab == v).values; per = [T.pts.values[msk & (yr >= a) & (yr <= b)].mean() if (msk & (yr >= a) & (yr <= b)).sum() >= 15 else np.nan for _, a, b in PER]
                out.append((v, msk.sum(), T.pts.values[msk].mean(), per))
            if len(out) < 2: continue
            out.sort(key=lambda z: -z[2]); b_, w_ = out[0], out[-1]; okp = [(x, y) for x, y in zip(b_[3], w_[3]) if np.isfinite(x) and np.isfinite(y)]
            cons = len(okp) >= 2 and all(x > y for x, y in okp); SUM.append((en, fn, b_[2] - w_[2], cons, b_[0], w_[0], len(okp)))
            print(f'  {fn:<30} spread {b_[2] - w_[2]:+6.2f}{" *" if cons else "  "} | ' + ' | '.join(f'{v} n{n} {mu:+.2f} (' + '/'.join(f'{x:+.1f}' if np.isfinite(x) else '.' for x in pp) + ')' for v, n, mu, pp in out), flush=True)
    print('\nSUMMARY: * = best bucket beats worst in every period with 15+ trades (at least 2 periods)', flush=True)
    for en, fn, sp, cons, b_, w_, np_ in sorted(SUM, key=lambda z: (not z[3], -z[2])):
        if cons: print(f'  * {en:<20} {fn:<30} spread {sp:+6.2f}  best {b_} / worst {w_}  ({np_} periods)', flush=True)
    print('  (not consistent: ' + ', '.join(f'{en.split()[0]}:{fn}' for en, fn, sp, cons, b_, w_, np_ in SUM if not cons) + ')', flush=True)
