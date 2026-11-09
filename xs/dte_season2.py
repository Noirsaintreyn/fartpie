"""Nov 8 2026: THE SEASON IS GAMMA, NOT THE SIGNAL. DECLARED BEFORE RESULTS.
xs/dte_season.py: choosing a signal per regime on 2014-21 failed out of sample (book Sharpe 0.78 vs the vector alone 1.09); the per-cell picks were noise.
What repeated: trend signals did worst when the prior close's QQQ gamma was in its top third (dealers long gamma dampen moves), the same reason noise momentum
only worked in negative gamma. One rule tested, nothing tuned:
 1) each signal, points per trade, gamma top third vs the rest, 2014-17 / 2018-21 / 2022-25; GEX sign (2018+)
 2) books at 1 contract, 2022 - Jan 2026: each trend signal with and without 'off when gamma is top third'; the vector by year
 3) as an engine in the dealer-regime model: V15 = vector 15m flips, exit on the opposite flip, stop 0.15 ADR, off when gamma is top third - (a) all day,
    (b) afternoon only (flips 12:00 - 15:30, after Rule A is flat). Weight 0.3 (3 micros) fixed in advance, no YZ sizing on it.
    Lucid 50K / 150K: U3 + YZ (current best) vs + V15a vs + V15b; accounts started Feb 2018 - Jun 2025, and those started 2022+."""
_y = open('/home/claude/fartpie/xs/dte_yz.py').read(); exec(_y[:_y.rindex("if __name__ == '__main__':")])
import sys; sys.path.insert(0, '/home/claude/fartpie/xs'); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']; NSL = {}
exec(open('/home/claude/fartpie/xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0], NSL); sys.argv = _a; vec = NSL['vec']
G1o = pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); G1o.index = G1o.index.normalize()
g2 = (G1o.call_gex - G1o.put_gex); DY['g2p'] = g2.rolling(252, min_periods=120).rank(pct=True).reindex(DY.index, method='ffill').shift(1)
H['b15'] = H.dt.dt.floor('15min'); B15 = H.groupby('b15').agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), day=('day', 'first'))
B15['reg'] = np.asarray(vec(B15[['open', 'high', 'low', 'close']])); B15['end'] = B15.index + pd.Timedelta(minutes=15); tpos5 = {t: i for i, t in enumerate(H.dt.values)}
_s = open('/home/claude/fartpie/xs/dte_season.py').read(); exec(_s[_s.index('ema_ = lambda'):_s.index('def run_sig')])
def run_sig2(col, lo=585, hi=930):
    s = np.nan_to_num(B[col].values.astype(float)); ends = B.end.values; dys = B.day.values; mm_ = (pd.DatetimeIndex(ends).hour * 60 + pd.DatetimeIndex(ends).minute).values; rows = []; pos_end = {}
    bidx = {t: q for q, t in enumerate(B.index.values)}
    for q in range(1, len(B)):
        if s[q] == s[q - 1] or s[q] == 0 or not (lo <= mm_[q] <= hi): continue
        d = pd.Timestamp(dys[q])
        if not ok(d): continue
        i = tpos5.get(np.datetime64(ends[q], 'ns'))
        if i is None or dd[i] != np.datetime64(d): continue
        cnt, busy = pos_end.get(d, (0, -1))
        if cnt >= 4 or i <= busy: continue
        dr = int(s[q]); e = o[i]; S = e - dr * .15 * DY.adr[d]; ix = gi[d]; j = int(np.where(ix == i)[0][0]); res = None; worst = 0.0
        for p in range(j, len(ix)):
            ii = ix[p]
            if m[ii] >= 955: res = ((c[ix[p - 1]] - e) * dr, ii); break
            if (l[ii] <= S) if dr > 0 else (h[ii] >= S): res = ((S - e) * dr, ii); worst = min(worst, (S - e) * dr); break
            worst = min(worst, ((l[ii] if dr > 0 else h[ii]) - e) * dr)
            if (m[ii] + 5) % 15 == 0 and p + 1 < len(ix):
                bq = bidx.get(H.dt.values[ii] - np.timedelta64(10, 'm'))
                if bq is not None and s[bq] == -dr: res = ((o[ix[p + 1]] - e) * dr, ix[p + 1]); break
        if res is None: res = ((c[ix[-1]] - e) * dr, ix[-1])
        rows.append((d, m[i], dr, res[0] - 1.0, worst - 1.0)); pos_end[d] = (cnt + 1, res[1])
    return pd.DataFrame(rows, columns=['day', 'm', 'dir', 'pts', 'mae'])
hig = lambda T: (T.day.map(DY.g2p) >= 2 / 3).values
if __name__ == '__main__':
    TR = {col[2:]: run_sig2(col) for col in SIGS}
    PER = [(2014, 2017), (2018, 2021), (2022, 2026)]
    print('1) points per trade: gamma TOP THIRD vs the rest | 2014-17 / 2018-21 / 2022-25 | GEX sign 2018+ (negative / positive)', flush=True)
    for k, T in TR.items():
        yr = T.day.dt.year.values; hg = hig(T); kn = T.day.map(DY.g2p).notna().values; gx = T.day.map(DY.gex).values
        f = lambda msk: ' / '.join(f'{T.pts.values[msk & (yr >= a) & (yr <= b)].mean():+5.2f}' for a, b in PER)
        print(f'  {k:<8} top third {f(hg)}  | rest {f(kn & ~hg)}  | GEX<0 {T.pts.values[gx < 0].mean():+5.2f} GEX>0 {T.pts.values[gx > 0].mean():+5.2f}', flush=True)
    AD22 = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2022-01-01') and ok(d)])
    def book(T, lab):
        T = T[T.day >= '2022-01-01']; P = T.groupby('day').pts.sum().reindex(AD22).fillna(0.0); cum = P.cumsum()
        print(f'  {lab:<34} trades {len(T):5d} avg {T.pts.mean():+6.2f} | {P.mean():+5.2f} pts/day, Sharpe {P.mean() / P.std() * np.sqrt(252):+.2f}, max DD {(cum - cum.cummax()).min():+5.0f}', flush=True)
    print('\n2) out-of-sample books 2022 - Jan 2026, 1 contract, all day:', flush=True)
    for k in ('vector', 'ema921', 'ema2050', 'donch', 'super', 'macd', 'vwap'):
        T = TR[k]; book(T, f'{k} every flip'); book(T[~hig(T)], f'{k} off in top-third gamma')
    T = TR['vector']; yr = T.day.dt.year
    print('  vector by year, every flip / off in top-third gamma (pts/trade): ' + ', '.join(f'{y} {T.pts[yr == y].mean():+.1f}/{T[~hig(T)].pts[T[~hig(T)].day.dt.year == y].mean():+.1f}'
          for y in range(2014, 2026)), flush=True)
    # 3) Lucid
    gx = (DY.gex < 0); M1t, R1t = m1_trades(), r1_trades(); R1g = R1t[R1t.day.map(gx) == False]; M1g = M1t[M1t.day.map(gx) == True]
    M1gg = M1g[[not (fget(d, 'dexp') >= 2 / 3) for d in M1g.day]]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    OB = on_list(0, 600, .35, 'all'); RAs = ra_full2(True)
    OBpos = OB[[bool(DY.gex.get(d, np.nan) > 0) for d in OB.day]]; OBneg = OB[[bool(DY.gex.get(d, np.nan) < 0) for d in OB.day]]
    V = TR['vector']; V15a = V[~hig(V)]; Vb = run_sig2('s_vector', 720, 930); V15b = Vb[~hig(Vb)]
    RAd = RAs.groupby('day').pts.sum(); PA = V15a.groupby('day').pts.sum(); PB = V15b.groupby('day').pts.sum(); J = pd.DataFrame({'ra': RAd, 'a': PA, 'b': PB}).reindex(D_).fillna(0)
    print(f'\n3) V15a (all day) {len(V15a[V15a.day >= "2018-02-01"])} trades since Feb 2018, avg {V15a[V15a.day >= "2018-02-01"].pts.mean():+.2f}; V15b (afternoon) {len(V15b[V15b.day >= "2018-02-01"])}, '
          f'avg {V15b[V15b.day >= "2018-02-01"].pts.mean():+.2f}; daily correlation with Rule A: V15a {J.ra.corr(J.a):+.2f}, V15b {J.ra.corr(J.b):+.2f}', flush=True)
    RULE = {'RA': (5, 'low', 'mid'), 'M1': (5, 'high', 'low'), 'R1': (1, 'low', 'mid'), 'ON+': (5, 'high', 'mid'), 'ON-': (1, 'high', 'low')}
    def lists(spec, phase):
        L = [[] for _ in range(NS)]
        for T, u, yzrule in spec:
            for r_ in T.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * (r_.sz * (r_.ce if phase == 'e' else r_.cf) if hasattr(r_, 'sz') else 1.0)
                if yzrule is not None:
                    n, best, worst = yzrule; v = DY[f'yz{n}p'].get(pd.Timestamp(r_.day), np.nan); b3 = t3(np.array([v]))[0]; uu *= 1.5 if b3 == best else (.5 if b3 == worst else 1.0)
                if uu > 0: L[k].append((r_.m, ((min(uu, 1.75),), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    base = [(RAs, 1.0, RULE['RA']), (M1gg, .3, RULE['M1']), (R1g, .5, RULE['R1']), (OBpos, .4, RULE['ON+']), (OBneg, .1, RULE['ON-'])]
    VAR = {'U3 + YZ (current best)': base, '+ V15a vector all day': base + [(V15a, .3, None)], '+ V15b vector afternoon': base + [(V15b, .3, None)]}
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); s22 = [s for s in starts if D_[s] >= pd.Timestamp('2022-01-01')]; LOCK['on'] = False
    print(f'\n  {"version":<26} {"acct":<5} {"starts":<6} pass  paid  payouts x size  blown | per acct-month | 1st payout  every  payouts/mo funded', flush=True)
    for nm, spec in VAR.items():
        DE = lists(spec, 'e'); DF = lists(spec, 'f')
        for a in ('50K', '150K'):
            A = ACC[a]
            for slab, ST in (('all', starts), ('2022+', s22)):
                Rr = []
                for s in ST:
                    res, k2 = ev(A, s, NS)
                    if res != 'pass': Rr.append((res, k2, [], 'none', k2 if res == 'fail' else NS)); continue
                    fr, k3, pays = fu(A, k2 + 1, NS, 5); Rr.append((res, k2, pays, fr, NS if fr == 'open' else k3))
                net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in Rr]); life = np.array([(r[4] - s + 1) / 21 for r, s in zip(Rr, ST)])
                fund = [r for r in Rr if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
                fp = [(D_[min(r[2][0][0], NS - 1)] - D_[s]).days / 30.44 for r, s in zip(Rr, ST) if r[2]]; flife = sum((r[4] - r[1]) / 21 for r in fund)
                print(f'  {nm:<26} {a:<5} {slab:<6} {np.mean([r[0] == "pass" for r in Rr]):4.0%} {np.mean([len(r[2]) > 0 for r in Rr]):5.0%}  '
                      f'{np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x {np.mean(pays) / 20 if pays else 0:3.0f} pts {np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | '
                      f'{net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {len(pays) / flife if flife else 0:10.2f}', flush=True)
