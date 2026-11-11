"""ICT / TA concept screen on V15a and Rule A trades (entry-time features from closed 5m bars; NQ 24h + ES pair). Nov 11 2026.
Concepts: FVG, displacement, MSS (structure state), premium/discount (PD range, session range), open-relative (midnight/8:30/9:30 opens: buy below open), NWOG gap, daily bias, draw-on-liquidity (distance to PDH/PDL in trade dir), OTE (62-79% retrace of last leg), SMT (NQ vs ES divergence), silver-bullet window, TA: RSI, ADX, BB-width, Kaufman efficiency.
Every bucket: n, win, mean pts, both eras (A 2014-21 / B 2022+), PASS = same sign, |t|>=2, n>=30 in both eras. ~60 buckets x2 algos => expect a few false PASS by chance; a PASS needs a mechanism to be trusted."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O, N = pickle.load(open(SC + 's6.pkl', 'rb')); L = dict(zip(N, O['V2']))
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet').sort_values('dt').reset_index(drop=True)
E = pd.read_parquet('/tmp/claude-0/ml/es5_real.parquet'); E = E.sort_values('dt').drop_duplicates('dt').set_index('dt')
IDX = pd.DatetimeIndex(X.dt); o, h, l, c = X.open.values, X.high.values, X.low.values, X.close.values; sym = X.sym.values; mm = X.m.values; rth = X.rth.values
Eh, El = E.high.reindex(IDX).values, E.low.reindex(IDX).values
# whole-series TA
cs = pd.Series(c); d1 = cs.diff(); up = d1.clip(lower=0).ewm(alpha=1/14).mean(); dn = (-d1.clip(upper=0)).ewm(alpha=1/14).mean(); RSI = (100 - 100 / (1 + up / dn.replace(0, np.nan))).values
hh, ll = pd.Series(h), pd.Series(l); tr = pd.concat([hh - ll, (hh - cs.shift()).abs(), (ll - cs.shift()).abs()], axis=1).max(axis=1)
pdm = (hh.diff()).where((hh.diff() > -ll.diff()) & (hh.diff() > 0), 0); ndm = (-ll.diff()).where((-ll.diff() > hh.diff()) & (-ll.diff() > 0), 0)
atr = tr.ewm(alpha=1/14).mean(); pdi = 100 * pdm.ewm(alpha=1/14).mean() / atr; ndi = 100 * ndm.ewm(alpha=1/14).mean() / atr
ADX = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).ewm(alpha=1/14).mean().values; ATR = atr.values
bw = (4 * cs.rolling(20).std() / cs.rolling(20).mean()); BBR = (bw / bw.rolling(500).median()).values
ER = ((cs - cs.shift(24)).abs() / cs.diff().abs().rolling(24).sum()).values
BODY = np.abs(c - o); MEDB = pd.Series(BODY).rolling(100).median().values
# daily RTH table
R_ = X[X.rth].copy(); R_['day'] = pd.to_datetime(R_.day).dt.normalize()
dd = R_.groupby('day').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'), sym=('sym', 'first'))
dd['adr'] = (dd.h - dd.l).rolling(14).mean().shift(1)
DAYS = list(dd.index); DI = {d: i for i, d in enumerate(DAYS)}
def swings(a, b):   # confirmed fractal swings (2/2) among bars [a,b)
    SH, SL = [], []
    for i in range(a + 2, b - 2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]: SH.append(i)
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]: SL.append(i)
    return SH, SL
def feats(D, te, dr):
    p = IDX.searchsorted(te); a = IDX.searchsorted(D - pd.Timedelta(hours=6)); b = p   # window: 18:00 prev eve .. entry (bars < p)
    if b - a < 100 or sym[a] != sym[b - 1] or D not in DI or DI[D] < 15: return None
    px = c[b - 1]; adr = dd.adr[D]; f = {}
    # FVG (last 12 bars)
    bull = bear = 0
    for i in range(b - 12, b):
        if l[i] > h[i-2] and l[i] - h[i-2] >= 0.1 * ATR[i]: bull += 1
        if h[i] < l[i-2] and l[i-2] - h[i] >= 0.1 * ATR[i]: bear += 1
    f['fvg_with'] = np.sign(dr * (bull - bear))
    # displacement (largest body of last 6)
    k = b - 6 + int(np.argmax(BODY[b-6:b])); f['disp'] = int(np.sign(dr * (c[k] - o[k]))) if BODY[k] >= 2 * MEDB[k] else 0
    # structure state (MSS)
    SH, SL = swings(a, b); st = 0; lsh = lsl = np.nan; si = sj = 0
    for k in range(a + 4, b):
        while si < len(SH) and SH[si] + 2 <= k - 1: lsh = h[SH[si]]; si += 1
        while sj < len(SL) and SL[sj] + 2 <= k - 1: lsl = l[SL[sj]]; sj += 1
        if c[k] > lsh: st = 1; lsh = np.nan
        elif c[k] < lsl: st = -1; lsl = np.nan
    f['mss'] = dr * st
    # OTE
    csh = [i for i in SH if i + 2 <= b - 1]; csl = [i for i in SL if i + 2 <= b - 1]; f['ote'] = np.nan
    if csh and csl:
        ih, il = csh[-1], csl[-1]; rng_ = h[ih] - l[il]
        if rng_ > 0:
            if dr == 1 and il < ih: r = (h[ih] - px) / rng_
            elif dr == -1 and ih < il: r = (px - l[il]) / rng_
            else: r = np.nan
            f['ote'] = r
    # premium/discount
    pdh, pdl = dd.h.iloc[DI[D] - 1], dd.l.iloc[DI[D] - 1]; f['pd_disc'] = np.sign(dr * (0.5 - (px - pdl) / (pdh - pdl)))
    sh_, sl_ = h[a:b].max(), l[a:b].min(); f['sess_disc'] = np.sign(dr * (0.5 - (px - sl_) / (sh_ - sl_)))
    # open-relative: +1 = buying below / selling above the open
    for nm, hr in (('mid', 6), ('o830', 14.5 - 6 + 6 - 6 + 0)):
        pass
    mid = o[IDX.searchsorted(D)]; f['open_mid'] = np.sign(dr * (mid - px))
    j830 = IDX.searchsorted(D + pd.Timedelta(hours=8, minutes=30)); f['open_830'] = np.sign(dr * (o[j830] - px)) if j830 < b else 0
    j930 = IDX.searchsorted(D + pd.Timedelta(hours=9, minutes=30)); f['open_930'] = np.sign(dr * (o[j930] - px)) if j930 < b else 0
    # gap / daily bias
    pv = dd.iloc[DI[D] - 1]; gap = dd.o[D] - pv.c; f['gap'] = np.sign(dr * gap) if dd.sym[D] == pv.sym and abs(gap) > 0.1 * adr else 0
    f['prior_day'] = np.sign(dr * (pv.c - pv.o)); loc = (pv.c - pv.l) / (pv.h - pv.l); f['prior_loc'] = dr * (1 if loc > 2/3 else -1 if loc < 1/3 else 0)
    # draw on liquidity: distance to PDH (long) / PDL (short) in ADR
    f['dol'] = ((pdh - px) if dr == 1 else (px - pdl)) / adr
    # SMT vs ES (12 bars vs prior 60)
    def newx(arr, f_): return f_(arr[b-12:b]) > f_(arr[b-72:b-12])
    nh, eh, nl, el = newx(h, np.nanmax), newx(Eh, np.nanmax), newx(l, np.nanmin) == False, None
    nhi, ehi = newx(h, np.nanmax), newx(Eh, np.nanmax); nlo = np.nanmin(l[b-12:b]) < np.nanmin(l[b-72:b-12]); elo = np.nanmin(El[b-12:b]) < np.nanmin(El[b-72:b-12])
    bear_smt = (nhi != ehi); bull_smt = (nlo != elo)
    f['smt'] = dr * (int(bull_smt) - int(bear_smt)) if not (bull_smt and bear_smt) else 0
    # TA
    f['rsi'] = dr * (RSI[b-1] - 50); f['adx'] = ADX[b-1]; f['bbr'] = BBR[b-1]; f['er'] = ER[b-1]
    m = int(round((te - D).total_seconds() / 60)); f['win'] = 'open 9:30-10' if m < 600 else 'silver 10-11' if m < 660 else 'late 11+'
    return f
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
def line(lab, s, out):
    a = s[s.era == 'A'].pts; b = s[s.era == 'B'].pts
    ok = len(a) >= 30 and len(b) >= 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
    print(f' {lab:34} n {len(s):4d} win {np.mean(s.pts>0) if len(s) else 0:3.0%} mean {s.pts.mean() if len(s) else 0:+6.1f} | A n{len(a):3d} {a.mean() if len(a) else 0:+6.1f} ({tstat(a):+.1f}) | B n{len(b):3d} {b.mean() if len(b) else 0:+6.1f} ({tstat(b):+.1f}) {"PASS" if ok else ""}')
    if ok: out.append(lab)
for nm in ('V15a', 'RA'):
    T = L[nm][0]; T = T[T.sz > 0].copy(); T['day'] = pd.to_datetime(T.day); rows = []
    for r in T.itertuples():
        D = pd.Timestamp(r.day).normalize(); f = feats(D, D + pd.Timedelta(minutes=int(r.m)), int(r.dir))
        if f: f.update(day=D, m=int(r.m), pts=r.pts, dir=int(r.dir)); rows.append(f)
    R = pd.DataFrame(rows); R['era'] = np.where(R.day < '2022-01-01', 'A', 'B'); passes = []
    print(f'\n== {nm}: {len(R)} trades, mean {R.pts.mean():+.1f}, win {np.mean(R.pts>0):.0%}')
    for col, lab in (('fvg_with', 'FVG net (+1 w/ trade)'), ('disp', 'displacement (+1 w/ trade)'), ('mss', 'structure state (+1 = MSS w/ trade)'), ('pd_disc', 'PD range: +1 = buy discount/sell prem'),
                     ('sess_disc', 'session range: +1 disc'), ('open_mid', 'midnight open: +1 buy below'), ('open_830', '8:30 open: +1 buy below'), ('open_930', '9:30 open: +1 buy below'),
                     ('gap', 'NWOG gap (+1 = w/ gap)'), ('prior_day', 'prior day candle (+1 w/)'), ('prior_loc', 'prior close loc (+1 w/)'), ('smt', 'SMT vs ES (+1 = w/ trade)'), ('win', 'time window')):
        print(f'- {lab}')
        for v in sorted(R[col].dropna().unique()): line(f'   {v}', R[R[col] == v], passes)
    R['ote_in'] = np.where(R.ote.isna(), 'no leg', np.where((R.ote >= .62) & (R.ote <= .79), 'OTE 62-79%', np.where(R.ote < .62, 'shallow <62', 'deep >79')))
    print('- OTE'); [line(f'   {v}', R[R.ote_in == v], passes) for v in sorted(R.ote_in.unique())]
    R['dol_b'] = pd.cut(R.dol, [-9, 0, .3, .7, 99], labels=['PDH/L taken', '<0.3 ADR', '0.3-0.7', '>0.7 ADR'])
    print('- draw on liquidity (dist to PDH long / PDL short)'); [line(f'   {v}', R[R.dol_b == v], passes) for v in R.dol_b.cat.categories]
    for col, lab in (('rsi', 'RSI aligned'), ('adx', 'ADX(14)'), ('bbr', 'BB width vs 500-bar median (squeeze<1)'), ('er', 'Kaufman efficiency (24 bars)')):
        q = pd.qcut(R[col], 3, labels=['low', 'mid', 'high'], duplicates='drop'); print(f'- {lab} terciles'); [line(f'   {v}', R[q == v], passes) for v in q.cat.categories]
    print(f' >> PASS buckets ({nm}): {passes}')
    R.to_pickle(SC + f'ict_{nm}.pkl')
