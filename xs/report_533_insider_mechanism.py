"""Report 533 (Nov 28 2026): Rey: "look into these leads, see why they work, then refine". LEAD 2: clustered insider buying. Until now tested only on flips (n 46-124, no power). Here: all sub-$5 stock-days (power).
DECLARED BEFORE RESULTS. Flags from Form 4 open-market purchases (P), known on filing date, window = prior 90 calendar days: ANY (>=1 insider), ROLE (officer/director), CLUS (>=2 distinct owners), BIGV (>= $25k total), CLUSBIG (CLUS & BIGV).
Hypotheses (mechanism, from Consensus #6 literature): H1 insiders are better informed in high-asymmetry names -> lift larger in HEALTH/small names; H2 purchases signal quality when firm is in adverse situation (Chen & Li 2023: 'signal quality / battle shorts') -> lift larger when the stock is a past loser (DD252 below train median) and when days-to-cover is high; H3 clustered buying beats single buyer (Bylin & Wallin).
Template: month x ATRP decile x price bucket within-cell flag(1)-flag(0) lift of P(+100% in 40d); eras 2018/2019-21/2022/2023-26; PASS pooled |t|>=3 & same sign >=3/4 eras. Trials: 5 flags + sector splits (7) + 2 interactions x2 = report all; interaction contrasts judged vs the pooled flag.
Also as a MECHANISM check (ex-post, not a signal): share of flagged stock-days followed by a PR/DEAL 8-K in the next 30d vs unflagged (do insiders buy ahead of news?).
"""
import numpy as np, pandas as pd, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
P['d'] = P.d.astype('datetime64[ns]'); Q = pd.read_pickle(X + 'runs515_panel2.pkl'); Q['d'] = Q.d.astype('datetime64[ns]'); P = P.merge(Q[['tk', 'd', 'DD252', 'DTC']].drop_duplicates(['tk', 'd']), on=['tk', 'd'], how='left')
P = P[P.y.notna()].copy(); P['pb'] = np.select([P.c < 1, P.c < 2.5], [0, 1], 2); P['ab10'] = pd.qcut(P.ATRP.rank(method='first'), 10, labels=False); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
I = pd.read_pickle(X + 'insider_P_hist.pkl'); I = I[I.TRANS_CODE == 'P'].copy(); I['fd'] = pd.to_datetime(I.fd).astype('datetime64[ns]'); I['role'] = I.RPTOWNER_RELATIONSHIP.astype(str).str.lower().str.contains('officer|director'); I['val'] = pd.to_numeric(I.TRANS_SHARES, errors='coerce') * pd.to_numeric(I.TRANS_PRICEPERSHARE, errors='coerce')
IG = {t: g.sort_values('fd') for t, g in I.groupby('ISSUERTRADINGSYMBOL')}; out = {k: np.zeros(len(P), bool) for k in ('ANY', 'ROLE', 'CLUS', 'BIGV')}; ix = pd.Series(range(len(P))).groupby(P.tk.values).indices
for tk, ii in ix.items():
    g = IG.get(tk)
    if g is None: continue
    fd = g.fd.values; d = P.d.values[ii]; lo = np.searchsorted(fd, d - np.timedelta64(90, 'D'), side='right'); hi = np.searchsorted(fd, d, side='right'); has = hi > lo
    if not has.any(): continue
    cik = g.RPTOWNERCIK.values; rl = g.role.values; vl = np.nan_to_num(g.val.values)
    for k in np.where(has)[0]:
        a, b = lo[k], hi[k]; o = ii[k]; out['ANY'][o] = True; out['ROLE'][o] = bool(rl[a:b].any()); out['CLUS'][o] = len(set(cik[a:b])) >= 2; out['BIGV'][o] = vl[a:b].sum() >= 25000
for k, v in out.items(): P[k] = v.astype(int)
P['CLUSBIG'] = P.CLUS & P.BIGV; tr = P.yr.between(2019, 2021); P['LOSER'] = (P.DD252 < P.loc[tr, 'DD252'].median()).astype(int); P['HIDTC'] = (P.DTC > P.loc[tr, 'DTC'].quantile(.67)).astype(int)
print('stock-days', len(P), 'base y', round(P.y.mean(), 4), '| flag share %', {k: round(P[k].mean() * 100, 2) for k in ('ANY', 'ROLE', 'CLUS', 'BIGV', 'CLUSBIG')})
def lift2(Gx, col):
    r = []
    for (m, b, p), gp in Gx.groupby(['mon', 'ab10', 'pb']):
        a, c = gp[gp[col] == 1].y, gp[gp[col] == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); s = mm.std(); return mm.mean(), (mm.mean() / (s / np.sqrt(len(mm))) if s and s > 0 else np.nan)
def row(lab, Gx, col):
    r = dict(cell=lab, n1=int((Gx[col] == 1).sum()))
    for e in ('2018', '2019-21', '2022', '2023-26'): d, t = lift2(Gx[Gx.era4 == e], col); r[e] = None if d != d else round(d * 100, 1)
    d, t = lift2(Gx, col); r['pooled_pp'] = None if d != d else round(d * 100, 1); r['t'] = None if t != t else round(t, 1); return r
pd.set_option('display.width', 200); rows = [row(k, P, k) for k in ('ANY', 'ROLE', 'CLUS', 'BIGV', 'CLUSBIG')]
print('\nPOOLED lift on P(+100% in 40d), pp'); print(pd.DataFrame(rows).to_string(index=False))
rows = [row(f'CLUS | {s}', P[P.sector == s], 'CLUS') for s in ('HEALTH', 'TECH', 'ENERGY', 'MINING', 'FIN', 'UTIL', 'OTHER')]
rows += [row('CLUS | loser', P[P.LOSER == 1], 'CLUS'), row('CLUS | non-loser', P[P.LOSER == 0], 'CLUS'), row('CLUS | high DTC', P[P.HIDTC == 1], 'CLUS'), row('CLUS | not high DTC', P[P.HIDTC == 0], 'CLUS'), row('ANY | loser', P[P.LOSER == 1], 'ANY'), row('ANY | non-loser', P[P.LOSER == 0], 'ANY')]
print('\nCLUS (>=2 insiders, 90d) splits'); print(pd.DataFrame(rows).to_string(index=False))
K = pd.read_csv(X + '8k_wide.csv').dropna(subset=['items']); K['filed'] = pd.to_datetime(K.filed); it = K['items'].astype(str).str.split(','); K = K[it.apply(lambda l: ('7.01' in l or '8.01' in l or '1.01' in l or '2.01' in l) and '2.02' not in l)]; KG = {t: np.sort(g.filed.values) for t, g in K.groupby('ticker')}
nxt = np.zeros(len(P), bool)
for tk, ii in ix.items():
    a = KG.get(tk)
    if a is None: continue
    d = P.d.values[ii]; j = np.searchsorted(a, d, side='right'); ok = j < len(a); nxt[ii] = ok & ((a[np.clip(j, 0, len(a) - 1)] - d) <= np.timedelta64(30, 'D'))
P['NEWS30'] = nxt.astype(int); print('\nMECHANISM (ex-post): P(PR/DEAL 8-K in next 30d)  ANY=1 %.3f vs 0 %.3f | CLUS=1 %.3f vs 0 %.3f | base y when news follows %.3f vs not %.3f' % (P[P.ANY == 1].NEWS30.mean(), P[P.ANY == 0].NEWS30.mean(), P[P.CLUS == 1].NEWS30.mean(), P[P.CLUS == 0].NEWS30.mean(), P[P.NEWS30 == 1].y.mean(), P[P.NEWS30 == 0].y.mean()))
