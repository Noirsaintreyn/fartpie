"""Nov 4 2026: what hints at BIG vs BAD vector trades (not trade-by-trade ranking). DECLARED BEFORE RESULTS. 17,447 live-rule trades, 54 names, 2019-21 / 2022-26.
Outcome classes: BIG ret >= +5% (7.5% of trades, 62% of gross profit), RUN ret >= +15%, BAD ret <= -3% (6.9% of trades, 34% of gross loss). Everything else is chop.
Features known at entry (prior RTH close or the entry bar's open):
  context file: gap, runup, z20, prior, margin, tod, open_bar, qqq, crowd, whip, tkrec, rv, rvrel, dnext, dlast, spy200, vts, vixp, spy20, h1 track, options ratio, sector, regime
  non-price file: egap, esurp, rel60, rel20, hi52, dratio
  new from bars: atr_pct (ATR14/price), squeeze (ATR14 now / its 120-day median), bbw (20d Bollinger width percentile, 1y), d_hi20/d_hi80 (close vs 20/80-day high), base (close / 80-day low - 1),
                 trend50 (close vs daily EMA50), e10e20 (daily EMA10 > EMA20), egap_open (entry-day open / prior close - 1), rvol (prior-day RTH volume / 20-day average), dd_days (sessions since the 80-day high)
PART A: every feature by quintile (within-year ranks), BIG / RUN / BAD rates and avg return, both halves. A hint counts only if top-vs-bottom quintile goes the same direction in both halves
        and the BIG (or BAD) rate differs by >= 1.3x in both.
PART B: gradient boosting, train one half / test the other (both ways): AUC for BIG and for BAD, and what the top/bottom predicted deciles actually returned.
PART C: in-trade: after 1 and 3 sessions, P(BIG at exit) by open P&L bucket (how early a big trade shows itself)."""
exec(open('xs/ema_common.py').read())
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
C = pd.read_parquet('/tmp/claude-0/ml/ctx_rank_X.parquet').drop_duplicates(['tk', 'entry', 'h1'])
N = pd.read_parquet('/tmp/claude-0/ml/nonprice_X.parquet')[['tk', 'entry', 'h1', 'egap', 'esurp', 'rel60', 'rel20', 'hi52', 'dratio']].drop_duplicates(['tk', 'entry', 'h1'])
ctxc = ['gap', 'runup', 'z20', 'prior', 'margin', 'tod', 'open_bar', 'qqq', 'crowd', 'whip', 'tkrec', 'rv', 'rvrel', 'dnext', 'dlast', 'spy200', 'vts', 'vixp', 'spy20', 'sec', 'regime']
T = XL.merge(C[['tk', 'entry', 'h1'] + ctxc], on=['tk', 'entry', 'h1'], how='left').merge(N, on=['tk', 'entry', 'h1'], how='left')
# ---- bar features
VOL = {}
for tk in BAR:
    H = pd.read_parquet(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')[0]); m = H.datetime.dt.hour * 60 + H.datetime.dt.minute; R = H[(m >= 570) & (m < 960)]
    v = R.groupby(R.datetime.dt.normalize()).volume.sum(); c = R.groupby(R.datetime.dt.normalize()).close.last()
    bb = (c.rolling(20).std() * 4 / c.rolling(20).mean()); VOL[tk] = pd.DataFrame({'rvol': v / v.rolling(20).mean(), 'bbw': bb.rolling(252, min_periods=120).rank(pct=True),
                                                                                    'dd_days': c.rolling(80).apply(lambda a: len(a) - 1 - np.argmax(a), raw=True)})
rows = []
for i, r in T.iterrows():
    B = BAR.get(r.tk); j = None if B is None else B['pos'].get(r.entry)
    if B is None or j is None or np.ndim(j) or j < 1: rows.append((i,) + (np.nan,) * 8); continue
    j = int(j); dc = B['d_c'][j]; at = B['d_atr'][j]
    rows.append((i, at / dc, dc / B['d_hi20'][j] - 1, dc / B['d_hi80'][j] - 1, dc / B['d_lo80'][j] - 1, dc / B['d_e50'][j] - 1, float(B['d_e10'][j] > B['d_e20'][j]), B['o'][j] / dc - 1, j))
F = pd.DataFrame(rows, columns=['i', 'atr_pct', 'd_hi20', 'd_hi80', 'base', 'trend50', 'e10e20', 'egap_open', 'j']).set_index('i'); T = T.join(F)
sq = []
for tk, g in T.groupby('tk'):
    B = BAR[tk]; d = pd.Series(B['d_atr'] / B['d_c'], index=pd.DatetimeIndex(B['t'])); d = d.groupby(d.index.normalize()).last(); rel = d / d.rolling(120, min_periods=60).median()
    V = VOL[tk].shift(1); g2 = g[['entry']].copy(); g2['dd_'] = g.entry.dt.normalize()
    sq.append(pd.DataFrame({'squeeze': rel.shift(1).reindex(g2.dd_).values, 'rvol': V.rvol.reindex(g2.dd_).values, 'bbw': V.bbw.reindex(g2.dd_).values, 'dd_days': V.dd_days.reindex(g2.dd_).values}, index=g.index))
T = T.join(pd.concat(sq))
T['per'] = np.where(T.entry.dt.year <= 2021, '19-21', '22-26'); T['BIG'] = T.ret >= .05; T['RUN'] = T.ret >= .15; T['BAD'] = T.ret <= -.03
T['regime_c'] = T.regime.astype('category').cat.codes; T['sec_c'] = T.sec.astype('category').cat.codes; T['h1f'] = T.h1.astype(float)
for k in ('spy200',): T[k] = T[k].astype(float)
T.to_parquet('/tmp/claude-0/ml/tails_T.parquet')
num = ['gap', 'runup', 'z20', 'prior', 'margin', 'tod', 'open_bar', 'qqq', 'crowd', 'whip', 'tkrec', 'rv', 'rvrel', 'dnext', 'dlast', 'spy200', 'vts', 'vixp', 'spy20', 'ratio', 'h1f',
       'egap', 'esurp', 'rel60', 'rel20', 'hi52', 'dratio', 'atr_pct', 'squeeze', 'bbw', 'd_hi20', 'd_hi80', 'base', 'trend50', 'e10e20', 'egap_open', 'rvol', 'dd_days']
print('trades', len(T), ' base rates', {p: dict(BIG=f"{g.BIG.mean():.1%}", RUN=f"{g.RUN.mean():.1%}", BAD=f"{g.BAD.mean():.1%}", avg=f"{g.ret.mean():+.2%}") for p, g in T.groupby('per')}, flush=True)
# ---------------- PART A
print('\nPART A: bottom vs top quintile (within-year rank). cells: BIG% / RUN% / BAD% / avg  -- ratio = top/bottom BIG rate, BAD rate', flush=True)
hint = []
for f in num:
    s = T[f]
    if s.notna().sum() < 2000 or s.nunique() < 3:
        if s.nunique() == 2 and s.notna().sum() > 2000: q = s.astype(float) * 4
        else: continue
    else: q = T.groupby(T.entry.dt.year)[f].transform(lambda z: pd.qcut(z.rank(method='first'), 5, labels=False) if z.notna().sum() > 50 else z * np.nan)
    out = f'  {f:<10}'; rb, rd = [], []
    for p in ('19-21', '22-26'):
        lo, hi = T[(q == 0) & (T.per == p)], T[(q == 4) & (T.per == p)]
        cell = lambda z: f"{z.BIG.mean()*100:4.1f}/{z.RUN.mean()*100:3.1f}/{z.BAD.mean()*100:4.1f}/{z.ret.mean()*100:+.2f}"
        out += f' | {p} low {cell(lo)}  high {cell(hi)}'; rb.append(hi.BIG.mean() / max(lo.BIG.mean(), 1e-9)); rd.append(hi.BAD.mean() / max(lo.BAD.mean(), 1e-9))
    tag = ''
    if all(x >= 1.3 for x in rb) or all(x <= 1 / 1.3 for x in rb): tag += f'  BIG-HINT x{rb[0]:.2f}/{rb[1]:.2f}'
    if all(x >= 1.3 for x in rd) or all(x <= 1 / 1.3 for x in rd): tag += f'  BAD-HINT x{rd[0]:.2f}/{rd[1]:.2f}'
    print(out + tag, flush=True)
    if tag: hint.append(f)
print('  consistent hints:', hint, flush=True)
# skew: does high-magnitude context carry positive skew? big/bad ratio by ATR% quintile
q = T.groupby(T.entry.dt.year).atr_pct.transform(lambda z: pd.qcut(z.rank(method='first'), 5, labels=False))
print('\n  ATR% quintile -> BIG% / BAD% / avg / share of all BIG trades', flush=True)
for p in ('19-21', '22-26'):
    z = T[T.per == p]; print('   ', p, ' | '.join(f"q{k} {z[q[z.index] == k].BIG.mean():.1%}/{z[q[z.index] == k].BAD.mean():.1%}/{z[q[z.index] == k].ret.mean():+.2%}/{z[q[z.index] == k].BIG.sum() / z.BIG.sum():.0%}" for k in range(5)), flush=True)
# ---------------- PART B
print('\nPART B: gradient boosting, out of sample', flush=True)
Xf = T[num].astype(float)
for tgt in ('BIG', 'RUN', 'BAD'):
    for a, b in (('19-21', '22-26'), ('22-26', '19-21')):
        tr, te = T.per == a, T.per == b
        mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=300, min_samples_leaf=80, l2_regularization=1.0, random_state=0).fit(Xf[tr], T[tgt][tr])
        p = mdl.predict_proba(Xf[te])[:, 1]; auc = roc_auc_score(T[tgt][te], p); z = T[te].assign(p=p); dz = pd.qcut(z.p.rank(method='first'), 10, labels=False)
        top, bot = z[dz == 9], z[dz == 0]
        mv = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=300, min_samples_leaf=80, random_state=0).fit(Xf[tr][['atr_pct', 'rv', 'h1f']], T[tgt][tr])
        auc_v = roc_auc_score(T[tgt][te], mv.predict_proba(Xf[te][['atr_pct', 'rv', 'h1f']])[:, 1])
        print(f'  {tgt} train {a} test {b}: AUC {auc:.3f} (vol+track only {auc_v:.3f})  top decile {tgt} {top[tgt].mean():.1%} avg {top.ret.mean():+.2%} | bottom decile {bot[tgt].mean():.1%} avg {bot.ret.mean():+.2%} | base {z[tgt].mean():.1%} {z.ret.mean():+.2%}', flush=True)
# ---------------- PART C
print('\nPART C: in-trade. P(BIG at exit | open P&L at the close after k sessions), trades still open', flush=True)
pc = []
for i, r in T.iterrows():
    B = BAR.get(r.tk); j = r.j
    if B is None or j != j: continue
    j = int(j); ex = B['pos'].get(r.exit); ex = int(ex) if ex is not None and not np.ndim(ex) else None
    dts = pd.DatetimeIndex(B['t'][j:j + 26 * 8]).normalize(); m = B['m'][j:j + 26 * 8]; c = B['c'][j:j + 26 * 8]; px = B['o'][j]
    rth = (m >= 570) & (m < 960); ud = pd.unique(dts[rth]); rec = dict(i=i)
    for k in (1, 3):
        if len(ud) > k and (ex is None or ex > j + np.where((dts == ud[k - 1]) & rth)[0].max()):
            ix = np.where((dts == ud[k - 1]) & rth)[0].max(); rec[f'p{k}'] = c[ix] / px - 1
    pc.append(rec)
P = pd.DataFrame(pc).set_index('i'); T = T.join(P)
for k in (1, 3):
    col = f'p{k}'; z = T[T[col].notna()]; b = pd.cut(z[col], [-1, -.02, 0, .02, .05, 1], labels=['< -2%', '-2..0', '0..+2', '+2..+5', '> +5%'])
    print(f'  after {k} session(s): still open {len(z)} ({len(z)/len(T):.0%} of trades)', flush=True)
    for p in ('19-21', '22-26'):
        zz = z[z.per == p]; bb = b[zz.index]
        print(f'    {p}: ' + ' | '.join(f"{lab} n{(bb == lab).sum()} BIG {zz[bb == lab].BIG.mean():.0%} RUN {zz[bb == lab].RUN.mean():.1%} avg {zz[bb == lab].ret.mean():+.1%}" for lab in b.cat.categories), flush=True)
T.to_parquet('/tmp/claude-0/ml/tails_T.parquet')
