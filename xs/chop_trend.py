"""Nov 9 2026: CHOP vs TREND WITHOUT VOLATILITY (Rey: 'is there any model other than volatility that can tell chop from trending, or should I do options flow or
individual-stock VP / TPO'). DECLARED BEFORE RESULTS.
Candidates = the live filter (S&P point-in-time, options ratio >= 1.5): long bull flips, sticky 1h stop (xs/ext/wide_T.parquet), 2019 - Oct 2026.
Outcomes per flip: return | CHOP-OUT = a loss closed the same or the next session | RUNNER = return >= +4% (about the top 10%).
FEATURES, all known before the entry (daily ones at the prior close):
 A. daily trend vs chop, not volatility: ER10 / ER20 Kaufman efficiency ratio (net move / path) | ADX 14 | DIB (DI+ - DI-) / (DI+ + DI-) | CHOP 14 choppiness index |
    R2S signed R^2 of a 20-day log-price line | VR5 variance ratio 5-day vs 1-day over 60 days | HURST from lag 2-20 differences over 100 days |
    AC1 lag-1 autocorrelation of daily returns over 60 days | SPY_ER20 / SPY_ADX the market's own trendiness | NFLIP candidates that flipped earlier the same session
 B. per-stock volume profile / TPO from the 15m bars (only where lse_clean exists = the Mac), prior session unless noted:
    VAW value area (70%) width / day range | POCP POC position in the range | CLVA close vs value (0 = VAL, 1 = VAH) | IBF first-hour range / day range |
    SPF single-print share of the TPO profile | BAL days in balance (consecutive sessions whose value overlaps the day before) | MIG value migration / ATR |
    at entry: EVA entry vs prior value (below / inside / above) | ECOMP entry vs the 5-session composite value | OPEN today's open vs prior value |
    EDEV entry vs today's developing POC / ATR (flips after the first bar)
 References tested before: ATRP 14-day ATR / price (volatility) | GX per-stock net GEX sign (options flow; 18 names, to Dec 2025).
TEST 1 per trade: buckets (terciles cut on 2019-22 flips): return, win, chop-out, runner on 2019-22 (CHOOSE) and 2023-26 (JUDGE). A feature counts only if its best and
       worst buckets on 2019-22 keep the same order on 2023-26.
TEST 2 the 6-slot book (8% breaker, 2 bp, stand-in daily path as xs/scr_slots.py): per feature, SKIP its worst bucket, or PICK ORDER ratio + feature (bucket scores from
       2019-22). The single best rule on 2019-22, judged on 2023-26 against live (ratio order, no skip)."""
import glob, os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(rel):
    for base in (ROOT, '/mnt/user-data/uploads/fartpie', '/home/claude/fartpie'):
        if glob.glob(os.path.join(base, rel)): return os.path.join(base, rel)
    return os.path.join(ROOT, rel)
J = pd.read_parquet(P('xs/ext/site_wide_daily.parquet')); D = J.index
T = pd.read_parquet(P('xs/ext/wide_T.parquet'))
T = T[T.sp & (T.ratio >= 1.5) & (T.exit < D[-1] + pd.Timedelta(days=1))].sort_values(['entry', 'ratio', 'tk'], ascending=[True, False, True], kind='stable').reset_index(drop=True)
T['day'] = T.entry.dt.normalize(); T['hold'] = D.searchsorted(T.exit.dt.normalize()) - D.searchsorted(T.day)
T['chop'] = (T.ret < 0) & (T.hold <= 1); T['run'] = T.ret >= .04; T['win'] = T.ret > 0; T['per'] = np.where(T.entry < '2023-01-01', 'A', 'B')
TKS = sorted(T.tk.unique())
def bars15(tk):
    f = glob.glob(P(f'lse_clean/{tk}_15m_*.parquet'))
    if not f: return None
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute
    return H[(m >= 570) & (m < 960)].drop_duplicates('dt').set_index('dt').sort_index()[['open', 'high', 'low', 'close', 'volume']].astype(float)
def dbars(tk, H):
    f = P(f'data/yahoo/daily_v/{tk}.parquet')
    if os.path.exists(f):
        Y = pd.read_parquet(f); Y.index = pd.DatetimeIndex(Y.index).as_unit('ns')
        if Y.index.tz is not None: Y.index = Y.index.tz_localize(None)
        return Y[['open', 'high', 'low', 'close', 'volume']].astype(float).dropna(), 'Yahoo daily'
    if H is None: return None, None
    g = H.groupby(H.index.normalize())
    return pd.DataFrame({'open': g.open.first(), 'high': g.high.max(), 'low': g.low.min(), 'close': g.close.last(), 'volume': g.volume.sum()}), '15m RTH'
def dfeat(Y):
    h, l, c = Y.high, Y.low, Y.close; lc = np.log(c); r = lc.diff(); F = pd.DataFrame(index=Y.index)
    for n in (10, 20): F[f'ER{n}'] = (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff(); pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=Y.index); ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=Y.index)
    w = lambda x: x.ewm(alpha=1 / 14, adjust=False).mean(); at = w(tr); pdi, ndi = 100 * w(pdm) / at, 100 * w(ndm) / at
    F['ADX'] = w(100 * (pdi - ndi).abs() / (pdi + ndi)); F['DIB'] = (pdi - ndi) / (pdi + ndi)
    F['CHOP'] = 100 * np.log10(tr.rolling(14).sum() / (h.rolling(14).max() - l.rolling(14).min())) / np.log10(14)
    rho = lc.rolling(20).corr(pd.Series(np.arange(len(Y)), index=Y.index, dtype=float)); F['R2S'] = np.sign(rho) * rho ** 2
    F['VR5'] = (lc - lc.shift(5)).rolling(60).var() / (5 * r.rolling(60).var())
    ks = np.arange(2, 21); S = np.column_stack([np.log((lc - lc.shift(k)).rolling(100).std()).values for k in ks]); lk = np.log(ks) - np.log(ks).mean()
    F['HURST'] = (S - S.mean(axis=1, keepdims=True)) @ lk / (lk ** 2).sum()
    F['AC1'] = r.rolling(60).corr(r.shift(1)); F['ATRP'] = (h - l).rolling(14).mean() / c
    return F.shift(1)                                                  # known at the prior close -> the next session's flips
def prof(hi, lo, vo, nb):
    """volume profile (bar volume spread evenly over its high-low) + TPO counts; returns POC, VAH, VAL (70%), single-print share, low, high"""
    L, U = lo.min(), hi.max()
    if not np.isfinite(L) or U <= L: return None
    ed = np.linspace(L, U, nb + 1); ov = np.clip(np.minimum(ed[None, 1:], hi[:, None]) - np.maximum(ed[None, :-1], lo[:, None]), 0, None)
    sp = hi - lo; ok = sp > 0; fr = np.zeros_like(ov); fr[ok] = ov[ok] / sp[ok, None]; tp = ov > 0
    if (~ok).any():
        rw = np.flatnonzero(~ok); ix = np.clip(np.searchsorted(ed, lo[~ok], 'right') - 1, 0, nb - 1); fr[rw, ix] = 1; tp[rw, ix] = True
    vp = (fr * vo[:, None]).sum(0); tpo = tp.sum(0)
    if vp.sum() <= 0: vp = tpo.astype(float)
    pc = int(vp.argmax()); a = b = pc; acc = vp[pc]; tot = vp.sum()
    while acc < .7 * tot and (a > 0 or b < nb - 1):
        u_ = vp[b + 1] if b < nb - 1 else -1.0; d_ = vp[a - 1] if a > 0 else -1.0
        if u_ >= d_: b += 1; acc += u_
        else: a -= 1; acc += d_
    return (ed[pc] + ed[pc + 1]) / 2, ed[b + 1], ed[a], float((tpo == 1).mean()), L, U
def vpfeat(H, atr, Tk):
    day = H.index.normalize(); hi, lo, vo, op, cl = (H[c].values for c in ('high', 'low', 'volume', 'open', 'close'))
    cut = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1, len(H)]; days = day[cut[:-1]]; start = dict(zip(days, cut[:-1])); R = []
    for j in range(len(days)):
        s, e = cut[j], cut[j + 1]
        if e - s < 8: continue
        p = prof(hi[s:e], lo[s:e], vo[s:e], 40); q = prof(hi[cut[max(0, j - 4)]:e], lo[cut[max(0, j - 4)]:e], vo[cut[max(0, j - 4)]:e], 60)
        if p is None or q is None: continue
        R.append(dict(day=days[j], poc=p[0], vah=p[1], val=p[2], SPF=p[3], L=p[4], U=p[5], c=cl[e - 1], o=op[s], ibr=hi[s:s + 4].max() - lo[s:s + 4].min(), c5h=q[1], c5l=q[2]))
    S = pd.DataFrame(R).set_index('day'); rng = S.U - S.L
    S['VAW'] = (S.vah - S.val) / rng; S['POCP'] = (S.poc - S.L) / rng; S['CLVA'] = ((S.c - S.val) / (S.vah - S.val)).clip(-3, 4); S['IBF'] = S.ibr / rng
    pvh, pvl = S.vah.shift(1), S.val.shift(1); ovl = ((np.minimum(S.vah, pvh) - np.maximum(S.val, pvl)) > 0).values; bal = np.zeros(len(S)); run = 0
    for i, v in enumerate(ovl): run = run + 1 if v else 0; bal[i] = min(run, 10)
    S['BAL'] = bal; S['MIG'] = ((S.vah + S.val) / 2 - (pvh + pvl) / 2) / S.index.map(atr)
    Sp = S.shift(1).reindex(Tk.day.values); Sp.index = Tk.index; out = Sp[['VAW', 'POCP', 'CLVA', 'IBF', 'SPF', 'BAL', 'MIG']].copy()
    a_ = Tk.day.map(atr).values; k = H.index.get_indexer(Tk.entry.values); px = np.where(k >= 0, op[np.maximum(k, 0)], np.nan)
    loc = lambda x, hi_, lo_: np.where(~(np.isfinite(x) & np.isfinite(hi_) & np.isfinite(lo_)), None, np.where(x > hi_, 'above', np.where(x < lo_, 'below', 'inside')))
    out['EVA'] = loc(px, Sp.vah.values, Sp.val.values); out['ECOMP'] = loc(px, Sp.c5h.values, Sp.c5l.values)
    out['OPEN'] = loc(S.o.reindex(Tk.day.values).values, Sp.vah.values, Sp.val.values); ed = np.full(len(Tk), np.nan)
    for i, (kk, d) in enumerate(zip(k, Tk.day.values)):
        s = start.get(pd.Timestamp(d))
        if kk >= 0 and s is not None and kk - s >= 2:
            p = prof(hi[s:kk], lo[s:kk], vo[s:kk], 30)
            if p: ed[i] = (px[i] - p[0]) / a_[i]
    out['EDEV'] = ed; return out
# ---------------------------------------------------------------- features
src = set(); FE = []; HAVE15 = False
for tk in TKS + ['SPY']:
    H = bars15(tk); Y, s_ = dbars(tk, H)
    if Y is None: print('  no bars', tk, flush=True); continue
    src.add(s_); F = dfeat(Y); atr = (Y.high - Y.low).rolling(14).mean().shift(1)
    if tk == 'SPY': SPYF = F; continue
    Tk = T[T.tk == tk]; X = F.reindex(Tk.day.values); X.index = Tk.index
    if H is not None: HAVE15 = True; X = X.join(vpfeat(H, atr, Tk))
    FE.append(X)
FE = pd.concat(FE); T = T.join(FE)
if 'SPYF' in dir(): T['SPY_ER20'] = SPYF.ER20.reindex(T.day.values).values; T['SPY_ADX'] = SPYF.ADX.reindex(T.day.values).values
T['NFLIP'] = T.groupby('day').entry.rank(method='min') - 1
GXF = glob.glob(os.path.join(os.path.dirname(P('xs/ext/wide_T.parquet')), 'gex', '*.csv')); T['GX'] = None
if GXF:
    g_ = []
    for f in GXF:
        g = pd.read_csv(f, parse_dates=['date']).sort_values('date'); g['tk'] = os.path.basename(f)[:-4]; g['nxt'] = g.date.shift(-1); g_.append(g[['tk', 'nxt', 'net_gex']].dropna(subset=['nxt']))
    g_ = pd.concat(g_).set_index(['tk', 'nxt']).net_gex; v = np.array([g_.get((t, d), np.nan) for t, d in zip(T.tk, T.day)])
    T['GX'] = np.where(np.isnan(v), None, np.where(v > 0, 'pos gamma', 'neg gamma'))
A_ = ['ER10', 'ER20', 'ADX', 'DIB', 'CHOP', 'R2S', 'VR5', 'HURST', 'AC1', 'SPY_ER20', 'SPY_ADX', 'NFLIP']
B_ = ['VAW', 'POCP', 'CLVA', 'IBF', 'SPF', 'BAL', 'MIG', 'EVA', 'ECOMP', 'OPEN', 'EDEV'] if HAVE15 else []
REF = ['ATRP', 'GX']; FEATS = [f for f in A_ + B_ + REF if f in T]
CATS = {'BAL': ([-.5, .5, 2.5, 99], ['0 days', '1-2 days', '3+ days']), 'NFLIP': ([-.5, .5, 3.5, 1e9], ['first', '2nd-4th', '5th+'])}
def bucket(f):
    x = T[f]
    if not pd.api.types.is_numeric_dtype(x): return x.astype(object).where(x.notna(), None)
    if f in CATS: return pd.cut(x, CATS[f][0], labels=CATS[f][1]).astype(object).where(x.notna(), None)
    q1, q2 = x[T.per == 'A'].quantile([1 / 3, 2 / 3])
    return pd.Series(np.where(x.isna(), None, np.where(x < q1, '1 low', np.where(x < q2, '2 mid', '3 high'))), index=T.index)
# ---------------------------------------------------------------- the 6-slot book (as xs/scr_intraday.py, no replacement)
def taken(X, N=6, brk=.08):
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []
    for e, x, rt, tk in zip(X.entry, X.exit, X.ret, X.tk):
        done = sorted([p for p in pend if p[0] <= e]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
        op = [h for h in op if h > e]
        if len(op) >= N: continue
        if brk and hist:
            t0 = e - pd.Timedelta(days=28)
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [hh for _, hh in win]) < np.log1p(-brk): continue
        op.append(x); pend.append((x, rt)); out.append((tk, e, x, rt))
    return out
def daily(TK, N=6):
    R = np.zeros(len(D)); W = np.zeros(len(D))
    for tk, e, x, r in TK:
        a, b = D.searchsorted(pd.Timestamp(e).normalize()), D.searchsorted(pd.Timestamp(x).normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / N; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w; W[a:b + 1] += w
    return pd.Series(R * np.where(W > 1, 1 / np.maximum(W, 1e-9), 1.0), index=D)
def cagr(s, a, b):
    z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min()
def st2(s): return ' | '.join(f'{c_:6.1%} / DD {d_:6.1%}' for c_, d_ in (cagr(s, '2019-01-01', '2023-01-01'), cagr(s, '2023-01-01', '2027-01-01')))
if __name__ == '__main__':
    print(f'candidates {len(T):,} ({T.tk.nunique()} names) | daily bars from {", ".join(sorted(src))} | VP/TPO from 15m bars: {"yes" if HAVE15 else "no (needs lse_clean - run on the Mac)"}', flush=True)
    for p_, lab in (('A', '2019-22'), ('B', '2023-26')):
        z = T[T.per == p_]; print(f'  {lab}: n {len(z):,}  ret {z.ret.mean():+.2%}  win {z.win.mean():.0%}  chop-out {z.chop.mean():.0%}  runner {z.run.mean():.0%}', flush=True)
    print('\n=== TEST 1 per trade: bucket | 2019-22 (CHOOSE): n, ret, win, chop-out, runner | 2023-26 (JUDGE) ===', flush=True); SUM = []; SC = {}
    for f in FEATS:
        b = bucket(f); ok = b.notna()
        if ok.sum() < 300: continue
        gA, gB = T[ok & (T.per == 'A')].groupby(b[ok & (T.per == 'A')]), T[ok & (T.per == 'B')].groupby(b[ok & (T.per == 'B')])
        mA, mB, cA, cB = gA.ret.mean(), gB.ret.mean(), gA.chop.mean(), gB.chop.mean()
        if len(mA) < 2 or len(mB) < 2: continue
        print(f'  {f}', flush=True)
        for k in sorted(mA.index):
            a, bb = T[ok & (T.per == 'A') & (b == k)], T[ok & (T.per == 'B') & (b == k)]
            print(f'    {k:<10} n {len(a):5,} {a.ret.mean():+.2%} win {a.win.mean():3.0%} chop {a.chop.mean():3.0%} run {a.run.mean():3.0%} | '
                  f'n {len(bb):5,} {bb.ret.mean() if len(bb) else np.nan:+.2%} win {bb.win.mean() if len(bb) else np.nan:3.0%} chop {bb.chop.mean() if len(bb) else np.nan:3.0%} '
                  f'run {bb.run.mean() if len(bb) else np.nan:3.0%}', flush=True)
        hi_, lo_ = mA.idxmax(), mA.idxmin(); jB = mB.get(hi_, np.nan) - mB.get(lo_, np.nan); jC = cB.get(lo_, np.nan) - cB.get(hi_, np.nan)
        SUM.append((f, hi_, lo_, mA[hi_] - mA[lo_], jB, cA[lo_] - cA[hi_], jC)); SC[f] = (b, mA)
    print('\n  SUMMARY (best / worst bucket chosen on 2019-22): ret spread 2019-22 -> same buckets 2023-26 | chop-out gap (worst - best) 2019-22 -> 2023-26', flush=True)
    for f, hi_, lo_, sA, sB, ccA, ccB in sorted(SUM, key=lambda z: -np.nan_to_num(z[4], nan=-9)):
        print(f'    {f:<9} best {str(hi_):<10} worst {str(lo_):<10} {sA:+.2%} -> {sB:+.2%} | chop {ccA:+.0%} -> {ccB:+.0%}  {"HOLDS" if sB > 0 and ccB >= 0 else ""}', flush=True)
    print('\n=== TEST 2 the 6-slot book: CAGR / max DD 2019-22 (CHOOSE) | 2023-26 (JUDGE) ===', flush=True)
    live = daily(taken(T)); print(f'  {"live (ratio order, no skip)":<34} {st2(live)}', flush=True); RES = {}
    for f, (b, mA) in SC.items():
        sc = b.map(mA).astype(float).fillna(float(mA.median())); X = T.assign(_k=T.groupby('entry').ratio.rank(ascending=False) + sc.groupby(T.entry).rank(ascending=False))
        RES[f'{f}: skip {mA.idxmin()}'] = daily(taken(T[b != mA.idxmin()]))
        RES[f'{f}: order ratio + {f}'] = daily(taken(X.sort_values(['entry', '_k', 'tk'], kind='stable')))
    ca = {k: cagr(s, '2019-01-01', '2023-01-01')[0] for k, s in RES.items()}; cl = cagr(live, '2019-01-01', '2023-01-01')[0]; cb = cagr(live, '2023-01-01', '2027-01-01')[0]
    for k in sorted(RES, key=lambda k_: -ca[k_])[:12]: print(f'  {k:<34} {st2(RES[k])}', flush=True)
    nb = sum(1 for k, s in RES.items() if ca[k] > cl and cagr(s, '2023-01-01', '2027-01-01')[0] > cb)
    best = max(RES, key=lambda k_: ca[k_]); jb = cagr(RES[best], '2023-01-01', '2027-01-01')[0]
    print(f'  -> {len(RES)} rules; {nb} beat live in BOTH periods | chosen on 2019-22: {best} -> 2023-26 {jb:.1%} vs live {cb:.1%} ({"beats" if jb > cb else "does not beat"} live)', flush=True)
    yr = lambda s: (1 + s).groupby(s.index.year).prod() - 1; yl, yb = yr(live), yr(RES[best])
    print('  by year, live vs chosen: ' + ' | '.join(f'{y} {yl[y]:+.0%} vs {yb[y]:+.0%}' for y in yl.index), flush=True)
    # ADDED Nov 9 after TEST 1 (so a look, not a test): the accuracy view for discretionary options on flips - chop-outs and win rate, not mean return
    if 'ECOMP' in T:
        print('\n=== ACCURACY VIEW (added after TEST 1): flips by where they enter vs the 5-session composite value, alone and with per-stock gamma ===', flush=True)
        for lab, m in (('above 5-day value', T.ECOMP == 'above'), ('inside 5-day value', T.ECOMP == 'inside'), ('below 5-day value', T.ECOMP == 'below'),
                       ('above value + pos gamma', (T.ECOMP == 'above') & (T.GX == 'pos gamma')), ('above value + neg gamma', (T.ECOMP == 'above') & (T.GX == 'neg gamma')),
                       ('above value + 5th+ flip of the day', (T.ECOMP == 'above') & (T.NFLIP >= 4))):
            print(f'  {lab:<36} ' + ' | '.join(f'{p_}: n {len(z):5,} win {z.win.mean():3.0%} chop {z.chop.mean():3.0%} run {z.run.mean():3.0%} ret {z.ret.mean():+.2%} med {z.ret.median():+.2%}'
                                                   for p_, z in (('2019-22', T[m & (T.per == 'A')]), ('2023-26', T[m & (T.per == 'B')]))), flush=True)
