"""Nov 24 2026 report 456. Rey: explain every flip - if it comes off a low, WHY did the stock drop (market, sector, its own news/earnings), is the flip an exit or did we over-extend, what should price be / what is being priced; also on longer timeframes. DECLARED BEFORE RESULTS.
UNIVERSE the 54 live-screener names (xs/ctx_rank.py SEC map). Flips: (a) 15m and (b) 1h flips = xs/ext/wide_T.parquet rows (live filter: S&P member and options ratio >= 1.5, sticky-stop realized return `ret`, 2019-01..2026-10); (c) DAILY bull vector flips computed here from daily bars built from the 15m RTH cache (entry open of the bar after the confirm bar as in 451, exit open after the bear flip, 10bp, ungated). Cost conventions for (a)(b) as in wide_T.
LEG (daily resolution, all known at the prior close p): PEAK = highest close in the 60 sessions ending p; LEG = close_p / close_peak - 1; length = p - peak. Flips with LEG > -3% or length < 3 sessions are 'no leg' and excluded from X2, X5 (still in X1/X3/X4 where defined).
DECOMPOSITION: betas of the stock's daily log returns on SPY and (sector peers ex self - SPY), 250 sessions ending at PEAK (min 120); SYSTEMATIC share = (beta_spy*SPY_leg + beta_sec*(SEC-SPY)_leg) / stock_leg.  EVENT = an 8-K Item 2.02 (xs/ext/8k_items.csv; accepted after 16:00 ET counts for the next session) whose reaction session lies in (PEAK, p].  VAL = percentile of price/sales (TTM revenue from quarterly SEC facts sec_q.parquet, Q4 = FY - Q1..Q3, filed-date point in time; latest shares) within its own trailing 750 sessions.  SECTOR STATE = daily vector state (xs/vector_pine) of the equal-weight index of the same-sector names ex self at p.  EXT = LEG / (daily sigma over 60d * sqrt(length)).
HYPOTHESES (declared signs; each month-clustered |t|>=2 in the declared direction AND same sign in train 2019-21 and test 2022+; 5 hypotheses x 3 timeframes = 15 tests so ~1 chance pass is expected):
 X1 no-news low beats news low (Chan 2003: drops without news revert, drops with news drift): flips with NO earnings in the leg > flips with an earnings report in the leg  (legs only).
 X2 systematic low beats stock-specific low: SYSTEMATIC share >= 0.5 > < 0.5  (legs only).
 X3 cheap beats expensive: VAL <= 1/3 > VAL >= 2/3.
 X4 sector aligned: sector vector bull at p > bear.
 X5 over-extended legs bounce: EXT <= -2 > EXT > -1  (legs only).
Also reported (descriptive): explanation labels {market/sector-driven, earnings-driven, stock-specific no news} counts and mean return, eras 19-21 | 22 | 23-26."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
src = open('xs/ctx_rank.py').read(); i = src.index('SEC = dict('); SEC = eval(src[i + 6:src.index("\nX['sec']")].replace('\n', ' ').strip()); NAMES = sorted(SEC)
def daily(tk):
    d = pd.read_parquet(glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')[0]).rename(columns={'datetime': 'ts'}); d['dt'] = pd.to_datetime(d.ts, utc=True).dt.tz_convert(ET).dt.tz_localize(None); m = d.dt.dt.hour * 60 + d.dt.dt.minute
    d = d[(m >= 570) & (m < 960)].sort_values('dt'); d['d'] = d.dt.dt.normalize(); g = d.groupby('d'); return pd.DataFrame(dict(open=g.open.first(), high=g.high.max(), low=g.low.min(), close=g.close.last(), vol=g.volume.sum()))
D = {t: daily(t) for t in NAMES + ['SPY']}; idx = D['SPY'].index; print('daily bars built', len(D), 'sessions', len(idx), flush=True)
for t in D: D[t] = D[t].reindex(idx)
C = pd.DataFrame({t: D[t].close for t in D}).ffill(); O = pd.DataFrame({t: D[t].open for t in D}).ffill(); Hh = pd.DataFrame({t: D[t].high for t in D}).ffill(); Ll = pd.DataFrame({t: D[t].low for t in D}).ffill(); R = np.log(C).diff()
pos = {d: k for k, d in enumerate(idx)}; n = len(idx)
# sector peers ex self: factor return and index OHLC for the vector state
secidx, secstate = {}, {}
for t in NAMES:
    peers = [p for p in NAMES if p != t and SEC[p] == SEC[t]]
    if not peers: secidx[t] = None; continue
    pc = C[peers].shift(1); r = (C[peers] / pc - 1).mean(1).fillna(0); hi = (Hh[peers] / pc - 1).mean(1).fillna(0); lo = (Ll[peers] / pc - 1).mean(1).fillna(0)
    ic = (1 + r).cumprod() * 100; ip = ic.shift(1).fillna(100); X = pd.DataFrame(dict(high=ip * (1 + hi), low=ip * (1 + lo), close=ic)); regs, lv, a15, fls = run_state_machine(X[['high', 'low', 'close']])
    secidx[t] = np.log(1 + r).values; secstate[t] = np.asarray(regs, float)
# events: 8-K 2.02 reaction sessions
K8 = pd.read_csv(UP + 'xs/ext/8k_items.csv'); K8 = K8[K8['items'].astype(str).str.contains('2.02') & K8.ticker.isin(NAMES)].copy(); K8['acc'] = pd.to_datetime(K8.accept, utc=True).dt.tz_convert(ET).dt.tz_localize(None)
K8['react'] = np.where(K8.acc.dt.hour >= 16, K8.acc.dt.normalize() + pd.Timedelta(days=1), K8.acc.dt.normalize()); EV = {t: np.sort(np.array([idx.searchsorted(d) for d in pd.to_datetime(g.react)])) for t, g in K8.groupby('ticker')}
# value: price/sales percentile from quarterly SEC facts
Q = pd.read_parquet(SP + 'sec_q.parquet'); Q['start'] = pd.to_datetime(Q.start); Q['end'] = pd.to_datetime(Q.end); Q['filed'] = pd.to_datetime(Q.filed); Q['dur'] = (Q.end - Q.start).dt.days
def ps_series(tk):
    r = Q[(Q.tk == tk) & (Q.kind == 'rev')]; sh = Q[(Q.tk == tk) & (Q.kind == 'sh')].sort_values('filed')
    if not len(r) or not len(sh): return None
    qs = r[(r.dur >= 80) & (r.dur <= 100)].sort_values('filed').drop_duplicates('end', keep='first')[['end', 'filed', 'val']]; fy = r[(r.dur >= 350) & (r.dur <= 380)].sort_values('filed').drop_duplicates('end', keep='first'); rows = [tuple(x) for x in qs.values]
    for _, f in fy.iterrows():
        inyr = qs[(qs.end > f.end - pd.Timedelta(days=360)) & (qs.end < f.end - pd.Timedelta(days=20))]
        if len(inyr) == 3: rows.append((f.end, f.filed, f.val - inyr.val.sum()))
    qd = pd.DataFrame(rows, columns=['end', 'filed', 'val']).sort_values('end').drop_duplicates('end')
    if len(qd) < 4: return None
    ttm = pd.Series(qd.val.rolling(4).sum().values, index=qd.filed.values).dropna(); ttm = ttm[~ttm.index.duplicated(keep='last')].sort_index(); s = sh.drop_duplicates('filed').set_index('filed').val
    tt = ttm.reindex(idx, method='ffill'); ss = s.reindex(idx, method='ffill'); ps = C[tk] * ss / tt; ps = ps.where((tt > 0) & (ss > 0)); return ps.rolling(750, min_periods=250).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True).shift(0)
VAL = {t: ps_series(t) for t in NAMES}; print('value series', sum(v is not None for v in VAL.values()), 'names', flush=True)
beta_cache = {}
def betas(t, pk):
    k = (t, pk)
    if k in beta_cache: return beta_cache[k]
    a, b = max(1, pk - 250), pk
    if b - a < 120 or secidx[t] is None: beta_cache[k] = None; return None
    y = R[t].values[a:b + 1]; f1 = R['SPY'].values[a:b + 1]; f2 = secidx[t][a:b + 1] - f1; m = np.isfinite(y) & np.isfinite(f1) & np.isfinite(f2)
    if m.sum() < 100: beta_cache[k] = None; return None
    X = np.c_[f1[m], f2[m]]; bb = np.linalg.lstsq(X, y[m], rcond=None)[0]; beta_cache[k] = bb; return bb
def feats(t, p):
    """features for ticker t at prior-close session index p"""
    out = dict(leg=np.nan, length=np.nan, sysh=np.nan, event=np.nan, val=np.nan, sec=np.nan, ext=np.nan)
    if p < 70: return out
    cw = C[t].values[p - 59:p + 1]; pk = p - 59 + int(np.nanargmax(cw)); leg = C[t].values[p] / C[t].values[pk] - 1; length = p - pk; out['leg'] = leg; out['length'] = length
    ss = secstate.get(t)
    if ss is not None: out['sec'] = ss[p]
    if VAL[t] is not None: out['val'] = VAL[t].values[p]
    if leg <= -0.03 and length >= 3:
        sig = np.nanstd(R[t].values[p - 59:p + 1]); out['ext'] = np.log1p(leg) / (sig * np.sqrt(length)) if sig > 0 else np.nan
        ev = EV.get(t, np.array([], int)); out['event'] = float(((ev > pk) & (ev <= p)).any()); bb = betas(t, pk)
        if bb is not None:
            sl = np.log(C[t].values[p] / C[t].values[pk]); spy = np.log(C['SPY'].values[p] / C['SPY'].values[pk]); secl = np.nansum(secidx[t][pk + 1:p + 1]); out['sysh'] = (bb[0] * spy + bb[1] * (secl - spy)) / sl if sl < 0 else np.nan
    return out
rows = []
W = pd.read_parquet(UP + 'xs/ext/wide_T.parquet'); W = W[W.tk.isin(NAMES) & W.sp & (W.ratio >= 1.5)].copy(); W['d'] = W.entry.dt.normalize()
for r in W.itertuples():
    if r.d not in pos: continue
    p = pos[r.d] - 1; f = feats(r.tk, p); f.update(tk=r.tk, tf='1h' if r.h1 else '15m', entry=r.entry, ret=r.ret); rows.append(f)
print('wide flips featurized', len(rows), flush=True)
for t in NAMES:
    regs, lv, a15, fls = run_state_machine(pd.DataFrame(dict(high=Hh[t], low=Ll[t], close=C[t]))[['high', 'low', 'close']]); regs = np.asarray(regs)
    for i_, dd in fls:
        if dd != 1: continue
        e = i_ + 2
        if e + 2 >= n or idx[e] < pd.Timestamp('2019-01-01'): continue
        x = next((j for j in range(e + 1, n) if regs[j - 1] == -1), None)
        if x is None: continue
        f = feats(t, e - 1); f.update(tk=t, tf='daily', entry=idx[e], ret=O[t].values[x] / O[t].values[e] - 1 - 1e-3); rows.append(f)
T = pd.DataFrame(rows); T['mon'] = T.entry.dt.to_period('M'); T['half'] = np.where((T.entry >= '2019-01-01') & (T.entry < '2022-01-01'), 'train', 'test'); T['era'] = pd.cut(T.entry.dt.year, [2018, 2021, 2022, 2100], labels=['19-21', '22', '23-26'])
print(T.groupby('tf').agg(n=('ret', 'size'), mean_bp=('ret', lambda x: x.mean() * 1e4), leg_share=('leg', lambda x: (x <= -0.03).mean())), flush=True)
def ctest(A, B):
    if len(A) < 10 or len(B) < 10: return np.nan, np.nan
    dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); gm = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x.ret[x.s == 1].sum(), s0=x.ret[x.s == -1].sum()))); m1, m0 = A.ret.mean(), B.ret.mean()
    se = np.sqrt((((gm.s1 - gm.n1 * m1) / len(A) - (gm.s0 - gm.n0 * m0) / len(B)) ** 2).sum()); return (m1 - m0) * 1e4, (m1 - m0) / se
def show(nm, A, B):
    d, t_ = ctest(A, B); tr = ctest(A[A.half == 'train'], B[B.half == 'train']); te = ctest(A[A.half == 'test'], B[B.half == 'test']); ok = 'PASS' if (abs(t_) >= 2 and d > 0 and tr[0] > 0 and te[0] > 0) else ('WRONG' if (abs(t_) >= 2 and d < 0) else '    ')
    print(f"{ok:5s} {nm:34s} A {A.ret.mean()*1e4:7.1f}bp (n{len(A):5d}) B {B.ret.mean()*1e4:7.1f}bp (n{len(B):5d}) diff {d:+7.1f} t {t_:+5.2f} | train {tr[0]:+7.1f} (t {tr[1]:+.1f}) test {te[0]:+7.1f} (t {te[1]:+.1f})")
for tf in ('15m', '1h', 'daily'):
    S = T[T.tf == tf]; L = S[S.event.notna()]; print(f"\n===== {tf}: {len(S)} flips, mean {S.ret.mean()*1e4:.1f}bp, with a leg {len(L)}")
    show('X1 no-news vs earnings in leg', L[L.event == 0], L[L.event == 1]); M = S[S.sysh.notna()]; show('X2 systematic>=.5 vs stock-specific', M[M.sysh >= .5], M[M.sysh < .5])
    V = S[S.val.notna()]; show('X3 cheap(<=1/3) vs expensive(>=2/3)', V[V.val <= 1 / 3], V[V.val >= 2 / 3]); Z = S[S.sec.notna()]; show('X4 sector vector bull vs bear', Z[Z.sec == 1], Z[Z.sec == -1])
    E = S[S.ext.notna()]; show('X5 extended(<=-2) vs shallow(>-1)', E[E.ext <= -2], E[E.ext > -1])
    lab = np.where(S.event == 1, 'earnings-driven', np.where(S.sysh >= .5, 'market/sector-driven', np.where(S.event == 0, 'stock-specific no news', 'no leg'))); S2 = S.assign(lab=lab)
    print('   labels (n, mean bp): ' + ' | '.join(f"{k} {len(g)} {g.ret.mean()*1e4:+.0f}" for k, g in S2.groupby('lab')))
    print('   by era, no-news vs earnings (bp): ' + ' | '.join(f"{e} {L[(L.event==0)&(L.era==e)].ret.mean()*1e4:+.0f}/{L[(L.event==1)&(L.era==e)].ret.mean()*1e4:+.0f}" for e in ('19-21', '22', '23-26')))
T.to_pickle(SP + 'T456.pkl')
