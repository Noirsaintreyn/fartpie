"""Nov 15 2026 DOES PAIRING THE VECTOR WITH SUPERTREND / ULTIMATE MA / ALPHATREND HELP (Rey). Declared before results. Universe and flips: report_90 daily flip file (30,323 bull flips, ~1,570 names, 2019-Mar 2026; survivorship-biased), trade = entry next open after the flip bar, exit open after the vector bear flip, 30bp round trip. TRAIN 2019-21, TEST 2022+. Indicators, evaluated at the flip bar close (no look-ahead): SUPERTREND ATR10 x3 (hl2, RMA ATR; bull = price above the lower band direction); ALPHATREND (Kivanc: ATR = SMA(TR,14), coeff 1, MFI(14) >= 50 picks the lower/upper rail, ratcheting; bull = AlphaTrend > AlphaTrend[2], i.e. its buy-signal state); ULTIMATE MA = PROXY because the indicator is a configurable MA and I do not know your settings: bull = close above a rising HMA(21) (UM1) or EMA(20) (UM2). Hypotheses (declared sign, +): flip with the indicator bull at the flip bar beats flips without it (agreement filter), more so when all three agree; and an indicator-bear exit (exit open after the first bar the indicator turns bear, or the vector bear flip, whichever first) beats vector-only exit on net mean and drawdown per trade. Metrics: n, mean/median net, win10 share (>=+10%), huge share (>=+100%), ex-top-3 mean, agree-minus-not-agree mean with month-clustered t, on TRAIN and TEST separately. A lead needs + on BOTH eras and TEST month-clustered t >= 2. PART 2 repeats the agreement filter on the 3,436 1H flips (report_105 set, 60 days, ONE regime, lead only)."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SC = X[:-4]; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); F = pd.read_pickle(X + 'stack_flips.pkl'); COST = .003
def rma(x, n):
    o = np.full(len(x), np.nan); 
    if len(x) < n: return o
    o[n - 1] = np.nanmean(x[:n]); a = 1 / n
    for k in range(n, len(x)): o[k] = o[k - 1] * (1 - a) + x[k] * a
    return o
def wma(x, n):
    w = np.arange(1, n + 1, dtype=float); o = np.full(len(x), np.nan)
    if len(x) >= n: o[n - 1:] = np.convolve(x, w[::-1] / w.sum(), 'valid')
    return o
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values
def indic(o, h, l, c, v):
    n = len(c); pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); out = {}
    atr = rma(tr, 10); hl2 = (h + l) / 2; ub = hl2 + 3 * atr; lb = hl2 - 3 * atr; up = np.full(n, np.nan); lo_ = np.full(n, np.nan); dr = np.zeros(n)
    for k in range(1, n):
        if np.isnan(atr[k]): continue
        if np.isnan(lo_[k - 1]): lo_[k] = lb[k]; up[k] = ub[k]; dr[k] = 1; continue
        lo_[k] = lb[k] if (lb[k] > lo_[k - 1] or c[k - 1] < lo_[k - 1]) else lo_[k - 1]; up[k] = ub[k] if (ub[k] < up[k - 1] or c[k - 1] > up[k - 1]) else up[k - 1]
        dr[k] = 1 if (dr[k - 1] == -1 and c[k] > up[k - 1]) else (-1 if (dr[k - 1] == 1 and c[k] < lo_[k - 1]) else dr[k - 1])
    out['ST'] = dr == 1
    tp = (h + l + c) / 3; raw = tp * v; dtp = np.r_[0, np.diff(tp)]; pos = pd.Series(np.where(dtp > 0, raw, 0)).rolling(14).sum().values; neg = pd.Series(np.where(dtp < 0, raw, 0)).rolling(14).sum().values; mfi = 100 - 100 / (1 + pos / np.where(neg == 0, np.nan, neg)); mfi = np.where(neg == 0, 100, mfi)
    atr14 = pd.Series(tr).rolling(14).mean().values; upT = l - atr14; dnT = h + atr14; at = np.full(n, np.nan)
    for k in range(14, n):
        p = at[k - 1] if not np.isnan(at[k - 1]) else (upT[k] if mfi[k] >= 50 else dnT[k])
        at[k] = (upT[k] if upT[k] >= p else p) if mfi[k] >= 50 else (dnT[k] if dnT[k] <= p else p)
    a2 = np.r_[np.nan, np.nan, at[:-2]]; out['AT'] = at > a2
    nn = 21; hm = wma(2 * wma(c, nn // 2) - wma(c, nn), int(np.sqrt(nn))); out['UM1'] = (c > hm) & (hm > np.r_[np.nan, hm[:-1]]); e20 = ema(c, 20); out['UM2'] = (c > e20) & (e20 > np.r_[np.nan, e20[:-1]])
    for k in out: out[k] = np.where(np.isnan(c), False, out[k])
    return out
rows = []; Fg = {t: g for t, g in F.groupby('tk')}
for tk, g in Fg.items():
    q = D.get(tk)
    if q is None or len(q) < 300: continue
    q = q[['open', 'high', 'low', 'close', 'volume']].copy(); q = q[~q.index.duplicated()].dropna(); q.index = pd.to_datetime(q.index).tz_localize(None).normalize()
    o = q.open.values; h = q.high.values; l = q.low.values; c = q.close.values; v = q.volume.values.astype(float); n = len(q); reg, lev, atr, _ = run_state_machine(q.assign(datetime=q.index)); I = indic(o, h, l, c, v); pos = {d: k for k, d in enumerate(q.index)}
    for _, r in g.iterrows():
        i = pos.get(r.d)
        if i is None or i + 2 >= n: continue
        j = i + 1
        while j < n and reg[j] != -1: j += 1
        if j + 1 >= n: continue
        e = o[i + 1]; row = dict(tk=tk, d=r.d, test=r.test, r0=o[j + 1] / e - 1, lr=r.lr)
        for k in I:
            row[k] = bool(I[k][i]); kk = i + 1
            while kk <= j and I[k][kk]: kk += 1
            row['rx_' + k] = (o[min(kk, j) + 1] / e - 1)
        rows.append(row)
A = pd.DataFrame(rows); A['mo'] = A.d.dt.to_period('M').astype(str); A['n_agree'] = A.ST.astype(int) + A.AT + A.UM1; print('flips', len(A), 'train', (~A.test).sum(), 'test', A.test.sum())
def stt(z, col='r0'):
    r = z[col] - COST; s = r.sort_values(ascending=False); return f'n {len(z):5d} mean {r.mean()*100:+6.2f}% med {r.median()*100:+6.2f}% win10 {(r>=.1).mean()*100:5.1f}% huge {(r>=1).mean()*100:4.2f}% ex-top3 {s.iloc[3:].mean()*100:+6.2f}%'
def diff(z, m, col='r0'):
    a = z.assign(r=z[col] - COST, m=m); g = a.groupby(['mo', 'm']).r.mean().unstack(); d = (g[True] - g[False]).dropna(); return d.mean() * 100, d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 5 else np.nan
print('\nPART 1 AGREEMENT FILTERS (vector flip + indicator bull at the flip bar), net of 30bp, hold to vector bear flip')
for era, z in (('TRAIN 2019-21', A[~A.test]), ('TEST 2022+', A[A.test])):
    print(f'\n {era}'); print('  all flips                ', stt(z))
    for k in ('ST', 'AT', 'UM1', 'UM2'):
        m = z[k]; d, t = diff(z, m); print(f'  {k:3s} bull  {stt(z[m])}\n  {k:3s} NOT   {stt(z[~m])} | bull minus not {d:+.2f}pp month-t {t:+.2f}')
    for lab, m in (('ST&AT&UM1 all 3', z.n_agree == 3), ('any 2+ of ST/AT/UM1', z.n_agree >= 2), ('none of ST/AT/UM1', z.n_agree == 0)):
        d, t = diff(z, m); print(f'  {lab:22s} {stt(z[m])} | vs rest {d:+.2f}pp month-t {t:+.2f}')
print('\nPART 1b indicator-bear EXIT (earlier of indicator turning bear or vector bear flip), net of 30bp')
for era, z in (('TRAIN', A[~A.test]), ('TEST', A[A.test])):
    print(f' {era} vector-only exit   ', stt(z))
    for k in ('ST', 'AT', 'UM1', 'UM2'): print(f' {era} exit on {k:3s} bear   ', stt(z, 'rx_' + k))
print('\nPART 1c same filter inside the frozen score top 20% (cut from report_90 train distribution)'); cut = F[~F.test].lr.quantile(.8)
for era, z in (('TRAIN', A[~A.test]), ('TEST', A[A.test])):
    s = z[z.lr >= cut]; print(f' {era} score top20 all   ', stt(s))
    for k in ('ST', 'AT', 'UM1'): print(f' {era} score top20 & {k:3s} bull ', stt(s[s[k]]), '| not', stt(s[~s[k]])[:60])
A.to_pickle(SC + 'ext/trend_pairing.pkl')
# PART 2: 1H
H = pd.read_pickle(SC + 'ext/flips_1h_filters.pkl'); M5 = pickle.load(open(SC + 'lowpx_5m.pkl', 'rb'))
def cl(x):
    x = x[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); x.columns = ['open', 'high', 'low', 'close', 'volume']; x.index = x.index.tz_convert('America/New_York') if x.index.tz is not None else x.index; x = x[~x.index.duplicated()]; m = x.index.hour * 60 + x.index.minute; return x[(m >= 570) & (m < 960)]
fl = {}
for tk, g in H.groupby('tk'):
    x = cl(M5[tk]); x = x[x.volume > 0]; B = x.resample('60min', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna(subset=['open']); B = B[B.index.hour * 60 + B.index.minute >= 570]
    I = indic(B.open.values, B.high.values, B.low.values, B.close.values, B.volume.values.astype(float)); ix = {t: k for k, t in enumerate(B.index)}
    for _, r in g.iterrows():
        k = ix.get(r.t)
        if k is not None: fl[(tk, r.t)] = {m: bool(I[m][k]) for m in I}
for m in ('ST', 'AT', 'UM1', 'UM2'): H[m] = [fl.get((a, b), {}).get(m, np.nan) for a, b in zip(H.tk, H.t)]
H = H.dropna(subset=['ST']); H['n_agree'] = H.ST.astype(int) + H.AT.astype(int) + H.UM1.astype(int); H['half'] = np.where(H.d < '2026-10-06', 'train', 'test'); print('\nPART 2: 1H flips (close-to-close, 100bp, 60 days, ONE regime)')
def s1(z): r = z.r_base; return f'n {len(z):5d} mean {r.mean()*100:+6.2f}% med {r.median()*100:+6.2f}% win {(r>0).mean()*100:5.1f}% >=10% {(r>=.1).mean()*100:4.1f}%'
for half in ('train', 'test'):
    z = H[H.half == half]; print(f' {half}: all {s1(z)}')
    for k in ('ST', 'AT', 'UM1', 'UM2'): print(f'   {k:3s} bull {s1(z[z[k].astype(bool)])} | not {s1(z[~z[k].astype(bool)])}')
    print(f'   all 3 {s1(z[z.n_agree == 3])} | none {s1(z[z.n_agree == 0])}')
