"""WHAT SEPARATES GOOD FROM BAD VECTOR TRADES (other than volatility)? Hypothesis-first feature audit, fixed before results. Every feature is computed from data known at the flip bar's close (no look-ahead), signed so that POSITIVE = 'in favour of the trade direction'.
Trades: every 15m vector flip, enter next open (08:00-15:45 ET), exit at the open after the opposite flip (as live). Outcome = signed return in bp (rank/Spearman IC, robust to fat tails) + tercile spread.
Panel 1: 8 mega-caps 15m 2019-26 (train 2019-21 / test 2022+, QQQ for market context). Panel 2: 22 small/mid names 1h RTH ~Dec23-Nov26 (halves; QQQ-free features only).
Features: stop_atr (ratchet-stop distance / ATR = risk taken), er (Kaufman efficiency ratio of the prior W bars = trendiness), prev_len (bars the previous regime lasted), chop (flips in last 3W bars), gap (overnight gap / ADR), day_mv (move since open / ADR), pos_rng (location in day range so far), d20 (distance beyond prior-20d extreme / ADR), ret5 (5-day return / ADR), mkt (QQQ move today in direction / its ADR), rs (own minus QQQ move), vol_rel (recent volume vs 96-bar mean), atr_pct (vol percentile, included as a control), d1 (close vs 20d mean of daily closes), h1 (1h vector state agrees, panel 1), ema (price vs EMA50 agrees), impulse (|close - close 4 bars ago| / ATR), body (flip-bar body / range), tod (minute of day), dow.
Test: Spearman IC(feature, trade return) per era; family-wise permutation p for the best |IC| of ~20 features on TRAIN (shuffle returns across trades 1000x, max |IC| per shuffle); then the REPLICATION read of the same sign and single-test p on TEST. Only features that pass train FWER and replicate in test count as 'quantifying good vs bad'."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from scipy.stats import spearmanr, rankdata
from vector_pine import run_state_machine
rng = np.random.default_rng(21)
def load15(tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_2018-01-01_*.parquet')[0]; H = pd.read_parquet(f, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.drop(columns='datetime').dropna().sort_index(); H = H[~H.index.duplicated()]
    m = H.index.hour * 60 + H.index.minute; return H[(m >= 240) & (m < 960)]
def feats(B, W, Q=None, H1=True):
    n = len(B); reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); o, h, l, c, v = (B[k].values for k in ('open', 'high', 'low', 'close', 'volume'))
    m = (B.index.hour * 60 + B.index.minute).values; day = B.index.normalize(); rth = (m >= 570) & (m < 960)
    d = pd.DataFrame({'o': B.open.where(rth).groupby(day).first(), 'h': B.high.where(rth).groupby(day).max(), 'l': B.low.where(rth).groupby(day).min(), 'c': B.close.where(rth).groupby(day).last()}).dropna()
    rgp = (d.h - d.l) / d.c; adr = rgp.rolling(20).mean().shift(); dc = d.c; sma20 = dc.rolling(20).mean().shift(); hi20 = d.h.rolling(20).max().shift(); lo20 = d.l.rolling(20).min().shift(); pc = dc.shift()
    adr_b = adr.reindex(day).values; pc_b = pc.reindex(day).values; do_b = d.o.reindex(day).values; sma_b = sma20.reindex(day).values; hi_b = hi20.reindex(day).values; lo_b = lo20.reindex(day).values
    c5 = (dc / dc.shift(5) - 1).shift().reindex(day).values
    dayhi = B.high.where(rth).groupby(day).cummax().values; daylo = B.low.where(rth).groupby(day).cummin().values
    er = np.full(n, np.nan); cs = pd.Series(c); er = (cs.diff(W).abs() / cs.diff().abs().rolling(W).sum()).values
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values; atrp = pd.Series(atr).rolling(500, min_periods=100).apply(lambda z: (z[:-1] < z[-1]).mean(), raw=True).values
    vr = pd.Series(v).rolling(4).mean().values / pd.Series(v).rolling(96).mean().values
    if H1:
        h1 = B.resample('60min', origin='start_day', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna(); r1 = run_state_machine(h1.assign(datetime=h1.index))[0]
        ser = pd.Series(r1, index=h1.index + pd.Timedelta(minutes=45)); ser = ser[~ser.index.duplicated()]; h1s = ser.reindex(B.index, method='ffill').fillna(0).values
    else: h1s = np.zeros(n)
    if Q is not None:
        qc = Q['close'].reindex(B.index, method='ffill').values; qd = Q['close'].where((Q.index.hour * 60 + Q.index.minute >= 570) & (Q.index.hour * 60 + Q.index.minute < 960)).groupby(Q.index.normalize())
        qpc = qd.last().shift().reindex(day).values; qh = Q['high'].where((Q.index.hour * 60 + Q.index.minute >= 570)).groupby(Q.index.normalize()).max(); ql = Q['low'].where((Q.index.hour * 60 + Q.index.minute >= 570)).groupby(Q.index.normalize()).min()
        qadr = (((qh - ql) / qd.last()).rolling(20).mean().shift()).reindex(day).values; qmv = (qc / qpc - 1) / qadr
    ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]; rows = []
    for a in range(1, len(ch) - 1):
        i, j = ch[a], ch[a + 1]; st = i + 1
        if j + 1 >= n or not (480 <= m[st] < 945) or np.isnan(adr_b[i]) or np.isnan(atr[i]) or np.isnan(er[i]) or np.isnan(sma_b[i]): continue
        dr = int(reg[i]); e = o[st]; x = o[j + 1]; ret = dr * (x / e - 1) * 1e4; A = adr_b[i] * c[i]
        f = dict(ret=ret, R=float(np.clip(ret / max(abs(c[i] - lev[i]) / c[i] * 1e4, 1.0), -10, 10)), yr=B.index[st].year, ts=B.index[st], stop_atr=abs(c[i] - lev[i]) / atr[i], er=er[i], prev_len=ch[a] - ch[a - 1], chop=sum(1 for k in ch[max(0, a - 12):a] if i - k <= 3 * W),
                 gap=dr * (do_b[i] / pc_b[i] - 1) / adr_b[i] if rth[i] else np.nan, day_mv=dr * (c[i] / pc_b[i] - 1) / adr_b[i], pos_rng=(((c[i] - daylo[i]) / max(dayhi[i] - daylo[i], 1e-9)) if dr > 0 else ((dayhi[i] - c[i]) / max(dayhi[i] - daylo[i], 1e-9))) if rth[i] else np.nan,
                 d20=(dr * ((c[i] - hi_b[i]) if dr > 0 else (lo_b[i] - c[i]))) / A, ret5=dr * c5[i] / adr_b[i], vol_rel=vr[i], atr_pct=atrp[i], d1=dr * np.sign(c[i] - sma_b[i]), h1=dr * h1s[i], ema=dr * np.sign(c[i] - ema50[i]),
                 impulse=dr * (c[i] - c[i - 4]) / atr[i], body=dr * (c[i] - o[i]) / max(h[i] - l[i], 1e-9), tod=m[st], dow=B.index[st].dayofweek)
        if Q is not None: f['mkt'] = dr * qmv[i]; f['rs'] = dr * ((c[i] / pc_b[i] - 1) / adr_b[i]) - dr * qmv[i]
        rows.append(f)
    return pd.DataFrame(rows)
FEATS = ['stop_atr', 'er', 'prev_len', 'chop', 'gap', 'day_mv', 'pos_rng', 'd20', 'ret5', 'mkt', 'rs', 'vol_rel', 'atr_pct', 'd1', 'h1', 'ema', 'impulse', 'body', 'tod', 'dow']
def ic(x, y):
    mk = ~(np.isnan(x) | np.isnan(y)); 
    return spearmanr(x[mk], y[mk])[0] if mk.sum() > 50 else np.nan
def audit(D, tr, te, label, nperm=300, Y='ret'):
    D = D.assign(ret=D[Y]); fs = [f for f in FEATS if f in D.columns and D[f].notna().sum() > 200]; A = D[tr(D)]; T = D[te(D)]; y = A.ret.values; yt = T.ret.values
    print(f'\n################ {label}: trades train {len(A)} test {len(T)} | mean ret train {A.ret.mean():+.1f} bp test {T.ret.mean():+.1f} bp | win% train {100*(A.ret>0).mean():.1f} test {100*(T.ret>0).mean():.1f}')
    ict = {f: ic(A[f].values.astype(float), y) for f in fs}; ice = {f: ic(T[f].values.astype(float), yt) for f in fs}
    # permutation FWER on train
    ranks = {f: rankdata(np.where(np.isnan(A[f].values.astype(float)), np.nanmedian(A[f].values.astype(float)), A[f].values.astype(float))) for f in fs}
    mx = np.zeros(nperm); ry = rankdata(y)
    for z in range(nperm):
        py = rng.permutation(ry); mx[z] = max(abs(np.corrcoef(ranks[f], py)[0, 1]) for f in fs)
    print('feature       IC train   FWER-p(train)   IC test   p(test,single)   tercile spread test (top-bottom, bp)   verdict')
    out = []
    for f in sorted(fs, key=lambda k: -abs(ict[k])):
        x = T[f].values.astype(float); mk = ~np.isnan(x); sp = np.nan
        if mk.sum() > 90:
            q1, q2 = np.nanquantile(x, [1 / 3, 2 / 3]); lo_, hi_ = yt[mk & (x <= q1)], yt[mk & (x >= q2)]
            if len(lo_) > 20 and len(hi_) > 20: sp = hi_.mean() - lo_.mean()
        n_ = mk.sum(); p_te = 2 * (1 - __import__('scipy.stats', fromlist=['norm']).norm.cdf(abs(ice[f]) * np.sqrt(n_ - 3))) if n_ > 50 else np.nan
        fw = (mx >= abs(ict[f])).mean(); ok = fw < .10 and np.sign(ict[f]) == np.sign(ice[f]) and p_te < .05
        print(f'{f:10s}  {ict[f]:+7.3f}   {fw:8.3f}      {ice[f]:+7.3f}   {p_te:8.3f}        {sp:+8.1f}      {"REPLICATES" if ok else ""}', flush=True); out.append((f, ok))
    return out
TK = ['AAPL', 'AMZN', 'AVGO', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']; Q = load15('QQQ')
D1 = pd.concat([feats(load15(tk), 32, Q).assign(tk=tk) for tk in TK]); print('panel1 trades', len(D1), flush=True)
audit(D1, lambda D: D.yr <= 2021, lambda D: D.yr >= 2022, 'PANEL 1: 8 mega-caps 15m (train 2019-21 / test 2022+)')
import yfinance as yf
def h1y(tk):
    d = yf.download(tk, period='730d', interval='1h', prepost=False, progress=False, auto_adjust=False); d.columns = [c[0].lower() for c in d.columns]
    d.index = d.index.tz_convert('America/New_York').tz_localize(None); d = d[['open', 'high', 'low', 'close', 'volume']].dropna(); return d[~d.index.duplicated()]
NAMES = ['KGS','LCID','RIOT','Z','IREN','PLUG','UPST','CLOV','TLRY','BB','SOFI','FUBO','OPEN','ETSY','DKNG','ROKU','CROX','FIVE','CHWY','MTCH','LYFT','BBWI']; rows = []
for tk in NAMES:
    try:
        B = h1y(tk)
        if len(B) > 1500: rows.append(feats(B, 8, None, H1=False).assign(tk=tk))
    except Exception as e: print('skip', tk, str(e)[:60])
D2 = pd.concat(rows); print('panel2 trades', len(D2), flush=True)
audit(D2, lambda D: D.ts < pd.Timestamp('2025-01-01'), lambda D: D.ts >= pd.Timestamp('2025-01-01'), 'PANEL 2: 22 small/mid names 1h (half 1 = Dec23-24 / half 2 = 2025+)')
print('\n#### STOP-DISTANCE BUCKETS (cut points = TRAIN terciles, applied unchanged to test): mean bp / win% / n')
for lab, D, trm, tem in (('mega-caps', D1, lambda D: D.yr <= 2021, lambda D: D.yr >= 2022), ('small/mid 1h', D2, lambda D: D.ts < pd.Timestamp('2025-01-01'), lambda D: D.ts >= pd.Timestamp('2025-01-01'))):
    A = D[trm(D)]; T = D[tem(D)]
    for f in ('stop_atr', 'atr_pct'):
        q1, q2 = A[f].quantile([1 / 3, 2 / 3]); print(f'  {lab} by {f} (cuts {q1:.2f}/{q2:.2f})')
        for era, X in (('train', A), ('test ', T)):
            r = []
            for nm, mk in (('low third', X[f] <= q1), ('mid third', (X[f] > q1) & (X[f] <= q2)), ('high third', X[f] > q2)): s = X[mk]; r.append(f'{nm}: {s.ret.mean():+7.1f} bp {100 * (s.ret > 0).mean():4.1f}% n={len(s)}')
            print(f'    {era}  ' + ' | '.join(r))

print("\n\n================ SECOND PASS: outcome = R-multiple (return / stop distance, clipped +-10) so wide-stop trades are not scored just for being bigger ================")
audit(D1, lambda D: D.yr <= 2021, lambda D: D.yr >= 2022, 'PANEL 1 (R)', Y='R')
audit(D2, lambda D: D.ts < pd.Timestamp('2025-01-01'), lambda D: D.ts >= pd.Timestamp('2025-01-01'), 'PANEL 2 (R)', Y='R')
