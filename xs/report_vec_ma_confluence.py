"""VECTOR x MOVING-AVERAGE CONFLUENCE ON HIGH-VOL DAYS. Hypothesis (from reading charts): when the 15m vector state agrees with price-vs-trend-filter, the vector's call is better. DECLARED BEFORE RESULTS.
Data: 8 mega-caps + none else with long 15m history (lse_clean, 04:00-16:00 ET). Train 2019-21 (select), test 2022+ (frozen). 5m is not available locally so timeframes tested are 15m and 1h (resampled, 1h bias known only after the 1h bar closes).
Filters (bias = sign(close - filter)): SMA L=20/50/100/200, EMA same, Kalman local-level (q=.3/.03/.003), Kalman local-linear-trend slope (q_slope=1e-3/1e-4/1e-5), causal a-trous Haar wavelet smooth level 3-6. Length unit = bars of that timeframe.
Position at bar t = vector state at t-1 close IF bias at t-1 close agrees, else flat; scored bar t open->close, RTH bars only, class HV-pre (prev day's range in top third of own history = known at the open; no hindsight).
Stat = mean bp per ACTIVE bar. Null = circular shifts of the position series vs returns (400, per ticker independent), keeps persistence/exposure. Family-wise p for the best of 36 configs = share of shifts whose MAX z over configs beats the observed best z (train). Selected config is frozen and read once on 2022+.
Also reported: vector alone (same mask), MA alone, and 'conflict' (vector active where bias disagrees) = does the filter carry information.
Chart-reading caveat: eyeballing days that were ALREADY high-vol/high-volume is the HV-post (hindsight) class, which earlier showed +2 bp/bar for any state; this test only uses what is known at the open."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
rng = np.random.default_rng(5)
TK = ['AAPL', 'AMZN', 'AVGO', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
def bars(tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_2018-01-01_*.parquet')[0]; H = pd.read_parquet(f, columns=['datetime', 'open', 'high', 'low', 'close'])
    H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.drop(columns='datetime').dropna().sort_index(); H = H[~H.index.duplicated()]
    m = H.index.hour * 60 + H.index.minute; return H[(m >= 240) & (m < 960)]
def sma(x, L): return pd.Series(x).rolling(L).mean().values
def ema(x, L): return pd.Series(x).ewm(span=L, adjust=False).mean().values
def kal_level(y, q):
    x = np.empty(len(y)); P = 1.0; m = y[0]; 
    for i, v in enumerate(y): Pp = P + q; K = Pp / (Pp + 1); m = m + K * (v - m); P = (1 - K) * Pp; x[i] = m
    return x
def kal_trend(y, qs, ql=.1):
    n = len(y); lev = np.empty(n); sl = np.empty(n); x = np.array([y[0], 0.]); P = np.eye(2); F = np.array([[1, 1.], [0, 1]]); Q = np.diag([ql, qs]); 
    for i, v in enumerate(y):
        x = F @ x; P = F @ P @ F.T + Q; S = P[0, 0] + 1; K = P[:, 0] / S; x = x + K * (v - x[0]); P = P - np.outer(K, P[0]); lev[i] = x[0]; sl[i] = x[1]
    return lev, sl
def haar(x, J):
    c = x.copy()
    for j in range(J): lag = 2 ** j; d = np.r_[np.full(lag, np.nan), c[:-lag]]; c = (c + d) / 2
    return c
def biases(c, scale):
    """c = close series of one timeframe; returns dict name->bias(+1/-1/0) at that bar's close"""
    out = {}; y = np.log(c) / scale
    for L in (20, 50, 100, 200): out[f'SMA{L}'] = np.sign(c - sma(c, L)); out[f'EMA{L}'] = np.sign(c - ema(c, L))
    for q in (.3, .03, .003): out[f'KalLvl{q}'] = np.sign(y - kal_level(y, q))
    for qs in (1e-3, 1e-4, 1e-5): out[f'KalTrend{qs:g}'] = np.sign(kal_trend(y, qs)[1])
    for J in (3, 4, 5, 6): out[f'Wav{J}(~{2**J}b)'] = np.sign(c - haar(c, J))
    return {k: np.nan_to_num(v) for k, v in out.items()}
def prep(tk):
    B = bars(tk); n = len(B); reg = run_state_machine(B.assign(datetime=B.index))[0]; s = np.r_[0, reg[:-1]].astype(float)
    day = B.index.normalize(); m = B.index.hour * 60 + B.index.minute; rth = (m >= 570) & (m < 960)
    d = pd.DataFrame({'h': B.high.where(rth).groupby(day).max(), 'l': B.low.where(rth).groupby(day).min()}).dropna(); rg = d.h - d.l
    pre = rg.shift() / rg.shift().rolling(20).mean().shift(); rk = pre.expanding(min_periods=60).apply(lambda z: (z.iloc[:-1] < z.iloc[-1]).mean(), raw=False)
    hv = (rk.reindex(day) >= 2 / 3).fillna(False).values; r = B.close.values / B.open.values - 1; yr = B.index.year.values
    c15 = B.close.values; sc15 = np.std(np.diff(np.log(c15[:5000])))
    cols = {}; bb = biases(c15, sc15)
    for k, v in bb.items(): cols['15m ' + k] = np.r_[0, v[:-1]]
    h = B.resample('60min', origin='start_day', offset='30min').agg({'close': 'last'}).dropna(); sc1 = np.std(np.diff(np.log(h.close.values[:1500]))); bh = biases(h.close.values, sc1)
    key = h.index + pd.Timedelta(minutes=45)                                    # last 15m bar of the 1h bin: its close = the 1h close
    for k, v in bh.items():
        ser = pd.Series(v, index=key); ser = ser[~ser.index.duplicated()]; cols['1h ' + k] = np.r_[0, ser.reindex(B.index, method='ffill').fillna(0).values[:-1]]
    return dict(n=n, s=s, r=r, rth=rth, hv=hv, yr=yr, cols=cols)
P = {tk: prep(tk) for tk in TK}; names = list(next(iter(P.values()))['cols'].keys()); C = len(names); print('configs', C, flush=True)
def build(p):                                                                    # position matrices
    s = p['s'][:, None]; Bm = np.column_stack([p['cols'][k] for k in names]); agree = (Bm == s) & (s != 0)
    return {'conf': np.where(agree, s, 0.), 'vec': p['s'][:, None], 'ma': Bm, 'confl': np.where((Bm == -s) & (s != 0), s, 0.)}
for tk, p in P.items(): p['pos'] = build(p)
def era_idx(p, lo, hi): return np.where(p['hv'] & p['rth'] & (p['yr'] >= lo) & (p['yr'] <= hi))[0]
def stat(kind, lo, hi, nsh=0):
    """pooled mean bp per active bar for each column; with nsh>0 also null matrix (nsh x C)"""
    obs_n = np.zeros(C); obs_d = np.zeros(C); IDX = {}
    for tk, p in P.items():
        idx = era_idx(p, lo, hi); IDX[tk] = idx; X = p['pos'][kind]; X = X if X.shape[1] == C else np.repeat(X, C, 1)
        obs_n += (X[idx] * p['r'][idx, None]).sum(0); obs_d += (X[idx] != 0).sum(0)
    obs = obs_n / np.maximum(obs_d, 1) * 1e4
    if not nsh: return obs, obs_d, None
    null = np.zeros((nsh, C))
    for k in range(nsh):
        nn = np.zeros(C); dd = np.zeros(C)
        for tk, p in P.items():
            idx = IDX[tk]; sh = rng.integers(200, p['n'] - 200); rows = (idx - sh) % p['n']; X = p['pos'][kind]; X = X if X.shape[1] == C else np.repeat(X, C, 1)
            nn += (X[rows] * p['r'][idx, None]).sum(0); dd += (X[rows] != 0).sum(0)
        null[k] = nn / np.maximum(dd, 1) * 1e4
    return obs, obs_d, null
NS = 400
tr, trn, trnull = stat('conf', 2019, 2021, NS); te, ten, tenull = stat('conf', 2022, 2026, NS)
vtr, _, vtrnull = stat('vec', 2019, 2021, NS); vte, _, vtenull = stat('vec', 2022, 2026, NS)
mtr, _, _ = stat('ma', 2019, 2021); mte, _, _ = stat('ma', 2022, 2026); ctr, _, _ = stat('confl', 2019, 2021); cte, cten, _ = stat('confl', 2022, 2026)
ztr = (tr - trnull.mean(0)) / (trnull.std(0) + 1e-9); zte = (te - tenull.mean(0)) / (tenull.std(0) + 1e-9)
mx = ((trnull - trnull.mean(0)) / (trnull.std(0) + 1e-9)).max(1); fw = np.array([(mx >= z).mean() for z in ztr])   # family-wise: max-z over configs per shift
print(f'\nVECTOR ALONE on HV-pre bars: train {vtr[0]:+.2f} bp/bar (null mean {vtrnull[:,0].mean():+.2f}) | test {vte[0]:+.2f} bp/bar (null {vtenull[:,0].mean():+.2f}, p={(vtenull[:,0] >= vte[0]).mean():.3f})')
print('\nconfig                  | TRAIN 2019-21: n_active  conf bp  z   FWER-p | TEST 2022+: n_active conf bp  z  p(single) | MA-alone test | vec-vs-MA-conflict test (n)')
order = np.argsort(-ztr)
for c in order: print(f'{names[c]:22s} | {int(trn[c]):7d} {tr[c]:+7.2f} {ztr[c]:+5.2f} {fw[c]:6.3f} | {int(ten[c]):7d} {te[c]:+7.2f} {zte[c]:+5.2f} {(tenull[:,c] >= te[c]).mean():6.3f} | {mte[c]:+7.2f} | {cte[c]:+7.2f} ({int(cten[c])})')
b = order[0]; print(f'\nSELECTED ON TRAIN (best z): {names[b]}  -> frozen TEST: confluence {te[b]:+.2f} bp/bar vs vector alone {vte[0]:+.2f}; single-test p={(tenull[:,b] >= te[b]).mean():.3f}; conflict-bars {cte[b]:+.2f}')
print(f'configs positive on BOTH eras: {int(((tr > vtr[0]) & (te > vte[0])).sum())} of {C} (beat vector-alone in both); mean test conf edge across configs {te.mean():+.2f} bp vs vector-alone {vte[0]:+.2f}')
