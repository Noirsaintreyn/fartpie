"""Nov 22 2026 report 374. Rey's 'Vector Quant Research Suite': five confirmations evaluated at the flip bar for each of the 12,360 live bull flips (b302), 15m, Pine defaults, no look-ahead (value at the flip bar's close, entry is the next open).
H1 DC bull: directional-change leg up >= 0.75 ATR (dcLegRatio>=1 with dcDirection=1). H2 CUSUM bull (up>=3 and > down; drift .25, 50-bar standardised). H3 Semivariance balance > 0.20 (12 bars). H4 change-point mean-shift z >= 1.5 (20 vs prior 20). H5 noise-adjusted RSI >= 60. H6 count of aligned flags >= 3.
Mechanism: a flip that arrives with momentum / direction already confirmed by independent measures is more likely to continue. Direction declared (aligned = better). Aligned vs not, mean trade difference, week-clustered t, train 2019-21 | test 2022+. Pass: same sign both halves and |t|>=2 in at least one; 6 tests so ~0.3 luck passes."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet').reset_index(drop=True); b['entry'] = pd.to_datetime(b.entry); F = {k: np.full(len(b), np.nan) for k in ('dc', 'cusum', 'semi', 'cp', 'rsi')}
def feats(H, atr):
    c = H.close.values; hi = H.high.values; lo = H.low.values; n = len(c); r = np.r_[0, np.log(c[1:] / c[:-1])]; rs = pd.Series(r)
    vol = rs.rolling(50).std(ddof=0).values; sr = np.where(vol > 0, r / vol, 0); sr = np.nan_to_num(sr)
    up = np.zeros(n); dn = np.zeros(n)
    for i in range(1, n): up[i] = max(0, up[i - 1] + sr[i] - .25); dn[i] = max(0, dn[i - 1] - sr[i] - .25)
    cus = (up >= 3) & (up > dn)
    ps = pd.Series(np.where(r > 0, r * r, 0)).rolling(12).sum().values; ns = pd.Series(np.where(r < 0, r * r, 0)).rolling(12).sum().values; tot = ps + ns; semi = np.where(tot > 0, (ps - ns) / np.where(tot > 0, tot, 1), 0) > .20
    rm = rs.rolling(20).mean(); pm = rm.shift(20); sd = rs.rolling(20).std(ddof=0); z = ((rm - pm) / (sd * np.sqrt(2 / 20))).values; cp = z >= 1.5
    nf = rs.abs().rolling(30).mean().values * .5; eu = np.maximum(r - nf, 0); ed = np.maximum(-r - nf, 0); au = pd.Series(eu).ewm(alpha=1 / 14, adjust=False).mean().values; ad = pd.Series(ed).ewm(alpha=1 / 14, adjust=False).mean().values
    rsi = np.where(ad == 0, np.where(au == 0, 50, 100), 100 - 100 / (1 + au / np.where(ad == 0, 1, ad))) >= 60
    d = 0; ext = np.nan; anchor = np.nan; dcb = np.zeros(n, bool)
    for i in range(n):
        a = atr[i]
        if not a > 0: continue
        if d == 0: d = 1 if c[i] >= c[max(i - 1, 0)] else -1; ext = hi[i] if d == 1 else lo[i]; anchor = c[i]
        elif d == 1:
            ext = max(ext, hi[i])
            if c[i] <= ext - a * .75: d = -1; ext = lo[i]; anchor = c[i]
        else:
            ext = min(ext, lo[i])
            if c[i] >= ext + a * .75: d = 1; ext = hi[i]; anchor = c[i]
        dcb[i] = d == 1 and abs(ext - anchor) / a / .75 >= 1
    return dict(dc=dcb, cusum=cus, semi=semi, cp=cp, rsi=rsi)
for tk in sorted(b.tk.unique()):
    f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    regs, lv, atr, fl = run_state_machine(H); S = feats(H, atr); m = np.where(b.tk.values == tk)[0]; ix = H.index.get_indexer(b.entry.values[m]) - 1; ok = ix >= 0
    for k, k2 in (('dc', 'dc'), ('cusum', 'cusum'), ('semi', 'semi'), ('cp', 'cp'), ('rsi', 'rsi')): F[k][m[ok]] = S[k2][ix[ok]].astype(float)
    print(tk, end=' ', flush=True)
Fd = pd.DataFrame(F); Fd['count'] = Fd[['dc', 'cusum', 'semi', 'cp', 'rsi']].sum(axis=1); Fd.to_parquet(SP + 'F374.parquet')
b['wk'] = b.entry.dt.to_period('W').astype(str); b['half'] = np.where(b.entry < '2022-01-01', 'train', 'test'); print('\nbase mean %+.2f%% | flag prevalence: %s' % (b.ret.mean() * 100, {k: round(float(Fd[k].mean()), 2) for k in F}))
def clus(T, R):
    Z = pd.concat([T.assign(x=1.0), R.assign(x=0.0)]); m = sm.OLS(Z.ret.values, sm.add_constant(Z.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Z.wk)[0]}); return m.params[1] * 100, m.tvalues[1]
for k, lab in (('dc', 'H1 DC bull leg'), ('cusum', 'H2 CUSUM bull'), ('semi', 'H3 semivar>0.2'), ('cp', 'H4 changepoint z>=1.5'), ('rsi', 'H5 noise RSI>=60'), ('count', 'H6 count>=3')):
    fl_ = (Fd[k] >= 3) if k == 'count' else (Fd[k] == 1); out = []
    for h in ('train', 'test'):
        D = b[b.half == h]; T = D[fl_[D.index]]; R = D[~fl_[D.index]]; d, t = clus(T, R); out.append((d, t, len(T), len(R), T.ret.mean() * 100, R.ret.mean() * 100, (T.ret > 0).mean() * 100, (R.ret > 0).mean() * 100))
    ok = np.sign(out[0][0]) == np.sign(out[1][0]) == 1 and max(abs(out[0][1]), abs(out[1][1])) >= 2
    print('%-22s TRAIN aligned %+.2f%% (n %d, win %.0f%%) vs not %+.2f%% (n %d, win %.0f%%): diff %+.2f t %+.1f | TEST aligned %+.2f%% (n %d, win %.0f%%) vs not %+.2f%% (n %d, win %.0f%%): diff %+.2f t %+.1f  %s' % (lab, out[0][4], out[0][2], out[0][6], out[0][5], out[0][3], out[0][7], out[0][0], out[0][1], out[1][4], out[1][2], out[1][6], out[1][5], out[1][3], out[1][7], out[1][0], out[1][1], 'PASS' if ok else ''))
