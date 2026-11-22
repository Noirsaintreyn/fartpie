"""Nov 22 2026 report 359. Rey: 'test everything you can find on arXiv/SSRN or brute force it.'  Declared before running.
SCOPE: ~110 pre-entry features per flip (12,360 flips, 54 names, 2019-26), all from data through the PRIOR close (Yahoo daily + b302 flip-bar columns), grouped by the literature they come from:
 MOM   r1..r250, 12-1, skip-month, risk-adjusted momentum (Li-Yuan-Zhou), residual / idiosyncratic momentum vs the 54-name index (Blitz et al.), 52w-high / 52w-low distance (George-Hwang), drawdown, recovery and days-since-high (Maximum drawdown & momentum, arXiv 1403.8125)
 TUG  overnight vs intraday return sums over 20/60/120d (Lou-Polk-Skouras 'tug of war'), gap on the entry day
 LOTT MAX / MIN 20d, realised skew/kurtosis 20/60, downside/upside vol, idiosyncratic vol (Bali-Cakici-Whitelaw, Ang et al.)
 LIQ  dollar volume, volume trend 5/60, Amihud illiquidity, range / ATR measures, ATRP, Yang-Zhang, ratio (live), flip-bar rvol / range
 PEER lead-lag / network momentum (arXiv 2501.07135): top-3 correlated peers' 5d / 20d return, peers' flips in the last 5 days, sector peers' 20d return, own return minus sector
 NAME name-level persistence: last 3 / 10 flips' mean return and hit rate of THIS name's vector (exits before this entry), days since last exit, flips in the last 60 sessions (chop)
 CLOCK minute of the day of the flip, day of week, month
 MKT  breadth, idx_vr, corr60, spy trend (already in the table)
TEST (no cherry-picking): rank-IC of feature vs flip return per entry month (Fama-MacBeth), t over months.  TRAIN = 2019-21 (36 mo), TEST = 2022-26.
PASS = train |t| >= 2 with Benjamini-Hochberg q < 0.10 over ALL features, same sign in test with |t| >= 1.65.  NULL: shuffle returns within month 100x, rerun the full scan, count how many features pass by luck.
Survivors then go to the 6-slot book as pick order (high or low first as signed by train), and a train-frozen ridge of the survivors as a composite.  Everything else = live book (23.6%/-23.5%/2.60; train 33.8 | test 17.5)."""
import sys, glob, os, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
B = pd.read_parquet(SP + 'b302.parquet').reset_index(drop=True); B['entry'] = pd.to_datetime(B.entry); B['exit'] = pd.to_datetime(B.exit); B['d'] = B.entry.dt.normalize(); B['mo'] = B.entry.dt.to_period('M').astype(str)
tks = sorted(B.tk.unique()); Dly = {}
for tk in tks: Y = pd.read_parquet(f'data/yahoo/daily_v/{tk}.parquet'); Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); Dly[tk] = Y[~Y.index.duplicated()]
C = pd.DataFrame({t: Dly[t].close for t in tks}).sort_index(); O = pd.DataFrame({t: Dly[t].open for t in tks}).reindex(C.index); H = pd.DataFrame({t: Dly[t].high for t in tks}).reindex(C.index); L = pd.DataFrame({t: Dly[t].low for t in tks}).reindex(C.index); V = pd.DataFrame({t: Dly[t].volume for t in tks}).reindex(C.index)
R = C.pct_change(); lr = np.log(C).diff(); on = np.log(O / C.shift()); idr = np.log(C / O); DV = C * V; mk = R.mean(axis=1)
F = {}
for k in (1, 2, 3, 5, 10, 20, 60, 120, 250): F[f'r{k}'] = C / C.shift(k) - 1
F['r12_1'] = C.shift(21) / C.shift(252) - 1; F['r60_skip5'] = C.shift(5) / C.shift(65) - 1; F['r120_skip20'] = C.shift(20) / C.shift(140) - 1
v20 = R.rolling(20).std(); v60 = R.rolling(60).std(); F['vol20'] = v20; F['vol60'] = v60; F['vol250'] = R.rolling(250).std(); F['volratio'] = v20 / v60
F['rm60'] = F['r60'] / v60; F['rm20'] = F['r20'] / v20; F['rm120'] = F['r120'] / R.rolling(120).std(); F['rm250'] = F['r250'] / F['vol250']
bt = R.rolling(120).cov(mk).div(mk.rolling(120).var(), axis=0); F['beta120'] = bt; res = R - bt.shift(1).mul(mk, axis=0)
for k in (20, 60, 120): F[f'resid{k}'] = res.rolling(k).sum()
F['idio60'] = res.rolling(60).std(); F['idio20'] = res.rolling(20).std(); F['resid_rm60'] = F['resid60'] / F['idio60']
hi = C.rolling(252, min_periods=120).max(); lo = C.rolling(252, min_periods=120).min(); F['hi52d'] = C / hi - 1; F['lo52d'] = C / lo - 1; F['hilo_pos'] = (C - lo) / (hi - lo)
for k in (60, 120, 250): cm = C.rolling(k, min_periods=k // 2).max(); F[f'dd{k}'] = (C / cm - 1).rolling(k, min_periods=k // 2).min(); F[f'ddnow{k}'] = C / cm - 1
F['recov60'] = (C - C.rolling(60).min()) / (C.rolling(60).max() - C.rolling(60).min()); F['days_since_hi'] = C.rolling(252, min_periods=120).apply(lambda a: len(a) - 1 - np.argmax(a), raw=True)
for k in (20, 60, 120): F[f'on{k}'] = on.rolling(k).sum(); F[f'id{k}'] = idr.rolling(k).sum(); F[f'on_minus_id{k}'] = F[f'on{k}'] - F[f'id{k}']
F['max20'] = R.rolling(20).max(); F['min20'] = R.rolling(20).min(); F['max60'] = R.rolling(60).max(); F['min60'] = R.rolling(60).min(); F['skew20'] = R.rolling(20).skew(); F['skew60'] = R.rolling(60).skew(); F['kurt60'] = R.rolling(60).kurt()
F['down20'] = R.clip(upper=0).pow(2).rolling(20).mean().pow(.5); F['up20'] = R.clip(lower=0).pow(2).rolling(20).mean().pow(.5); F['updown20'] = F['up20'] / F['down20']
F['ldv20'] = np.log(DV.rolling(20).mean()); F['ldv250'] = np.log(DV.rolling(250, min_periods=100).mean()); F['dvtrend'] = DV.rolling(5).mean() / DV.rolling(60).mean(); F['vtrend20'] = V.rolling(20).mean() / V.rolling(120).mean(); F['amihud20'] = (R.abs() / DV).rolling(20).mean()
pc = C.shift(); tr_ = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs())); atr = tr_.ewm(alpha=1 / 14, adjust=False).mean(); F['atrp'] = atr / C; F['atrp_rel'] = F['atrp'] / F['atrp'].rolling(252, min_periods=100).median(); F['range20'] = ((H - L) / C).rolling(20).mean(); F['clv20'] = ((2 * C - H - L) / (H - L)).rolling(20).mean(); F['updays20'] = (R > 0).rolling(20).mean(); F['updays60'] = (R > 0).rolling(60).mean()
F['absret5'] = R.abs().rolling(5).mean(); F['gapabs20'] = on.abs().rolling(20).mean(); F['ext_sma50'] = C / C.rolling(50).mean() - 1; F['ext_sma200'] = C / C.rolling(200).mean() - 1; F['sma50_200'] = C.rolling(50).mean() / C.rolling(200).mean() - 1
SEC = json.load(open('sp500_sectors.json')); EXTRA = {'GOOG': 'Communication Services', 'META': 'Communication Services', 'NFLX': 'Communication Services', 'TSLA': 'Consumer Discretionary', 'AMZN': 'Consumer Discretionary', 'BRKB': 'Financials', 'COIN': 'Financials', 'HOOD': 'Financials', 'PLTR': 'Information Technology', 'CRWD': 'Information Technology', 'APP': 'Information Technology', 'CVNA': 'Consumer Discretionary'}
sec = pd.Series({t: EXTRA.get(t) or SEC.get(t, 'Other') for t in tks}); sec_mean20 = {s: F['r20'][sec.index[sec == s]].mean(axis=1) for s in sec.unique()}; sec_mean60 = {s: F['r60'][sec.index[sec == s]].mean(axis=1) for s in sec.unique()}
F['rel_sector20'] = pd.DataFrame({t: F['r20'][t] - sec_mean20[sec[t]] for t in tks}); F['rel_sector60'] = pd.DataFrame({t: F['r60'][t] - sec_mean60[sec[t]] for t in tks}); F['sector_mom20'] = pd.DataFrame({t: sec_mean20[sec[t]] for t in tks}); F['sector_mom60'] = pd.DataFrame({t: sec_mean60[sec[t]] for t in tks})
P5 = F['r5'] * 0; P20 = P5.copy(); P60 = P5.copy(); mstart = pd.date_range(C.index[130], C.index[-1], freq='MS')
for m0 in mstart:
    w = lr.loc[:m0 - pd.Timedelta(days=1)].tail(120); cr = w.corr().values.copy(); np.fill_diagonal(cr, -9); nxt = C.index[(C.index >= m0) & (C.index < m0 + pd.offsets.MonthBegin(1))]
    for i, t in enumerate(tks):
        pe = [tks[j] for j in np.argsort(-cr[i])[:3]]; P5.loc[nxt, t] = F['r5'].loc[nxt, pe].mean(axis=1); P20.loc[nxt, t] = F['r20'].loc[nxt, pe].mean(axis=1); P60.loc[nxt, t] = F['r60'].loc[nxt, pe].mean(axis=1)
F['peer_r5'] = P5; F['peer_r20'] = P20; F['peer_r60'] = P60; F['own_minus_peer20'] = F['r20'] - P20
# shift all one day (value known at the PRIOR close of the entry date)
Fs = {k: v.shift(1) for k, v in F.items()}
rows = {}
di = C.index; pos = di.searchsorted(B.d.values, side='left'); ti = {t: i for i, t in enumerate(tks)}; ci = B.tk.map(ti).values
for k, v in Fs.items(): rows[k] = v.values[np.clip(pos, 0, len(di) - 1), ci]
X = pd.DataFrame(rows); X['gap_entry'] = (O.values[np.clip(pos, 0, len(di) - 1), ci] / C.values[np.clip(pos - 1, 0, len(di) - 1), ci] - 1)
for c in ('ratio', 'rvol', 'rngatr', 'hi52', 'flipm', 'room', 'bearlen', 's1', 'mkt', 'ldv', 'breadth', 'idx_vr', 'corr60', 'spy_trend', 'trend', 'mom', 'beta', 'vol20', 'vr', 'ac1', 'mom20', 'rs20'): X['b_' + c] = B[c].values
X['clock_min'] = B.entry.dt.hour * 60 + B.entry.dt.minute; X['dow'] = B.entry.dt.dayofweek; X['month_of_year'] = B.entry.dt.month
# name-level persistence and peers' flips (only exits <= this entry, so nothing leaks)
nm = {k: np.full(len(B), np.nan) for k in ('last1', 'last3', 'last10', 'hit3', 'hit10', 'days_since_exit', 'flips60')}; peerfl = np.full(len(B), np.nan)
for t, g in B.groupby('tk'):
    g = g.sort_values('entry'); ex = g.exit.values; en = g.entry.values; ret = g.ret.values; idxs = g.index.values; order = np.argsort(ex)
    for j in range(len(g)):
        done = [q for q in order if ex[q] <= en[j]]
        if done:
            r_ = ret[done]; nm['last1'][idxs[j]] = r_[-1]; nm['last3'][idxs[j]] = r_[-3:].mean(); nm['last10'][idxs[j]] = r_[-10:].mean(); nm['hit3'][idxs[j]] = (r_[-3:] > 0).mean(); nm['hit10'][idxs[j]] = (r_[-10:] > 0).mean(); nm['days_since_exit'][idxs[j]] = (en[j] - ex[done[-1]]) / np.timedelta64(1, 'D')
        nm['flips60'][idxs[j]] = ((en < en[j]) & (en >= en[j] - np.timedelta64(86, 'D'))).sum()
for k, v in nm.items(): X['nm_' + k] = v
# peers' flips in prior 5 days: top-3 peers from the same month table (P5 construction above used top-3 by 120d corr) -> use sector peers instead (cheap, no leak)
tk_ = B.tk.values; secid = B.tk.map(sec).values; en_all = B.entry.values; pf = np.zeros(len(B))
for s in np.unique(secid):
    ii = np.where(secid == s)[0]; e = en_all[ii]; tkk = tk_[ii]
    for a, i in enumerate(ii): pf[i] = ((e < e[a]) & (e >= e[a] - np.timedelta64(5, 'D')) & (tkk != tkk[a])).sum()
X['sector_flips5d'] = pf
X = X.replace([np.inf, -np.inf], np.nan); y = B.ret.values; tr = ((B.entry >= '2019-01-01') & (B.entry < '2022-01-01')).values; te = (B.entry >= '2022-01-01').values
def scan(yv):
    out = {}
    for c in X.columns:
        x = X[c].values; res = []
        for mask in (tr, te):
            ics = []
            for m, ii in pd.Series(np.where(mask)[0]).groupby(B.mo.values[mask]):
                ii = ii.values; xv = x[ii]; yy = yv[ii]; ok = ~np.isnan(xv)
                if ok.sum() >= 8 and np.nanstd(xv[ok]) > 0: ics.append(stats.spearmanr(xv[ok], yy[ok])[0])
            ics = np.array(ics); res.append((np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics, ddof=1) / np.sqrt(len(ics))) if len(ics) > 3 else np.nan, len(ics)))
        out[c] = res
    return out
def passed(out):
    t_tr = np.array([out[c][0][1] for c in X.columns]); t_te = np.array([out[c][1][1] for c in X.columns]); p = 2 * (1 - stats.norm.cdf(np.abs(np.nan_to_num(t_tr)))); o = np.argsort(p); m = len(p); thr = np.zeros(m, bool)
    for r_, i in enumerate(o): thr[i] = p[i] <= .10 * (r_ + 1) / m
    k = np.max(np.where(thr[o])[0]) if thr.any() else -1; bh = np.zeros(m, bool)
    if k >= 0: bh[o[:k + 1]] = True
    return bh & (np.abs(t_tr) >= 2) & (np.sign(t_tr) == np.sign(t_te)) & (np.abs(np.nan_to_num(t_te)) >= 1.65)
out = scan(y); ok = passed(out); cols = list(X.columns)
print(f'{len(cols)} features | flips {len(B)} (train {tr.sum()} | test {te.sum()})\n')
print('TOP 25 by |train t| (IC = mean monthly Spearman with flip return, t over months):')
tab = pd.DataFrame({'ic_tr': [out[c][0][0] for c in cols], 't_tr': [out[c][0][1] for c in cols], 'ic_te': [out[c][1][0] for c in cols], 't_te': [out[c][1][1] for c in cols], 'PASS': ok}, index=cols).sort_values('t_tr', key=np.abs, ascending=False)
print(tab.head(25).round(3).to_string()); print('\nPASSING features:', list(tab.index[tab.PASS]))
print('\nNULL: shuffle returns within month, rescan (30 draws; features passing by luck):')
rng = np.random.default_rng(11); cnt = []
for d in range(30):
    ys = y.copy()
    for m, ii in pd.Series(np.arange(len(B))).groupby(B.mo.values): ii = ii.values; ys[ii] = rng.permutation(y[ii])
    cnt.append(int(passed(scan(ys)).sum()))
print(f'  false passes per scan: mean {np.mean(cnt):.2f}, max {np.max(cnt)}, draws with >= real count ({int(ok.sum())}): {np.mean(np.array(cnt) >= ok.sum()):.0%}')
X.assign(ret=y, tr=tr).to_parquet(SP + 'X359.parquet'); tab.to_csv(SP + 'tab359.csv')
