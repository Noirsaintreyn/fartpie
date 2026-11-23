"""Nov 22 2026 report 377: Cross-Market Propagation Index (CMPI) from an outside assistant.  Stock return predicted from lagged SPY and lagged sector-residual returns: market slope (stock_t on SPY_{t-1}, 100-bar rolling, shifted) x SPY_{t-1}; sector slope (stock_t on sector residual_{t-1}) x residual_{t-1}; sector residual = sector - beta_s x SPY (beta from prior 100 bars).  DATA SUBSTITUTION: the cache has no sector ETFs (only SPY, QQQ, IWM, DIA), so the sector series is the equal-weighted return of the OTHER same-sector names among the 54.  CMPI = 50 + (100/pi) atan(predicted / rolling 100-bar sd of the stock's return).
Tests (tercile on train 2019-21, two-sided, week-clustered t, raw and beta-hedged outcomes, features at the flip bar): MKTC (market contribution /sd), SECC (sector contribution /sd), CMPI (combined), ACC (CMPI minus CMPI 5 bars earlier), AGREE (CMPI>50 vs not).  Rey's priority question: is the combined effect present when each component alone is not.  5 features x 2 outcomes; ~1 WEAK and ~0.1 SOLID expected by luck."""
import glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet').reset_index(drop=True); b['entry'] = pd.to_datetime(b.entry); b['exit'] = pd.to_datetime(b.exit); names = sorted(b.tk.unique()); SEC = json.load(open('sp500_sectors.json'))
def load(tk):
    f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); return H.sort_index()
HS = {t: load(t) for t in names + ['SPY']}; idx = HS['SPY'].index; C = pd.DataFrame({t: HS[t].close.reindex(idx) for t in names + ['SPY']}); O = pd.DataFrame({t: HS[t].open.reindex(idx) for t in names + ['SPY']})
R = np.log(C).diff(); newday = pd.Series(idx.normalize(), index=idx) != pd.Series(idx.normalize(), index=idx).shift(); R[newday.values] = 0.0; R = R.fillna(0.0)
rs = R.SPY; sec = {t: SEC.get(t) for t in names}; print('sector coverage', sum(v is not None for v in sec.values()), 'of', len(names), flush=True)
E = pd.DataFrame(index=idx); BETA = {}
for t in names:
    bt = (R[t].rolling(500).cov(rs) / rs.rolling(500).var()).shift(1); BETA[t] = bt; E[t] = R[t] - bt * rs
feat = {}
for t in names:
    peers = [u for u in names if u != t and sec[u] is not None and sec[u] == sec[t]]
    if sec[t] is None or not peers: continue
    sr = R[peers].mean(axis=1); bs = (sr.rolling(100).cov(rs) / rs.rolling(100).var()).shift(1); res = sr - bs * rs; r = R[t]
    ms = (r.rolling(100).cov(rs.shift(1)) / rs.shift(1).rolling(100).var()).shift(1); ss = (r.rolling(100).cov(res.shift(1)) / res.shift(1).rolling(100).var()).shift(1); sd = r.rolling(100).std()
    mc = ms * rs.shift(1) / sd; sc = ss * res.shift(1) / sd; cm = 50 + 100 / np.pi * np.arctan(mc + sc)
    feat[t] = pd.DataFrame({'MKTC': mc, 'SECC': sc, 'CMPI': cm, 'ACC': cm - cm.shift(5), 'AGREE': (cm > 50).astype(float)}); BETA[t] = (R[t].rolling(500).cov(rs) / rs.rolling(500).var()).shift(1); print(t, end=' ', flush=True)
rows = []
for t, F in feat.items():
    m = b[b.tk == t]; pos = idx.get_indexer(m.entry.values) - 1; ok = pos >= 0
    d = F.iloc[pos[ok]].copy(); d.index = m.index[ok]
    so = O.SPY; en = so.reindex(idx).values[idx.get_indexer(m.entry.values[ok])]; ex = so.reindex(idx).values[np.minimum(idx.get_indexer(m.exit.values[ok], method='pad'), len(idx) - 1)]; d['spyret'] = ex / en - 1; d['bprior'] = BETA[t].iloc[pos[ok]].values; rows.append(d)
D = pd.concat(rows); b = b.join(D, how='inner'); b['res'] = b.ret - b.bprior * b.spyret; b = b[b.entry <= '2026-08-15']; b['wk'] = b.entry.dt.to_period('W').astype(str); b['half'] = np.where(b.entry < '2022-01-01', 'train', 'test'); tr_ = b.half == 'train'
print('\nflips used %d (train %d, test %d)' % (len(b), tr_.sum(), (~tr_).sum()), flush=True); b.to_parquet(SP + 'A377.parquet')
def clus(T, Rr, y):
    Z = pd.concat([T.assign(x=1.0), Rr.assign(x=0.0)]); m = sm.OLS(Z[y].values, sm.add_constant(Z.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Z.wk)[0]}); return m.params[1] * 100, m.tvalues[1]
for y, lab in (('ret', 'raw return'), ('res', 'beta-hedged residual')):
    print('--', lab, '| base train %+.2f%% test %+.2f%%' % (b[tr_][y].mean() * 100, b[~tr_][y].mean() * 100))
    for k in ('MKTC', 'SECC', 'CMPI', 'ACC', 'AGREE'):
        v = b[k]; lo, hi = (.5, .5) if k == 'AGREE' else v[tr_].quantile([1 / 3, 2 / 3]); out = {}
        for h in ('train', 'test'):
            S = b[(b.half == h) & v.notna()]; T = S[S[k] >= hi] if k != 'AGREE' else S[S[k] == 1]; Rr = S[S[k] <= lo] if k != 'AGREE' else S[S[k] == 0]; out[h] = clus(T, Rr, y) + (T[y].mean() * 100, Rr[y].mean() * 100, len(T))
        same = np.sign(out['train'][0]) == np.sign(out['test'][0]); tier = 'SOLID' if same and min(abs(out['train'][1]), abs(out['test'][1])) >= 2 else ('WEAK' if same and max(abs(out['train'][1]), abs(out['test'][1])) >= 2 else '')
        print('%-7s train top %+.2f%% vs bottom %+.2f%% diff %+.2f t %+.1f (n %d) | test top %+.2f%% vs bottom %+.2f%% diff %+.2f t %+.1f (n %d) %s' % (k, out['train'][2], out['train'][3], out['train'][0], out['train'][1], out['train'][4], out['test'][2], out['test'][3], out['test'][0], out['test'][1], out['test'][4], tier), flush=True)
print('CMPI>50 share of flips: %.0f%% train | %.0f%% test; mean |market contribution|/sd %.3f, |sector contribution|/sd %.3f' % (b.AGREE[tr_].mean() * 100, b.AGREE[~tr_].mean() * 100, b.MKTC.abs().mean(), b.SECC.abs().mean()))
