"""Nov 22 2026 report 376: Lead-Lag Propagation Index (LLPI), proposed by an outside assistant and approved by Rey: 'OHLC across multiple symbols, improve the 15m vector crosses'.
Construction (15m log returns, overnight bar set to 0, everything known at the flip bar's close):
 eps_i = r_i - beta_i r_SPY (beta = prior-500-bar rolling cov/var, shifted one bar)  = residual after the market
 esec_i = mean of eps over the OTHER names in i's sector among the 54 (sp500_sectors.json; names with no sector peer are dropped)
 theta_i = prior-500-bar rolling slope of eps_i(t) on esec_i(t-1) (delayed transmission of the sector shock into the stock, shifted one bar)
 PROP = theta_i x (esec_i(t)+esec_i(t-1)+esec_i(t-2))  = how much of the sector's last-3-bar shock is still expected to arrive in the stock.
Also: THETA, SECRUN (8-bar sum of esec: sector leads), STKRES (8-bar sum of eps_i: stock-specific run-up), MKTRUN (8-bar sum of beta r_SPY: market-driven run-up).
Hypothesis: positive propagation (PROP high) = an independent source still transmitting, so more continuation; stock-specific run-up (STKRES high) = continuation not explained by market or sector. Two-sided terciles on train 2019-21, top-minus-bottom, week-clustered t.  Outcomes: raw trade return AND beta-hedged residual (ret - beta_prior x SPY return over the hold).  SOLID = same sign and |t|>=2 in both halves.  5 features x 2 outcomes = 10 tests: expect ~0.1 SOLID, ~1 WEAK by luck.  SPY 15m ends 2026-08-15, so later flips have no value."""
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
    es = E[peers].mean(axis=1); es1 = es.shift(1)
    theta = (E[t].rolling(500).cov(es1) / es1.rolling(500).var()).shift(1)
    prop = theta * (es + es1 + es.shift(2)); F = pd.DataFrame({'PROP': prop, 'THETA': theta, 'SECRUN': es.rolling(8).sum(), 'STKRES': E[t].rolling(8).sum(), 'MKTRUN': (BETA[t] * rs).rolling(8).sum()}); feat[t] = F; print(t, end=' ', flush=True)
rows = []
for t, F in feat.items():
    m = b[b.tk == t]; pos = idx.get_indexer(m.entry.values) - 1; ok = pos >= 0
    d = F.iloc[pos[ok]].copy(); d.index = m.index[ok]
    so = O.SPY; en = so.reindex(idx).values[idx.get_indexer(m.entry.values[ok])]; ex = so.reindex(idx).values[np.minimum(idx.get_indexer(m.exit.values[ok], method='pad'), len(idx) - 1)]; d['spyret'] = ex / en - 1; d['bprior'] = BETA[t].iloc[pos[ok]].values; rows.append(d)
D = pd.concat(rows); b = b.join(D, how='inner'); b['res'] = b.ret - b.bprior * b.spyret; b = b[b.entry <= '2026-08-15']; b['wk'] = b.entry.dt.to_period('W').astype(str); b['half'] = np.where(b.entry < '2022-01-01', 'train', 'test'); tr_ = b.half == 'train'
print('\nflips used %d (train %d, test %d)' % (len(b), tr_.sum(), (~tr_).sum()), flush=True); b.to_parquet(SP + 'A376.parquet')
def clus(T, Rr, y):
    Z = pd.concat([T.assign(x=1.0), Rr.assign(x=0.0)]); m = sm.OLS(Z[y].values, sm.add_constant(Z.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Z.wk)[0]}); return m.params[1] * 100, m.tvalues[1]
for y, lab in (('ret', 'raw return'), ('res', 'beta-hedged residual')):
    print('--', lab, '| base train %+.2f%% test %+.2f%%' % (b[tr_][y].mean() * 100, b[~tr_][y].mean() * 100))
    for k in ('PROP', 'THETA', 'SECRUN', 'STKRES', 'MKTRUN'):
        v = b[k]; lo, hi = v[tr_].quantile([1 / 3, 2 / 3]); out = {}
        for h in ('train', 'test'):
            S = b[(b.half == h) & v.notna()]; T = S[S[k] >= hi]; Rr = S[S[k] <= lo]; out[h] = clus(T, Rr, y) + (T[y].mean() * 100, Rr[y].mean() * 100, len(T))
        same = np.sign(out['train'][0]) == np.sign(out['test'][0]); tier = 'SOLID' if same and min(abs(out['train'][1]), abs(out['test'][1])) >= 2 else ('WEAK' if same and max(abs(out['train'][1]), abs(out['test'][1])) >= 2 else '')
        print('%-7s train top %+.2f%% vs bottom %+.2f%% diff %+.2f t %+.1f (n %d) | test top %+.2f%% vs bottom %+.2f%% diff %+.2f t %+.1f (n %d) %s' % (k, out['train'][2], out['train'][3], out['train'][0], out['train'][1], out['train'][4], out['test'][2], out['test'][3], out['test'][0], out['test'][1], out['test'][4], tier), flush=True)
th = b.THETA; print('theta: mean %.3f, share >0 %.0f%% (train) | %.0f%% (test)' % (th.mean(), (th[tr_] > 0).mean() * 100, (th[~tr_] > 0).mean() * 100))
