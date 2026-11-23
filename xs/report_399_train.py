"""Nov 23 2026 report 399 (part 2): POOLED, cross-market models on flips.  PRE-REGISTERED before any result.
Data: windows from report_399_build.py (large caps A + sub-$5 small caps B), 9 channels incl. SPY co-movement and rolling correlation with SPY, plus per-flip scalars: consolidation/AMD flags (report 397 definitions, 8), book flag, log entry price, options ratio (A only, 0 + mask for B), and two flip-clustering counts (number of pooled flips that entered in the prior 60 min, all and same-book).
Targets: flip return winsorised 1/99% on train and standardised WITHIN book; Huber (+BCE win head for NNs).  Train 2019-01..2021-12 pooled; last 15% by time = early-stop validation; tests never touched: A 2018, A 2022-26 (and ratio>=1.5 subset), B 2022-26 (and entry price < $5 = the live small-cap book).
Models (fixed): M1 CNN (3 conv blocks 32/48/64, kernel 5, GELU, global mean+max pool, scalar branch) ; M2 CNN-BiLSTM (as report 398 + scalar branch) ; M3 HistGradientBoosting on window summaries (last/mean/min/max/std of the 9 channels) + scalars (depth 3, lr 0.03, l2, min leaf 200; iterations picked on the validation block).  NNs: 3 seeds averaged.  CONTROL: M1 and M3 with train labels permuted.
Metrics per test set: Spearman IC (mean weekly cross-section IC and its t), top-minus-bottom tercile flip-return diff (cuts on TRAIN scores of the same book, week-clustered t), AUC.  Declared pass: IC > 0 and tercile diff > 0 with t >= 2 in BOTH A 2022-26 and B(<$5) 2022-26, not negative in A 2018, and clearly above the matching control."""
import numpy as np, pandas as pd, torch, torch.nn as nn, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; torch.set_num_threads(2)
Z = np.load(SP + 'win399.npz', allow_pickle=True); X = Z['X']; ret = Z['ret'].astype(float); entry = pd.to_datetime(Z['entry']); tk = Z['tk']; ds = Z['ds']; px = Z['px'].astype(float)
M = pd.DataFrame({'ds': ds, 'tk': tk, 'entry': entry}); am = pd.read_parquet(SP + 'amd397.parquet'); am['entry'] = pd.to_datetime(am.entry).astype('datetime64[ns]'); cols = ['C_15m', 'C_1h', 'C_4h', 'C_D', 'S_15m', 'S_1h', 'S_4h', 'S_D']
M['entry'] = M.entry.astype('datetime64[ns]'); M = M.merge(am[['ds', 'tk', 'entry'] + cols].drop_duplicates(['ds', 'tk', 'entry']), on=['ds', 'tk', 'entry'], how='left'); print('amd flags matched %.1f%%' % (M[cols[0]].notna().mean() * 100), flush=True); M[cols] = M[cols].fillna(0)
r = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/optvol_ratio_sp.csv', parse_dates=['date']); r['date'] = r.date.astype('datetime64[ns]'); RS = {t: g.sort_values('date').set_index('date').r.rolling(252, min_periods=126).median().shift(1) for t, g in r.groupby('tk')}
M['ratio'] = [RS[t].reindex([e.normalize()], method='ffill').values[0] if (d == 'A' and t in RS) else np.nan for t, e, d in zip(M.tk, M.entry, M.ds)]; M['rmask'] = M.ratio.notna().astype(float); M['ratio'] = M.ratio.fillna(0).clip(0, 10) / 3
et = entry.values.astype('int64'); srt = np.sort(et); M['b_all'] = np.log1p(np.searchsorted(srt, et, 'left') - np.searchsorted(srt, et - 3600 * 10**9, 'left'))
b_same = np.zeros(len(M))
for d in ('A', 'B'):
    m = (ds == d); s2 = np.sort(et[m]); b_same[m] = np.log1p(np.searchsorted(s2, et[m], 'left') - np.searchsorted(s2, et[m] - 3600 * 10**9, 'left'))
M['b_same'] = b_same; M['small'] = (ds == 'B').astype(float); M['lpx'] = np.log(np.clip(px, .05, 2000)) / 5
SC = M[cols + ['small', 'lpx', 'ratio', 'rmask', 'b_all', 'b_same']].values.astype(np.float32); SC[:, -2:] /= 3
tr = (entry >= '2019-01-01') & (entry < '2022-01-01'); isA = ds == 'A'; isB = ds == 'B'; print('windows', X.shape, 'train', tr.sum(), '(A %d, B %d)' % ((tr & isA).sum(), (tr & isB).sum()), flush=True)
lo = {}; mu = {}; sd = {}; yz = np.zeros(len(ret), np.float32)
for d, m in (('A', isA), ('B', isB)):
    l, h = np.percentile(ret[tr & m], [1, 99]); y = np.clip(ret[m], l, h); mu[d], sd[d] = y[tr[m]].mean(), y[tr[m]].std(); yz[m] = ((y - mu[d]) / sd[d]).astype(np.float32)
fm, fs = X[tr].reshape(-1, 9).mean(0), X[tr].reshape(-1, 9).std(0) + 1e-6; Xn = ((X - fm) / fs).astype(np.float32); Xt = torch.from_numpy(Xn); St = torch.from_numpy(SC)
class Net(nn.Module):
    def __init__(s, kind):
        super().__init__(); s.kind = kind
        s.c1 = nn.Sequential(nn.Conv1d(9, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 48, 5, padding=2, dilation=1), nn.GELU(), nn.MaxPool1d(2), nn.Conv1d(48, 64, 5, padding=2), nn.GELU()) if kind == 'cnn' else nn.Sequential(nn.Conv1d(9, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, padding=2), nn.GELU(), nn.MaxPool1d(2))
        if kind == 'lstm': s.l = nn.LSTM(32, 32, batch_first=True, bidirectional=True); fd = 128
        else: fd = 128
        s.sc = nn.Sequential(nn.Linear(SC.shape[1], 16), nn.GELU()); s.h = nn.Sequential(nn.Dropout(.3), nn.Linear(fd + 16, 48), nn.GELU(), nn.Dropout(.3), nn.Linear(48, 2))
    def forward(s, x, sc):
        h = s.c1(x.transpose(1, 2))
        if s.kind == 'lstm': o, _ = s.l(h.transpose(1, 2)); z = torch.cat([o.mean(1), o[:, -1]], 1)
        else: z = torch.cat([h.mean(2), h.amax(2)], 1)
        return s.h(torch.cat([z, s.sc(sc)], 1))
def fit_nn(kind, ytr, seed):
    torch.manual_seed(seed); np.random.seed(seed); idx = np.nonzero(tr)[0]; cut = int(len(idx) * .85); ti, vi = idx[:cut], idx[cut:]; net = Net(kind); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2); best, bs, pat = -9, None, 0; yr = torch.from_numpy(ytr); wn = torch.from_numpy((ytr > 0).astype(np.float32))
    for ep in range(25):
        net.train(); p = np.random.permutation(ti)
        for i in range(0, len(p), 256):
            b = p[i:i + 256]; out = net(Xt[b], St[b]); loss = nn.functional.huber_loss(out[:, 0], yr[b]) + nn.functional.binary_cross_entropy_with_logits(out[:, 1], wn[b]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): sv = net(Xt[vi], St[vi])[:, 0].numpy()
        ic = spearmanr(sv, ytr[vi]).correlation
        if ic > best: best, bs, pat = ic, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 4: break
    net.load_state_dict(bs); net.eval()
    with torch.no_grad(): s = np.concatenate([net(Xt[i:i + 4096], St[i:i + 4096])[:, 0].numpy() for i in range(0, len(Xt), 4096)])
    print('  %s seed %d: epochs %d, best val IC %+.3f' % (kind, seed, ep + 1, best), flush=True); return s
def summ():
    return np.concatenate([X[:, -1, :], X.mean(1), X.min(1), X.max(1), X.std(1), SC], 1)
FE = summ()
def fit_gb(ytr, seed=0):
    idx = np.nonzero(tr)[0]; cut = int(len(idx) * .85); ti, vi = idx[:cut], idx[cut:]; g = HistGradientBoostingRegressor(max_depth=3, learning_rate=.03, max_iter=300, l2_regularization=5., min_samples_leaf=200, early_stopping=False, random_state=seed).fit(FE[ti], ytr[ti])
    ics = [spearmanr(p, ytr[vi]).correlation for p in g.staged_predict(FE[vi])]; nb = int(np.argmax(ics)) + 1; print('  gb: best iters %d, val IC %+.3f' % (nb, max(ics)), flush=True)
    g2 = HistGradientBoostingRegressor(max_depth=3, learning_rate=.03, max_iter=nb, l2_regularization=5., min_samples_leaf=200, early_stopping=False, random_state=seed).fit(FE[ti], ytr[ti]); return g2.predict(FE)
rng = np.random.default_rng(0); yperm = yz.copy(); yperm[tr] = rng.permutation(yz[tr])
P = {}; P['M1 CNN'] = np.mean([fit_nn('cnn', yz, s_) for s_ in (1, 2, 3)], 0); P['M2 CNN-BiLSTM'] = np.mean([fit_nn('lstm', yz, s_) for s_ in (1, 2, 3)], 0); P['M3 GBM'] = fit_gb(yz); P['CTRL M1'] = np.mean([fit_nn('cnn', yperm, s_) for s_ in (11, 12)], 0); P['CTRL M3'] = fit_gb(yperm, 1)
D = pd.DataFrame({'ds': ds, 'tk': tk, 'entry': entry, 'ret': ret, 'px': px, 'ratio': M.ratio.values * 3, 'rmask': M.rmask.values}); D['wk'] = D.entry.dt.to_period('W').astype(str)
for k, v in P.items(): D[k] = v
D.to_parquet(SP + 'scores399.parquet'); print('scores saved', flush=True)
def ev(d, col, cuts, lab):
    ics = [spearmanr(g[col], g.ret).correlation for _, g in d.groupby('wk') if len(g) >= 20]; ics = np.array([i for i in ics if np.isfinite(i)]); ic = spearmanr(d[col], d.ret).correlation; tt = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 3 else np.nan
    z = d[(d[col] <= cuts[0]) | (d[col] >= cuts[1])].copy(); z['top'] = (z[col] >= cuts[1]).astype(int)
    if z.top.nunique() < 2 or len(z) < 40: print('    %-24s n %6d | IC %+.3f | n/a' % (lab, len(d), ic)); return
    m = smf.ols('ret ~ top', z).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(z.wk)[0]}); print('    %-24s n %6d | IC %+.3f (wk t %+.1f) | top %+.2f%% bot %+.2f%% diff %+.2fpp t %+.1f | AUC %.3f' % (lab, len(d), ic, tt, z[z.top == 1].ret.mean() * 100, z[z.top == 0].ret.mean() * 100, m.params['top'] * 100, m.tvalues['top'], roc_auc_score(d.ret > 0, d[col])), flush=True)
sets = (('A 2018 (held out)', (D.ds == 'A') & (D.entry < '2019-01-01')), ('A 2022-26', (D.ds == 'A') & (D.entry >= '2022-01-01')), ('A 2022-26 gated ratio>=1.5', (D.ds == 'A') & (D.entry >= '2022-01-01') & (D.ratio >= 1.5)), ('B 2022-26 all', (D.ds == 'B') & (D.entry >= '2022-01-01')), ('B 2022-26 px<$5 (book)', (D.ds == 'B') & (D.entry >= '2022-01-01') & (D.px < 5)))
for nm, m in sets:
    print('\n== ' + nm)
    for col in P:
        book = 'A' if nm.startswith('A') else 'B'; trm = (D.ds == book) & tr; cuts = np.percentile(D[trm][col], [100 / 3, 200 / 3]); ev(D[m], col, cuts, col)
