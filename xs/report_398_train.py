"""Nov 23 2026 report 398 (part 2): CNN-BiLSTM on the flip windows (report_398_build.py).  PRE-REGISTERED before any result:
Target: flip return (winsorised 1/99% on train, standardised; Huber) + win/loss (BCE).  Train 2019-01..2021-12 ONLY; the last 15% of train by time is the early-stopping validation (patience 4 on validation Spearman IC); test = 2018 (never seen) and 2022-26 (never seen).  Architecture fixed: Conv1d(7,32,5)-GELU-Conv1d(32,32,5)-GELU-MaxPool2 -> BiLSTM(32) -> [mean,last] pooling -> dropout 0.3 -> Linear(2); AdamW lr 2e-3 wd 1e-2, batch 256, <= 25 epochs; 3 seeds averaged.  No tuning after seeing test.  CONTROL: identical recipe with the train labels randomly permuted (2 seeds) - shows what pure noise scores.
Score = standardised regression output.  Tests per test period (ungated flips, and the live-gated subset ratio >= 1.5): Spearman IC of score vs flip return (mean of weekly cross-section ICs, t of the weekly mean), top-minus-bottom tercile flip-return diff (cuts on TRAIN scores, week-clustered OLS t), AUC for win.
Declared pass: IC > 0 and tercile diff > 0 with |t| >= 2 in BOTH 2018 and 2022-26, and clearly above the control."""
import numpy as np, pandas as pd, torch, torch.nn as nn, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; torch.set_num_threads(2)
Z = np.load(SP + 'win398.npz', allow_pickle=True); X = Z['X']; ret = Z['ret'].astype(float); entry = pd.to_datetime(Z['entry']); tk = Z['tk']
tr = (entry >= '2019-01-01') & (entry < '2022-01-01'); t18 = entry < '2019-01-01'; t22 = entry >= '2022-01-01'; print('windows', X.shape, 'train', tr.sum(), 'test2018', t18.sum(), 'test2022-26', t22.sum(), flush=True)
lo, hi = np.percentile(ret[tr], [1, 99]); y = np.clip(ret, lo, hi); mu, sd = y[tr].mean(), y[tr].std(); yz = ((y - mu) / sd).astype(np.float32); win = (ret > 0).astype(np.float32)
fm, fs = X[tr].reshape(-1, 7).mean(0), X[tr].reshape(-1, 7).std(0) + 1e-6; Xn = ((X - fm) / fs).astype(np.float32)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.c = nn.Sequential(nn.Conv1d(7, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, padding=2), nn.GELU(), nn.MaxPool1d(2)); s.l = nn.LSTM(32, 32, batch_first=True, bidirectional=True); s.d = nn.Dropout(.3); s.o = nn.Linear(128, 2)
    def forward(s, x):
        h = s.c(x.transpose(1, 2)).transpose(1, 2); o, _ = s.l(h); return s.o(s.d(torch.cat([o.mean(1), o[:, -1]], 1)))
def fit(ytr, seed):
    torch.manual_seed(seed); np.random.seed(seed); idx = np.nonzero(tr)[0]; cut = int(len(idx) * .85); ti, vi = idx[:cut], idx[cut:]; net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2); best, bs, pat = -9, None, 0; Xt = torch.from_numpy(Xn)
    yr = torch.from_numpy(ytr); wn = torch.from_numpy((ytr * sd + mu > 0).astype(np.float32))
    for ep in range(25):
        net.train(); p = np.random.permutation(ti)
        for i in range(0, len(p), 256):
            b = p[i:i + 256]; out = net(Xt[b]); loss = nn.functional.huber_loss(out[:, 0], yr[b]) + nn.functional.binary_cross_entropy_with_logits(out[:, 1], wn[b]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): sv = net(Xt[vi])[:, 0].numpy()
        ic = spearmanr(sv, ytr[vi]).correlation
        if ic > best: best, bs, pat = ic, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 4: break
    net.load_state_dict(bs); net.eval()
    with torch.no_grad(): s = net(Xt)[:, 0].numpy()
    print('  seed %d: epochs %d, best val IC %+.3f' % (seed, ep + 1, best), flush=True); return s
S = np.mean([fit(yz, sd_) for sd_ in (1, 2, 3)], 0); print('real model trained', flush=True)
yperm = yz.copy(); rng = np.random.default_rng(0); yperm[tr] = rng.permutation(yz[tr]); C = np.mean([fit(yperm, sd_) for sd_ in (11, 12)], 0); print('control trained', flush=True)
r = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/optvol_ratio_sp.csv', parse_dates=['date']); r['date'] = r.date.astype('datetime64[ns]'); RS = {t: g.sort_values('date').set_index('date').r.rolling(252, min_periods=126).median().shift(1) for t, g in r.groupby('tk')}
D = pd.DataFrame({'tk': tk, 'entry': entry, 'ret': ret, 'S': S, 'C': C}); D['ratio'] = [RS[t].reindex([e.normalize()], method='ffill').values[0] if t in RS else np.nan for t, e in zip(D.tk, D.entry)]; D['wk'] = D.entry.dt.to_period('W').astype(str); D['split'] = np.where(tr, 'train', np.where(t18, '2018', np.where(t22, '2022-26', 'x'))); D.to_parquet(SP + 'scores398.parquet')
q1, q2 = np.percentile(D[tr].S, [100 / 3, 200 / 3]); c1, c2 = np.percentile(D[tr].C, [100 / 3, 200 / 3])
def ev(d, col, a, b, lab):
    ics = []
    for _, g in d.groupby('wk'):
        if len(g) >= 20: ics.append(spearmanr(g[col], g.ret).correlation)
    ics = np.array([i for i in ics if np.isfinite(i)]); ic = spearmanr(d[col], d.ret).correlation; t_ic = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 3 else np.nan
    z = d[(d[col] <= a) | (d[col] >= b)].copy(); z['top'] = (z[col] >= b).astype(int); m = smf.ols('ret ~ top', z).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(z.wk)[0]}); auc = roc_auc_score(d.ret > 0, d[col])
    print('%-34s n %6d | IC %+.3f (weekly IC t %+.1f) | top %+.2f%% bot %+.2f%% diff %+.2fpp t %+.1f | AUC %.3f' % (lab, len(d), ic, t_ic, z[z.top == 1].ret.mean() * 100, z[z.top == 0].ret.mean() * 100, m.params['top'] * 100, m.tvalues['top'], auc), flush=True)
for nm, d in (('TRAIN (in-sample, for reference)', D[D.split == 'train']), ('2018 held-out', D[D.split == '2018']), ('2022-26 test', D[D.split == '2022-26'])):
    print('\n== ' + nm)
    for g, gd in (('all flips', d), ('gated ratio>=1.5', d[d.ratio >= 1.5])):
        ev(gd, 'S', q1, q2, 'MODEL ' + g); ev(gd, 'C', c1, c2, 'CONTROL(shuffled) ' + g)
