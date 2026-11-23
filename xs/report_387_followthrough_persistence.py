"""Nov 23 2026 report 387. Rey: 'all my confirmations are arbitrary models of follow-through; think outside the box'.  Hypothesis: follow-through is persistent in the vector's OWN track record (no external feature).  Mechanism: the ratchet works when the name/tape is trending (flips resolve cleanly); in chop flips fail repeatedly, and that regime persists for days.
Features (strictly past: only flips whose EXIT is before this flip's entry): (F1) the same name's mean return over its last 3 completed flips; (F2) the same name's last-1 flip win; (F3) the cross-section's mean return over the last 30 completed flips across all 54 names; (F4) cross-section win rate of the last 30.
Test on b302 (12,360 flips): terciles cut on train (2019-21), top-minus-bottom mean flip return with week-clustered OLS t, train | test; admission: same sign in both halves and |t| >= 2 both."""
import numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore')
b = pd.read_parquet('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/b302.parquet'); b['entry'] = pd.to_datetime(b.entry); b['exit'] = pd.to_datetime(b.exit); b = b.sort_values('entry').reset_index(drop=True)
ex = b.sort_values('exit')[['exit', 'ret', 'tk']].reset_index(drop=True); exv = ex.exit.values; exr = ex.ret.values
# cross-section trailing: flips with exit < entry, last 30 by exit time
idx = np.searchsorted(exv, b.entry.values, side='left'); F3 = np.full(len(b), np.nan); F4 = F3.copy()
for i, k in enumerate(idx):
    if k >= 30: w = exr[k - 30:k]; F3[i] = w.mean(); F4[i] = (w > 0).mean()
b['F3'] = F3; b['F4'] = F4
f1 = np.full(len(b), np.nan); f2 = f1.copy()
for tk, g in b.groupby('tk'):
    g = g.sort_values('entry'); e = g.entry.values; x = g.exit.values; r = g.ret.values
    for j, i in enumerate(g.index):
        prior = [r[m] for m in range(j) if x[m] < e[j]]
        if len(prior) >= 3: f1[i] = np.mean(prior[-3:])
        if len(prior) >= 1: f2[i] = float(prior[-1] > 0)
b['F1'] = f1; b['F2'] = f2; b['tr'] = b.entry < '2022-01-01'; b['wk'] = b.entry.dt.to_period('W').astype(str)
def test(col, lab):
    d = b.dropna(subset=[col]).copy()
    if d[col].nunique() <= 2: d['T'] = np.where(d[col] > 0, 'hi', 'lo'); hi, lo = 'hi', 'lo'
    else:
        q1, q2 = d[d.tr][col].quantile([1 / 3, 2 / 3]); d['T'] = np.where(d[col] <= q1, 'lo', np.where(d[col] >= q2, 'hi', 'mid')); hi, lo = 'hi', 'lo'
    out = []
    for h, D in (('train', d[d.tr]), ('test', d[~d.tr])):
        D = D[D['T'].isin([hi, lo])].copy(); D['top'] = (D['T'] == hi).astype(int); m = smf.ols('ret ~ top', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.wk)[0]}); out.append('%s: hi %+.2f%% lo %+.2f%% diff %+.2fpp t %+.1f (n %d)' % (h, D[D.top == 1].ret.mean() * 100, D[D.top == 0].ret.mean() * 100, m.params['top'] * 100, m.tvalues['top'], len(D)))
    print('%-42s %s | %s' % (lab, out[0], out[1]), flush=True)
test('F1', 'F1 same-name last-3 mean return'); test('F2', 'F2 same-name last flip win'); test('F3', 'F3 cross-section last-30 mean return'); test('F4', 'F4 cross-section last-30 win rate')
