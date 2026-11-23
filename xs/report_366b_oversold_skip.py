"""Report 366b (post-hoc, flagged): report 366 found bull flips taken after a 5-day drop of more than 2 sigma (oversold, z on the prior close) did WORSE in both halves, opposite to the declared direction (z_yz train -0.97% t -2.2 | test -0.39% t -2.5; z_reg -0.65 | -0.41 t -3.1 test).  Mechanism: the flip is a bounce inside a downtrend (falling knife).  Test as a SKIP rule in the 6-slot book (live order otherwise): CAGR / DD / Sharpe, train | test, plus placebo of skipping the same number of random flips (200 draws).  Three variants looked at: a forward-log hypothesis, not an admission."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; A = pd.read_parquet(SP + 'Z366.parquet'); A['date'] = A.date.astype('datetime64[ns]')
B0['day'] = pd.to_datetime(B0.day).astype('datetime64[ns]'); B0['pd_'] = B0.day - pd.Timedelta(days=1); B0 = B0.sort_values('pd_')
B0 = pd.merge_asof(B0, A.sort_values('date')[['tk', 'date', 'z_reg', 'z_yz', 'z_har']], left_on='pd_', right_on='date', by='tk', direction='backward', tolerance=pd.Timedelta(days=5)).sort_values('entry').reset_index(drop=True)
def live(Bs): return Bs.assign(_k=Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values).sort_values(['entry', '_k', 'tk'], kind='stable')
order = live
def run(Bx):
    TK, _ = sim_swap(Bx, None); z = spread(TK); z = z[z.index >= '2019-01-01']; return cagr(z), mdd(z), sharpe(z.values), cagr(z[z.index < '2022-01-01']), cagr(z[z.index >= '2022-01-01']), len(TK)
def show(lab, r): print('%-26s CAGR %.1f%% DD %.1f%% Sharpe %.2f | train %.1f test %.1f | %d trades' % (lab, r[0] * 100, r[1] * 100, r[2], r[3] * 100, r[4] * 100, r[5]), flush=True)
show('LIVE', run(B0))
for z in ('z_reg', 'z_yz', 'z_har'):
    sk = (B0[z] < -2).fillna(False); show('skip %s < -2 (%d flips)' % (z, sk.sum()), run(B0[~sk].reset_index(drop=True)))
sk = ((B0.z_reg < -2) | (B0.z_yz < -2) | (B0.z_har < -2)).fillna(False); show('skip any of the three (%d)' % sk.sum(), run(B0[~sk].reset_index(drop=True)))
rng = np.random.default_rng(3); n = int((B0.z_yz < -2).sum()); res = []
for _ in range(60): m = np.ones(len(B0), bool); m[rng.choice(len(B0), n, replace=False)] = False; r = run(B0[m].reset_index(drop=True)); res.append((r[0], r[2], r[3], r[4]))
res = np.array(res); print('placebo skip %d random flips (60 draws): CAGR mean %.1f%% p95 %.1f%% | Sharpe mean %.2f p95 %.2f | test CAGR mean %.1f%% p95 %.1f%%' % (n, res[:, 0].mean() * 100, np.percentile(res[:, 0], 95) * 100, res[:, 1].mean(), np.percentile(res[:, 1], 95), res[:, 3].mean() * 100, np.percentile(res[:, 3], 95) * 100))
