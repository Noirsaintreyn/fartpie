"""Nov 22 2026 report 360. Brute force the REAL objective: every one of the 114 pre-entry features of report 359 (b_mkt dropped: same-day market return) used as the pick order of the 6-slot book, HIGH-first and LOW-first (228 books), vs a NULL of 200 random pick orders.
Report 359 scored features by rank IC with flip return; that over-weights hit rate (low-vol names win more often) while the book is a heavy-tail mean (high-vol names make the big winners), so only the book decides.
Declared pass rule: beats the live order's CAGR in BOTH halves (train 33.8 / test 17.5), full-sample CAGR above the 95th percentile of the random-order null, and drawdown no worse than live.  228 trials -> ~11 are expected above a 95th-percentile line by luck, so a single hit is not evidence; a CLUSTER of related features in the same direction, or a hit that survives a +-1 neighbour feature, is.
Output: top 15 books by Sharpe with halves, the null quantiles, and the passers."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
from multiprocessing import Pool
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
X = pd.read_parquet(SP + 'X359.parquet'); b = pd.read_parquet(SP + 'b302.parquet').reset_index(drop=True); b['entry'] = pd.to_datetime(b.entry); X['tk'] = b.tk.values; X['entry'] = b.entry.values
X = X.rename(columns={c: 'f_' + c for c in X.columns if c not in ('ret', 'tr', 'tk', 'entry')})
feats = [c for c in X.columns if c not in ('ret', 'tr', 'tk', 'entry', 'f_b_mkt')]
B0 = B0.merge(X.drop(columns=['ret', 'tr']), on=['tk', 'entry'], how='left')
def cagr_of(TK):
    z = spread(TK); z = z[z.index >= '2019-01-01']; a = z[z.index < '2022-01-01']; c = z[z.index >= '2022-01-01']; return cagr(z), mdd(z), sharpe(z.values), cagr(a), cagr(c)
def one(arg):
    col, hi, seed = arg
    global order
    def order(Bs):
        if col is None: k = np.random.default_rng(seed).random(len(Bs))
        else:
            v = Bs[col].fillna(B0[col].median()).values; k = -v if hi else v
        return Bs.assign(_k=k).sort_values(['entry', '_k', 'tk'], kind='stable')
    TK, _ = sim_swap(B0, None); return (col, hi, seed) + cagr_of(TK)
if __name__ == '__main__':
    def liveorder(Bs): return Bs.assign(_k=Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values).sort_values(['entry', '_k', 'tk'], kind='stable')
    order = liveorder; TK, _ = sim_swap(B0, None); L = cagr_of(TK); print('LIVE order: CAGR %.1f%% DD %.1f%% Sharpe %.2f train %.1f test %.1f' % (L[0] * 100, L[1] * 100, L[2], L[3] * 100, L[4] * 100), flush=True)
    jobs = [(c, h, 0) for c in feats for h in (True, False)] + [(None, True, s) for s in range(200)]
    with Pool(max(1, 2)) as p: res = p.map(one, jobs, chunksize=4)
    R = pd.DataFrame(res, columns=['col', 'hi', 'seed', 'cagr', 'dd', 'sh', 'tr', 'te']); N = R[R.col.isna()]; F = R[R.col.notna()].copy()
    print('NULL (200 random orders): CAGR mean %.1f%% p5 %.1f%% p95 %.1f%% | Sharpe p95 %.2f | train p95 %.1f%% test p95 %.1f%%' % (N.cagr.mean() * 100, N.cagr.quantile(.05) * 100, N.cagr.quantile(.95) * 100, N.sh.quantile(.95), N.tr.quantile(.95) * 100, N.te.quantile(.95) * 100))
    F['pass'] = (F.tr > L[3]) & (F.te > L[4]) & (F.cagr > N.cagr.quantile(.95)) & (F.dd >= L[1] - .005)
    print('\nTOP 20 by Sharpe:'); print(F.sort_values('sh', ascending=False).head(20).assign(cagr=lambda d: d.cagr * 100, dd=lambda d: d.dd * 100, tr=lambda d: d.tr * 100, te=lambda d: d.te * 100).round(2).drop(columns='seed').to_string(index=False))
    print('\nBeat live in BOTH halves:', int(((F.tr > L[3]) & (F.te > L[4])).sum()), 'of', len(F), '| PASS (+null p95, DD ok):', F[F['pass']][['col', 'hi']].values.tolist())
    F.to_csv(SP + 'tab360.csv', index=False)
