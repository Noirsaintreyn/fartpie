"""Nov 23 2026 report 392. Rey: 'decide if a flip is going to be a rejection or a break'.  Define the label from the vector's own level: REJ = within the first 4 entry bars (1 hour) any 15m close back BELOW the old bear level the flip just broke (the break did not hold); BRK otherwise.
Hypothesis (market-structure mechanism): a level that has been tested and rejected several times has supply sitting at it, so a break of a heavily-tested level is a stronger, more informative break (stops above it get taken) - OR the opposite: repeated tests exhaust the buyers.  Both directions are admissible; the sign must be the same in both halves.  Bar anatomy: a break bar that closes near its high with little upper wick shows acceptance beyond the level; a long upper wick shows rejection inside the bar itself.
Features known at the flip bar (strictly before entry):
 T1 prior tests of the broken level during the bear run = bars with high within 0.25 ATR of the level that closed at/below it;  T2 same, last 30 bars only;  A1 close location in the flip bar (close-low)/(high-low);  A2 body in ATR (close-open)/ATR;  A3 upper wick fraction (high-close)/(high-low);  A4 overshoot already in report 388 (G2: not admitted).
Reported: label base rate by half, mean flip return by label, then for each feature top-vs-bottom third (cuts on train 2019-21): rejection-rate diff and mean flip return diff with week-clustered t; admit = same sign both halves, |t|>=2 both (on return)."""
import sys, glob, os, numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
from vector_pine import run_state_machine
import report_295_flip_quality as q
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet'); b['entry'] = pd.to_datetime(b.entry); b = b[b.entry >= '2019-01-01'].reset_index(drop=True)
rows = []; fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob(U + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
for tk in sorted(b.tk.unique()):
    H = q.build(fs[tk]); r_, l_, a_, fl = run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt)); r_ = np.asarray(r_); l_ = np.asarray(l_); a_ = np.asarray(a_)
    o, hi, lo, c = H.open.values, H.high.values, H.low.values, H.close.values; pos = {t: i for i, t in enumerate(H.dt.values)}
    for _, e in b[b.tk == tk].iterrows():
        j = pos.get(np.datetime64(e.entry))
        if j is None or j + 4 >= len(H): continue
        k = j - 1; n = 0
        while k > 1 and not (r_[k] == 1 and r_[k - 1] != 1):
            k -= 1; n += 1
            if n > 40: k = None; break
        if k is None or k < 40: continue
        old = l_[k - 1]; a = a_[k]
        if not a > 0 or np.isnan(old): continue
        s = k - 1
        while s > 0 and r_[s] == -1: s -= 1
        idx = np.arange(s + 1, k); tests = (hi[idx] >= old - .25 * a) & (c[idx] <= old); rng = hi[k] - lo[k]
        rej = bool((c[j:j + 4] < old).any())
        rows.append(dict(tk=tk, entry=e.entry, ret=e.ret, REJ=int(rej), T1=int(tests.sum()), T2=int(tests[-30:].sum()), A1=(c[k] - lo[k]) / rng if rng > 0 else np.nan, A2=(c[k] - o[k]) / a, A3=(hi[k] - c[k]) / rng if rng > 0 else np.nan))
    print(tk, len(rows), flush=True)
F = pd.DataFrame(rows); F['tr'] = F.entry < '2022-01-01'; F['wk'] = F.entry.dt.to_period('W').astype(str); F.to_parquet(SP + 'rej392.parquet'); print('matched', len(F), 'of', len(b))
print('\nLABEL: rejection rate train %.1f%% test %.1f%%; mean flip return REJ vs BRK: train %+.2f%% vs %+.2f%% | test %+.2f%% vs %+.2f%%' % (F[F.tr].REJ.mean() * 100, F[~F.tr].REJ.mean() * 100, F[F.tr & (F.REJ == 1)].ret.mean() * 100, F[F.tr & (F.REJ == 0)].ret.mean() * 100, F[~F.tr & (F.REJ == 1)].ret.mean() * 100, F[~F.tr & (F.REJ == 0)].ret.mean() * 100))
def test(col, lab):
    d = F.dropna(subset=[col]).copy(); q1, q2 = d[d.tr][col].quantile([1 / 3, 2 / 3])
    if q1 == q2: q1, q2 = q1 - 1e-9, q2
    d['T'] = np.where(d[col] <= q1, 'lo', np.where(d[col] >= q2, 'hi', 'mid')); out = []
    for h, D in (('train', d[d.tr]), ('test', d[~d.tr])):
        D = D[D['T'].isin(['hi', 'lo'])].copy(); D['top'] = (D['T'] == 'hi').astype(int); m = smf.ols('ret ~ top', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.wk)[0]})
        out.append('%s: REJ hi %.0f%% lo %.0f%% | ret diff %+.2fpp t %+.1f (n %d)' % (h, D[D.top == 1].REJ.mean() * 100, D[D.top == 0].REJ.mean() * 100, m.params['top'] * 100, m.tvalues['top'], len(D)))
    print('%-34s %s | %s' % (lab, out[0], out[1]), flush=True)
for c, l in (('T1', 'T1 prior tests of level (bear run)'), ('T2', 'T2 prior tests, last 30 bars'), ('A1', 'A1 close location in flip bar'), ('A2', 'A2 flip-bar body / ATR'), ('A3', 'A3 upper-wick fraction')): test(c, l)
