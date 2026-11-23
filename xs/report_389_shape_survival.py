"""Nov 23 2026 report 389. Rey: 'look for more market shape instead of arbitrary unrelated values' (TDA / geometry of the vector's own structure).
Shape features at each live flip (b302, 12,360 flips), computed only from bars at or before the flip bar:
 S1 total persistence = sum of peak prominences (0-dim persistent homology of the close series; for a 1-D series a local maximum's persistence equals its topographic prominence) over the prior 60 bars / ATR  -> how much structured swing there is
 S2 efficiency = |net move over 60 bars| / total persistence  -> trending (1) vs looping/choppy (0) 'topology' of the window
 S3 persistent-peak count = number of peaks with prominence >= 1 ATR in the window
 S4 persistence-weighted skew = (sum persistence of troughs - sum persistence of peaks) / total  -> dump shape (deep troughs under shallow peaks) vs top shape
Survival (vector's own regime durations, per name, expanding, only runs that ended before the flip):
 V1 = stretch = prior bear-run length / median of the name's previous completed bear runs  -> how overextended the dump is relative to the name's own history
 V2 = expected bull duration = median of the name's previous completed bull runs (bars)
Test: tercile cuts on train 2019-21, top-minus-bottom mean flip return, week-clustered OLS t, train | test; admit = same sign both halves, |t|>=2 both."""
import sys, glob, os, numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
from scipy.signal import find_peaks, peak_prominences
from vector_pine import run_state_machine
import report_295_flip_quality as q
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet'); b['entry'] = pd.to_datetime(b.entry); b = b[b.entry >= '2019-01-01'].reset_index(drop=True)
def pers(x):
    pk, _ = find_peaks(x); tr, _ = find_peaks(-x)
    pp = peak_prominences(x, pk)[0] if len(pk) else np.array([]); tp = peak_prominences(-x, tr)[0] if len(tr) else np.array([]); return pp, tp
rows = []; fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob(U + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
for tk in sorted(b.tk.unique()):
    H = q.build(fs[tk]); r_, l_, a_, fl = run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt)); r_ = np.asarray(r_); a_ = np.asarray(a_); close = H.close.values; pos = {t: i for i, t in enumerate(H.dt.values)}
    # completed runs: (start, end, regime)
    runs = []; s = 0
    for i in range(1, len(r_)):
        if r_[i] != r_[i - 1]: runs.append((s, i - 1, r_[i - 1])); s = i
    run_end = np.array([x[1] for x in runs]); run_len = np.array([x[1] - x[0] + 1 for x in runs]); run_reg = np.array([x[2] for x in runs])
    for _, e in b[b.tk == tk].iterrows():
        j = pos.get(np.datetime64(e.entry))
        if j is None: continue
        k = j - 1; n = 0
        while k > 0 and not (r_[k] == 1 and r_[k - 1] != 1):
            k -= 1; n += 1
            if n > 40: k = None; break
        if k is None or k < 70: continue
        w = close[k - 59:k + 1]; pp, tp = pers(w); a = a_[k]; tot = pp.sum() + tp.sum()
        if tot <= 0 or not a > 0: continue
        done = run_end < k; bear = run_len[done & (run_reg == -1)]; bull = run_len[done & (run_reg == 1)]
        cur_bear = run_len[done & (run_reg == -1)][-1] if len(bear) else np.nan
        rows.append(dict(tk=tk, entry=e.entry, ret=e.ret, S1=tot / a, S2=abs(w[-1] - w[0]) / tot, S3=int((pp >= a).sum() + (tp >= a).sum()), S4=(tp.sum() - pp.sum()) / tot,
                         V1=(cur_bear / np.median(bear[:-1])) if len(bear) > 3 else np.nan, V2=np.median(bull) if len(bull) > 3 else np.nan))
    print(tk, len(rows), flush=True)
F = pd.DataFrame(rows); F['tr'] = F.entry < '2022-01-01'; F['wk'] = F.entry.dt.to_period('W').astype(str); F.to_parquet(SP + 'shape389.parquet'); print('matched', len(F), 'of', len(b))
def test(col, lab):
    d = F.dropna(subset=[col]).copy(); q1, q2 = d[d.tr][col].quantile([1 / 3, 2 / 3]); d['T'] = np.where(d[col] <= q1, 'lo', np.where(d[col] >= q2, 'hi', 'mid')); out = []
    for h, D in (('train', d[d.tr]), ('test', d[~d.tr])):
        D = D[D['T'].isin(['hi', 'lo'])].copy(); D['top'] = (D['T'] == 'hi').astype(int); m = smf.ols('ret ~ top', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.wk)[0]}); out.append('%s: hi %+.2f%% (n %d) lo %+.2f%% diff %+.2fpp t %+.1f' % (h, D[D.top == 1].ret.mean() * 100, D.top.sum(), D[D.top == 0].ret.mean() * 100, m.params['top'] * 100, m.tvalues['top']))
    print('%-36s %s | %s' % (lab, out[0], out[1]), flush=True)
print('\nhi = top third, lo = bottom third (cuts on train)')
for c, l in (('S1', 'S1 total persistence / ATR'), ('S2', 'S2 efficiency (trend vs loop)'), ('S3', 'S3 persistent peak count'), ('S4', 'S4 persistence skew (trough-heavy)'), ('V1', 'V1 bear-run stretch vs own history'), ('V2', 'V2 expected bull duration')): test(c, l)
