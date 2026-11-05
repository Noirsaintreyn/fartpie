"""Nov 4 2026: the prop lifecycle (nq_prop_lifecycle.py, same rules) with the VOLATILITY / REGIME models and the VOLUME PROFILE in the sizing. DECLARED BEFORE RESULTS.
Flags per trade, all known before entry (labels from nq_flip_types.py / nq_vp_nodes.py):
  THIN  60-day volume profile thin on the path to the 75-pt target (bottom third, cut-offs from 2018-21)
  BEAR  2-state HMM on QQQ daily returns in the bear state at the prior close (fit 2016-21, forward-filtered)
  NOISY wavelet low-frequency share of the last 64 5m returns in its bottom third (noisy tape)
  HARH  HAR volatility forecast in its top third (1-year percentile, prior close)
  ZUP   QQQ 20d z-score >= 0 (the flip did better when QQQ was below its 20d mean)
Sizing = contracts as a fraction of 1 NQ (10 MNQ); eval stays 1 NQ unless the name says 'all stages'.
  B   funded: 0.5 on THIN (the earlier pick)          V1 funded: 0.5 on BEAR or NOISY (regime models only)      V2 funded: 0.5 on HARH (vol model only)
  V3  funded: 0.5 on THIN or BEAR or NOISY             V4 funded: 1 with no flag, 0.5 with one, 0.3 with two or more of THIN / BEAR / NOISY / HARH
  V5  V4 in the eval AND funded (all stages)           V6 funded: V4 flags + ZUP counted as a fifth flag"""
_lc = open('xs/nq_prop_lifecycle.py').read(); exec(_lc[:_lc.index("rows = {")])
L = pd.read_parquet('/tmp/claude-0/ml/nq_flip_types_C.parquet')[['day', 'j', 'HMM', 'WAVE', 'HAR', 'QZ']]
X = C.loc[R.index, ['day', 'j', 'AHEAD_60D']].rename_axis('ix').reset_index().merge(L, on=['day', 'j'], how='left').set_index('ix')
print(f'trades {len(R)}, labels matched {X.HMM.notna().mean():.0%}', flush=True)
F = pd.DataFrame({'THIN': X.AHEAD_60D == 'LVN path ahead', 'BEAR': X.HMM == 'HMM bear', 'NOISY': X.WAVE == 'wavelet noisy', 'HARH': X.HAR == 'HAR high', 'ZUP': X.QZ == 'QQQ z20 >= 0'}, index=R.index)
post = R.day >= P0
for k in F: print(f'  {k:<5} on {F[k][post].mean():4.0%} of 2022-26 trades | avg pts flagged {R.net[post & F[k]].mean():+5.1f} vs not {R.net[post & ~F[k]].mean():+5.1f}', flush=True)
def dv(w): return (R.net * 20 * w).groupby(R.day).sum().reindex(D_).fillna(0.0).values
n4 = F[['THIN', 'BEAR', 'NOISY', 'HARH']].sum(1); n5 = n4 + F.ZUP
sz4 = np.select([n4 == 0, n4 == 1], [1.0, .5], .3); sz5 = np.select([n5 == 0, n5 == 1], [1.0, .5], .3)
rows = {'A  flat 1 NQ (no models)': summarize(dvA),
        'B  0.5 on THIN (volume profile)': summarize(dvA, dv(np.where(F.THIN, .5, 1.0))),
        'V1 0.5 on BEAR or NOISY (regime)': summarize(dvA, dv(np.where(F.BEAR | F.NOISY, .5, 1.0))),
        'V2 0.5 on HARH (vol model)': summarize(dvA, dv(np.where(F.HARH, .5, 1.0))),
        'V3 0.5 on THIN or BEAR or NOISY': summarize(dvA, dv(np.where(F.THIN | F.BEAR | F.NOISY, .5, 1.0))),
        'V4 1 / 0.5 / 0.3 by flag count (4 flags)': summarize(dvA, dv(sz4)),
        'V5 V4 in eval and funded': summarize(dv(sz4), dv(sz4)),
        'V6 V4 + ZUP as 5th flag': summarize(dvA, dv(sz5))}
exec(_lc[_lc.index("T = pd.DataFrame(rows).T"):])
