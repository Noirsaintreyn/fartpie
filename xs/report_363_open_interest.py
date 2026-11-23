"""Nov 22 2026 report 363. Rey: his raw ThetaData DID include open interest (kind=='oi' rows; theta_compact.py had dropped them), so test the OI / 'insider-opening' ideas that report 362 could not.
Data: xs/oi_compact.py -> xs/ext/oi_daily/<TK>.csv (per trade date T: call/put EOD volume, call/put OI before, positive and net OI change over T, contracts with 0-60 DTE).  OI is published ~06:30 and reflects the PRIOR close, so the change over trade date T is known before the open of T+1; a flip on day D uses T = the last trading day before D.  Own-history z-scores over the trailing 126 sessions.
Declared before running (7 hypotheses; same split, train 2019-21 / test 2022-26, terciles cut on train, week-clustered t, 38 option names; same flips as report 362):
 O1 CALL TURNOVER (call volume / call OI): higher -> BETTER (speculation relative to existing positions).
 O2 CALL OI BUILD (net change in call OI over the day / call OI): higher -> BETTER (new bullish positions; Pan-Poteshman open-buy).
 O3 PUT OI BUILD (net change in put OI / put OI): higher -> WORSE (opening puts = informed bearish flow or hedging).
 O4 CALL OPENING SHARE (sum of positive per-contract call OI changes / call volume): higher -> BETTER (volume that opens rather than closes).
 O5 OI CALL/PUT (log call OI / put OI): higher -> BETTER (positioning).
 O6 NET BULLISH BUILD (call OI build minus put OI build): higher -> BETTER.
 O7 PUT TURNOVER (put volume / put OI): higher -> WORSE.
Pass rule: same sign in both halves in the declared direction and |t| >= 2 in at least one half (week-clustered); with 7 hypotheses a single |t| >= 2.7 is the solid bar.  Report 362 (volume-only versions) found 0 of 9, so the prior is low."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
M = pd.read_parquet(SP + 'M362.parquet')[['tk', 'day', 'entry', 'ret', 'big', 'wk', 'half']].copy(); M['day'] = pd.to_datetime(M.day).astype('datetime64[ns]')
rows = []
for f in sorted(glob.glob(U + 'xs/ext/oi_daily/*.csv')):
    tk = os.path.basename(f)[:-4]; d = pd.read_csv(f, parse_dates=['date']).sort_values('date').set_index('date')
    d = d[(d.call_oi > 0) & (d.put_oi > 0) & (d.call_vol > 0)]
    q = pd.DataFrame({'o1': np.log(d.call_vol / d.call_oi), 'o2': d.c_dnet / d.call_oi, 'o3': d.p_dnet / d.put_oi, 'o4': d.c_dpos / d.call_vol, 'o5': np.log(d.call_oi / d.put_oi)})
    q['o6'] = q.o2 - q.o3; q['o7'] = np.log((d.put_vol + 1) / d.put_oi)
    for k in list(q.columns): s = q[k]; q[k + '_z'] = (s - s.rolling(126, min_periods=60).mean()) / s.rolling(126, min_periods=60).std()
    q = q[[c for c in q.columns if c.endswith('_z')]].shift(1); q['tk'] = tk; rows.append(q.reset_index())
F = pd.concat(rows); F['date'] = pd.to_datetime(F.date).astype('datetime64[ns]'); M = M.sort_values('day')
M = pd.merge_asof(M, F.sort_values('date'), left_on='day', right_on='date', by='tk', direction='backward', tolerance=pd.Timedelta(days=5))
print('names %d, flips %d, with OI features %d' % (F.tk.nunique(), len(M), M.o1_z.notna().sum()), flush=True)
H_ = [('O1 call turnover vol/OI', 'o1_z', 1), ('O2 call OI build', 'o2_z', 1), ('O3 put OI build', 'o3_z', -1), ('O4 call opening share', 'o4_z', 1), ('O5 OI call/put', 'o5_z', 1), ('O6 net bullish build', 'o6_z', 1), ('O7 put turnover', 'o7_z', -1)]
tr = M[M.half == 'train']
print('\n%-26s %-48s %-48s  pass' % ('', 'TRAIN top-bot mean (t) | >=5% top/bot', 'TEST top-bot mean (t) | >=5% top/bot'))
for lab, col, sgn in H_:
    lo, hi = tr[col].quantile([1 / 3, 2 / 3]); res = {}
    for h in ('train', 'test'):
        D = M[(M.half == h) & M[col].notna()]; T = D[D[col] >= hi]; Bo = D[D[col] <= lo]; Z = pd.concat([T.assign(x=1.0), Bo.assign(x=0.0)])
        mod = sm.OLS(Z.ret.values, sm.add_constant(Z.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Z.wk)[0]}); res[h] = (mod.params[1] * 100, mod.tvalues[1], T.big.mean() * 100, Bo.big.mean() * 100, len(T), len(Bo))
    s = lambda h: '%+.2f%% (t %+.1f) | %.1f/%.1f%% n%d/%d' % res[h]
    same = np.sign(res['train'][0]) == np.sign(res['test'][0]) == sgn; tmax = max(abs(res['train'][1]), abs(res['test'][1])); ok = same and tmax >= 2
    print('%-26s %-48s %-48s %s' % (lab, s('train'), s('test'), 'PASS' if ok else ('same sign, |t|<2' if same else 'fail')), flush=True)
