"""Nov 23 2026 report 393. Rey: the vector fires at pivots, i.e. it should trade the DISTRIBUTION leg of an AMD / Po3 sequence, not accumulation (compression, already tested: report 342 flat) and not manipulation; test variants of that phase logic on the live flips.
Mechanism (declared before the run): in Po3 the manipulation is a sweep of an obvious liquidity reference (prior-day low, the opening-range low, the prior week's low) that traps late sellers; the vector's bull flip that comes AFTER such a sweep is the confirmation that the sweep was a trap, so it should follow through better than a flip with no prior sweep.  A flip inside the first 30 minutes (the classic 'Judas' window) is still the manipulation phase and should be worse.  Would stop working: when sweeps are routine noise (every day takes some prior low) so the flag carries no information.
Variants (all known at the flip bar, RTH only; direction declared):
 P1 SWEEP_PDL: the session low so far is below the prior session's low (+ expected)
 P2 SWEEP_OR: after 10:00, a low below the opening-range low (first 30 min) has printed and the flip bar closes back above it (+)
 P3 range position at the flip = (close - session low)/(session high - session low), low third vs high third (low position = bought near the low after a manipulation, +)
 P4 FIRST30: the flip bar is in 09:30-10:00 (manipulation window, -)
 P5 WEEKLY: the week's low so far is below the prior week's low AND the flip closes above the week open (weekly Judas then distribution, +)
 P6 TEXTBOOK = (P1 or P2) and flip at/after 10:00 and not P4  (+)
Also descriptive: mean flip return by weekday and by flip hour.  Test: mean flip return yes-minus-no (P3: low vs high third, cuts on train), week-clustered OLS t, train 2019-21 | test 2022-26; admit = same sign both halves, |t|>=2 both."""
import sys, glob, os, numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
from vector_pine import run_state_machine
import report_295_flip_quality as q
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet'); b['entry'] = pd.to_datetime(b.entry); b = b[b.entry >= '2019-01-01'].reset_index(drop=True)
rows = []; fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob(U + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
for tk in sorted(b.tk.unique()):
    H = q.build(fs[tk]); r_ = np.asarray(run_state_machine(H[['open', 'high', 'low', 'close']].assign(datetime=H.dt))[0]); pos = {t: i for i, t in enumerate(H.dt.values)}
    R = H[H.m >= 570].copy(); R['wk'] = R.dt.dt.to_period('W').astype(str); g = R.groupby('day'); R['cc'] = g.cumcount(); R['dlow'] = g.low.cummin(); R['dhigh'] = g.high.cummax()
    dl = g.low.min(); R['pdl'] = R.day.map(dl.shift(1)); R['orl'] = g.low.transform(lambda s: s.iloc[:2].min()); R['swor'] = ((R.low < R.orl) & (R.cc >= 2)).groupby(R.day).cummax()
    gw = R.groupby('wk'); R['wlow'] = gw.low.cummin(); R['wopen'] = gw.open.transform('first'); wl = gw.low.min(); wkorder = R.wk.drop_duplicates().values; prev = pd.Series(wkorder[:-1], index=wkorder[1:]); R['pwl'] = R.wk.map(prev).map(wl)
    R['P1'] = (R.dlow < R.pdl).astype(float); R['P2'] = (R.swor & (R.cc >= 2) & (R.close > R.orl)).astype(float); R['P3'] = (R.close - R.dlow) / (R.dhigh - R.dlow).replace(0, np.nan); R['P4'] = (R.m < 600).astype(float)
    R['P5'] = ((R.wlow < R.pwl) & (R.close > R.wopen)).astype(float); R['hr'] = R.dt.dt.hour; R['dow'] = R.dt.dt.dayofweek; Rx = R.set_index('dt')
    for _, e in b[b.tk == tk].iterrows():
        j = pos.get(np.datetime64(e.entry))
        if j is None: continue
        k = j - 1; n = 0
        while k > 0 and not (r_[k] == 1 and r_[k - 1] != 1):
            k -= 1; n += 1
            if n > 40: k = None; break
        if k is None: continue
        t = H.dt.values[k]
        if t not in Rx.index: continue
        x = Rx.loc[t]; x = x.iloc[0] if isinstance(x, pd.DataFrame) else x
        rows.append(dict(tk=tk, entry=e.entry, ret=e.ret, P1=x.P1, P2=x.P2, P3=x.P3, P4=x.P4, P5=x.P5, hr=x.hr, dow=x.dow, cc=x.cc))
    print(tk, len(rows), flush=True)
F = pd.DataFrame(rows); F['P6'] = (((F.P1 == 1) | (F.P2 == 1)) & (F.cc >= 2) & (F.P4 == 0)).astype(float); F['tr'] = F.entry < '2022-01-01'; F['wk'] = F.entry.dt.to_period('W').astype(str); F.to_parquet(SP + 'po3_393.parquet'); print('rows', len(F), 'of', len(b))
def diff(d, lab, a, bb):
    out = []
    for h, D in (('train', d[d.tr]), ('test', d[~d.tr])):
        D = D[a(D) | bb(D)].copy(); D['top'] = a(D).astype(int)
        if D.top.nunique() < 2 or len(D) < 30: out.append(h + ': n/a'); continue
        m = smf.ols('ret ~ top', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.wk)[0]}); out.append('%s: yes %+.2f%% (n %d) no %+.2f%% (n %d) diff %+.2fpp t %+.1f' % (h, D[D.top == 1].ret.mean() * 100, D.top.sum(), D[D.top == 0].ret.mean() * 100, (1 - D.top).sum(), m.params['top'] * 100, m.tvalues['top']))
    print('%-34s %s | %s' % (lab, out[0], out[1]), flush=True)
print('\nyes = condition true / low position third, no = condition false / high third')
diff(F, 'P1 SWEEP_PDL (+)', lambda D: D.P1 == 1, lambda D: D.P1 == 0); diff(F[F.cc >= 2], 'P2 SWEEP_OR reclaimed (+)', lambda D: D.P2 == 1, lambda D: D.P2 == 0)
d3 = F.dropna(subset=['P3']).copy(); q1, q2 = d3[d3.tr].P3.quantile([1 / 3, 2 / 3]); diff(d3, 'P3 low third vs high third (+)', lambda D: D.P3 <= q1, lambda D: D.P3 >= q2)
diff(F, 'P4 FIRST30 vs later (-)', lambda D: D.P4 == 1, lambda D: D.P4 == 0); diff(F, 'P5 WEEKLY sweep+above open (+)', lambda D: D.P5 == 1, lambda D: D.P5 == 0); diff(F, 'P6 TEXTBOOK distribution (+)', lambda D: D.P6 == 1, lambda D: D.P6 == 0)
print('\nDescriptive mean flip return %% by weekday (train | test): ' + ' '.join('%s %+.2f|%+.2f' % (['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][i], F[F.tr & (F.dow == i)].ret.mean() * 100, F[~F.tr & (F.dow == i)].ret.mean() * 100) for i in range(5)))
print('by flip hour (train | test): ' + ' '.join('%d:%+.2f|%+.2f' % (h, F[F.tr & (F.hr == h)].ret.mean() * 100, F[~F.tr & (F.hr == h)].ret.mean() * 100) for h in sorted(F.hr.unique())))
print('shares of flips: P1 %.0f%%, P2 %.0f%% (of >=10:00), P4 %.0f%%, P5 %.0f%%, P6 %.0f%%' % (F.P1.mean() * 100, F[F.cc >= 2].P2.mean() * 100, F.P4.mean() * 100, F.P5.mean() * 100, F.P6.mean() * 100))
