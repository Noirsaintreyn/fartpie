"""Nov 14 2026 COMMITMENT AFTER THE BREAK (declared before results). Mechanism: if a flip is large orders being worked, participation should PERSIST after the break (not just spike at it), price should be ACCEPTED above the level (VWAP held, shallow pullback), and progress should come easily. Universe/trades = live screener (xs/ext/wide_T.parquet, sp, ratio >= 1.5), 15m track (h1 False), trades still open 4 bars after entry. Features measured over the 4 bars from entry (known at bar 4): BV = volume at the break bar / prior-26-bar mean (the spike, the thing we had); PV = mean volume of the 4 bars / prior-26-bar mean (persistence); VW = share of the 4 bars closing above session VWAP; EFF = net move / sum of ranges; PB = max adverse excursion in prior-26-bar average ranges (shallow = commitment, sign flipped); UVS = up-bar volume share. Outcome = REMAINDER return from the close of bar 4 to the trade's exit (no look-ahead; 2 bp not deducted; compare like with like). State = expanding PIT tercile (min 300). Hypothesis fixed in advance: HIGH PV, VW, EFF, UVS and SHALLOW PB give the better remainder; BV (spike) does not add. PASS = high > low in BOTH eras (2019-21, 2022-26), pooled |t| >= 2."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
T0 = pd.read_parquet('xs/ext/wide_T.parquet'); T0 = T0[T0.sp & (T0.ratio >= 1.5) & (~T0.h1)].copy().reset_index(drop=True); T0['entry'] = pd.to_datetime(T0.entry); T0['exit'] = pd.to_datetime(T0.exit)
rows = []
for tk, g in T0.groupby('tk'):
    f = glob.glob(f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    idx = pd.DatetimeIndex(H.dt); o = H.open.values; h = H.high.values; l = H.low.values; c = H.close.values; v = H.volume.values.astype(float)
    day = idx.normalize(); pv_ = np.cumsum(np.where(np.r_[True, day[1:] != day[:-1]], 0, 0))  # placeholder
    tp = (h + l + c) / 3; cumpv = pd.Series(tp * v).groupby(day).cumsum().values; cumv = pd.Series(v).groupby(day).cumsum().values; vw = cumpv / np.maximum(cumv, 1)
    for ix, e, x, r in zip(g.index, g.entry, g.exit, g.ret):
        j = idx.searchsorted(e)
        if j < 30 or j + 4 >= len(idx) or idx[j] != e: continue
        if x < idx[j + 4] or idx[j + 3].normalize() != idx[j].normalize(): continue
        base = v[j - 27:j - 1].mean(); rng = (h[j - 27:j - 1] - l[j - 27:j - 1]).mean()
        if base <= 0 or rng <= 0: continue
        sl = slice(j, j + 4); en = o[j]; p4 = c[j + 3]; ex = en * (1 + r); rem = ex / p4 - 1
        bar_up = c[sl] > o[sl]
        rows.append(dict(ix=ix, t=e, rem=rem, BV=v[j - 1] / base, PV=v[sl].mean() / base, VW=(c[sl] > vw[sl]).mean(), EFF=(p4 - en) / max((h[sl] - l[sl]).sum(), 1e-9), PB=-(en - l[sl].min()) / rng, UVS=(v[sl] * bar_up).sum() / max(v[sl].sum(), 1)))
P = pd.DataFrame(rows).sort_values('t').reset_index(drop=True); print('trades', len(P))
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
for c_ in ('BV', 'PV', 'VW', 'EFF', 'PB', 'UVS'):
    v_ = P[c_].values; o_ = np.full(len(v_), np.nan)
    for i in range(300, len(v_)): q1, q2 = np.quantile(v_[:i], [1 / 3, 2 / 3]); o_[i] = 0 if v_[i] <= q1 else 2 if v_[i] >= q2 else 1
    P['T_' + c_] = o_
print('REMAINDER return after bar 4 (%), low | high | high-low t   (PB high = shallow pullback)')
for c_ in ('BV', 'PV', 'VW', 'EFF', 'PB', 'UVS'):
    line = f' {c_:4s}'
    for e, (a, b) in (('19-21', ('2019', '2022')), ('22-26', ('2022', '2027')), ('pooled', ('2019', '2027'))):
        s = P[(P.t >= a) & (P.t < b)]; lo = s[s['T_' + c_] == 0].rem; hi = s[s['T_' + c_] == 2].rem; line += f' | {e}: {lo.mean() * 100:+.2f} {hi.mean() * 100:+.2f} (n {len(lo)}/{len(hi)}) t {tt(hi, lo):+.1f}'
    print(line)
P['COMP'] = (P[['PV', 'VW', 'EFF', 'PB', 'UVS']].rank(pct=True)).mean(axis=1); v_ = P.COMP.values; o_ = np.full(len(v_), np.nan)
for i in range(300, len(v_)): q1, q2 = np.quantile(v_[:i], [1 / 3, 2 / 3]); o_[i] = 0 if v_[i] <= q1 else 2 if v_[i] >= q2 else 1
P['T_COMP'] = o_; line = ' COMP'
for e, (a, b) in (('19-21', ('2019', '2022')), ('22-26', ('2022', '2027')), ('pooled', ('2019', '2027'))):
    s = P[(P.t >= a) & (P.t < b)]; lo = s[s.T_COMP == 0].rem; hi = s[s.T_COMP == 2].rem; line += f' | {e}: {lo.mean() * 100:+.2f} {hi.mean() * 100:+.2f} (n {len(lo)}/{len(hi)}) t {tt(hi, lo):+.1f}'
print(line + '   (composite of PV, VW, EFF, PB, UVS)')
