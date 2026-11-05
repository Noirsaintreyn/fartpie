"""Nov 4 2026: BREAK TYPES of the vector flip for the NQ NY AM trade. DECLARED BEFORE RESULTS.
Trades: base (QQQ 15m flip 09:30-11:30, both ways, <= 2 a day, 1 NQ 75/50, flat 12:00, 1 pt cost, QQQ proxy -> NQ pts), 2018 - Oct 2026.
LABELS on the flip bar / entry (only data up to the entry):
  STRENGTH  hard break = flip bar range >= its 2018-21 top third in ATR AND close in the outer 25% of the bar in the trade's direction AND volume above the
            20-day same-time average | weak break = small range (bottom third) OR close in the inner half of the bar | normal = the rest
  CLOSE     flip bar close location in the trade's direction: strong (>= .75) / middle / weak (< .5)
  RANGE-D   entry within TODAY's range so far (from 09:30), in the trade's direction: breaking out at the extreme (>= .8) / mid / from the opposite end (< .2, reversal)
  RANGE-PD  entry vs the PRIOR day's RTH range in the trade's direction: beyond it (new high for longs / new low for shorts) / top fifth / middle / bottom fifth / beyond the opposite side
  RANGE-20  entry within the prior 20 sessions' range in the trade's direction: top fifth / middle / bottom fifth (outside counted with the nearest fifth)
  NODE      entry at an HVN / LVN / mid node / outside of the monthly (21-day) and 60-day profiles (from nq_vp_nodes.py)
  STRENGTH x NODE_M and STRENGTH x AHEAD_60D (does a HARD break through thin profile run, where a weak one fails?)
A level counts if it beats its period's average in 2018-21, 2022-23 AND 2024-26 with n >= 30 in each."""
_s = open('xs/nq_vp_nodes.py').read(); exec(_s[:_s.index("R = run_l(C); Z = ")])
L = pd.read_parquet('/tmp/claude-0/ml/nq_flip_types_C.parquet')[['day', 'j', 'disp', 'vol5']]
C = C.rename_axis('ix').reset_index().merge(L, on=['day', 'j'], how='left').set_index('ix')
end_i = pd.Series(np.arange(len(H15)), index=pd.DatetimeIndex(H15.end)); kk = end_i.reindex(pd.DatetimeIndex(t5[C.j.values])).values.astype(int) - 1   # the flip bar
hh, ll, cc_ = H15.high.values, H15.low.values, H15.close.values
clv = np.where(hh[kk] > ll[kk], (cc_[kk] - ll[kk]) / (hh[kk] - ll[kk]), .5); C['clv'] = np.where(C.dir > 0, clv, 1 - clv)
isb = C.day < P0; dq_ = C.loc[isb, 'disp'].quantile([1 / 3, 2 / 3]).values
hard = (C.disp >= dq_[1]) & (C.clv >= .75) & (C.vol5 > 1); weak = (C.disp < dq_[0]) | (C.clv < .5)
C['STRENGTH'] = np.where(hard, 'hard break', np.where(weak, 'weak break', 'normal break')); C['CLOSE'] = np.where(C.clv >= .75, 'strong close', np.where(C.clv < .5, 'weak close', 'middle close'))
RT = pd.DataFrame({'h': h5, 'l': l5, 'd': d5})[rthm].groupby('d').agg(h=('h', 'max'), l=('l', 'min')); RT.index = pd.DatetimeIndex(RT.index); PDr = RT.shift(1)
H20, L20 = RT.h.rolling(20).max().shift(1), RT.l.rolling(20).min().shift(1)
rd, rpd, r20 = [], [], []
for j, d_, day in zip(C.j.values, C.dir.values, C.day.values):
    e = o5[j]; i0 = j
    while i0 > 0 and d5[i0 - 1] == d5[j] and m5[i0 - 1] >= 570: i0 -= 1
    hi_, lo_ = (h5[i0:j].max(), l5[i0:j].min()) if j > i0 else (np.nan, np.nan)
    p = (e - lo_) / (hi_ - lo_) if hi_ > lo_ else np.nan; rd.append(p if d_ > 0 else 1 - p)
    day = pd.Timestamp(day); ph, pl = PDr.h.get(day, np.nan), PDr.l.get(day, np.nan); q = (e - pl) / (ph - pl) if ph > pl else np.nan; rpd.append(q if d_ > 0 else 1 - q)
    a, b = H20.get(day, np.nan), L20.get(day, np.nan); z = (e - b) / (a - b) if a > b else np.nan; r20.append(z if d_ > 0 else 1 - z)
C['rd'], C['rpd'], C['r20'] = rd, rpd, r20
C['RANGE-D'] = np.where(C.rd >= .8, 'at the extreme (breakout)', np.where(C.rd < .2, 'from the opposite end', 'mid range'))
C['RANGE-PD'] = np.select([C.rpd > 1, C.rpd >= .8, C.rpd >= .2, C.rpd >= 0], ['beyond prior range (new hi/lo)', 'top fifth of prior range', 'middle of prior range', 'bottom fifth of prior range'], 'beyond the opposite side')
C['RANGE-20'] = np.where(C.r20 >= .8, '20d top fifth', np.where(C.r20 < .2, '20d bottom fifth', '20d middle'))
C['STRENGTH x NODE_M'] = C.STRENGTH + ' @ ' + C.NODE_M.astype(str); C['STRENGTH x AHEAD_60D'] = C.STRENGTH + ' / ' + C.AHEAD_60D.astype(str)
R = run_l(C); Z = C.join(R[['net']], how='inner'); Z['per'] = np.where(Z.day < P0, '18-21', np.where(Z.day < '2024-01-01', '22-23', '24-26')); base = Z.groupby('per').net.mean(); PERS = ('18-21', '22-23', '24-26')
print(f'trades {len(Z)}; avg 18-21 {base["18-21"]:+.1f} | 22-23 {base["22-23"]:+.1f} | 24-26 {base["24-26"]:+.1f}; hard {np.mean(hard):.0%} weak {np.mean(weak & ~hard):.0%} of flips', flush=True)
for col in ('STRENGTH', 'CLOSE', 'RANGE-D', 'RANGE-PD', 'RANGE-20', 'NODE_M', 'NODE_60D', 'STRENGTH x NODE_M', 'STRENGTH x AHEAD_60D'):
    print(f'\n  -- {col}', flush=True)
    for v_, g in Z.groupby(col):
        p = {k: g[g.per == k] for k in PERS}; ok = all(len(p[k]) >= 30 and p[k].net.mean() > base[k] for k in PERS); bad = all(len(p[k]) >= 30 and p[k].net.mean() < base[k] for k in PERS)
        if len(g) < 25: continue
        cells = ' | '.join(f'{k} {p[k].net.mean():+6.1f} (n{len(p[k]):>3} w{p[k].net.gt(0).mean():.0%})' if len(p[k]) else f'{k}    n/a' for k in PERS)
        print(f'  {str(v_):<40} {cells}' + ('   <- beats all three' if ok else ('   <- worse in all three' if bad else '')), flush=True)
