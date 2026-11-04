"""Nov 4 2026: (1) does the SMA cross WORK by itself, (2) as a tie-breaker / rank bonus on the vector trades. DECLARED BEFORE RESULTS. 54 names with bars; 40 matched runs vs live; 2019-21 / 2022-26.
PART 1 standalone: enter at the first RTH open (before 14:00) after the state turns bull, exit at the first RTH open after it turns bear, S&P + ratio >= 1.5 as live, 2bp cost, 6 slots, ratio tie-break.
   systems: D20 daily SMA20>SMA50 | D50 daily SMA50>SMA200 | H20 4h SMA20>SMA50 | H50 4h SMA50>SMA200 | the vector (live trades) for reference. Bare per-trade stats too.
PART 2 on the live vector trades (nothing dropped): K = options ratio with the cross as the SECOND key:
   T1 ratio rounded to 1.0 buckets, then #bull states of (D20, H20)   T2 ratio rounded to 0.5, then same   T3 ratio x (1 + 0.10 x n)   T4 ratio x (1 + 0.25 x n)   T5 ratio + 0.5 x n
   n = number of bull states among {D20, H20, D50, H50} (T3-T5) or {D20, H20} (T1-T2);  T6 fresh cross: bull state that began within the last 10 sessions (bonus, same as T3 weight .25)."""
src = open('xs/sma_cross.py').read(); exec(src[:src.index("rows = []")])
RTs = RT.sort_values('d')
def asof_ratio(tk, e):
    z = RTs[RTs.tk == tk]; d = pd.Timestamp(e).normalize(); k = z.d.searchsorted(d, side='right') - 1
    return (z.ratio.iloc[k], bool(z.sp.iloc[k])) if k >= 0 else (np.nan, False)
def cross_trades(key):
    out = []
    for tk, B in BAR.items():
        s = B[key]; m = B['m']; o = B['o']; t = B['t']; pos = False; prev = np.nan; e = None
        for j in range(1, len(o)):
            if m[j] < 570 or m[j] >= 960: continue
            st = s[j - 1]
            if np.isnan(st): continue
            if not pos:
                if st == 1.0 and prev != 1.0 and m[j] < 840 and not np.isnan(prev): pos, e = True, j
            elif st == 0.0: out.append((tk, t[e], t[j], o[j] / o[e] - 1 - 2e-4)); pos = False
            prev = st
    Y = pd.DataFrame(out, columns=['tk', 'entry', 'exit', 'ret'])
    if not len(Y): return Y
    ra = [asof_ratio(a, b) for a, b in zip(Y.tk, Y.entry)]; Y['ratio'] = [r[0] for r in ra]; Y['sp'] = [r[1] for r in ra]
    return Y[(Y.sp) & (Y.ratio >= 1.5) & (Y.entry >= '2019-01-01')].sort_values('entry').reset_index(drop=True)
B0 = ev(XL); print('PART 1: standalone crosses, 6 slots, same engine\n'); line('L vector (live trades)', B0)
for nm, key in (('D20 daily SMA20>50', 'D20'), ('D50 daily SMA50>200', 'D50'), ('H20 4h SMA20>50', 'H20'), ('H50 4h SMA50>200', 'H50')):
    Y = cross_trades(key); per = np.where(Y.entry.dt.year <= 2021, '19-21', '22-26')
    print(f'  {nm}: {len(Y)} trades, avg ' + ' | '.join(f'{p} {Y[per == p].ret.mean()*100:+.2f}% win {Y[per == p].ret.gt(0).mean():.0%} n{(per == p).sum()} med-days {((Y[per == p].exit - Y[per == p].entry).dt.days).median():.0f}' for p in ('19-21', '22-26')))
    line(nm, ev(Y), B0)
print('\nPART 2: cross as tie-breaker / rank bonus on the vector trades (nothing dropped)\n')
rows = []
for i, r in XL.iterrows():
    B = BAR.get(r.tk); j = None if B is None else B['pos'].get(r.entry)
    if B is None or j is None or np.ndim(j) or j < 1: rows.append((i, np.nan, np.nan, np.nan, np.nan, 0)); continue
    j = int(j); fresh = 0
    for key in ('D20', 'H20'):
        s = B[key]; w = 10 * 26 if key == 'D20' else 10 * 6                      # ~10 sessions of 15m bars (26/session) or of 4h bars (3/session x ... approx)
        a = s[max(0, j - 1 - 260):j]; 
        if len(a) and a[-1] == 1.0 and np.nanmin(a[-260:] if len(a) >= 260 else a) == 0.0: fresh += 1
    rows.append((i, B['D20'][j - 1], B['H20'][j - 1], B['D50'][j - 1], B['H50'][j - 1], fresh))
Z = XL.join(pd.DataFrame(rows, columns=['i', 'D20', 'H20', 'D50', 'H50', 'fresh']).set_index('i'))
n2 = Z[['D20', 'H20']].fillna(0).sum(axis=1); n4 = Z[['D20', 'H20', 'D50', 'H50']].fillna(0).sum(axis=1)
def tryk(nm, K):
    Y = XL.copy(); Y['ratio'] = K.values; line(nm, ev(Y), B0)
tryk('T1 ratio (1.0 buckets), then cross count', np.floor(Z.ratio) + .01 * n2); tryk('T2 ratio (0.5 buckets), then cross count', np.floor(Z.ratio * 2) / 2 + .01 * n2)
tryk('T3 ratio x (1 + .10 n4)', Z.ratio * (1 + .10 * n4)); tryk('T4 ratio x (1 + .25 n4)', Z.ratio * (1 + .25 * n4)); tryk('T5 ratio + .5 n4', Z.ratio + .5 * n4)
tryk('T6 ratio x (1 + .25 fresh cross)', Z.ratio * (1 + .25 * Z.fresh))
tryk('T7 REVERSE: ratio x (1 - .10 n4) (bear-state first)', Z.ratio * (1 - .10 * n4))
