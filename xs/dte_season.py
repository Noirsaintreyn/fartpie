"""Nov 8 2026: SIGNALS IN SEASON - let the dealer / volatility regime pick which simple signal to trade. OUT OF SAMPLE: the signal for each regime is chosen on 2014-2021
and judged on 2022 - Jan 2026. DECLARED BEFORE RESULTS.
Real NQ futures, 15m bars built from 5m (RTH, carried across days); entry at the next 5m open after a 15m close; exit when the signal turns back (next open), stop 0.15 ADR
on 5m bars, flat 15:55; one position at a time per signal, <= 4 a day; signals on bars ending 09:45 - 15:30; 1 pt cost.
Signals: EMA 9/21 | EMA 20/50 | DONCHIAN 20-bar breakout (stop-and-reverse on the opposite break) | VWAP cross | SUPERTREND (10, 3) | MACD histogram sign | VECTOR 15m flip |
ORB 30 (first break of the 09:30 - 10:00 range, exit at the other side or the close).
Regimes at the prior close: QQQ GEX percentile from the greeks file (low / mid / high third, 2014+) x 20-day Yang-Zhang (top third vs not) = 6 cells.
Report: the full grid (points per trade 2014-21 / 2022-26), the 2014-21 pick per cell, and the out-of-sample book of 'trade the pick of today's cell' vs the vector alone."""
_r = open('/home/claude/fartpie/xs/dte_meanrev.py').read(); exec(_r[:_r.rindex("if __name__ == '__main__':")])
_y = open('/home/claude/fartpie/xs/dte_yz.py').read(); exec(_y[_y.index('lr2 = lambda'):_y.index('def t3')])
ema_ = lambda s, n: s.ewm(span=n, adjust=False).mean()
B = B15.copy(); cl = B.close
B['s_ema921'] = np.sign(ema_(cl, 9) - ema_(cl, 21)); B['s_ema2050'] = np.sign(ema_(cl, 20) - ema_(cl, 50))
hh20 = B.high.rolling(20).max().shift(1); ll20 = B.low.rolling(20).min().shift(1); dch = np.where(cl > hh20, 1, np.where(cl < ll20, -1, np.nan)); B['s_donch'] = pd.Series(dch, index=B.index).ffill().fillna(0)
vw_end = pd.Series(vw, index=H.dt.values); B['vwap'] = vw_end.reindex(B.index + pd.Timedelta(minutes=10)).values; B['s_vwap'] = np.sign(cl - B.vwap)
tr_ = np.maximum.reduce([B.high - B.low, (B.high - cl.shift(1)).abs(), (B.low - cl.shift(1)).abs()]); at = pd.Series(tr_, index=B.index).rolling(10).mean()
mid = (B.high + B.low) / 2; up, dn = (mid + 3 * at).values, (mid - 3 * at).values; st_ = np.zeros(len(B)); fu_, fl_ = up.copy(), dn.copy(); cv_ = cl.values
for i in range(1, len(B)):
    fu_[i] = min(up[i], fu_[i - 1]) if cv_[i - 1] <= fu_[i - 1] else up[i]; fl_[i] = max(dn[i], fl_[i - 1]) if cv_[i - 1] >= fl_[i - 1] else dn[i]
    st_[i] = 1 if cv_[i] > fu_[i - 1] else (-1 if cv_[i] < fl_[i - 1] else st_[i - 1])
B['s_super'] = st_; mac = ema_(cl, 12) - ema_(cl, 26); B['s_macd'] = np.sign(mac - ema_(mac, 9)); B['s_vector'] = B.reg
SIGS = ['s_ema921', 's_ema2050', 's_donch', 's_vwap', 's_super', 's_macd', 's_vector']
def run_sig(col):
    s = np.nan_to_num(B[col].values.astype(float)); ends = B.end.values; dys = B.day.values; mm_ = (pd.DatetimeIndex(ends).hour * 60 + pd.DatetimeIndex(ends).minute).values; rows = []; pos_end = {}
    for q in range(1, len(B)):
        if s[q] == s[q - 1] or s[q] == 0 or not (585 <= mm_[q] <= 930): continue
        d = pd.Timestamp(dys[q])
        if not ok(d): continue
        i = tpos5.get(np.datetime64(ends[q], 'ns'))
        if i is None or dd[i] != np.datetime64(d): continue
        cnt, busy = pos_end.get(d, (0, -1))
        if cnt >= 4 or i <= busy: continue
        dr = int(s[q]); e = o[i]; S = e - dr * .15 * DY.adr[d]; ix = gi[d]; j = int(np.where(ix == i)[0][0]); res = None
        for p in range(j, len(ix)):
            ii = ix[p]
            if m[ii] >= 955: res = ((c[ix[p - 1]] - e) * dr, ii); break
            if (l[ii] <= S) if dr > 0 else (h[ii] >= S): res = ((S - e) * dr, ii); break
            if (m[ii] + 5) % 15 == 0 and p + 1 < len(ix):
                bq = B.index.searchsorted(H.dt.values[ii] - np.timedelta64(10, 'm'))
                if bq < len(B) and s[bq] == -dr and B.index[bq] == H.dt.values[ii] - np.timedelta64(10, 'm'): res = ((o[ix[p + 1]] - e) * dr, ix[p + 1]); break
        if res is None: res = ((c[ix[-1]] - e) * dr, ix[-1])
        rows.append((d, dr, res[0] - 1.0)); pos_end[d] = (cnt + 1, res[1])
    return pd.DataFrame(rows, columns=['day', 'dir', 'pts'])
def orb30():
    rows = []
    for d in days:
        if not ok(d): continue
        ix = gi[d]; rng = ix[m[ix] < 600]
        if len(rng) < 6: continue
        hi_, lo_ = h[rng].max(), l[rng].min(); after = ix[m[ix] >= 600]
        for p, i in enumerate(after[:-1]):
            if c[i] > hi_ or c[i] < lo_:
                dr = 1 if c[i] > hi_ else -1; e = o[after[p + 1]]; S = lo_ if dr > 0 else hi_
                if (e - S) * dr <= 0: break
                j = int(np.where(ix == after[p + 1])[0][0]); rows.append((d, dr, walk(ix, j, dr, e, e + dr * 1e9, S) - 1.0)); break
    return pd.DataFrame(rows, columns=['day', 'dir', 'pts'])
if __name__ == '__main__':
    TR = {col[2:]: run_sig(col) for col in SIGS}; TR['orb30'] = orb30()
    gp = DY.g2p; yz = DY.yz20p
    def cell(d):
        g, v = gp.get(d, np.nan), yz.get(d, np.nan)
        if not (np.isfinite(g) and np.isfinite(v)): return None
        return ('gamma low' if g <= 1 / 3 else ('gamma high' if g >= 2 / 3 else 'gamma mid')) + (' / vol high' if v >= 2 / 3 else ' / vol normal')
    CELLS = [f'gamma {a} / vol {b}' for a in ('low', 'mid', 'high') for b in ('high', 'normal')]
    for k, T in TR.items(): T['cell'] = [cell(pd.Timestamp(d)) for d in T.day]; T['yr'] = T.day.dt.year
    print('points per trade by regime cell: 2014-21 (n) / 2022-26 (n)', flush=True)
    print(f'  {"signal":<9} ' + ' | '.join(f'{c_:<24}' for c_ in CELLS), flush=True)
    PICK = {}
    for k, T in TR.items():
        cells_txt = []
        for c_ in CELLS:
            a = T[(T.cell == c_) & (T.yr <= 2021)].pts; b = T[(T.cell == c_) & (T.yr >= 2022)].pts
            cells_txt.append(f'{a.mean():+5.1f} ({len(a):3d}) / {b.mean():+5.1f} ({len(b):3d})')
            if len(a) >= 50 and (c_ not in PICK or a.mean() > PICK[c_][1]): PICK[c_] = (k, a.mean())
        print(f'  {k:<9} ' + ' | '.join(f'{t:<24}' for t in cells_txt), flush=True)
    print('\n2014-21 PICK per cell -> its 2022-26 result (out of sample):', flush=True)
    rows = []
    for c_ in CELLS:
        k, a = PICK.get(c_, (None, np.nan))
        if k is None: continue
        T = TR[k]; b = T[(T.cell == c_) & (T.yr >= 2022)]; rows.append(b.assign(sig=k))
        print(f'  {c_:<24} pick {k:<9} 2014-21 {a:+5.2f} -> 2022-26 {b.pts.mean():+5.2f} (n{len(b)}, win {np.mean(b.pts > 0):.0%})', flush=True)
    SEL = pd.concat(rows); AD22 = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2022-01-01') and ok(d)])
    print('\nOUT-OF-SAMPLE BOOK 2022 - Jan 2026, 1 contract:', flush=True)
    for lab, T in (('trade the pick of today\'s cell', SEL), ('vector alone', TR['vector'][TR['vector'].yr >= 2022]), ('EMA 9/21 alone', TR['ema921'][TR['ema921'].yr >= 2022]),
                   ('best single signal 2014-21 overall', None)):
        if T is None:
            best = max(TR, key=lambda k: TR[k][TR[k].yr <= 2021].pts.mean()); T = TR[best][TR[best].yr >= 2022]; lab = f'{lab} ({best})'
        P = T.groupby('day').pts.sum().reindex(AD22).fillna(0.0); cum = P.cumsum()
        print(f'  {lab:<44} trades {len(T):5d} ({len(T) / len(AD22):.2f}/day) avg {T.pts.mean():+5.2f} | {P.mean():+5.2f} pts/day, Sharpe {P.mean() / P.std() * np.sqrt(252):+.2f}, '
              f'max DD {(cum - cum.cummax()).min():+.0f}, days >= +7.5 {np.mean(P >= 7.5) * 21:.1f}/mo', flush=True)
