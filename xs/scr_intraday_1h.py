"""Nov 9 2026: ROUGH LOOK (Yahoo HOURLY bars, the Mac's 15m bars unavailable) at the screener's slots with intraday information (Rey). DECLARED BEFORE RESULTS.
Hourly bars exist only for the last ~2 years, so: candidates Dec 2 2024 - Sep 25 2026, CHOOSE on Dec 2024 - Sep 2025, JUDGE on Oct 2025 - Sep 2026.
Hourly bars cannot see a 15m flip; features use the LAST COMPLETED HOUR before the entry, so only flips at / after 10:30 have them (first-hour flips, ~57% of all,
keep the live ratio order):
  SPD1h  the last completed hour's move (close - open) / 14-day ATR, on the trade's side (long)     RV1h  that hour's volume / the same hour's 14-session average
PICK ORDER when flips arrive together: ratio (live) | SPD1h | RV1h | ratio + SPD1h | ratio + RV1h (rank sums; a bar where nobody has the feature keeps the ratio order).
REPLACEMENT when all 6 slots are full and a new flip arrives (prices at the new flip's time from the hourly bars - entry and current prices are approximate, +-0.2-0.5%):
  R1 weakest holding losing and the new flip's ratio higher | R2 a holding whose first full hour after entry closed below entry, new ratio higher | R3 weakest holding down > 0.5 ATR.
Same 6 slots, 8% breaker, 2 bp, stand-in daily path as xs/scr_slots.py. A hint, not a decision: rerun xs/scr_intraday.py on the Mac's 15m bars for the real test."""
import os, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
from collections import deque
J = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/site_wide_daily.parquet'); D = J.index
T0 = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet')
T0 = T0[T0.sp & (T0.ratio >= 1.5) & (T0.entry >= '2024-12-02') & (T0.exit < D[-1] + pd.Timedelta(days=1))].copy(); TKS = sorted(T0.tk.unique())
CA = '/home/claude/fartpie/data/yahoo/h1'; os.makedirs(CA, exist_ok=True); YS = {'BRKB': 'BRK-B'}; BARS = {}
for tk in TKS:
    f = f'{CA}/{tk}.parquet'
    if not os.path.exists(f):
        Y = yf.download(YS.get(tk, tk), period='729d', interval='1h', auto_adjust=False, progress=False)
        if isinstance(Y.columns, pd.MultiIndex): Y.columns = Y.columns.get_level_values(0)
        if not len(Y): print('  no bars', tk); continue
        Y = Y.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']].dropna(); Y.index = pd.DatetimeIndex(Y.index).tz_convert('America/New_York').tz_localize(None); Y.to_parquet(f)
    H = pd.read_parquet(f); mm = H.index.hour * 60 + H.index.minute; H = H[(mm >= 570) & (mm < 960)].copy()
    H['end'] = np.minimum(H.index + pd.Timedelta(minutes=60), H.index.normalize() + pd.Timedelta(minutes=960)); BARS[tk] = H
D = D[D >= '2024-12-02']; J = J.loc[D]
def feat_rows():
    out = []
    for tk, H in BARS.items():
        day = H.index.normalize(); dd = H.groupby(day).agg(h=('high', 'max'), l=('low', 'min')); atr = (dd.h - dd.l).rolling(14).mean().shift(1)
        hr = H.index.hour * 60 + H.index.minute; vavg = H.volume.groupby(hr).transform(lambda s_: s_.shift(1).rolling(14, min_periods=8).mean())
        out.append(pd.DataFrame({'tk': tk, 'end': H.end.values, 'day': day, 'spd': ((H.close - H.open) / day.map(atr)).values, 'rv': (H.volume / vavg).values, 'atr': day.map(atr).values,
                                 'close': H.close.values}))
    return pd.concat(out).sort_values('end')
FR = feat_rows()
T0 = T0.sort_values('entry'); T0['day'] = T0.entry.dt.normalize()
FR['end'] = FR['end'].astype('datetime64[ns]'); T0['entry'] = T0['entry'].astype('datetime64[ns]')
M = pd.merge_asof(T0.reset_index(), FR.rename(columns={'day': 'fday'}), left_on='entry', right_on='end', by='tk', direction='backward')
M.loc[M.fday != M.day, ['spd', 'rv']] = np.nan; T0 = M.set_index('index'); T0['atrp'] = T0.atr / T0.close
def price_at(tk, t):
    H = BARS.get(tk)
    if H is None: return np.nan
    k = H.index.searchsorted(t, 'left')
    if k < len(H) and H.index[k] == t: return H.open.iloc[k]
    e = H.end.values.searchsorted(np.datetime64(t), 'right') - 1
    if e >= 0 and pd.Timestamp(H.end.values[e]).normalize() == pd.Timestamp(t).normalize(): return H.close.iloc[e]
    k2 = H.index.searchsorted(pd.Timestamp(t).normalize() + pd.Timedelta(minutes=570), 'left')
    return H.open.iloc[k2] if k2 < len(H) and H.index[k2].normalize() == pd.Timestamp(t).normalize() else np.nan
def first_hour_close(tk, e):
    H = BARS.get(tk)
    if H is None: return np.nan, None
    k = H.index.searchsorted(e, 'left')
    if k < len(H) and H.index[k].normalize() == pd.Timestamp(e).normalize(): return H.close.iloc[k], H.end.iloc[k]
    return np.nan, None
T0['px'] = [price_at(t, e) for t, e in zip(T0.tk, T0.entry)]
FH = [first_hour_close(t, e) for t, e in zip(T0.tk, T0.entry)]; T0['fh_c'] = [x[0] for x in FH]; T0['fh_end'] = [x[1] for x in FH]
def taken(X, N=6, brk=.08, rule=None):
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []; nrep = 0
    for r in X.itertuples():
        e = r.entry
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr, _h in done: real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
        op = [h for h in op if h['x'] > e]
        if len(op) >= N and rule:
            cand = []
            for h in op:
                if h['e'] >= e or not np.isfinite(h['px']): continue
                cur = price_at(h['tk'], e) / h['px'] - 1
                if not np.isfinite(cur): continue
                if rule == 'R1' and cur < 0 and r.ratio > h['ratio']: cand.append((cur, h))
                if rule == 'R2' and h['fh_end'] is not None and h['fh_end'] <= e and h['fh_c'] < h['px'] and r.ratio > h['ratio']: cand.append((cur, h))
                if rule == 'R3' and np.isfinite(h['atrp']) and cur < -.5 * h['atrp']: cand.append((cur, h))
            if cand:
                cur, h = min(cand, key=lambda z: z[0]); op.remove(h); pend = [p for p in pend if p[2] is not h]; nrep += 1
                rr = (1 + cur) * (1 - 2e-4) - 1; out[h['k']] = (h['tk'], h['e'], e, rr); real += np.log1p(rr / N); hist.append((e, real)); win.append((e, real))
        if len(op) >= N: continue
        if brk and hist:
            t0 = e - pd.Timedelta(days=28)
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [hh for _, hh in win]) < np.log1p(-brk): continue
        h = dict(tk=r.tk, e=e, x=r.exit, ratio=r.ratio, px=r.px, fh_c=r.fh_c, fh_end=r.fh_end, atrp=r.atrp, k=len(out))
        op.append(h); pend.append((r.exit, r.ret, h)); out.append((r.tk, e, r.exit, r.ret))
    return out, nrep
def daily(TK, N=6):
    R = np.zeros(len(D)); W = np.zeros(len(D))
    for tk, e, x, r in TK:
        a, b = D.searchsorted(pd.Timestamp(e).normalize()), D.searchsorted(pd.Timestamp(x).normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / N; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w; W[a:b + 1] += w
    return pd.Series(R * np.where(W > 1, 1 / np.maximum(W, 1e-9), 1.0), index=D)
def halves(s):
    out = []
    for a, b in (('2024-12-01', '2025-10-01'), ('2025-10-01', '2026-10-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); out.append(f'{e.iloc[-1] - 1:+6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    e = (1 + s).cumprod(); return ' | '.join(out) + f' | whole {e.iloc[-1] - 1:+6.1%}'
def h1(s): z = s[s.index < '2025-10-01']; return (1 + z).prod()
def order(kind):
    X = T0.copy(); g = X.groupby('entry'); has = {f: g[f].transform(lambda s_: s_.notna().any()) for f in ('spd', 'rv')}
    rr = g.ratio.rank(ascending=False); rk = lambda f: np.where(has[f], g[f].rank(ascending=False, na_option='bottom'), rr)
    X['_k'] = {'ratio': rr, 'spd': rk('spd'), 'rv': rk('rv'), 'ratio+spd': rr + rk('spd'), 'ratio+rv': rr + rk('rv')}[kind]
    return X.sort_values(['entry', '_k', 'tk'], kind='stable')
if __name__ == '__main__':
    print(f'candidates {len(T0):,} (Dec 2024 - Sep 2026) | hourly bars for {len(BARS)} of {len(TKS)} names | entry price found {T0.px.notna().mean():.0%} | '
          f'hourly features (flips at / after 10:30) {T0.spd.notna().mean():.0%}', flush=True)
    print('\n=== pick order | return / max DD  Dec 2024 - Sep 2025 (CHOOSE) | Oct 2025 - Sep 2026 (JUDGE) | whole ===', flush=True); RES = {}
    for kind in ('ratio', 'spd', 'rv', 'ratio+spd', 'ratio+rv'):
        tk_, _ = taken(order(kind)); s_ = daily(tk_); RES[kind] = s_; print(f'  {kind:<12} {halves(s_)}', flush=True)
    bk = max(RES, key=lambda k_: h1(RES[k_])); print(f'  -> chosen on the first half: {bk}', flush=True)
    late = T0[T0.spd.notna()]; q = late[late.entry < '2025-10-01']
    for f in ('spd', 'rv'):
        c1, c2 = q[f].quantile([1 / 3, 2 / 3]); b_ = pd.Series(np.where(late[f] < c1, '1 low', np.where(late[f] < c2, '2 mid', '3 high')), index=late.index)
        print(f'  flips at / after 10:30 by {f} third: ' + ' | '.join(f'{k}: {g_.ret.mean():+.2%} (win {np.mean(g_.ret > 0):.0%}, n {len(g_)})' for k, g_ in late.groupby(b_)), flush=True)
    print('\n=== replacing weak holdings (live ratio order) | first half (CHOOSE) | second half (JUDGE) | whole ===', flush=True); RR = {}
    X0 = order('ratio')
    for rule in (None, 'R1', 'R2', 'R3'):
        tk_, nrep = taken(X0, rule=rule); s_ = daily(tk_); RR[rule] = s_; print(f'  {str(rule or "no replacement (live)"):<22} {halves(s_)} | swaps {nrep:,}', flush=True)
    br = max(['R1', 'R2', 'R3'], key=lambda k_: h1(RR[k_])); print(f'  -> chosen on the first half: {br} ({"beats" if h1(RR[br]) > h1(RR[None]) else "does not beat"} live there)', flush=True)
