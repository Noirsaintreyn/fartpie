"""Nov 18 2026 report 335. THE 5-MIN ORB (333/334) PAIRED WITH THE RANGE FORECAST + ANCHOR (Rey Nov 17: forecast the day's range, anchor the extreme, target the rest of the range; report 264).  Declared BEFORE running.
Forecast R^ = HAR + QQQ-implied-move model (report 264 model B; best out-of-sample R2 .557 | .392), known at 09:30 from prior data only; rolling OLS 500 sessions, refit every 20, bias-corrected.
ANCHOR: the first-candle extreme on the side AWAY from the break is the day's running extreme so far (long break: anchor A = candle low; short: A = candle high).  Projected day extreme = A +/- R^.
THEORY: the ORB is a trend-day engine (285/307); range is forecastable (253-270) but day type is not.  So the forecast can set HOW MUCH room a break has, not WHETHER it trends.  Tests (all fixed here):
 P1 ROOM: room = (A +/- R^ - entry)/ candle range (how many R of projected range remain above the entry).  HIGH room -> higher POINTS per trade (the 334 compression test failed because it used raw ADR and measured R, whose denominator moves).  Cuts: train terciles.
 P2 EXPANSION: R^ / ADR14 (forecast above normal = expected expansion day, IV/HAR say bigger than recent): HIGH -> higher points per trade (and R).
 P3 TARGET: exit at the projected extreme A +/- f x R^ (f = .618, 1.0) instead of holding to 15:55; skip the trade when the target is nearer than 1R.  The 264 diagnostic says the full-R^ target is too far (median remaining range .67 R^), so f=.618 is the live candidate.  Reported vs the same trades held to 15:55.
 P4 the 334 survivor: all of the above on GAP-ALIGNED breaks only, and the room/expansion filter ON TOP of the gap (does the forecast add to the gap?).
Stop = other end of the first candle (1R), cost .75 pt, same-bar ordering pessimistic (stop first).  Train 2019-21 | test 2018+2022-26.  Why it should fail: R^ is accurate in log range but its error (MAE ~60-90 pts) is large relative to the targets used; targets cut the right tail that carries the ORB's profit (top 5% of trades = all of R in 333)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
U = '/mnt/user-data/uploads/fartpie/'
N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); N = N[N.rth & (N.m >= 570) & (N.m <= 955)].copy(); N['day'] = pd.to_datetime(N.day).dt.normalize(); G = {d: x.sort_values('m') for d, x in N.groupby('day')}
dd = pd.DataFrame({d: dict(o=x.open.iloc[0], h=x.high.max(), l=x.low.min(), c=x.close.iloc[-1], n=len(x)) for d, x in G.items()}).T; dd = dd[dd.n >= 70].astype(float); dd['rng'] = dd.h - dd.l; dd['lr'] = np.log(dd.rng)
ov = pd.read_csv(U + 'options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); ov.index = ov.index.normalize(); k = np.clip(ov.index.searchsorted(dd.index.values.astype('datetime64[ns]')) - 1, 0, len(ov) - 1)
dd['imp'] = np.log(ov.atm_iv.values[k] / np.sqrt(252) * dd.o.values); dd['x1'] = dd.lr.shift(1); dd['x5'] = dd.lr.shift(1).rolling(5).mean(); dd['x22'] = dd.lr.shift(1).rolling(22).mean(); dd['adr'] = dd.rng.rolling(14).mean().shift(1)
dd = dd.dropna(subset=['x1', 'x5', 'x22', 'imp', 'adr']).copy(); cols = ['x1', 'x5', 'x22', 'imp']; X = np.c_[np.ones(len(dd)), dd[cols].values]; y = dd.lr.values; pred = np.full(len(dd), np.nan); sm = np.full(len(dd), np.nan); beta = None
for i in range(500, len(dd)):
    if beta is None or (i - 500) % 20 == 0: beta = np.linalg.lstsq(X[i - 500:i], y[i - 500:i], rcond=None)[0]; sm_ = np.mean(np.exp(y[i - 500:i] - X[i - 500:i] @ beta))
    pred[i] = X[i] @ beta; sm[i] = sm_
dd['Rh'] = np.exp(pred) * sm; dd = dd.dropna(subset=['Rh']); print(f'forecast available for {len(dd)} sessions {dd.index.min().date()} -> {dd.index.max().date()}; median R^ {dd.Rh.median():.0f} pts, median realised {dd.rng.median():.0f}')
cl = dd.c.shift(1); rows = []
for d, r in dd.iterrows():
    x = G[d]; hi, lo = x.high.iloc[0], x.low.iloc[0]; rg = hi - lo
    if rg < 2: continue
    rest = x.iloc[1:]; ex = rest[rest.m == 955].open.iloc[0] if (rest.m == 955).any() else rest.close.iloc[-1]; brk = None
    for j, (_, b) in enumerate(rest.iterrows()):
        up, dn = b.high >= hi, b.low <= lo
        if up and dn: break
        if up or dn: brk = (1 if up else -1, hi if up else lo, lo if up else hi, b, rest[rest.m > b.m]); break
    if brk is None: continue
    dr, ent, stop, b, later = brk; A = lo if dr == 1 else hi; out = {}
    for lab, f in (('hold', None), ('t618', .618), ('t100', 1.0)):
        tgt = None if f is None else A + dr * f * r.Rh
        if tgt is not None and (tgt - ent) * dr < rg: out[lab] = np.nan; continue      # target nearer than 1R: skip
        xp = ex; hit = False
        if (dr == 1 and b.low <= stop) or (dr == -1 and b.high >= stop): xp = stop; hit = True
        if not hit:
            for _, c in later.iterrows():
                if (dr == 1 and c.low <= stop) or (dr == -1 and c.high >= stop): xp = stop; break
                if tgt is not None and ((dr == 1 and c.high >= tgt) or (dr == -1 and c.low <= tgt)): xp = tgt; break
        out[lab] = (xp - ent) * dr - .75
    rows.append(dict(day=d, dir=dr, rg=rg, o=r.o, gap=(r.o - cl.get(d, np.nan)) / r.adr, room=(A + dr * r.Rh - ent) * dr / rg, expn=r.Rh / r.adr, Rh=r.Rh, **out))
T = pd.DataFrame(rows).set_index('day'); T['gal'] = np.sign(T.gap) == T.dir; T['R'] = T.hold / T.rg
tr = (T.index >= '2019-01-01') & (T.index < '2022-01-01'); te = (T.index >= '2018-01-01') & ~tr
def f(m, col='hold'):
    s = T.loc[m, col].dropna(); return '   n/a' if len(s) < 20 else f'{s.mean():+6.2f} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f},n{len(s):4d})'
print(f'ORB trades {len(T)}; pts/trade hold-to-close: train {f(tr)} | test {f(te)}')
def terc(col, lab, sub=None, label=''):
    s = np.ones(len(T), bool) if sub is None else sub.values; q = T.loc[tr & s, col].quantile([1 / 3, 2 / 3]).values; L, Mi, H = T[col] <= q[0], (T[col] > q[0]) & (T[col] < q[1]), T[col] >= q[1]
    print(f'  {lab:28s}{label} pts  train low {f(tr & s & L)} mid {f(tr & s & Mi)} high {f(tr & s & H)}\n  {"":28s}{"":9s}test  low {f(te & s & L)} mid {f(te & s & Mi)} high {f(te & s & H)}')
print('\nP1 room (projected range remaining above the entry, in R) | P2 expansion (R^/ADR14)'); terc('room', 'P1 room'); terc('expn', 'P2 expansion'); terc('Rh', 'forecast range (pts)')
print('\nP4 on GAP-ALIGNED breaks only'); print(f'  gap-aligned hold: train {f(tr & T.gal)} | test {f(te & T.gal)}'); terc('room', 'P1 room', T.gal, ' [gap]'); terc('expn', 'P2 expansion', T.gal, ' [gap]')
print('\nP3 range-anchored TARGET (skip if target < 1R away) vs HOLD on the SAME trades, pts/trade')
for lab in ('t618', 't100'):
    ok = T[lab].notna(); print(f'  {lab}: trades kept {ok[tr].mean():.0%} | {ok[te].mean():.0%};  target: train {f(tr & ok, lab)} | test {f(te & ok, lab)};  hold same trades: train {f(tr & ok)} | test {f(te & ok)};  gap-aligned only target: train {f(tr & ok & T.gal, lab)} | test {f(te & ok & T.gal, lab)}  hold: {f(tr & ok & T.gal)} | {f(te & ok & T.gal)}')
T.to_parquet('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/orb335.parquet')
