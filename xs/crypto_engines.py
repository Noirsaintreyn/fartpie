"""Nov 10 2026: DO THE ENGINES THAT WORKED (OR NOT) ON NQ WORK IN CRYPTO? BTCUSDT / ETHUSDT 5m spot (Binance public archive, UTC). DECLARED BEFORE RESULTS.
Session anchors (a 24h market has no RTH, so two clocks are tried): U0 = 00:00-23:55 UTC (the calendar day); US = 13:30-20:00 UTC (the NY cash session, 09:30-16:00 ET, DST ignored).
Costs per round trip as a fraction of price: 0.06% base (taker fee + spread of a funded crypto account), 0.03% and 0.10% as sensitivity. Returns in % of notional, per day (days with no trade = 0).
FAMILIES (same code as xs/new_models.py, ported to % returns):
  N noise-area momentum: sigma_t = mean |close_t / anchor-open - 1| over the prior 14 sessions at the same time of day; bands = open x (1 +- k sigma) (prior close also used as on NQ); decisions every 30 min from 1 hour after the anchor; exit on a close back inside max(UB, VWAP)/min(LB, VWAP); reverse allowed; flat at the session end; stop = 0.5 x sigma_end of the session (or none).
  M inside-band fade to VWAP (z 1.5).   O opening-range breakout 30 / 60 min, flat at the end.   L last-hours momentum: sign of the first 4h return -> trade the last 2h.   T time-series: sign of the prior session's return -> trade the whole session (daily trend).
PASS = mean % per day > 0 with t >= 2 in BOTH periods (2018-21 | 2022-26); neighbouring parameters must agree. ~30 variants x 2 coins x 2 clocks -> expect a few chance passes, so only a pattern across coins / neighbours counts."""
import numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings('ignore')
D = '/tmp/claude-0/crypto'; BAR = 5
def load(sym):
    d = pd.read_parquet(f'{D}/{sym}_5m.parquet'); d = d[~d.index.duplicated()]; return d[(d.index >= '2018-01-01')]
def prep(d, a0, a1):
    """sessions: arrays o,h,l,c,v on the 5m grid between minute-of-day a0 and a1 (exclusive end)"""
    m = d.index.hour * 60 + d.index.minute; s = d[(m >= a0) & (m < a1)].copy(); s['day'] = s.index.normalize(); out = {}
    n = (a1 - a0) // BAR
    for day, g in s.groupby('day'):
        if len(g) < n * .9: continue
        ix = ((g.index.hour * 60 + g.index.minute - a0) // BAR).values; a = np.full((5, n), np.nan)
        a[0, ix], a[1, ix], a[2, ix], a[3, ix], a[4, ix] = g.open.values, g.high.values, g.low.values, g.close.values, g.volume.values
        out[day] = a
    return out, n
def sig_table(S, n):
    days = sorted(S); AB = pd.DataFrame({d: np.abs(S[d][3] / S[d][0][~np.isnan(S[d][0])][0] - 1) for d in days}).T
    return AB.rolling(14, min_periods=10).mean().shift(1)
def run(S, n, SIG, fam, cost, k=1.0, step=6, stop_m=.5, start=12, side=0, z=1.5, orm=6, hrs=48):
    rows = []; days = sorted(S); prev_close = None
    for i, d in enumerate(days):
        o, h, l, c, v = S[d]; op = o[~np.isnan(o)][0]; sg = SIG.loc[d].values if d in SIG.index else np.full(n, np.nan)
        pc = c[~np.isnan(c)][-1]
        if fam == 'N' or fam == 'M':
            sgk = sg * k; send = sgk[-1] if np.isfinite(sgk[-1]) else np.nan; pcl = prev_close if prev_close is not None else op
            tp = (h + l + c) / 3; vw = np.cumsum(np.nan_to_num(tp * v)) / np.maximum(np.cumsum(np.nan_to_num(v)), 1e-9)
            pos = 0; e = None; pend = None; worst = 0.0; nt = 0
            for j in range(n):
                if np.isnan(c[j]): continue
                if pend is not None: pos, e, worst = pend, o[j], 0.0; pend = None; nt += 1
                if pos != 0:
                    lo = ((l[j] / e - 1) if pos > 0 else (1 - h[j] / e))
                    if stop_m and np.isfinite(send) and lo <= -stop_m * send: rows.append((d, pos, -stop_m * send - cost)); pos = 0; continue
                if j == n - 1:
                    if pos != 0: rows.append((d, pos, ((c[j] / e - 1) * pos) - cost)); pos = 0
                    break
                if np.isnan(o[j + 1]) or not np.isfinite(sg[j]): continue
                ub, lb = max(op, pcl) * (1 + sg[j] * k), min(op, pcl) * (1 - sg[j] * k); nx = o[j + 1]
                if fam == 'N':
                    if (j + 1) % step != 0 or j + 1 < start: continue
                    if pos == 1 and c[j] < max(ub, vw[j]): rows.append((d, 1, nx / e - 1 - cost)); pos = 0
                    elif pos == -1 and c[j] > min(lb, vw[j]): rows.append((d, -1, 1 - nx / e - cost)); pos = 0
                    if pos == 0 and c[j] > ub and side >= 0: pend = 1
                    elif pos == 0 and c[j] < lb and side <= 0: pend = -1
                else:
                    if j + 1 < start or j + 1 > n - 12: continue
                    zz = (c[j] - vw[j]) / (op * sg[j] * k) if sg[j] > 0 else 0
                    if pos != 0:
                        if (pos > 0 and c[j] >= vw[j]) or (pos < 0 and c[j] <= vw[j]) or (pos > 0 and c[j] < lb) or (pos < 0 and c[j] > ub): rows.append((d, pos, (nx / e - 1) * pos - cost)); pos = 0
                    if pos == 0 and nt < 2 and lb < c[j] < ub:
                        if zz >= z: pend = -1
                        elif zz <= -z: pend = 1
        elif fam == 'O':
            e5 = slice(0, orm)
            if np.isnan(h[e5]).all(): prev_close = pc; continue
            orh, orl = np.nanmax(h[e5]), np.nanmin(l[e5]); ent = None
            for j in range(orm, n - 12):
                if c[j] > orh and not np.isnan(o[j + 1]): ent = (1, o[j + 1]); jj = j + 1; break
                if c[j] < orl and not np.isnan(o[j + 1]): ent = (-1, o[j + 1]); jj = j + 1; break
            if ent:
                dr, e = ent; st = orl if dr > 0 else orh; px = None
                for j in range(jj, n):
                    if np.isnan(c[j]): continue
                    if (l[j] <= st if dr > 0 else h[j] >= st): px = st; break
                rows.append((d, dr, ((px if px is not None else c[-1]) / e - 1) * dr - cost))
        elif fam == 'L':
            f = int(hrs // 5 * 0 + n * .5)
            r1 = c[f - 1] / op - 1 if np.isfinite(c[f - 1]) else np.nan
            if np.isfinite(r1) and not np.isnan(o[n - 24]):
                dr = 1 if r1 > 0 else -1; rows.append((d, dr, (c[-1] / o[n - 24] - 1) * dr - cost))
        elif fam == 'T':
            if prev_close is not None and np.isfinite(S[days[i - 1]][3][-1]):
                r0 = prev_close / S[days[i - 1]][0][~np.isnan(S[days[i - 1]][0])][0] - 1; dr = 1 if r0 > 0 else -1; rows.append((d, dr, (c[-1] / op - 1) * dr - cost))
        prev_close = pc
    return pd.DataFrame(rows, columns=['day', 'dir', 'r'])
def stat(T, days, label, show=True):
    s = T.groupby('day').r.sum().reindex(days).fillna(0.0) * 100; res = []; ok = True
    for lab, mm in (('18-21', s.index < '2022-01-01'), ('22-26', s.index >= '2022-01-01')):
        x = s[mm]; t = x.mean() / (x.std() / np.sqrt(len(x))) if x.std() > 0 else 0; res.append((x.mean(), t)); ok &= (x.mean() > 0 and t >= 2)
    L, Sh = T[T.dir > 0].r.mean() * 100 if (T.dir > 0).any() else np.nan, T[T.dir < 0].r.mean() * 100 if (T.dir < 0).any() else np.nan
    if show: print(f'  {label:<46} n {len(T):5d} ({len(T) / len(days):.2f}/d) | %/day ' + ' | '.join(f'{lab} {m:+6.3f} (t {t:+5.2f})' for lab, (m, t) in zip(('18-21', '22-26'), res)) + f' | long {L:+.3f} short {Sh:+.3f} %/trade{"   <== PASS" if ok else ""}', flush=True)
    return s, ok
if __name__ == '__main__':
    for sym in ('BTCUSDT', 'ETHUSDT'):
        d = load(sym)
        for clk, (a0, a1) in (('U0 00:00-24:00 UTC', (0, 1440)), ('US 13:30-20:00 UTC', (810, 1200))):
            S, n = prep(d, a0, a1); SIG = sig_table(S, n); days = pd.DatetimeIndex(sorted(S)); days = days[(days >= '2018-02-01')]
            print(f'\n===== {sym} {clk}: {len(days)} sessions, {n} bars; buy-and-hold in-session mean {np.mean([S[x][3][-1] / S[x][0][~np.isnan(S[x][0])][0] - 1 for x in days]) * 100:+.3f} %/day =====')
            for cost in (.0006, .0003, .0010):
                print(f' -- cost {cost * 100:.2f}% round trip')
                grid = [('N base k1 30m stop .5s', dict(fam='N')), ('N k .75', dict(fam='N', k=.75)), ('N k 1.25', dict(fam='N', k=1.25)), ('N 15m', dict(fam='N', step=3)), ('N no stop', dict(fam='N', stop_m=None)), ('N long only', dict(fam='N', side=1)), ('N short only', dict(fam='N', side=-1))]
                if cost == .0006: grid += [('M fade z1.5', dict(fam='M')), ('O ORB30', dict(fam='O', orm=6)), ('O ORB60', dict(fam='O', orm=12)), ('L first-half -> last 2h', dict(fam='L')), ('T prior session trend', dict(fam='T'))]
                for nm, kw in grid:
                    T = run(S, n, SIG, cost=cost, **kw); stat(T, days, nm)
