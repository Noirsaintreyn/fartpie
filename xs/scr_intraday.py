"""Nov 9 2026: THE SCREENER'S SLOTS WITH INTRADAY INFORMATION - SPEED INTO THE FLIP, FLIP-BAR RELATIVE VOLUME, AND REPLACING WEAK HOLDINGS (Rey). DECLARED BEFORE RESULTS.
Needs the per-stock 15m bars (lse_clean/{TK}_15m_2018-01-01_*.parquet, on the Mac) + xs/ext/wide_T.parquet + xs/ext/site_wide_daily.parquet.
Candidates: the live filter (S&P point-in-time, options ratio >= 1.5), 6 slots, 8% breaker, 2 bp a trade, stand-in daily path as xs/scr_slots.py (identical for all variants).
FEATURES at entry (the flip bar = the 15m bar before the entry bar; ATR = 14-day average daily high-low built from the 15m bars):
  SPD   move over the last 2 bars into the flip close, / ATR (speed into the flip - the NQ finding)
  RV15  flip-bar volume / the average volume of that same time-of-day bar over the prior 14 sessions ('stocks in play' relative volume)
PICK ORDER when flips arrive together: ratio (live) | SPD | RV15 | ratio + SPD | ratio + RV15 | ratio + SPD + RV15 (rank sums).
REPLACEMENT (live ratio order otherwise): when a new flip arrives and all 6 slots are full, sell one holding at that bar's open and take the new flip if
  R1  the weakest holding is losing and the new flip's ratio is higher than that holding's
  R2  a holding whose FIRST 15m bar after entry closed below its entry, and the new flip's ratio is higher (the lowest such holding)
  R3  the weakest holding is down more than 0.5 ATR, whatever the ratios
  RPOS (added Nov 9 after the rough hourly look, xs/scr_hp_swap.py) the new flip's stock is in POSITIVE gamma and the weakest holding is losing (18 names with GEX, to Dec 2025)
ALLOCATION (from Rey's attached review, Nov 9; frozen before running): RES6 hold the 6th slot through the first 15-minute interval, then give it to the
best-ratio flip still valid, at the 09:45 price | BATCH flips arriving 09:30-09:59 allocated together at 10:00 by ratio, at 10:00 prices | QUEUE a flip skipped
because slots were full waits (same session, before 14:00, while its own trade is still open) and takes the next freed slot at that moment's price.
CHOSEN on 2019-22, JUDGED on 2023-26 (Sep 25)."""
import glob, os, sys, numpy as np, pandas as pd
from collections import deque
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(rel):
    for base in (ROOT, '/mnt/user-data/uploads/fartpie'):
        if glob.glob(os.path.join(base, rel)): return os.path.join(base, rel)
    return os.path.join(ROOT, rel)
SYN = '--synthetic' in sys.argv
if SYN:                                                        # smoke test only: random bars and candidates
    rng = np.random.default_rng(0); D = pd.bdate_range('2019-01-02', '2026-09-25'); J = pd.DataFrame({'m': 0.0, 'SPY': rng.normal(4e-4, .01, len(D)), 'QQQ': rng.normal(5e-4, .013, len(D))}, index=D)
    TKS = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG', 'HHH']; BARS = {}
    for tk in TKS:
        t = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + 15 * k) for d in D for k in range(26)]); r = rng.normal(0, .004, len(t)); c = 100 * np.exp(np.cumsum(r))
        BARS[tk] = pd.DataFrame({'open': c * (1 + rng.normal(0, .001, len(t))), 'high': c * 1.002, 'low': c * .998, 'close': c, 'volume': rng.integers(1e4, 1e5, len(t)).astype(float)}, index=t)
    rows = []
    for d in D[::2]:
        for tk in rng.choice(TKS, 7, replace=False):
            e = d + pd.Timedelta(minutes=570 + 15 * int(rng.integers(0, 12))); x = e + pd.Timedelta(days=int(rng.integers(0, 4)), minutes=60)
            rows.append(dict(tk=tk, entry=e, exit=x, ret=rng.normal(.003, .03), ratio=rng.uniform(1.5, 6), h1=False, liq=True, sp=True))
    T0 = pd.DataFrame(rows)
else:
    J = pd.read_parquet(P('xs/ext/site_wide_daily.parquet')); D = J.index
    T0 = pd.read_parquet(P('xs/ext/wide_T.parquet')); T0 = T0[T0.sp & (T0.ratio >= 1.5) & (T0.exit < D[-1] + pd.Timedelta(days=1))].copy(); TKS = sorted(T0.tk.unique()); BARS = {}
    for tk in TKS:
        f = glob.glob(P(f'lse_clean/{tk}_15m_2018-01-01_*.parquet').replace('*', '') + '*') or glob.glob(os.path.join(ROOT, f'lse_clean/{tk}_15m_*.parquet')) or glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: print('  no bars', tk); continue
        H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        H = H[(H.dt.dt.hour * 60 + H.dt.dt.minute >= 570) & (H.dt.dt.hour * 60 + H.dt.dt.minute < 960)].drop_duplicates('dt').set_index('dt').sort_index()
        BARS[tk] = H[['open', 'high', 'low', 'close', 'volume']].astype(float)
T0 = T0.sort_values(['entry', 'ratio', 'tk'], ascending=[True, False, True], kind='stable').reset_index(drop=True)
FE = []
for tk, H in BARS.items():
    day = H.index.normalize(); dd = H.groupby(day).agg(h=('high', 'max'), l=('low', 'min')); atr = (dd.h - dd.l).rolling(14).mean().shift(1)
    tod = H.index.hour * 60 + H.index.minute; vavg = H.volume.groupby(tod).transform(lambda s_: s_.shift(1).rolling(14, min_periods=8).mean())
    A = pd.DataFrame({'o': H.open.values, 'c': H.close.values, 'v': H.volume.values, 'vavg': vavg.values, 'atr': day.map(atr).values, 'day': day}, index=H.index)
    A['spd'] = (A.c.shift(1) - A.c.shift(3)) / A.atr; A['rv15'] = A.v.shift(1) / A.vavg.shift(1)
    A['b1'] = (A.c - A.o) / A.atr; A.loc[A.day != A.day.shift(1), ['spd', 'rv15']] = np.nan; A.loc[A.day != A.day.shift(2), 'spd'] = np.nan
    FE.append(A.assign(tk=tk)[['tk', 'o', 'spd', 'rv15', 'b1', 'atr']])
FE = pd.concat(FE).reset_index().rename(columns={'index': 'entry', 'dt': 'entry'}); FE = FE.set_index(['tk', 'entry'])
T0 = T0.join(FE, on=['tk', 'entry']); T0['atrp'] = T0.atr / T0.o
GXF = glob.glob(os.path.join(os.path.dirname(P('xs/ext/wide_T.parquet')), 'gex', '*.csv')); T0['gx'] = np.nan
if GXF and not SYN:                                            # per-stock net GEX (18 names); prior close -> next session
    _g = []
    for f in GXF:
        g = pd.read_csv(f, parse_dates=['date']).sort_values('date'); g['tk'] = os.path.basename(f)[:-4]; g['nxt'] = g.date.shift(-1); _g.append(g[['tk', 'nxt', 'net_gex']].dropna(subset=['nxt']))
    _g = pd.concat(_g).set_index(['tk', 'nxt']).net_gex; T0['gx'] = [_g.get((t, d), np.nan) for t, d in zip(T0.tk, T0.entry.dt.normalize())]
print(f'candidates {len(T0):,}, with bars {T0.o.notna().mean():.0%}, SPD {T0.spd.notna().mean():.0%}, RV15 {T0.rv15.notna().mean():.0%}', flush=True)
def price_at(tk, t):
    H = BARS.get(tk)
    if H is None: return np.nan
    k = H.index.searchsorted(t, 'left')
    if k < len(H) and H.index[k] == t: return H.open.iloc[k]
    return H.close.iloc[k - 1] if k > 0 else np.nan
def taken(X, N=6, brk=.08, rule=None):
    """returns list of (tk, entry, exit, ret) incl. replacements; X sorted by pick order"""
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
                if rule == 'R2' and h['b1'] < 0 and e > h['e'] + pd.Timedelta(minutes=15) and r.ratio > h['ratio']: cand.append((cur, h))
                if rule == 'R3' and np.isfinite(h['atrp']) and cur < -.5 * h['atrp']: cand.append((cur, h))
                if rule == 'RPOS' and np.isfinite(r.gx) and r.gx > 0 and cur < 0: cand.append((cur, h))
            if cand:
                cur, h = min(cand, key=lambda z: z[0]); op.remove(h); pend = [p for p in pend if p[2] is not h]; nrep += 1
                rr = (1 + cur) * (1 - 2e-4) - 1; out[h['k']] = (h['tk'], h['e'], e, rr); real += np.log1p(rr / N); hist.append((e, real)); win.append((e, real))
        if len(op) >= N: continue
        if brk and hist:
            t0 = e - pd.Timedelta(days=28)
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [hh for _, hh in win]) < np.log1p(-brk): continue
        h = dict(tk=r.tk, e=e, x=r.exit, ratio=r.ratio, px=r.o, b1=r.b1 if np.isfinite(r.b1) else 0.0, atrp=r.atrp, k=len(out))
        op.append(h); pend.append((r.exit, r.ret, h)); out.append((r.tk, e, r.exit, r.ret))
    return out, nrep
def daily(TK, N=6):
    R = np.zeros(len(D)); W = np.zeros(len(D))
    for tk, e, x, r in TK:
        a, b = D.searchsorted(pd.Timestamp(e).normalize()), D.searchsorted(pd.Timestamp(x).normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / N; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w; W[a:b + 1] += w
    return pd.Series(R * np.where(W > 1, 1 / np.maximum(W, 1e-9), 1.0), index=D)
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25
        out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    e = (1 + s).cumprod(); return ' | '.join(out) + f' | all {e.iloc[-1] ** (1 / ((s.index[-1] - s.index[0]).days / 365.25)) - 1:6.1%}'
def cagr_a(s): z = s[s.index < '2023-01-01']; return (1 + z).prod()
def order(kind):
    X = T0.copy(); g = X.groupby('entry'); rk = lambda f, asc=False: g[f].rank(ascending=asc, na_option='bottom')
    X['_k'] = {'ratio': -X.ratio, 'spd': rk('spd'), 'rv15': rk('rv15'), 'ratio+spd': rk('ratio') + rk('spd'), 'ratio+rv15': rk('ratio') + rk('rv15'),
               'ratio+spd+rv15': rk('ratio') + rk('spd') + rk('rv15')}[kind]
    return X.sort_values(['entry', '_k', 'tk'], kind='stable')
import heapq
def delayed(r, t):
    """the same signal entered later at time t (real later price); None if it already exited or no price"""
    if t >= r.exit or not np.isfinite(r.o): return None
    p = price_at(r.tk, t); X = r.o * (1 + r.ret)
    if not np.isfinite(p) or p <= 0: return None
    return dict(tk=r.tk, entry=t, exit=r.exit, ret=X / p - 1, ratio=r.ratio)
def taken_alloc(mode, N=6, brk=.08, qmax_min=840):
    """mode: 'RES6' hold the 6th slot through the first 15-minute interval, then give it to the best-ratio flip still valid (later price) |
       'BATCH' flips arriving 09:30-09:59 are pooled and allocated together at 10:00 by ratio (later prices) |
       'QUEUE' a flip skipped because all slots were full waits (same session, until 14:00, while its own trade would still be open);
               when a slot frees, the best-ratio queued flip enters at that moment's price. No holding is ever sold early."""
    X = order('ratio'); ev = []; seq = 0
    for r in X.itertuples():
        tod = r.entry.hour * 60 + r.entry.minute
        if mode == 'BATCH' and 570 <= tod < 600:
            d = delayed(r, r.entry.normalize() + pd.Timedelta(minutes=600))
            if d: heapq.heappush(ev, (d['entry'], -d['ratio'], seq, 'c', d)); seq += 1
            continue
        heapq.heappush(ev, (r.entry, -r.ratio, seq, 'c', dict(tk=r.tk, entry=r.entry, exit=r.exit, ret=r.ret, ratio=r.ratio, _r=r))); seq += 1
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []; queue = []; held_res = {}; n_q = 0; n_res = 0
    def free_slots(): return N - len(op)
    def breaker_ok(e):
        nonlocal lvl0
        if not (brk and hist): return True
        t0 = e - pd.Timedelta(days=28)
        while win and win[0][0] < t0: lvl0 = win.popleft()[1]
        return real - max([lvl0] + [hh for _, hh in win]) >= np.log1p(-brk)
    def settle(upto):
        nonlocal real, op, pend
        done = sorted([p for p in pend if p[0] <= upto], key=lambda z: z[0]); pend = [p for p in pend if p[0] > upto]
        for xt, rr in done:
            real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
            if mode == 'QUEUE': heapq.heappush(ev, (xt, 0.0, -1, 'x', None))
        op = [x_ for x_ in op if x_ > upto]
    def take(c):
        op.append(c['exit']); pend.append((c['exit'], c['ret'])); out.append((c['tk'], c['entry'], c['exit'], c['ret']))
    while ev:
        e = ev[0][0]; grp = []
        while ev and ev[0][0] == e: grp.append(heapq.heappop(ev))
        settle(e)
        cands = sorted([g for g in grp if g[3] == 'c'], key=lambda g: (g[1], g[2]))
        if mode == 'QUEUE' and any(g[3] == 'x' for g in grp) and free_slots() > 0 and not cands:
            tod = e.hour * 60 + e.minute; queue = [q for q in queue if q['_r'].entry.normalize() == e.normalize() and tod < qmax_min and e < q['exit']]
            while queue and free_slots() > 0 and breaker_ok(e):
                q = max(queue, key=lambda z: z['ratio']); queue.remove(q); d = delayed(q['_r'], e)
                if d: take(d); n_q += 1
        tod = e.hour * 60 + e.minute; day = e.normalize()
        if mode == 'RES6' and tod == 585 and day in held_res:
            pool = [dict(c, _late=True) for c in held_res.pop(day)] + [g[4] for g in cands]
            best = sorted(pool, key=lambda z: -z['ratio'])
            for c in best:
                if free_slots() <= 0 or not breaker_ok(e): break
                if c.get('_late'):
                    d = delayed(c['_r'], e)
                    if d: take(d); n_res += 1
                    continue
                take(c)
            continue
        for g in cands:
            c = g[4]
            limit = N - 1 if (mode == 'RES6' and tod == 570) else N
            if len(op) >= limit:
                if mode == 'RES6' and tod == 570 and len(op) < N: held_res.setdefault(day, []).append(c); continue
                if mode == 'QUEUE' and '_r' in c and tod < qmax_min: queue.append(c)
                continue
            if not breaker_ok(e): break
            take(c)
    return out, dict(queued_entries=n_q, reserved_entries=n_res)
if __name__ == '__main__':
    print('\n=== pick order when flips arrive together | CAGR / max DD 2019-22 (CHOOSE) | 2023-26 (JUDGE) | all ===', flush=True); RES = {}
    for kind in ('ratio', 'spd', 'rv15', 'ratio+spd', 'ratio+rv15', 'ratio+spd+rv15'):
        tk_, _ = taken(order(kind)); s_ = daily(tk_); RES[kind] = s_; print(f'  {kind:<16} {st2(s_)}', flush=True)
    bk = max(RES, key=lambda k_: cagr_a(RES[k_])); print(f'  -> chosen on 2019-22: {bk}', flush=True)
    q = T0[T0.entry < '2023-01-01']
    for f in ('spd', 'rv15'):
        c1, c2 = q[f].quantile([1 / 3, 2 / 3]); b_ = np.where(T0[f] < c1, 'low', np.where(T0[f] < c2, 'mid', 'high'))
        print(f'  all candidates by {f} third (cuts {c1:.2f} / {c2:.2f}): ' + ' | '.join(f'{k}: {g_.ret.mean():+.2%} (win {np.mean(g_.ret > 0):.0%})' for k, g_ in T0.groupby(b_) if k != 'nan'), flush=True)
    print('\n=== replacing weak holdings with a new, higher-rated flip (live ratio order) | 2019-22 (CHOOSE) | 2023-26 (JUDGE) | all ===', flush=True)
    X0 = order('ratio'); RR = {}
    for rule in (None, 'R1', 'R2', 'R3', 'RPOS'):
        tk_, nrep = taken(X0, rule=rule); s_ = daily(tk_); RR[rule] = s_; print(f'  {str(rule or "no replacement (live)"):<22} {st2(s_)} | replacements {nrep:,}', flush=True)
    br = max(['R1', 'R2', 'R3'], key=lambda k_: cagr_a(RR[k_])); print(f'  -> chosen on 2019-22: {br} ({"beats" if cagr_a(RR[br]) > cagr_a(RR[None]) else "does not beat"} live there)', flush=True)
    if bk != 'ratio':
        tk_, nrep = taken(order(bk), rule=br); print(f'  combined: {bk} order + {br}: {st2(daily(tk_))}', flush=True)
    print('\n=== ALLOCATION (Rey review, added Nov 9): reserve the 6th slot / opening batch / valid-signal queue, real later prices | 2019-22 | 2023-26 | all ===', flush=True)
    print(f'  {"live":<28} {st2(RR[None])}', flush=True)
    for mode in ('RES6', 'BATCH', 'QUEUE'):
        tk_, inf = taken_alloc(mode); print(f'  {mode:<28} {st2(daily(tk_))} | {inf}', flush=True)
