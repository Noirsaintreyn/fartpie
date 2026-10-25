"""Oct 25 2026 batch (full-S&P screener, ratio >= 1.5, live rules, 6 slots, 8% breaker):
 IM  implied-move earnings guard: sell at the close before a report if the trade is up >= MIN but less than K x the name's typical
     earnings move (mean |gap| of its previous 8 reports, from the same bars) - the gain is small enough for one gap to erase
 FT  flip type at entry (daily closes, prior day): breakout = at a 20d closing high; reversal = below the 50d average;
     squeeze = 20d range < 1.5x its expected size; continuation = everything else. Avg trade by type, skip / tie-break by type
 SL  separate slot pools for the 15m and 1h tracks (each slot still 1/6)
 PB  pullback entries: instead of the open, a limit DIP below it, good for N bars; unfilled -> skip (or buy at market after N bars)
PART env picks one: IM, FT, SL, PB (default all)."""
from collections import deque
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('\nP = [(np.datetime64')[0])
exec('def sim3' + open('xs/equity_models.py').read().split('def sim3')[1].split('STRAT_MED =')[0]); STRAT_MED = 1
exec(open('xs/exit_lab2.py').read().split('BARS = {}')[1].split('VARS =')[0].replace('def bars', 'BARS = {}\ndef bars', 1))
T = T.sort_values('entry').reset_index(drop=True)
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z, sim=None, **kw):
    sim = sim or sim3; print(f'  {nm:<50}' + ''.join(f'{cc:>+9.1%} /{dd:>5.0%}' for cc, dd in (sim(Z, a, b, **kw) for a, b in P)), flush=True)
def head(t): print(f'\n{t:<52}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', T)
PART = os.getenv('PART', 'ALL')
TD = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet('etf_daily/SPY.parquet').datetime).dt.tz_localize(None).dt.normalize().unique()))
# ------------------------------------------------------------------ IM
if PART in ('ALL', 'IM'):
    EVS = {}
    for f in glob.glob('xs/ext/earn/*.csv'):
        tk = f.split('/')[-1][:-4]
        if tk.startswith('_'): continue
        try: e = pd.read_csv(f); t = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        except Exception: continue
        sp = pd.to_numeric(e['Surprise(%)'], errors='coerce')
        G = sorted({d if tt.hour < 12 else TD[min(TD.searchsorted(d, 'right'), len(TD) - 1)] for d, tt, s in zip(t.dt.normalize(), t, sp) if s == s})
        EVS[tk] = G
    GAPS = {}                                                                     # tk -> {G: |gap|}
    for tk, G in EVS.items():
        B = bars(tk)
        if B is None: continue
        d = B.dt.dt.normalize(); O = B.groupby(d).open.first(); C = B.groupby(d).close.last(); gg = {}
        for g in G:
            k = C.index.searchsorted(g)
            if 0 < k < len(C) and C.index[k] == g: gg[g] = abs(O.iloc[k] / C.iloc[k - 1] - 1)
        GAPS[tk] = gg
    VARS = [(mn, K) for mn in (.02, .03) for K in (1.0, 1.5, 2.0)] + [(.03, 99.0)]
    ALT = {v: (T.ret.values.copy(), T.exit.values.copy()) for v in VARS}; n_ev = 0; fired = {v: 0 for v in VARS}
    for tk, g in T.groupby('tk'):
        B = bars(tk); gg = GAPS.get(tk)
        if B is None or not gg: continue
        dt = B.dt.values; o, c = B.open.values, B.close.values; day = B.dt.dt.normalize().values; gs = set(np.datetime64(x) for x in gg)
        for i, e, x in zip(g.index, g.entry.values, g.exit.values):
            a = np.searchsorted(dt, e); b = np.searchsorted(dt, x)
            if a >= len(dt) or dt[a] != e or b >= len(dt) or b <= a: continue
            ends = [j for j in range(a, b) if day[j + 1] != day[j] and np.datetime64(day[j + 1]) in gs]
            if not ends: continue
            j = ends[0]; G0 = pd.Timestamp(day[j + 1]); prev = [v for k_, v in sorted(gg.items()) if k_ < G0][-8:]
            if len(prev) < 4: continue
            mv = float(np.mean(prev)); up = c[j] / o[a] - 1; base = o[b] / o[a] - 1; n_ev += 1
            for v in VARS:
                mn, K = v
                if up >= mn and up < K * mv:
                    r_, x_ = ALT[v]; r_[i] += up - base; x_[i] = dt[j]; fired[v] += 1
    head(f'IM implied-move guard ({n_ev} trades into reports)')
    for v in VARS:
        Z = T.copy(); Z['ret'], Z['exit'] = ALT[v]
        show(f'sell if up >= {v[0]:.0%} and < {v[1]:g}x typical move ({fired[v]})' if v[1] < 50 else f'sell if up >= {v[0]:.0%} (no move check) ({fired[v]})', Z)
# ------------------------------------------------------------------ FT
if PART in ('ALL', 'FT'):
    d = T.entry.dt.normalize()
    hi20 = (CL >= CL.rolling(20).max() * .995).shift(1); ma50 = (CL < CL.rolling(50).mean()).shift(1)
    rng = (CL.rolling(20).max() / CL.rolling(20).min() - 1); exp_ = CL.pct_change().rolling(60).std() * np.sqrt(20) * 2.5
    sq = (rng < 1.5 * exp_ / 2.5).shift(1)
    def look(M): return np.array([bool(M[t].asof(x)) if t in M.columns and M[t].asof(x) == M[t].asof(x) else False for t, x in zip(T.tk, d)])
    BO, RV, SQ = look(hi20), look(ma50), look(sq)
    T['ftype'] = np.where(BO, 'breakout', np.where(RV, 'reversal', np.where(SQ, 'squeeze', 'continuation')))
    T['per'] = np.where(T.entry.dt.year >= 2022, '22-26', '19-21')
    g = T.groupby(['ftype', 'per']).ret.agg(['mean', 'count']); g['mean'] *= 100
    print('\nFT avg trade % by flip type:'); print(g.unstack().round(2).to_string())
    head('FT flip type')
    for ty in ('breakout', 'reversal', 'squeeze', 'continuation'):
        show(f'skip {ty} flips', T[T.ftype != ty])
        T['k_ft'] = (T.ftype == ty).astype(float) * 10 + T.ratio / 100; show(f'tie-break: {ty} first', T, key='k_ft')
# ------------------------------------------------------------------ SL
if PART in ('ALL', 'SL'):
    def sim_pool(Z, a, b, n15=4, n1h=2, brk=.08):
        Z = Z[(Z.entry >= a) & (Z.entry < b)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
        op = {False: [], True: []}; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; w = 1 / 6
        for e, x, r_, h1 in zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.h1.values):
            done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
            for xt, rr in done: real += np.log1p(rr * w); hist.append((xt, real)); win.append((xt, real))
            for k in op: op[k] = [o for o in op[k] if o > e]
            if len(op[bool(h1)]) >= (n1h if h1 else n15): continue
            if brk and hist:
                t0 = e - np.timedelta64(28, 'D')
                while win and win[0][0] < t0: lvl0 = win.popleft()[1]
                if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
            op[bool(h1)].append(x); pend.append((x, r_))
        for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr * w); hist.append((xt, real))
        s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
        eq = np.exp(s.groupby(s.index.normalize()).last()); yy = (eq.index[-1] - eq.index[0]).days / 365.25
        return eq.iloc[-1] ** (1 / yy) - 1, (eq / eq.cummax() - 1).min()
    print('\nSL share of trades that are 1h-track:', f'{T.h1.mean():.0%}')
    head('SL separate slot pools (15m / 1h)')
    for a_, b_ in ((5, 1), (4, 2), (3, 3), (6, 1), (6, 2)): show(f'{a_} slots for 15m, {b_} for 1h' + (' (more than 6 = up to ' + f'{(a_ + b_) / 6:.0%} invested)' if a_ + b_ > 6 else ''), T, sim=sim_pool, n15=a_, n1h=b_)
# ------------------------------------------------------------------ PB
if PART in ('ALL', 'PB'):
    VARS = [(dp, n, fb) for dp in (.0025, .005, .01) for n in (2, 4, 8) for fb in ('skip', 'market')]
    ALT = {v: (T.ret.values.copy(), T.entry.values.copy(), np.ones(len(T), bool)) for v in VARS}
    for tk, g in T.groupby('tk'):
        B = bars(tk)
        if B is None: continue
        dt = B.dt.values; o, lo = B.open.values, B.low.values
        for i, e, x in zip(g.index, g.entry.values, g.exit.values):
            a = np.searchsorted(dt, e); b = np.searchsorted(dt, x)
            if a >= len(dt) or dt[a] != e or b >= len(dt) or b <= a: continue
            p0 = o[a]; px = o[b]
            for v in VARS:
                dp, n, fb = v; lim = p0 * (1 - dp); r_, e_, keep = ALT[v]
                hit = np.where(lo[a:min(a + n, b)] <= lim)[0]
                if len(hit): r_[i] = (1 + r_[i]) * (p0 / min(lim, o[a + hit[0]])) - 1; e_[i] = dt[a + hit[0]]
                elif fb == 'skip' or a + n >= b: keep[i] = False
                else: r_[i] = (1 + r_[i]) * (p0 / o[a + n]) - 1; e_[i] = dt[a + n]
    head('PB pullback entries')
    for v in VARS:
        r_, e_, keep = ALT[v]; Z = T.copy(); Z['ret'], Z['entry'] = r_, e_; Z = Z[keep]
        show(f'limit -{v[0]:.2%} for {v[1]} bars, else {v[2]} (fill {keep.mean():.0%})', Z)
