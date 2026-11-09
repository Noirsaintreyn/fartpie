"""Nov 9 2026: THE SCREENER'S SLOTS - WHERE IS RETURN LEFT ON THE TABLE? (Rey: 'the algo is naive with its slots'). DECLARED BEFORE RESULTS.
Trades: the live 6-slot S&P model's candidates (xs/ext/wide_T.parquet: point-in-time S&P 500, options ratio >= 1.5, sticky 1h stop), Jan 2019 - Sep 25 2026.
No per-stock bars here (they are on the Mac), so each trade's return is spread evenly (geometrically) over the trading days it was held - a stand-in for the daily
path, identical for every variant; checked against the real daily series first. 2 bp a trade. Idle money earns 0 unless a variant says otherwise.
Questions: how full are the slots? how good are the signals SKIPPED because 6 slots were full?
Variants: slot count 4 / 5 / 6 / 8 / 10 | idle slots held in QQQ or SPY (sold the moment a signal needs the slot) | breaker (8% on realized P&L over 28 days) off |
size by conviction: x clip(ratio / median ratio, 0.5, 1.5) of a slot, capped at 100% invested | combinations. CAGR, max drawdown, worst year, 2019-21 / 2022-26."""
import numpy as np, pandas as pd
from collections import deque
J = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/site_wide_daily.parquet'); D = J.index; END = D[-1] + pd.Timedelta(days=1)
T0 = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); T0 = T0[T0.sp & (T0.ratio >= 1.5) & (T0.exit < END)].copy()
T0 = T0.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); MED = T0.ratio.median()
def taken(N=6, brk=.08):
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []; skip = []
    for e, x, r, tk, ra in zip(T0.entry.values, T0.exit.values, T0.ret.values, T0.tk.values, T0.ratio.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= N: skip.append((tk, e, x, r, ra, 'full')); continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [hh for _, hh in win]) < np.log1p(-brk): skip.append((tk, e, x, r, ra, 'breaker')); continue
        op.append(x); pend.append((x, r)); out.append((tk, pd.Timestamp(e), pd.Timestamp(x), r, ra))
    return out, skip
DI = {d: i for i, d in enumerate(D)}
def daily(TK, N, idle=None, conv=False):
    R = np.zeros(len(D)); W = np.zeros(len(D))
    for tk, e, x, r, ra in TK:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = (1 / N) * (float(np.clip(ra / MED, .5, 1.5)) if conv else 1.0)
        R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w; W[a:b + 1] += w
    sc = np.where(W > 1, 1 / W, 1.0); R = R * sc; W = np.minimum(W, 1.0)
    if idle: R = R + (1 - W) * J[idle].values
    return pd.Series(R, index=D), W
def st(s):
    e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; yr = (1 + s).groupby(s.index.year).prod() - 1
    a, b = s[s.index < '2022-01-01'], s[s.index >= '2022-01-01']
    cg = lambda z: (1 + z).prod() ** (365.25 / max((z.index[-1] - z.index[0]).days, 1)) - 1
    return f'CAGR {e.iloc[-1] ** (1 / y) - 1:6.1%} | max DD {(e / e.cummax() - 1).min():6.1%} | worst yr {yr.min():6.1%} | 19-21 {cg(a):6.1%} / 22-26 {cg(b):6.1%} | $1k -> ${1000 * e.iloc[-1]:,.0f}'
if __name__ == '__main__':
    print(f'real daily series (bars):      {st(J.m)}')
    TK, SK = taken(); s0, W0 = daily(TK, 6); print(f'stand-in, 6 slots as live:     {st(s0)}')
    print(f'\nslots: average invested {W0.mean():.0%} | days fully invested {np.mean(W0 >= .999):.0%} | days with 0-2 slots used {np.mean(W0 <= 2 / 6 + 1e-9):.0%}')
    tk_r = np.array([t[3] for t in TK]); sf = np.array([s[3] for s in SK if s[5] == 'full']); sb = np.array([s[3] for s in SK if s[5] == 'breaker'])
    print(f'signals: taken {len(TK)} (avg {tk_r.mean():+.2%}, win {np.mean(tk_r > 0):.0%}) | skipped, slots full {len(sf)} (avg {sf.mean():+.2%}, win {np.mean(sf > 0):.0%}) | '
          f'skipped by the breaker {len(sb)} (avg {sb.mean() if len(sb) else np.nan:+.2%})', flush=True)
    print('\n=== variants ===')
    for lab, N, brk, idle, conv in (('6 slots (live)', 6, .08, None, False), ('4 slots', 4, .08, None, False), ('5 slots', 5, .08, None, False), ('8 slots', 8, .08, None, False),
                                    ('10 slots', 10, .08, None, False), ('6 slots, breaker off', 6, 0, None, False), ('6 slots, idle -> QQQ', 6, .08, 'QQQ', False),
                                    ('6 slots, idle -> SPY', 6, .08, 'SPY', False), ('6 slots, conviction size', 6, .08, None, True), ('8 slots, idle -> QQQ', 8, .08, 'QQQ', False),
                                    ('10 slots, idle -> QQQ', 10, .08, 'QQQ', False), ('6 slots, idle -> QQQ, conviction', 6, .08, 'QQQ', True)):
        tk_, _ = taken(N, brk); s_, w_ = daily(tk_, N, idle, conv); print(f'  {lab:<34} invested {w_.mean():4.0%} | {st(s_)}', flush=True)
    print(f'\n  QQQ: {st(J.QQQ)}\n  SPY: {st(J.SPY)}')
# --- follow-up (same day): does the pick order among signals that arrive together matter? and does the options ratio rank returns at all?
if __name__ == '__main__':
    print('\n=== pick order when several signals arrive at the same time (6 slots) ===')
    base_T0 = T0.copy(); same = T0.groupby('entry').tk.transform('size'); print(f'  signals arriving with others at the same bar: {np.mean(same > 1):.0%}')
    for lab, key in (('highest ratio first (live)', None), ('lowest ratio first', 'asc'), ('random 1', 1), ('random 2', 2), ('random 3', 3)):
        if key is None: T0 = base_T0
        elif key == 'asc': T0 = base_T0.sort_values(['entry', 'ratio'], ascending=[True, True], kind='stable')
        else: T0 = base_T0.assign(_r=np.random.default_rng(key).random(len(base_T0))).sort_values(['entry', '_r'], kind='stable')
        tk_, _ = taken(); s_, w_ = daily(tk_, 6); print(f'  {lab:<28} {st(s_)}', flush=True)
    T0 = base_T0; q = pd.qcut(T0.ratio, 5)
    print('  all candidates, avg trade return by options-ratio fifth: ' + ' | '.join(f'{iv.left:.2f}-{iv.right:.2f}: {g.ret.mean():+.2%}' for iv, g in T0.groupby(q, observed=True)))
# --- follow-up 2 (exploratory): put the 1h-confirmed signals (h1) first, then the highest ratio
if __name__ == '__main__':
    print('\n=== exploratory pick orders ===')
    for lab, srt in (('h1 first, then highest ratio', (['entry', 'h1', 'ratio'], [True, False, False])), ('highest ratio first, h1 breaks ties (h1 first)', (['entry', 'ratio', 'h1'], [True, False, False]))):
        T0 = base_T0.sort_values(srt[0], ascending=srt[1], kind='stable'); tk_, _ = taken(); s_, w_ = daily(tk_, 6); print(f'  {lab:<46} {st(s_)}', flush=True)
    T0 = base_T0
