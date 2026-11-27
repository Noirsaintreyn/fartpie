"""Report 505 (Nov 27 2026): Rey - 'before we take the 8% breaker off, how helpful is it in times like this?'
Live book clone (xs/hm_fill.py without the honorable-mention fills): large-cap flips (S&P, ratio>=1.5), 6 slots, ratio tie-break, no entries after 14:00, breaker = no new entries while realized book equity is > X% below its trailing 28-day high.
Compared: breaker OFF, 5%, 8% (live), 12%, 15%. By era + 2026. And the breaker episodes themselves: what the trades it BLOCKED went on to do (mean, win rate), what the book earned in the 20 trading days after each episode started.
Declared: the breaker helps only if the blocked trades are worse than the average trade (loss avoided) and max drawdown falls more than CAGR; the persistence result in report 504 (weak stretch -> better next flips) predicts it does not."""
import numpy as np, pandas as pd
from collections import deque
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); T = W[W.sp & (W.ratio >= 1.5)].copy(); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
T = T.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable').reset_index(drop=True)
def sim(a, b, brk=.08, n=6, cut=840):
    Z = T[(T.entry >= a) & (T.entry < b) & ((T.entry.dt.hour * 60 + T.entry.dt.minute) < cut)]; E, X, R = Z.entry.values, Z.exit.values, Z.ret.values
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = [0.0]; w = 1 / n; blocked = []; taken = []
    def realize(upto):
        nonlocal real, pend
        for xt, rr in sorted([p for p in pend if p[0] <= upto], key=lambda z: z[0]): real += np.log1p(rr * w); hist.append((xt, real)); win.append((xt, real))
        pend = [p for p in pend if p[0] > upto]
    def brk_on(t):
        if not brk or not hist: return False
        t0 = t - np.timedelta64(28, 'D')
        while win and win[0][0] < t0: lvl0[0] = win.popleft()[1]
        return real - max([lvl0[0]] + [h for _, h in win]) < np.log1p(-brk)
    for i in range(len(Z)):
        e = E[i]
        for x in sorted(x for x in op if x <= e): realize(x); op = [y for y in op if y > x]
        realize(e); op = [y for y in op if y > e]
        if brk_on(e): blocked.append((e, R[i], len(op) < n)); continue
        if len(op) >= n: continue
        op.append(X[i]); pend.append((X[i], R[i])); taken.append((e, R[i]))
    realize(np.datetime64('2100-01-01'))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); eq = np.exp(s.groupby(s.index.normalize()).last()); y = (eq.index[-1] - eq.index[0]).days / 365.25
    return dict(cagr=eq.iloc[-1] ** (1 / y) - 1, dd=(eq / eq.cummax() - 1).min(), n=len(taken), blocked=blocked, eq=eq, taken=taken)
P = {'2019-21': ('2019-01-01', '2022-01-01'), '2022-26': ('2022-01-01', '2027-01-01'), '2019-26': ('2019-01-01', '2027-01-01'), '2026 YTD': ('2026-01-01', '2027-01-01'), 'since Oct 13 26': ('2026-10-13', '2027-01-01')}
print(f"{'breaker':10s}" + ''.join(f'{k:>20s}' for k in P) + '      trades taken (all) / blocked'); R_ = {}
for brk in (0, .05, .08, .12, .15):
    row = {k: sim(np.datetime64(a), np.datetime64(b), brk=brk) for k, (a, b) in P.items()}; R_[brk] = row
    print(f"{('OFF' if not brk else f'{brk:.0%}'):10s}" + ''.join(f"{row[k]['cagr']:>+10.1%} /{row[k]['dd']:>6.1%}" for k in P) + f"      {row['2019-26']['n']} / {len(row['2019-26']['blocked'])}")
B = R_[.08]['2019-26']['blocked']; Bd = pd.DataFrame(B, columns=['t', 'r', 'slot_free']); Bd['t'] = pd.to_datetime(Bd.t); a = T[(T.entry < '2027-01-01') & ((T.entry.dt.hour * 60 + T.entry.dt.minute) < 840)].ret
print(f"\nTRADES THE 8% BREAKER BLOCKED (2019-26): {len(Bd)} signals ({Bd.slot_free.sum()} would have had a free slot); mean {Bd.r.mean() * 1e4:+.0f}bp, win {Bd.r.gt(0).mean():.0%} | all flips mean {a.mean() * 1e4:+.0f}bp, win {a.gt(0).mean():.0%}")
for e, (x, y_) in {'2019-21': ('2019', '2021'), '2022': ('2022', '2022'), '2023-26': ('2023', '2026')}.items():
    z = Bd[(Bd.t.dt.year >= int(x)) & (Bd.t.dt.year <= int(y_))]; za = T[(T.entry.dt.year >= int(x)) & (T.entry.dt.year <= int(y_))].ret
    print(f"   {e:8s} blocked n {len(z):4d} mean {z.r.mean() * 1e4:+6.0f}bp win {z.r.gt(0).mean():.0%} | all flips mean {za.mean() * 1e4:+6.0f}bp")
Bd['mo'] = Bd.t.dt.strftime('%Y-%m'); g = Bd.groupby('mo').agg(n=('r', 'size'), mean_bp=('r', lambda s: s.mean() * 1e4)); print('\nblocked trades by month where >= 20 were blocked:'); print(g[g.n >= 20].round(0).to_string())
# episodes: first blocked date of each cluster (gap > 14 days)
tt = Bd.t.sort_values().values; starts = [tt[0]] + [tt[i] for i in range(1, len(tt)) if tt[i] - tt[i - 1] > np.timedelta64(14, 'D')]
eq = R_[0]['2019-26']['eq']; print('\nBREAKER EPISODES (start, book return of the NEXT 20 trading days with the breaker OFF, and blocked-trade mean):'); rows = []
for s0 in starts:
    d0 = pd.Timestamp(s0).normalize(); w = eq[eq.index >= d0].iloc[:21]
    if len(w) < 21: continue
    z = Bd[(Bd.t >= d0) & (Bd.t < d0 + pd.Timedelta(days=30))]; rows.append((d0, w.iloc[-1] / w.iloc[0] - 1, z.r.mean() * 1e4, len(z))); print(f"  {d0:%Y-%m-%d}  next-20d book {w.iloc[-1] / w.iloc[0] - 1:+6.1%}  blocked in 30d: n {len(z):3d} mean {z.r.mean() * 1e4:+5.0f}bp")
rr = np.array([r[1] for r in rows]); print(f"\n{len(rows)} episodes: next-20d book return mean {rr.mean():+.1%}, positive in {np.mean(rr > 0):.0%}  (breaker ON means these trades were skipped)")
