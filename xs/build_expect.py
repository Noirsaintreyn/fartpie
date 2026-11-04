"""Nov 4 2026: backtest tables the morning report reads (live_data/expect_tables.json). Source: the 17,447 live-rule trades (S&P, ratio >= 1.5) tagged with
context at entry (xs/ctx_rank.py output). Every table is pooled over 2019-26 AND shows the two periods so the report can say when a cell is not stable.
Run after xs/ctx_rank.py: python3 xs/build_expect.py"""
import json, numpy as np, pandas as pd
X = pd.read_parquet('/tmp/claude-0/ml/ctx_rank_X.parquet'); U = X[X.ratio >= 1.5].copy()
U['days'] = (U.exit - U.entry).dt.total_seconds() / 86400; U['big'] = U.ret >= .15
def cell(z):
    if len(z) < 30: return None
    o = dict(n=int(len(z)), avg=round(float(z.ret.mean()) * 100, 2), win=round(float(z.ret.gt(0).mean()) * 100, 1), big=round(float(z.big.mean()) * 100, 2), med=round(float(z.ret.median()) * 100, 2))
    for p in ('19-21', '22-26'):
        q = z[z.per == p]; o[p] = round(float(q.ret.mean()) * 100, 2) if len(q) >= 30 else None
    return o
T = dict(all=cell(U), tracks={('1h' if h else '15m'): cell(g) for h, g in U.groupby('h1')},
         regime={k: cell(g) for k, g in U.groupby('regime')}, earn={str(k): cell(g) for k, g in U.groupby('earn_b', observed=True)},
         post={k: cell(g) for k, g in U.groupby('post_b')}, sector={k: cell(g) for k, g in U.groupby('sec')},
         vol={str(k): cell(g) for k, g in U.groupby('rvt', observed=True)}, volrel={str(k): cell(g) for k, g in U.groupby('rvrel_b', observed=True)},
         mom={k: cell(g) for k, g in U.groupby('mom')},
         ratio={k: cell(U[(U.ratio >= a) & (U.ratio < b)]) for k, (a, b) in {'1.5-2': (1.5, 2), '2-3': (2, 3), '3-5': (3, 5), '5+': (5, 1e9)}.items()})
T['age'] = {str(d): cell(U[U.days >= d]) for d in (0, 1, 2, 3, 5, 7, 10, 15, 20)}                   # trades still open after d calendar days (survivors)
T['age_track'] = {tr: {str(d): cell(g[g.days >= d]) for d in (0, 1, 3, 7, 15)} for tr, g in (('15m', U[~U.h1]), ('1h', U[U.h1]))}
w = U.sort_values('ret'); T['runners'] = dict(share_ge15=round(float(U.big.mean()) * 100, 2), share_of_pnl=round(float(w[w.big].ret.sum() / U.ret.clip(lower=0).sum()) * 100, 1), med_days_runner=round(float(U[U.big].days.median()), 1),
                                              med_days_all=round(float(U.days.median()), 1))
json.dump(T, open('live_data/expect_tables.json', 'w'), indent=1); print(json.dumps(T['all']), json.dumps(T['runners'])); print(json.dumps(T['age']['3']), json.dumps(T['age']['7']))
