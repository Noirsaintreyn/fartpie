"""Nov 9 2026: RETURNS WITH VP + VWAP TOGETHER IN THE 6-SLOT BOOK (Rey: 'so what's returns with VP and VWAP'). DECLARED BEFORE RESULTS. Mac (15m bars).
Features from xs/chop_trend.py (ECOMP entry vs the 5-session composite value, MIG value migration) and xs/vwap_flip.py (PVWD entry vs yesterday's VWAP / ATR,
VREC below VWAP / fresh reclaim / held above); cut points from 2019-22 only.
RULES:  F1 skip the worst combination only: inside-or-below 5-day value AND PVWD in its low third
        F2 take a flip only if it is above 5-day value OR PVWD is in its top third
        F3 the full checklist: above 5-day value AND PVWD top third AND not below today's VWAP
        O1 pick order ratio + a VP/VWAP score (PVWD rank + ECOMP above 2 / inside 1 / below 0 + VREC held 2 / reclaim 1 / below 0)
        O2 pick order ratio + MIG (low first) + VREC (the two order rules that held up on their own)
Chosen on 2019-22 (CAGR), judged on 2023-26 against live. Also: trades taken and average slot occupancy."""
import os, numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
_c = open(os.path.join(_HERE, 'vwap_flip.py')).read(); exec(_c[:_c.index("\nif __name__ == '__main__':")].replace("os.path.dirname(os.path.abspath(__file__))", repr(_HERE)), globals())
A_ = T.per == 'A'; q1, q2 = T.loc[A_, 'PVWD'].quantile([1 / 3, 2 / 3])
ab5 = T.ECOMP == 'above'; pv_hi = T.PVWD >= q2; pv_lo = T.PVWD < q1; notbelow = T.VREC.fillna('n/a') != 'below VWAP'
RULE = {'F1 skip inside/below value & PVWD low': T[~(~ab5 & pv_lo)], 'F2 above value OR PVWD top third': T[ab5 | pv_hi],
        'F3 full checklist': T[ab5 & pv_hi & notbelow]}
gr = T.groupby('entry'); rr = gr.ratio.rank(ascending=False)
score = T.PVWD.rank(pct=True).fillna(.5) + T.ECOMP.map({'above': 2, 'inside': 1, 'below': 0}).fillna(1) + T.VREC.map({'held above': 2, 'fresh reclaim': 1, 'below VWAP': 0}).fillna(1)
O1 = T.assign(_k=rr + score.groupby(T.entry).rank(ascending=False)).sort_values(['entry', '_k', 'tk'], kind='stable')
mg = T.MIG.rank(pct=True).fillna(.5); vr = T.VREC.map({'held above': 2, 'fresh reclaim': 1, 'below VWAP': 0}).fillna(1)
O2 = T.assign(_k=rr + mg.groupby(T.entry).rank(ascending=True) + vr.groupby(T.entry).rank(ascending=False)).sort_values(['entry', '_k', 'tk'], kind='stable')
RULE['O1 order ratio + VP/VWAP score'] = O1; RULE['O2 order ratio + MIG + VREC'] = O2
def occ(TK):
    w = np.zeros(len(D))
    for tk, e, x, r in TK:
        a, b = D.searchsorted(pd.Timestamp(e).normalize()), min(D.searchsorted(pd.Timestamp(x).normalize()), len(D) - 1); w[a:b + 1] += 1
    return np.minimum(w, 6).mean() / 6
if __name__ == '__main__':
    print(f'candidates {len(T):,}; PVWD thirds cut at {q1:.2f} / {q2:.2f} ATR (2019-22) | eligible: ' + ', '.join(f'{k.split()[0]} {len(v):,}' for k, v in RULE.items() if k[0] == 'F'), flush=True)
    print('\n=== the 6-slot book: CAGR / max DD 2019-22 (CHOOSE) | 2023-26 (JUDGE) | trades, slot use ===', flush=True)
    tl = taken(T); live = daily(tl); print(f'  {"live (ratio order, no filter)":<40} {st2(live)} | {len(tl):,} trades, slots used {occ(tl):.0%}', flush=True); RES = {}
    for k, X in RULE.items():
        tk_ = taken(X); RES[k] = daily(tk_)
        w = pd.DataFrame(tk_, columns=['tk', 'e', 'x', 'r']); w['A'] = w.e < '2023-01-01'
        print(f'  {k:<40} {st2(RES[k])} | {len(tk_):,} trades, slots used {occ(tk_):.0%} | win {np.mean(w.r > 0):.0%}, avg {w.r.mean():+.2%}', flush=True)
    ca = {k: cagr(s, '2019-01-01', '2023-01-01')[0] for k, s in RES.items()}; best = max(RES, key=lambda k_: ca[k_])
    jb, cb = cagr(RES[best], '2023-01-01', '2027-01-01')[0], cagr(live, '2023-01-01', '2027-01-01')[0]
    print(f'  -> chosen on 2019-22: {best} -> 2023-26 {jb:.1%} vs live {cb:.1%} ({"beats" if jb > cb else "does not beat"} live)', flush=True)
    yr = lambda s: (1 + s).groupby(s.index.year).prod() - 1; yl = yr(live)
    for k in RES: yb = yr(RES[k]); print(f'  by year {k[:2]} vs live: ' + ' | '.join(f'{y} {yb[y]:+.0%}/{yl[y]:+.0%}' for y in yl.index), flush=True)
