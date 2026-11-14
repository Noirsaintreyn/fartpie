"""Nov 14 2026: does the path-efficiency state (ER20, the one feature that passed in report_42 at the 63d horizon) also sort the 15m vector flips the live screener trades? 9 LSE names, report_38 events. ER_d = |20d net move| / sum|daily moves| on daily closes ending the prior day; ER_i = same on the last 26 15m bars (about one day) ending at the flip bar. State = expanding (PIT) tercile over all earlier events (min 300). Declared: HIGH efficiency better. PASS = high > low in both eras on hold-to-bear, pooled |t| >= 2."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
s38 = open('/home/claude/fartpie/xs/report_38_mtf_flip_baskets.py').read(); exec(s38[:s38.index("S={'A 15m")])
ERd = {}; ERi = {}
for f in sorted(glob.glob('/home/claude/fartpie/lse_clean/*.parquet')):
    tk = f.split('/')[-1].split('.')[0]; x = load(f); dc = x.close.groupby(x.index.normalize()).last()
    er = ((dc - dc.shift(20)).abs() / dc.diff().abs().rolling(20).sum()).shift(1); ERd[tk] = er
    c = x.close; ERi[tk] = ((c - c.shift(26)).abs() / c.diff().abs().rolling(26).sum())
P2 = P.sort_values('t').copy()
P2['ERd'] = [ERd[k].get(pd.Timestamp(t).normalize(), np.nan) for k, t in zip(P2.tk, P2.t)]; P2['ERi'] = [ERi[k].get(t, np.nan) for k, t in zip(P2.tk, P2.t)]
def pit(s, minp=300):
    v = s.values; o = np.full(len(v), np.nan)
    for i in range(minp, len(v)):
        h = v[:i]; h = h[~np.isnan(h)]
        if len(h) >= minp and not np.isnan(v[i]): q1, q2 = np.quantile(h, [1 / 3, 2 / 3]); o[i] = 0 if v[i] <= q1 else 2 if v[i] >= q2 else 1
    return o
for c in ('ERd', 'ERi'): P2['T_' + c] = pit(P2[c])
print(f'events {len(P2)} with ERd state {P2.T_ERd.notna().sum()}')
for c, lab in (('T_ERd', 'daily ER20 (prior close)'), ('T_ERi', '15m ER over last 26 bars')):
    print(f' {lab}')
    for ex in ('hold', 'r16'):
        line = f'   exit {ex:4s}'
        for e, (a, b) in list(ERA.items()) + [('pooled', ('2018-01-01', '2027-01-01'))]:
            s = P2[(P2.t >= a) & (P2.t < b)].dropna(subset=[ex, c]); lo = s[s[c] == 0][ex]; hi = s[s[c] == 2][ex]; line += f' | {e}: low {lo.mean():+6.1f} high {hi.mean():+6.1f} (n {len(lo)}/{len(hi)}) t {tt(hi, lo):+.1f}'
        print(line)
