"""Nov 24 2026 report 419. Rey: intraday - does negative-gamma vector flips over high option-vol stocks work?
DECLARED: live book's gated flips (ratio >= 1.5 = high option volume) for the 27 names with gex files.  GEX known at the prior day's close: NEG = net_gex < 0; LOW = own rolling-252d percentile rank of net_gex <= .25 (PIT, shifted 1d); HIGH = >= .75.  Mechanism: dealers short gamma amplify moves, so a bullish flip should continue more.  VOL3 = flip's ratio in top tercile of the gated set (more option activity).  Outcomes: (a) vector exit (live) and (b) same-session exit at the 15:45 bar close (day-trade version, a session boundary not a holding-period parameter), net 4bp.
PASS: NEG (or LOW) beats the other flips in win rate AND median under (a) in every era (18-19, 20-21, 22, 23-26), month-clustered |t| >= 2 pooled; same for (b).  Era-wise n shown; n<30 cells marked."""
import numpy as np, pandas as pd, os
_t = open('xs/report_402_five_ideas.py').read(); exec(_t[:_t.index("rows = []; TK0, s0")])
X = 'xs/ext/gex/'; G = {f[:-4]: pd.read_csv(X + f, parse_dates=['date']).set_index('date').sort_index().net_gex for f in os.listdir(X)}
B = B0[B0.tk.isin(G)].copy().reset_index(drop=True); B['entry'] = pd.to_datetime(B.entry); B['d0'] = B.entry.dt.normalize() - pd.Timedelta(days=1)
neg = []; pct = []
for tk, d0 in zip(B.tk, B.d0):
    g = G[tk]; pr = g.rolling(252, min_periods=126).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True).shift(1); gs = g.shift(1)
    neg.append(gs.reindex([d0], method='ffill').iloc[0] < 0 if d0 in gs.index or True else np.nan); pct.append(pr.reindex([d0], method='ffill').iloc[0])
B['NEG'] = neg; B['pct'] = pct; B = B[B.pct.notna()].copy(); B['LOW'] = B.pct <= .25; B['HIGH'] = B.pct >= .75; B['VOL3'] = B.ratio >= B.ratio.quantile(2 / 3)
B['a'] = B.ret - 4e-4
def sday(tk, e):
    ts = pd.Timestamp(e).normalize() + pd.Timedelta(hours=15, minutes=45)
    try: return px_at(tk, ts) / px_at(tk, e) - 1 - 4e-4
    except Exception: return np.nan
B['b'] = [sday(t, e) for t, e in zip(B.tk, B.entry)]
e_ = B.entry; B['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); B['mo'] = e_.dt.to_period('M'); print('flips', len(B), 'names', B.tk.nunique(), '| NEG share', B.NEG.mean().round(3), '| LOW', B.LOW.mean().round(3))
ARM = {'ALL': lambda d: d.NEG | ~d.NEG, 'NEG': lambda d: d.NEG, 'posGEX': lambda d: ~d.NEG, 'LOW': lambda d: d.LOW, 'HIGH': lambda d: d.HIGH, 'NEG&VOL3': lambda d: d.NEG & d.VOL3, 'LOW&VOL3': lambda d: d.LOW & d.VOL3}
for c, lab in (('a', '(a) vector exit'), ('b', '(b) same-session 15:45 exit')):
    D = B[B[c].notna()]; allm = D.groupby('mo')[c].mean(); print(f'\n=== {lab} === arm n | win% med% mean% | t vs ALL | era win/med 18-19 | 20-21 | 22 | 23-26')
    for nm, f in ARM.items():
        d = D[f(D)]
        if len(d) < 60: print(nm, len(d)); continue
        m = (d.groupby('mo')[c].mean() - allm).dropna(); t = m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 3 and m.std() > 0 else np.nan; s = f"{nm:9s}{len(d):5d} | {(d[c]>0).mean()*100:5.1f} {d[c].median()*100:5.2f} {d[c].mean()*100:5.2f} | {t:+.1f} |"
        for era in ('18-19', '20-21', '22', '23-26'): x = d[d.era == era]; s += f" {(x[c]>0).mean()*100:4.1f}/{x[c].median()*100:5.2f}(n{len(x)}) |" if len(x) >= 30 else ' n<30 |'
        print(s)
