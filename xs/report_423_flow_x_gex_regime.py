"""Nov 24 2026 report 423. Rey: other flows say WHEN/HOW BIG, GEX says IF it is happening, vector triggers.
DECLARED (before running): FLOW = put-tilt composite = mean of own-name 252d PIT percentiles of gamma_net, vanna_net, vega_net (report 422; low = put-heavy volume) <= .33.  REGIME = net_gex (OI-based gex files) < 0 at the prior close (dealers short gamma).  Trigger = live gated 15m flip, vector exit, net 4bp.  Names = those with both greek and gex data.
PRIMARY: FLOW&NEG vs all other flips in the set: mean and median higher in train (2019-21) and test (2018+2022-26), >= 3 of 4 eras (n>=30), month-clustered |t| >= 2.  Descriptive: the 2x2 (FLOW x REGIME), size of winners (share of trades >= +3%, mean of top-10% trades), and FLOW&NEG vs FLOW&posGEX (does the regime add anything).  One primary test, so no Bonferroni."""
import numpy as np, pandas as pd, os
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; B = pd.read_parquet(SP + 'greek422.parquet')
X = 'xs/ext/gex/'; G = {f[:-4]: pd.read_csv(X + f, parse_dates=['date']).set_index('date').sort_index().net_gex.shift(1) for f in os.listdir(X)}
B = B[B.tk.isin(G)].copy(); B['NEG'] = [G[tk].reindex([d0], method='ffill').iloc[0] < 0 for tk, d0 in zip(B.tk, B.d0)]; B['FLOW'] = B[['gamma_net', 'vanna_net', 'vega_net']].mean(axis=1) <= .33
print('flips', len(B), 'names', B.tk.nunique(), '| FLOW share', B.FLOW.mean().round(2), '| NEG share', B.NEG.mean().round(2))
def st(d): return f"n{len(d):5d} win {(d.n>0).mean()*100:5.1f} med {d.n.median()*100:5.2f} mean {d.n.mean()*100:5.2f} | >=3% {(d.n>=.03).mean()*100:4.1f}% top10% mean {d.n[d.n >= d.n.quantile(.9)].mean()*100:5.1f}%"
print('\n2x2'); 
for fl in (True, False):
    for ng in (True, False): print(f"FLOW={int(fl)} NEG={int(ng)}  {st(B[(B.FLOW == fl) & (B.NEG == ng)])}")
def cmp(a, b, lab):
    m = (a.groupby('mo').n.mean() - b.groupby('mo').n.mean()).dropna(); t = m.mean() / (m.std(ddof=1) / np.sqrt(len(m))); s = f"{lab}: mean diff {(a.n.mean()-b.n.mean())*100:+.2f}pp (t {t:+.1f})"
    for h in ('train', 'test'): s += f" | {h} mean {(a[a.half==h].n.mean()-b[b.half==h].n.mean())*100:+.2f} med {(a[a.half==h].n.median()-b[b.half==h].n.median())*100:+.2f} (n {int((a.half==h).sum())})"
    s += ' | eras'; 
    for era in ('18-19', '20-21', '22', '23-26'): x = a[a.era == era]; y = b[b.era == era]; s += (' +' if x.n.mean() > y.n.mean() else ' -') if len(x) >= 30 else ' ?'
    print(s)
P = B[B.FLOW & B.NEG]; R = B[~(B.FLOW & B.NEG)]; print(); cmp(P, R, 'PRIMARY FLOW&NEG vs rest'); cmp(B[B.FLOW & B.NEG], B[B.FLOW & ~B.NEG], 'regime adds? FLOW&NEG vs FLOW&posGEX'); cmp(B[B.FLOW], B[~B.FLOW], 'FLOW alone vs not')
