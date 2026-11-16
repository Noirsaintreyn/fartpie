"""Report 220: do the market-regime models we already have separate good and bad periods for 1H sub-$5 flips? DECLARED before results.
Sample: report 218's 7,260 1H flips (net 100bp), outcome RHY (hybrid exit) and R1H. Regime variables from market_state.pkl (wide universe, daily, value at the PRIOR daily close): BREADTH = share of names with the daily vector bull (sign +); THRUST = 5-day change in breadth (+); IDXREG = vector state of the equal-weight index (bull +); IDXDD = index drawdown from its high (deeper = worse, sign + on the value). Cuts = H1 terciles (IDXREG binary). PASS = same sign in H1 (2024-02..2025-06) and H2 (2025-07..2026-11) with day-clustered |t| >= 2.7 (Bonferroni 8) on top-minus-bottom, returns winsorized at the 99th pct. Earlier daily-flip tests of breadth / market health failed (reports 116, 123/124); this is the 1H, three-year, well-powered version."""
import numpy as np, pandas as pd
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; F=pd.read_pickle(SP+'flips1h_feats218.pkl'); M=pd.read_pickle(SP+'ext/market_state.pkl'); M['THRUST']=M.breadth.diff(5); M=M.shift(1)
d=F.entry.dt.normalize(); F['BREADTH']=d.map(M.breadth); F['THRUST']=d.map(M.THRUST); F['IDXREG']=d.map(M.idx_reg); F['IDXDD']=d.map(M.idx_dd)
def st(s,col): r=s[col]; return f'n{len(s):5d} mean {r.mean()*100:5.2f}% x3 {r.sort_values().iloc[:-3].mean()*100:5.2f}% win {np.mean(r>0)*100:3.0f}% >=+30% {np.mean(r>=.3)*100:4.1f}%'
def tt(a,b,col):
    cap=np.quantile(np.r_[a[col].values,b[col].values],.99); A=a.assign(w=np.minimum(a[col],cap)).groupby(a.entry.dt.normalize()).w.mean(); B=b.assign(w=np.minimum(b[col],cap)).groupby(b.entry.dt.normalize()).w.mean(); return (A.mean()-B.mean())/np.sqrt(A.var(ddof=1)/len(A)+B.var(ddof=1)/len(B))
for col in ('RHY','R1H'):
    print(f'=== {col}')
    for f in ('BREADTH','THRUST','IDXDD'):
        q=F[F.half=='H1'][f].quantile([1/3,2/3]).values
        for hf in ('H1','H2'):
            s=F[(F.half==hf)&F[f].notna()]; lo=s[s[f]<=q[0]]; hi=s[s[f]>=q[1]]; print(f'  {f:8s}{hf}: low {st(lo,col)} | high {st(hi,col)} | t {tt(hi,lo,col):+.1f}')
    for hf in ('H1','H2'):
        s=F[F.half==hf]; a=s[s.IDXREG==1]; b=s[s.IDXREG==-1]; print(f'  IDXREG  {hf}: index bear {st(b,col)} | index bull {st(a,col)} | t {tt(a,b,col):+.1f}')
m=F.groupby(F.entry.dt.to_period('Q')).agg(n=('RHY','size'),rhy=('RHY','mean'),r1h=('R1H','mean'),breadth=('BREADTH','mean')); print((m.assign(rhy=(m.rhy*100).round(1),r1h=(m.r1h*100).round(1),breadth=m.breadth.round(2))).to_string())
