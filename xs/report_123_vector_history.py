"""report_123 (Nov 15 2026): features from the VECTOR'S OWN HISTORY for daily flips (all known at flip-bar close). Declared before results, signs from mechanism:
H1 SHAKEOUT [+]: the bull flip follows a SHORT bear leg (<=10 bars) = failed breakdown / stop run (Wyckoff spring done by the vector itself). Also bear-leg depth below the prior bull peak [-] (shallow = absorbed).
H2 FLIP HEALTH [+]: trailing 60d mean of fwd-10d returns of flips that have already MATURED (flip date <= t-14 calendar days), market-wide $1-5 universe; self-exciting regime: when recent flips are working, new ones work.
H3 BREADTH THRUST [+]: change in share of the universe in bull state over 10 bars (liquidity arriving) and level (lagged 1d).
H4 WHIPSAW COUNT [-]: number of flips in this name in the last 120 bars (choppy names fail).
Thresholds from TRAIN terciles only. Output: flips with features -> ext/vh_features.pkl."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D=pd.read_pickle(X+'wide_daily_ohlcv.pkl'); F=pd.read_pickle(X+'stack_flips.pkl'); COST=.003
regs={}; info={}
for tk,g in F.groupby('tk'):
    q=D.get(tk)
    if q is None or len(q)<300: continue
    q=q[['open','high','low','close','volume']].copy(); q=q[~q.index.duplicated()].dropna(); q.index=pd.to_datetime(q.index).tz_localize(None).normalize()
    if len(q)<300: continue
    reg,lev,atr,_=run_state_machine(q.assign(datetime=q.index)); reg=np.asarray(reg); regs[tk]=pd.Series(reg,index=q.index)
    o,h,l,c=[q[k].values.astype(float) for k in ('open','high','low','close')]; n=len(q); pos={d:i for i,d in enumerate(q.index)}
    fl=np.where((reg[1:]==1)&(reg[:-1]!=1))[0]+1
    for _,r in g.iterrows():
        i=pos.get(r.d)
        if i is None or i<130 or i+12>=n: continue
        k=i-1; 
        while k>0 and reg[k]!=1: k-=1          # k = last bar of previous bull leg
        bear_len=i-1-k; 
        j=k
        while j>0 and reg[j]==1: j-=1          # start of previous bull leg
        peak=h[j:k+1].max() if k>j else h[k]; trough=l[k+1:i].min() if i-1>k else l[k]
        depth=trough/peak-1
        info[(tk,r.d)]=dict(BEARLEN=bear_len,DEPTH=depth,WHIP=float(((fl>i-120)&(fl<=i)).sum()),FWD10=o[i+11]/o[i+1]-1)
I=pd.DataFrame.from_dict(info,orient='index'); I.index=pd.MultiIndex.from_tuples(I.index,names=['tk','d']); A=F.merge(I.reset_index(),on=['tk','d'])
R=pd.DataFrame(regs); br=((R==1).sum(axis=1)/R.notna().sum(axis=1)); A['BRD']=A.d.map(br.shift(1)); A['BRDCH']=A.d.map((br-br.shift(10)).shift(1))
# flip health: market-wide matured flips (all $1-5 flips), trailing 60d mean FWD10 of flips with date <= t-14d
S=A[(A.px>=1)&(A.px<5)][['d','FWD10']].sort_values('d'); S=S.groupby('d').FWD10.agg(['sum','count'])
idx=pd.date_range(A.d.min(),A.d.max()); S=S.reindex(idx).fillna(0)
sm=S['sum'].rolling(60).sum().shift(14); ct=S['count'].rolling(60).sum().shift(14); H=(sm/ct).where(ct>=15)
A['HEALTH']=A.d.map(H); A['n']=A.r-COST; A['mo']=A.d.dt.to_period('M').astype(str)
A.to_pickle(X+'vh_features.pkl'); print(len(A),A[['BEARLEN','DEPTH','WHIP','BRD','BRDCH','HEALTH']].describe().T.to_string())
