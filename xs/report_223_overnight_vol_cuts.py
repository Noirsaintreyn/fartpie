"""Report 223: can overnight inventory or volatility state cut the bottom of the 1H sub-$5 flips? DECLARED before results.
Sample: report 221's 7,260 1H flips; outcomes RHY (1H entry, daily hold) and R1H, net 100bp; halves H1 2024-02..2025-06 | H2 2025-07..2026-11; cuts = H1 terciles.
Features known at the flip bar's close: ONG = overnight gap of the flip bar's day (first RTH open / prior RTH close - 1); two-sided (gap-and-go vs chase). ONSH = overnight share of the last 10 sessions' move = sum of overnight returns - sum of intraday returns (10 days ending at the flip bar); declared sign -: advances built intraday beat advances built overnight (overnight/intraday tug-of-war). VR = ATR5 / ATR60 on daily bars at the prior close (volatility expanding vs compressed); two-sided (compression 'coil' failed on daily flips, report 203). ATRP bottom third = the low-volatility names (Rey's 'cut the bottom'): expected fewer +30% trades.
PASS = same sign both halves and day-clustered |t| >= 2.7 (8 tests) on top-minus-bottom tercile mean (winsorized 99th pct). Shown per group: share >= +30%, >= +100%, median, mean."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; E=SP+'ext/'
F=pd.read_pickle(SP+'flips1h_feats221.pkl'); P=pickle.load(open(E+'lowpx_1h_3y.pkl','rb')); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); out={}
for tk,g in F.groupby('tk'):
    x=P[tk].copy(); x.columns=['open','high','low','close','volume']; x.index=x.index.tz_convert('America/New_York').tz_localize(None); x=x[~x.index.duplicated()].dropna(); m=x.index.hour*60+x.index.minute; x=x[(m>=570)&(m<960)]
    day=x.index.normalize(); dd=x.groupby(day).agg(o=('open','first'),c=('close','last')); dd['on']=dd.o/dd.c.shift(1)-1; dd['intra']=dd.c/dd.o-1; on10=dd.on.rolling(10).sum(); in9=dd.intra.shift(1).rolling(9).sum()
    d=D[tk]; d=d[~d.index.duplicated()]; pc=d.close.shift(1); tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1); vr=(tr.rolling(5).mean()/tr.rolling(60).mean()).shift(1)
    ix=x.index; c=x.close.values
    for fi,e in zip(g.index,g.entry):
        k=ix.searchsorted(e)-1
        if k<1: continue
        dy=ix[k].normalize()
        if dy not in dd.index: continue
        part=c[k]/dd.o[dy]-1; out[fi]=(dd.on[dy],on10[dy]-(in9[dy]+part),vr.get(dy,np.nan))
X=pd.DataFrame.from_dict(out,orient='index',columns=['ONG','ONSH','VR']); F=F.join(X)
def st(s,c): r=s[c]; return f'n{len(s):5d} >=+30% {np.mean(r>=.3)*100:5.1f}% >=+100% {np.mean(r>=1)*100:4.1f}% med {r.median()*100:5.1f}% mean {r.mean()*100:5.1f}%'
def tt(a,b,col):
    cap=np.quantile(np.r_[a[col].values,b[col].values],.99); A=a.assign(w=np.minimum(a[col],cap)).groupby(a.entry.dt.normalize()).w.mean(); B=b.assign(w=np.minimum(b[col],cap)).groupby(b.entry.dt.normalize()).w.mean(); return (A.mean()-B.mean())/np.sqrt(A.var(ddof=1)/len(A)+B.var(ddof=1)/len(B))
for col in ('RHY','R1H'):
    print('===',col)
    for f in ('ONG','ONSH','VR','ATRP'):
        q=F[F.half=='H1'][f].quantile([1/3,2/3]).values
        for hf in ('H1','H2'):
            s=F[(F.half==hf)&F[f].notna()]; lo=s[s[f]<=q[0]]; mi=s[(s[f]>q[0])&(s[f]<q[1])]; hi=s[s[f]>=q[1]]
            print(f'  {f:5s}{hf}: LOW {st(lo,col)} | MID {st(mi,col)} | HIGH {st(hi,col)} | t(high-low) {tt(hi,lo,col):+.1f}')
print('H1 tercile cuts:',{f:F[F.half=="H1"][f].quantile([1/3,2/3]).round(4).tolist() for f in ('ONG','ONSH','VR','ATRP')})
F.to_pickle(SP+'flips1h_feats223.pkl')
