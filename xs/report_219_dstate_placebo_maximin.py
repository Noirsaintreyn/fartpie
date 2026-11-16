"""Report 219: (a) placebo for report 218's one PASS (1H flip while the daily vector is bull, held with the hybrid exit); (b) Maximin pivot on 1H flips. DECLARED before results.
(a) Is it the 1H FLIP that pays, or simply being long while the daily vector is bull? Placebo = random 1H bars (1 in 8) of the same names where the daily vector is bull at the prior daily close AND the 1H vector is already bull (not a flip bar), open < $5, same hybrid exit rule, net 100bp. If flips do not beat these, the edge is the daily state + exit, not the 1H trigger. Day-clustered Welch t on returns winsorized at the 99th pct, halves H1 2024-02..2025-06 | H2 2025-07..2026-11. Also the mirror: random 1H-bull bars while the daily vector is bear vs 1H flips in daily bear.
(b) Maximin pivot LB = highest(lowest(low,20),20) on 1H bars (Rey's Pine). Feature at the flip bar: close/LB - 1. Buckets: below, 0-3% above, > 3% above. Two-sided; outcome RHY and R1H, both halves. (Large caps: every Maximin filter lowered the book, xs/report_maximin_scr.log.)"""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
E='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SP=E[:-4]
P=pickle.load(open(E+'lowpx_1h_3y.pkl','rb')); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); S0=pd.Timestamp('2024-02-01'); MID=pd.Timestamp('2025-07-01'); rows=[]; rng=np.random.default_rng(9)
for tk,h in P.items():
    x=h.copy(); x.columns=['open','high','low','close','volume']; x.index=x.index.tz_convert('America/New_York').tz_localize(None); x=x[~x.index.duplicated()].dropna(); m=x.index.hour*60+x.index.minute; x=x[(m>=570)&(m<960)]
    if len(x)<1500 or tk not in D: continue
    d=D[tk]; d=d[~d.index.duplicated()]; d=d[d.index>='2022-06-01'][['open','high','low','close','volume']].dropna()
    r1=np.asarray(run_state_machine(x.assign(datetime=x.index))[0]); rD=pd.Series(np.asarray(run_state_machine(d.assign(datetime=d.index))[0]),index=d.index)
    day=x.index.normalize(); o,c=x.open.values,x.close.values; n=len(x); last=np.r_[day[1:]!=day[:-1],True]; rd=rD.reindex(day,method='ffill').values; rdp=rD.shift(1).reindex(day,method='ffill').values; LB=x.low.rolling(20).min().rolling(20).max().values
    def hy(e):
        conf=False; j=e
        while j<n-1:
            if last[j]:
                if rd[j]==1: conf=True
                elif conf and rd[j]==-1: return j+1
            if not conf and r1[j]==-1: return j+1
            j+=1
        return None
    def one(i,kind):
        e=i+1
        if e>=n-1 or x.index[e]<S0 or o[e]>=5: return
        xi=hy(e); x2=n-1 if xi is None else xi; p2=c[n-1] if xi is None else o[x2]; g=o[e+1:x2+1]/c[e:x2] if x2>e else np.array([1.])
        if len(g) and (g.max()>2 or g.min()<.5): return
        j=e
        while j<n and r1[j]!=-1: j+=1
        p1=c[n-1] if j>=n-1 else o[min(j+1,n-1)]
        rows.append(dict(kind=kind,entry=x.index[e],RHY=p2/o[e]-1-.01,R1H=p1/o[e]-1-.01,DSTATE=rdp[e],MM=c[i]/LB[i]-1 if LB[i]==LB[i] else np.nan))
    for i in range(141,n-2):
        if r1[i]==1 and r1[i-1]!=1: one(i,'flip')
        elif r1[i]==1 and r1[i-1]==1 and rng.random()<.125: one(i,'rand')
F=pd.DataFrame(rows); F['half']=np.where(F.entry<MID,'H1','H2')
def st(s,col='RHY'):
    r=s[col]; rs=r.sort_values(); return f'n{len(s):5d} mean {r.mean()*100:5.2f}% x3 {rs.iloc[:-3].mean()*100:5.2f}% med {r.median()*100:5.2f}% win {np.mean(r>0)*100:3.0f}% >=+30% {np.mean(r>=.3)*100:4.1f}%'
def tt(a,b,col='RHY'):
    cap=np.quantile(np.r_[a[col].values,b[col].values],.99); A=a.assign(w=np.minimum(a[col],cap)).groupby(a.entry.dt.normalize()).w.mean(); B=b.assign(w=np.minimum(b[col],cap)).groupby(b.entry.dt.normalize()).w.mean(); return (A.mean()-B.mean())/np.sqrt(A.var(ddof=1)/len(A)+B.var(ddof=1)/len(B))
print('(a) hybrid-exit return: 1H FLIP bars vs random 1H-bull bars, by daily state')
for ds,nm in ((1,'daily BULL'),(-1,'daily BEAR')):
    for hf in ('H1','H2'):
        a=F[(F.kind=='flip')&(F.DSTATE==ds)&(F.half==hf)]; b=F[(F.kind=='rand')&(F.DSTATE==ds)&(F.half==hf)]; print(f'  {nm} {hf}: flips {st(a)} | random {st(b)} | t(flip-random) {tt(a,b):+.1f}')
print('\n(b) Maximin pivot (1H, L=20) at the flip bar')
fl=F[F.kind=='flip'].dropna(subset=['MM'])
for col in ('RHY','R1H'):
    for hf in ('H1','H2'):
        s=fl[fl.half==hf]; a=s[s.MM<0]; b=s[(s.MM>=0)&(s.MM<.03)]; c_=s[s.MM>=.03]; print(f'  {col} {hf}: below pivot {st(a,col)} | 0-3% above {st(b,col)} | >3% above {st(c_,col)} | t(above-below) {tt(s[s.MM>=0],a,col):+.1f}')
