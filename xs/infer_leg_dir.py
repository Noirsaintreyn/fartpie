"""Infer trade direction for legs without a 'dir' column (M1, R1, RULES) from the 5m price path: long if some later bar close ~ entry+pts(+cost) with min low - entry ~ mae, short if ~ entry-pts. Validated on RA/V15a (known dir). Output: ext/leg_dir.pkl {leg: Series indexed by (day,m)}"""
import pickle, numpy as np, pandas as pd
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').set_index('dt'); mm=b.index.hour*60+b.index.minute; b=b[(mm>=570)&(mm<=955)]
G={d:g for d,g in b.groupby(b.index.normalize())}
OUT,NAMES=pickle.load(open(S+'s6.pkl','rb')); res={}
def infer(day,m,pts,mae):
    g=G.get(day); 
    if g is None: return 0
    t=day+pd.Timedelta(minutes=int(m)); i=g.index.searchsorted(t)
    if i>=len(g): return 0
    E=g.open.iloc[i]; hi=g.high.values[i:]; lo=g.low.values[i:]; cl=g.close.values[i:]; ok=[]
    for dr in (1,-1):
        best=False
        for cost in (0.0,0.5,1.0):
            tgt=E+dr*(pts+cost)
            for q in range(len(cl)):
                adv=(lo[:q+1].min()-E) if dr>0 else (E-hi[:q+1].max())
                if abs(cl[q]-tgt)<=1.25 and abs(adv-mae)<=1.25: best=True; break
            if best: break
        ok.append(best)
    return 1 if ok[0] and not ok[1] else -1 if ok[1] and not ok[0] else 0
for (t,u,y),n in zip(OUT['V2'],NAMES):
    if n in ('ON+','ON-'): continue
    t=t[(pd.to_datetime(t.day)>='2018-02-01')]; d=pd.to_datetime(t.day).dt.normalize()
    dr=[infer(a,m_,p,e) for a,m_,p,e in zip(d,t.m,t.pts,t.mae)]; s=pd.Series(dr,index=pd.MultiIndex.from_arrays([d.values,t.m.values]))
    if 'dir' in t.columns: tr=t.dir.values; print(n,'validation vs known dir: match',np.mean(np.array(dr)==tr).round(3),'undetermined',np.mean(np.array(dr)==0).round(3))
    else: print(n,'inferred: long',(s==1).mean().round(2),'short',(s==-1).mean().round(2),'undetermined',(s==0).mean().round(2),'n',len(s))
    res[n]=s
pickle.dump(res,open(S+'ext/leg_dir.pkl','wb'))
