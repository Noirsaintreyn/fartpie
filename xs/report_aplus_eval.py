"""Eval sim on P-selected setups with size/bracket grid. Starts every 2nd trading day; MLL 2000 trailing (lock +2100), day stop 1000 (n*S*20<=1000), consistency 50%, target 3000, cap 4 NQ. Max 120 calendar days."""
import pickle,numpy as np,pandas as pd, itertools
exec(open('report_aplus.py').read().split("r=r.dropna")[0].split("def br")[0])  # loads r,b,G
exec("def br"+open('report_aplus.py').read().split("def br")[1].split("r=r.dropna")[0])
r=r.dropna(subset=['P']).copy(); r['dd']=pd.to_datetime(r.day)
r=r.sort_values(['dd','m']).drop_duplicates('dd',keep='first')   # one trade per day
days=pd.bdate_range('2017-06-01','2025-08-01')
def evalrun(sig,n,start):
    eq=pk=0.;mx=0.
    s=sig[(sig.dd>=start)&(sig.dd<=start+pd.Timedelta(days=120))]
    for d,o in zip(s.dd,s.o):
        v=o*n*20; eq+=v; pk=max(pk,eq); fl=min(pk-2000,100); mx=max(mx,v)
        if eq<=fl: return 'F',(d-start).days
        if eq>=3000 and mx<=.5*eq: return 'P',(d-start).days
    return 'R',120
res=[]
for T,S in ((20,20),(30,20),(40,20),(25,12),(40,25)):
    r['o']=[br(d,m,dr,T,S) for d,m,dr in zip(r.day,r.m,r.dir)]; X=r.dropna(subset=['o'])
    for q in (0,.5,.75,.9):
        sig=X[X.P>=X.P.quantile(q)] if q else X
        for n in (1,2,3,4):
            if n*S*20>1000: continue
            R=[evalrun(sig,n,s) for s in days[::2]]
            p=np.mean([a=='P' for a,_ in R]); f=np.mean([a=='F' for a,_ in R])
            dp=np.array([d for a,d in R if a=='P'])
            sh=lambda D:np.mean([a=='P' and d<=D for a,d in R])
            mo=np.mean([min(d,120) for a,d in R])/30.4
            res.append((T,S,q,n,len(sig),p,f,sh(30),sh(60),sh(90),mo/max(p,1e-9)))
D=pd.DataFrame(res,columns=['T','S','q','n','nsig','pass','fail','p30','p60','p90','mo_to_funded'])
pd.set_option('display.width',200); print(D.round(2).sort_values('mo_to_funded').head(25).to_string(index=False))
print('--- fail<=20%'); print(D[D.fail<=.2].round(2).sort_values('mo_to_funded').head(10).to_string(index=False))
print('reference current N4xECON K1.5: pass 87% fail 13% p30/60/90 22/58/73, fee-months 3.06, ~3.5 mo to funded')
