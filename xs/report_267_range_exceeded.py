"""Report 267: RANGE EXCEEDED (X = range so far / R^_t >= 1, the day already travelled further than the running model's expected full-day range). DECLARED before results; follows reports 265-266 (fading extremes loses most when X >= 1). Hypothesis: X >= 1 marks a trend/continuation day, so the CONTINUATION of the push wins and the rest of the day is more efficient.
Event = first 5m bar (10:00-14:00) where the session range through that bar >= thr x R^_t (R^_t = latest running forecast, report 266), thr in {1.0, 1.25}; direction = the side whose new extreme pushed the range over the line; entry next 5m open; stop = 0.25 x R^_t against; targets 1R, 2R, and hold to 15:55; cost 1 pt. Placebo: first bar where X >= 0.8 (not yet exceeded). Also: rest-of-day efficiency after the event vs after the same clock time on non-event days; main algo pts/day (deploy, shareable250 daily) on days flagged exceeded by 12:00. Train 2019-21 | test 2018+2022-26."""
import numpy as np, pandas as pd, pickle
src=open('/home/claude/fartpie/xs/report_266_running_zones.py').read(); exec(src.split("C['eh']=")[0])
SPd=SP; out=[]; days_flag=[]
for d,(o,h,l,c,v,mm,hodv,lodv) in Dd.items():
    if not any((d,ck) in Rt for ck in CK): continue
    done={}
    for i in range(7,len(o)-2):
        if mm[i]<600 or mm[i]>=840: continue
        ck=(mm[i]//30)*30; Rh=Rt.get((d,ck))
        if Rh is None: continue
        X_=(hodv[i]-lodv[i])/Rh
        for thr in (0.8,1.0,1.25):
            if thr in done or X_<thr: continue
            done[thr]=True
            up=h[i]>=hodv[i]-1e-9 and (hodv[i]>hodv[i-1]) 
            dn=l[i]<=lodv[i]+1e-9 and (lodv[i]<lodv[i-1])
            if up==dn: continue    # ambiguous (both or neither)
            dr=1 if up else -1; e=o[i+1]; Rs=.25*Rh; stop=e-dr*Rs
            res={}
            for kk,t_ in (('r1',e+dr*Rs),('r2',e+dr*2*Rs),('cl',None)):
                r_=None
                for j in range(i+1,len(o)):
                    hs=(l[j]<=stop) if dr==1 else (h[j]>=stop)
                    if hs: r_=-Rs;break
                    if t_ is not None and ((h[j]>=t_) if dr==1 else (l[j]<=t_)): r_=abs(t_-e);break
                    if mm[j]>=955: r_=dr*(c[j]-e);break
                if r_ is None: r_=dr*(c[-1]-e)
                res[kk]=r_-1
            ra=abs(c[-1]-c[i])/max(max(h[i+1:])-min(l[i+1:]),1e-9)
            out.append((d,thr,dr,Rs,res['r1'],res['r2'],res['cl'],ra,float(mm[i+1]),dr*(c[-1]-c[i])))
    days_flag.append((d,1.0 in done,0.8 in done))
T=pd.DataFrame(out,columns=['d','thr','dr','Rs','r1','r2','cl','erA','em','cont']); T['split']=np.where((T.d.dt.year>=2019)&(T.d.dt.year<=2021),'train','test')
def tt(x): x=x.dropna(); return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>2 and x.std()>0 else np.nan
print('median R^t-based stop',round(T.Rs.median()),'pts')
for thr in (0.8,1.0,1.25):
    g=T[T.thr==thr]; print(f'--- first bar with X >= {thr} ({"placebo" if thr==0.8 else "event"}): n {len(g)} ({len(g)/8:.0f}/yr)')
    for col,nm in (('r1','1R'),('r2','2R'),('cl','to close')):
        s=[]
        for sp in ('train','test'):
            q=g[g.split==sp][col]; s.append(f'n{len(q):4d} {q.mean():+6.1f} (t{tt(q):+.1f}) win {(q>0).mean():.0%}')
        print(f'    {nm:9s} {s[0]} | {s[1]}')
    print('    rest-of-day efficiency after the event (mean erA)',' | '.join(f'{sp} {x.erA.mean():.2f}' for sp,x in g.groupby('split')),'| continuation (pts from the event bar close to close, in push direction) mean',' | '.join(f'{sp} {x.cont.mean():+.1f}' for sp,x in g.groupby('split')))
g=T[T.thr==1.0]; print('by entry time (X>=1.0, to close):',{('<11:30' if k else '>=11:30'):(int(len(x)),round(x.cl.mean(),1)) for k,x in g.groupby(g.em<690)})
print('X>=1.0 by year (to close):',g.groupby(g.d.dt.year).cl.agg(['count','mean']).round(1).T.to_dict())
# daily flag vs main algo
Z=pickle.load(open(SP+'shareable250.pkl','rb')); alg=(Z['daily']/20)
F=pd.DataFrame(days_flag,columns=['d','ex1','ex08']).set_index('d'); F['alg']=alg.reindex(F.index); F['split']=np.where((F.index.year>=2019)&(F.index.year<=2021),'train','test')
print('--- main algo NQ pts/day (deploy size) on days range exceeded the running forecast by 14:00 vs not')
for sp in ('train','test'):
    f=F[F.split==sp]; print(f'  {sp}: exceeded n{int(f.ex1.sum())} {f[f.ex1].alg.mean():+.1f} | not n{int((~f.ex1).sum())} {f[~f.ex1].alg.mean():+.1f}')
pickle.dump((T,F),open(SP+'exceed267.pkl','wb'))
