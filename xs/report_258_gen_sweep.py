"""Report 258 (generator): Asia-range sweep + reclaim trades (rules of report 257, Asia levels only) with entry minute and MAE, targets 1R and 2R, non-overlapping (one position at a time), for the Lucid sim."""
import numpy as np, pandas as pd, pickle
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index()
rth=b.between_time('09:30','15:55'); D=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'h':D.high.max(),'l':D.low.min(),'n':D.size()}); dd=dd[dd.n>=70]; dd['atr']=(dd.h-dd.l).rolling(14).mean().shift(1)
out={1:[],2:[]}
for d,g in D:
    if d not in dd.index or not np.isfinite(dd.atr[d]): continue
    asia=b.loc[d-pd.Timedelta(hours=4):d+pd.Timedelta(hours=2)-pd.Timedelta(minutes=5)]
    if len(asia)<30: continue
    AH,AL=asia.high.max(),asia.low.min(); pre=b.loc[d+pd.Timedelta(hours=2):d+pd.Timedelta(hours=9,minutes=29)]
    o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;t=g.index; mm=t.hour*60+t.minute
    sw_h=bool(len(pre) and pre.high.max()>AH); sw_l=bool(len(pre) and pre.low.min()<AL); mh=pre.high.max() if len(pre) else -np.inf; ml=pre.low.min() if len(pre) else np.inf
    done={'S':False,'L':False}; sig=[]
    for i in range(len(g)-2):
        if mm[i]>690: break
        if h[i]>AH: sw_h=True; mh=max(mh,h[i])
        if l[i]<AL: sw_l=True; ml=min(ml,l[i])
        if sw_h and not done['S'] and c[i]<AH: done['S']=True; sig.append((i,-1,mh+1))
        if sw_l and not done['L'] and c[i]>AL: done['L']=True; sig.append((i,1,ml-1))
    for k in (1,2):
        last=-1
        for i,dr,stop in sorted(sig):
            e=i+1
            if e<=last: continue
            en=o[e]; R=(stop-en) if dr==-1 else (en-stop)
            if not (10<=R<=150): continue
            res=None; mae=0.0; j=e
            for j in range(e,len(g)):
                adv=(h[j]-en) if dr==-1 else (en-l[j]); fav=(en-l[j]) if dr==-1 else (h[j]-en)
                if adv>=R: res=-R; mae=R; break
                if fav>=k*R: res=k*R; mae=max(mae,adv); break
                mae=max(mae,adv)
                if mm[j]>=955: res=dr*(c[j]-en); break
            if res is None: res=dr*(c[-1]-en); j=len(g)-1
            out[k].append(dict(day=d,m=float(mm[e]),dir=dr,pts=res-1.0,mae=-mae,R=R)); last=j
for k in (1,2):
    X=pd.DataFrame(out[k]); X=X[X.day.dt.year>=2018]; yy=X.day.dt.year
    print(f'{k}R: n {len(X)} ({len(X)/8:.0f}/yr) net {X.pts.mean():+.2f} train {X[(yy>=2019)&(yy<=2021)].pts.mean():+.2f} test {X[(yy==2018)|(yy>=2022)].pts.mean():+.2f} median R {X.R.median():.0f} days {X.day.nunique()}')
    X.to_pickle(SP+f'sweep258_{k}R.pkl')
