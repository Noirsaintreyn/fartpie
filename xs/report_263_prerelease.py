"""Report 263: PRE-RELEASE compression fade (8:30 ET releases), DECLARED before results. Mirror of the FOMC finding: if markets compress/chop before a scheduled release, fading extremes before it should work.
NFP days = first Friday of the month (rule-based, known in advance; CPI/other dates not available without a calendar export). Window 04:00-08:25 ET, extremes measured from the 04:00 open; event = new window high (low) with extension >= 0.35 ATR14(daily RTH range) from the 04:00 open and a rejection bar (close back with >=50% of its range as wick); entry next 5m open, stop = bar extreme +/-1, R >= 8 pts (cap 120), targets 1R / 2R, ALWAYS flat at the 08:25 close (before the release), cost 1 pt, max 2 per day per direction. Compare NFP days to all other weekdays in the same window (placebo). Train 2019-21 | test 2018+2022-26. Also report the 08:30 bar range / ATR on NFP vs other days (does the release actually matter?)."""
import numpy as np, pandas as pd
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index()
rth=b.between_time('09:30','15:55'); D=rth.groupby(rth.index.normalize()); dd=pd.DataFrame({'h':D.high.max(),'l':D.low.min(),'n':D.size()}); dd=dd[dd.n>=70]; dd['atr']=(dd.h-dd.l).rolling(14).mean().shift(1)
idx=dd.index; nfp=pd.Series((idx.dayofweek==4)&(idx.day<=7),index=idx)
rows=[]; rel=[]
for d in idx:
    if d.year<2018 or not np.isfinite(dd.atr[d]): continue
    g=b.loc[d+pd.Timedelta(hours=4):d+pd.Timedelta(hours=8,minutes=25)]
    if len(g)<40: continue
    a=dd.atr[d]; o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values; o0=o[0]; mm=g.index.hour*60+g.index.minute
    r830=b.loc[d+pd.Timedelta(hours=8,minutes=30):d+pd.Timedelta(hours=8,minutes=34)]
    if len(r830): rel.append((d,nfp[d],(r830.high.max()-r830.low.min())/a))
    rh=-np.inf; rl=np.inf; cnt={1:0,-1:0}
    for i in range(len(g)-2):
        nh=h[i]>rh; nl=l[i]<rl; rh=max(rh,h[i]); rl=min(rl,l[i])
        for dr,new in ((-1,nh),(1,nl)):
            if not new or cnt[dr]>=2 or mm[i]<300: continue
            rng=h[i]-l[i]
            if rng<=0: continue
            if dr==-1:
                if h[i]-o0<.35*a or (h[i]-c[i])<.5*rng: continue
                stop=h[i]+1
            else:
                if o0-l[i]<.35*a or (c[i]-l[i])<.5*rng: continue
                stop=l[i]-1
            e=o[i+1]; R=abs(stop-e)
            if R<8 or R>120: continue
            cnt[dr]+=1; res={}
            for k in (1,2):
                tgt=e+(-1 if dr==-1 else 1)*k*R; r=None
                for j in range(i+1,len(g)):
                    hs=(h[j]>=stop) if dr==-1 else (l[j]<=stop); ht=(l[j]<=tgt) if dr==-1 else (h[j]>=tgt)
                    if hs: r=-R;break
                    if ht: r=k*R;break
                if r is None: r=(e-c[-1]) if dr==-1 else (c[-1]-e)
                res[k]=r-1
            rows.append((d,nfp[d],dr,R,res[1],res[2]))
T=pd.DataFrame(rows,columns=['d','nfp','dr','R','r1','r2']); T['split']=np.where((T.d.dt.year>=2019)&(T.d.dt.year<=2021),'train','test')
def tt(x): return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>2 and x.std()>0 else np.nan
for nm,m in (('NFP days',T.nfp),('other weekdays',~T.nfp)):
    x=T[m]; print(f'{nm:15s} n {len(x)} days {x.d.nunique()} | 1R {x.r1.mean():+.1f}(t{tt(x.r1):+.1f}) 2R {x.r2.mean():+.1f}(t{tt(x.r2):+.1f}) |',' | '.join(f'{sp} n{int((x.split==sp).sum())} 1R {x[x.split==sp].r1.mean():+.1f} 2R {x[x.split==sp].r2.mean():+.1f}' for sp in ('train','test')))
Rl=pd.DataFrame(rel,columns=['d','nfp','rg']); print('08:30 bar range/ATR: NFP median',round(Rl[Rl.nfp].rg.median(),2),'other median',round(Rl[~Rl.nfp].rg.median(),2),'| NFP days with big move (>=.3 ATR):',f'{(Rl[Rl.nfp].rg>=.3).mean():.0%}','other',f'{(Rl[~Rl.nfp].rg>=.3).mean():.0%}')
