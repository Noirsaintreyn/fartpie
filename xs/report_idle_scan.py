"""IDLE-DAY BRUTE-FORCE SCAN. Thousands of simple rules (feature x threshold x fade/follow x entry time x exit time x stop) on NQ 5m RTH.
Entry at NEXT bar open after decision bar, exit at fixed bar close (or stop), cost 0.5 pt. Scored on idle days (core legs flat; ex-post label), all days, eras <=2021 / 2022+.
Multiple-testing: sign-flip permutation max-|t| null across ALL configs. Walk-forward: pick best in one era by t, score in the other."""
import numpy as np, pandas as pd, itertools
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
mn=b.dt.dt.hour*60+b.dt.dt.minute; b['d']=b.dt.dt.normalize(); rth=b[(mn>=570)&(mn<960)].copy()
idle=pd.read_pickle(SC+'idle_state.pkl')['trade']; idle=~idle.astype(bool)
D=rth.groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last')); D['rng']=D.h-D.l
D['adr']=D.rng.rolling(20).mean().shift(); D['pdh']=D.h.shift(); D['pdl']=D.l.shift(); D['pdc']=D.c.shift()
eth=b.set_index('dt'); days=list(D.index)
ONH={};ONL={}
for i,d in enumerate(days):
    if i==0: continue
    x=eth.loc[days[i-1]+pd.Timedelta(hours=18):d+pd.Timedelta(hours=9,minutes=29),['high','low']]
    if len(x): ONH[d]=x.high.max(); ONL[d]=x.low.min()
keep=[];O=[];H=[];L=[];C=[];V=[]
for d,g in rth.groupby('d'):
    if len(g)==78 and d in ONH and D.adr.get(d)==D.adr.get(d):
        keep.append(d);O.append(g.open.values);H.append(g.high.values);L.append(g.low.values);C.append(g.close.values);V.append(g.volume.values)
O,H,L,C,V=map(np.array,(O,H,L,C,V)); keep=pd.DatetimeIndex(keep); n=len(keep)
adr=D.adr.reindex(keep).values; pdh=D.pdh.reindex(keep).values; pdl=D.pdl.reindex(keep).values; pdc=D.pdc.reindex(keep).values
onh=np.array([ONH[d] for d in keep]); onl=np.array([ONL[d] for d in keep])
tp=(H+L+C)/3; vw=np.cumsum(tp*V,1)/np.maximum(np.cumsum(V,1),1)
isidle=idle.reindex(keep).fillna(False).values.astype(bool); era2=(keep.year>=2022)
print(f'days {n}  idle {isidle.sum()}  era1 {(~era2).sum()} era2 {era2.sum()}',flush=True)
def feat(name,i):
    c=C[:,i]
    if name=='ret_open': return (c-O[:,0])/adr
    if name=='vwap_dev': return (c-vw[:,i])/adr
    if name=='pdc_dev':  return (c-pdc)/adr
    if name=='pd_pos':   return (c-pdl)/(pdh-pdl)-.5
    if name=='on_pos':   return (c-onl)/np.maximum(onh-onl,1)-.5
    if name=='ret_6':    return (c-C[:,max(i-6,0)])/adr
    if name=='ret_12':   return (c-C[:,max(i-12,0)])/adr
    if name=='day_pos':  return (c-L[:,:i+1].min(1))/np.maximum(H[:,:i+1].max(1)-L[:,:i+1].min(1),1)-.5
    if name=='gap':      return (O[:,0]-pdc)/adr
    if name=='none':     return np.ones(n)
FE=['ret_open','vwap_dev','pdc_dev','pd_pos','on_pos','ret_6','ret_12','day_pos','gap','none']
THR={'ret_open':[.15,.3,.5],'vwap_dev':[.1,.2,.35],'pdc_dev':[.15,.3,.5],'pd_pos':[.2,.35,.5],'on_pos':[.2,.35,.5],'ret_6':[.08,.15,.25],'ret_12':[.1,.2,.35],'day_pos':[.2,.3,.4],'gap':[.1,.25,.5],'none':[0]}
ENT=[(9*60+55-570)//5+0, ]  # placeholder
ent_min=[600,630,660,690,720,780,840]; ex_min=[690,720,780,840,900,955]
ei=lambda m:(m-570)//5-1   # decision bar index (bar CLOSING at m... bar start m-5)
cfg=[];P=[]
for fn in FE:
  for ent in ent_min:
    i=ei(ent)
    f=feat(fn,i)
    for thr in THR[fn]:
      for mode in ((['fade','follow'] if fn!='none' else ['long','short'])):
        if fn=='none': sg=np.full(n,1. if mode=='long' else -1.)
        else:
            s=np.where(f>thr,1,np.where(f<-thr,-1,0)).astype(float); sg=-s if mode=='fade' else s
        for ex in ex_min:
          if ex<=ent+10: continue
          j=(ex-570)//5-1
          e=O[:,i+1]; x=C[:,j]
          for stp in (0,.25,.5):
            if stp==0: pts=(x-e)*sg-.5
            else:
              lo_=L[:,i+1:j+1].min(1); hi_=H[:,i+1:j+1].max(1); sd=stp*adr
              hit=np.where(sg>0,e-lo_>=sd,hi_-e>=sd)
              pts=np.where(hit,-sd,(x-e)*sg)-.5
            pts=np.where(sg==0,np.nan,pts)
            cfg.append((fn,ent,thr,mode,ex,stp)); P.append(pts)
P=np.array(P); print('configs',len(cfg),flush=True)
def tstat(M,mask):
    X=M[:,mask]; k=(~np.isnan(X)).sum(1); mu=np.nanmean(X,1); sd=np.nanstd(X,1,ddof=1)
    return k,mu,mu/np.where(sd>0,sd/np.sqrt(np.maximum(k,1)),np.nan)
res=pd.DataFrame(cfg,columns=['feat','ent','thr','mode','ex','stop'])
for nm,mask in (('idle',isidle),('all',np.ones(n,bool)),('idle1',isidle&~era2),('idle2',isidle&era2),('all1',~era2),('all2',era2)):
    k,mu,t=tstat(P,mask); res['n_'+nm]=k; res['mu_'+nm]=mu; res['t_'+nm]=t
ok=res[(res.n_idle>=80)]
# permutation null for max |t| on idle days (sign-flip of day outcomes, shared across configs)
rng=np.random.default_rng(0); Pi=P[:,isidle]; mx=[]
for _ in range(300):
    s=rng.choice([-1,1],size=Pi.shape[1]); X=Pi*s
    k=(~np.isnan(X)).sum(1); mu=np.nanmean(X,1); sd=np.nanstd(X,1,ddof=1); t=mu/np.where(sd>0,sd/np.sqrt(np.maximum(k,1)),np.nan)
    mx.append(np.nanmax(t[k>=80]))
mx=np.array(mx); print(f'null max-t over configs (idle days): median {np.median(mx):.2f}  95% {np.quantile(mx,.95):.2f}',flush=True)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
cols=['feat','ent','thr','mode','ex','stop','n_idle','mu_idle','t_idle','mu_idle1','mu_idle2','mu_all','t_all','mu_all1','mu_all2']
both=ok[(ok.mu_idle1>0)&(ok.mu_idle2>0)].sort_values('t_idle',ascending=False)
print(f'\nconfigs with n_idle>=80: {len(ok)};  positive in BOTH eras on idle days: {len(both)}')
print('TOP 25 by idle t (both eras positive):'); print(both[cols].head(25).round(3).to_string(index=False))
print('\nAlso positive on ALL days both eras (deployable w/o knowing idle):')
dep=both[(both.mu_all1>0)&(both.mu_all2>0)&(both.t_all>2)]; print(len(dep)); print(dep[cols].head(15).round(3).to_string(index=False))
# walk-forward: best by idle t in era1 -> era2 and reverse
for a,b_ in (('idle1','idle2'),('idle2','idle1')):
    r=ok[ok['n_'+a]>=50].sort_values('t_'+a,ascending=False).head(20)
    print(f'\nWALK-FWD: top20 by t on {a} -> mean pts on {b_}: avg {r["mu_"+b_].mean():.2f}, share positive {(r["mu_"+b_]>0).mean():.0%}; in-sample avg {r["mu_"+a].mean():.2f}')
    print(r[['feat','ent','thr','mode','ex','stop','mu_'+a,'t_'+a,'mu_'+b_]].head(8).round(3).to_string(index=False))
# family summary: by feature, share of configs with positive mean in both eras (idle)
print('\nFEATURE FAMILY (idle days): share of configs positive both eras, mean of mu_idle')
print(ok.assign(pb=(ok.mu_idle1>0)&(ok.mu_idle2>0)).groupby(['feat','mode']).agg(cfgs=('pb','size'),pos_both=('pb','mean'),mu=('mu_idle','mean')).round(3).to_string())
res.to_pickle(SC+'idle_scan.pkl')
