"""Report 265: INTRADAY-UPDATING volatility + EXHAUSTION at the extreme (Rey Nov 17: not anchored at 9:30; learn through the day; candle volume, YZ/HAR, vanna/vega/vex tell the story; at the extreme, buyers/sellers failing = exhaustion). DECLARED before results.
Observation = checkpoint (10:30, 11:30, 12:30, 13:30, 14:30 ET) where the last close is within 0.15 x (range so far) of the RTH session high (side s=-1, fade short) or low (s=+1), and that extreme was set within the last 6 bars (fresh push).
Intraday volatility learning: R^_t = expected FULL-day range at the checkpoint from a rolling OLS (trailing 500 sessions, same checkpoint, bias-corrected) of log final range on [log R^0 (HAR+IV at 09:30), log range so far, log realized close-to-close vol so far, log Rogers-Satchell vol so far (candle based)]. X = range so far / R^_t = share of the expected day range already spent (updates every checkpoint).
Exhaustion features at the push (last 6 bars; est. delta = vol x (2c-h-l)/(h-l)), declared direction = exhaustion if: X high; effort/result high (relative volume of the last 6 bars / progress made in ATR); CVD divergence (est. cum. delta at this extreme weaker than at the previous session extreme on that side); aggressor fade (aggression of the last 3 bars in the push direction below the 3 bars before); close location of the last 3 bars far from the extreme; volume trend (last 3 bars vs prior 3) DOWN (classic dry-up; report 254 found the opposite on the 'later pushes' variant, so both are reported); Greeks: prior-close net vanna rolling z (252d), vanna flow, GEX>0 (positive gamma -> extremes hold).
Outcome: HELD = the day's final extreme equals this extreme (within .25pt). Also fade P&L: enter next 5m open, stop = extreme +/-1, target = retrace of 0.5 x range so far from the extreme, flat 15:55, cost 1pt, R>=10 (cap 150).
Check 1: AUC(feature, HELD) train 2019-21 | test 2018+2022-26; a feature COUNTS if on the declared side (>.52) in BOTH. Targets (Rey: need not be the whole rest): half the range so far, session VWAP, EMA20 (5m) at entry, mid of the range so far. Check 2: exhaustion score = # counting features on the exhaustion side (vs trailing-60-day median of the feature, causal); fade pts by score bucket, train | test."""
import numpy as np, pandas as pd, pickle
from scipy.stats import rankdata
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index(); rth=b.between_time('09:30','15:55'); D=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'o':D.open.first(),'h':D.high.max(),'l':D.low.min(),'c':D.close.last(),'n':D.size()}); dd=dd[dd.n>=70]; dd['rng']=dd.h-dd.l; dd['lr']=np.log(dd.rng); dd['atr']=dd.rng.rolling(14).mean().shift(1)
ov=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); ov.index=ov.index.normalize()
v2=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_v2_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); v2.index=v2.index.normalize()
gx=pd.read_csv('xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').sort_index()
def prior(s): k=s.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1; k=np.clip(k,0,len(s)-1); return pd.Series(s.values[k],index=dd.index)
dd['imp']=np.log(prior(ov.atm_iv)/np.sqrt(252)*dd.o); nv=v2.net_vanna; dd['vz']=prior((nv-nv.rolling(252,min_periods=120).mean())/nv.rolling(252,min_periods=120).std()); dd['iv5']=prior(ov.atm_iv.diff(5)); dd['vflow']=np.sign(prior(nv))*-np.sign(dd.iv5); dd['gexpos']=(prior(gx.net_gex)>0).astype(float)
dd['x1']=dd.lr.shift(1); dd['x5']=dd.lr.shift(1).rolling(5).mean(); dd['x22']=dd.lr.shift(1).rolling(22).mean(); dd=dd[dd.index>='2016-06-01'].dropna(subset=['x1','x5','x22'])
X=np.c_[np.ones(len(dd)),dd[['x1','x5','x22','imp']].values]; y=dd.lr.values; p0=np.full(len(dd),np.nan)
for i in range(500,len(dd)):
    if i==500 or (i-500)%20==0: Xt,yt=X[i-500:i],y[i-500:i]; be=np.linalg.lstsq(Xt,yt,rcond=None)[0]; sm=np.mean(np.exp(yt-Xt@be))
    p0[i]=X[i]@be
dd['R0']=np.exp(p0)*1.0; dd=dd[dd.index>='2018-01-01'].dropna(subset=['R0'])
CK=(630,690,750,810,870); rows=[]; CP=[]
for d,g in D:
    if d not in dd.index or not np.isfinite(dd.atr[d]): continue
    o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;v=g.volume.values.astype(float); mm=(g.index.hour*60+g.index.minute).values; atr=dd.atr[d]
    dl=v*(2*c-h-l)/np.maximum(h-l,1e-9); cum=np.cumsum(dl); lc=np.log(c); lo_=np.log(o); lh=np.log(h); ll_=np.log(l)
    rsb=lh-lc; rsb=(lh-lc)*(lh-lo_)+(ll_-lc)*(ll_-lo_); rvb=np.r_[0,np.diff(lc)]**2
    hodv=np.maximum.accumulate(h); lodv=np.minimum.accumulate(l)
    for ck in CK:
        ii=np.where(mm==ck)[0]
        if len(ii)==0: continue
        i=ii[0]-1   # last completed bar before the checkpoint minute
        if i<12 or i+3>=len(g): continue
        rs_t=hodv[i]-lodv[i]
        rv=np.sqrt(rvb[:i+1].sum())*o[0]; rsv=np.sqrt(max(rsb[:i+1].sum(),1e-12))*o[0]
        CP.append((d,ck,rs_t,rv,rsv,dd.R0[d],dd.rng[d]))
        row=dict(d=d,ck=ck,rs=rs_t,rv=rv,rsv=rsv,R0=dd.R0[d],fr=dd.rng[d],o0=o[0],atr=atr,vz=dd.vz[d],vflow=dd.vflow[d],gexpos=dd.gexpos[d])
        for s,ext,pos in ((-1,hodv[i],np.argmax(h[:i+1])),(1,lodv[i],np.argmin(l[:i+1]))):
            # last index where the extreme was set
            if s==-1: setb=np.where(h[:i+1]>=ext-1e-9)[0][-1]
            else: setb=np.where(l[:i+1]<=ext+1e-9)[0][-1]
            if i-setb>6 or abs(c[i]-ext)>.15*rs_t: continue
            if s==-1: prev=np.where(h[:setb]>=hodv[setb-1]-1e-9)[0] if setb>0 else []
            sb=slice(max(0,i-5),i+1); rel6=v[sb].sum()/(6*v[max(0,i-65):max(1,i-5)].mean()+1e-9)
            prog=abs(ext-(l[max(0,i-5)] if s==-1 else h[max(0,i-5)]))/atr; eff=rel6/max(prog,.02)
            # previous extreme CVD
            jj=np.arange(0,setb) ; 
            if s==-1: pe=jj[h[jj]>=np.maximum.accumulate(h[:setb]) -1e-9] if setb>0 else []
            else: pe=jj[l[jj]<=np.minimum.accumulate(l[:setb])+1e-9] if setb>0 else []
            div=np.nan
            if len(pe)>=1: div=-s*(cum[i]-cum[pe[-1]])/(6*v[max(0,i-65):max(1,i-5)].mean()+1e-9)   # positive = flow weaker than at prior extreme
            a3=-s*dl[i-2:i+1].sum()/max(v[i-2:i+1].sum(),1e-9); a3p=-s*dl[i-5:i-2].sum()/max(v[i-5:i-2].sum(),1e-9)   # push-direction aggression: s=-1 high push: buyers = +delta
            a3=-s*a3*-1 if False else a3
            aggr=(dl[i-2:i+1].sum()/max(v[i-2:i+1].sum(),1e-9))*(-s); aggrp=(dl[i-5:i-2].sum()/max(v[i-5:i-2].sum(),1e-9))*(-s)
            clo=np.mean(((c[i-2:i+1]-l[i-2:i+1])/np.maximum(h[i-2:i+1]-l[i-2:i+1],1e-9)) if s==-1 else ((h[i-2:i+1]-c[i-2:i+1])/np.maximum(h[i-2:i+1]-l[i-2:i+1],1e-9)))
            vtr=v[i-2:i+1].mean()/max(v[i-5:i-2].mean(),1e-9)
            e=o[i+2] if False else o[i+1]; stop=ext+1 if s==-1 else ext-1; Rs=abs(stop-e)
            tp_=(h[:i+1]+l[:i+1]+c[:i+1])/3; vw=(tp_*v[:i+1]).sum()/max(v[:i+1].sum(),1e-9)
            al=2/21; em=c[0]
            for q_ in c[1:i+1]: em=em+al*(q_-em)
            tgts={'half':(e-0.5*rs_t if s==-1 else e+0.5*rs_t),'vwap':vw,'ema20':em,'mid':(hodv[i]+lodv[i])/2}
            pn={}
            for tn,tgt in tgts.items():
                pn[tn]=np.nan
                if not (10<=Rs<=150): continue
                if (s==-1 and tgt>=e-3) or (s==1 and tgt<=e+3): continue
                r_=None
                for j in range(i+1,len(g)):
                    hs=(h[j]>=stop) if s==-1 else (l[j]<=stop); ht=(l[j]<=tgt) if s==-1 else (h[j]>=tgt)
                    if hs: r_=-Rs;break
                    if ht: r_=abs(tgt-e);break
                    if mm[j]>=955: r_=(e-c[j]) if s==-1 else (c[j]-e);break
                pn[tn]=(r_ if r_ is not None else ((e-c[-1]) if s==-1 else (c[-1]-e)))-1.0
            held=(dd.h[d]<=ext+.25) if s==-1 else (dd.l[d]>=ext-.25)
            rr=dict(row); rr.update(s=s,held=bool(held),eff=eff,div=div,agrfade=aggrp-aggr,aggr=aggr,clo=clo,vtr=vtr,pnl=pn['half'],p_vwap=pn['vwap'],p_ema=pn['ema20'],p_mid=pn['mid'],ext=ext); rows.append(rr)
R=pd.DataFrame(rows); R['yr']=R.d.dt.year; R['split']=np.where((R.yr>=2019)&(R.yr<=2021),'train','test')
# intraday vol learning per checkpoint on ALL days
C=pd.DataFrame(CP,columns=['d','ck','rs','rv','rsv','R0','fr']); Rt={}
for ck,u in C.groupby('ck'):
    u=u.sort_values('d').reset_index(drop=True); Xc=np.c_[np.ones(len(u)),np.log(u.R0),np.log(u.rs),np.log(u.rv),np.log(u.rsv)]; yc=np.log(u.fr.values)
    for k in range(250,len(u)):
        if k==250 or (k-250)%20==0: lo=max(0,k-500); be=np.linalg.lstsq(Xc[lo:k],yc[lo:k],rcond=None)[0]; sm=np.mean(np.exp(yc[lo:k]-Xc[lo:k]@be))
        Rt[(u.d[k],ck)]=np.exp(Xc[k]@be)*sm
R['Rt']=[Rt.get((d,ck),np.nan) for d,ck in zip(R.d,R.ck)]
R=R.dropna(subset=['Rt']); R['X']=R.rs/R.Rt
FE={'X':1,'eff':1,'div':1,'agrfade':1,'clo':-1,'vtr':-1,'vz':0,'vflow':0,'gexpos':1}
def auc(x,y):
    r=rankdata(x); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
print('obs',len(R),'| HELD rate train',round(R[R.split=="train"].held.mean(),3),'test',round(R[R.split=="test"].held.mean(),3),'| by checkpoint',R.groupby('ck').held.mean().round(2).to_dict())
print('intraday range model: corr(log R^t, log final range) vs corr(log R^0, ...) on obs:',round(np.corrcoef(np.log(R.Rt),np.log(R.fr))[0,1],3),round(np.corrcoef(np.log(R.R0),np.log(R.fr))[0,1],3))
cnt=[]
print('feature AUC for HELD (train | test); declared exhaustion side marked')
for f,sg in FE.items():
    z=R.dropna(subset=[f]); out=[auc(z[z.split==sp][f].values,z[z.split==sp].held.values.astype(int)) for sp in ('train','test')]
    ok=sg!=0 and all((o-.5)*sg>.02 for o in out); 
    if ok: cnt.append(f)
    print(f'  {f:8s} train {out[0]:.3f} test {out[1]:.3f}  declared {"+" if sg>0 else ("-" if sg<0 else "two-sided")} {"<- counts" if ok else ""}')
print('counting features:',cnt)
if cnt:
    sc=np.zeros(len(R))
    for f in cnt:
        med=R.groupby('ck')[f].transform(lambda s:s.shift(1).rolling(300,min_periods=60).median()); sg=FE[f]
        sc+=np.where(R[f].notna()&med.notna()&(((R[f]-med)*sg)>0),1,0)
    R['score']=sc; print('--- exhaustion score (# counting features on the exhaustion side): HELD rate and fade pnl (target .5 x range so far), train | test')
    for k in sorted(R.score.unique()):
        m=R.score==k; s=[]
        for sp in ('train','test'):
            x=R[m&(R.split==sp)]; q=x.pnl.dropna(); s.append(f'n{len(x):4d} held {x.held.mean():.0%} pnl {q.mean():+5.1f} (t{(q.mean()/(q.std()/np.sqrt(len(q))) if len(q)>2 else np.nan):+.1f})')
        print(f'  score {int(k)}: {s[0]} | {s[1]}')
print('--- fade pnl by target (all obs, then score>=2), train | test: half-range | VWAP | EMA20 | mid-range')
for nm,m in (('all',np.ones(len(R),bool)),)+((('score>=2',R.score>=2),) if 'score' in R else ()):
    for col in ('pnl','p_vwap','p_ema','p_mid'):
        s_=[]
        for sp in ('train','test'):
            q=R[m&(R.split==sp)][col].dropna(); s_.append(f'n{len(q):4d} {q.mean():+5.1f} (t{(q.mean()/(q.std()/np.sqrt(len(q))) if len(q)>2 else np.nan):+.1f}) win {(q>0).mean():.0%}')
        print(f'  {nm:9s} {col:7s} {s_[0]} | {s_[1]}')
pickle.dump(R,open(SP+'ex265.pkl','wb'))
