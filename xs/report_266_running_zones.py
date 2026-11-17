"""Report 266: RUNNING VOLATILITY ZONES (Rey Nov 17: do not anchor at 9:30 and then decide the high/low; keep the model running. If the current high is in, the low should be at high - R^_t; if the current low is in, the high should be at low + R^_t; targets can be VWAP / EMA / mid, not the whole rest). DECLARED before results.
R^_t = expected FULL-day range, re-learned every 30 min from 10:00 to 14:30 (rolling OLS, trailing 500 sessions per checkpoint, bias-corrected) on [log R^0 (HAR + QQQ implied move at 09:30), log range so far, log realized vol so far, log Rogers-Satchell vol so far]; the latest value is held until the next checkpoint.
Zones at bar i (information through bar i-1): Z_hi = session low so far + R^_t (where the high should be if the low is in); Z_lo = session high so far - R^_t.
Calibration: the day's final high vs Z_hi and final low vs Z_lo at each checkpoint, in units of R^_t (share within +/-0.15).
Trade: 5m bar 10:00-14:30 whose high reaches Z_hi - 0.10 R^_t (low reaches Z_lo + 0.10 R^_t) and that rejects (close back with >= 50% of its range as the wick) -> fade at the next open; stop = max(bar extreme, zone) +/- 0.10 R^_t (model-based stop, not extreme+1); entry-to-stop 10-250 pts; targets: session VWAP, EMA20 (5m) at entry, midpoint of the range so far (must be >= 3 pts beyond the entry); flat 15:55; cost 1 pt; max 2 per day per direction. PLACEBO: the same rejection bar at a new session extreme but NOT in the zone (extreme < Z_hi - 0.25 R^_t), same stop rule. Zone matters if the zone group beats the placebo in BOTH splits. Also by share of range spent X = range so far / R^_t (< .8 | .8-1 | >= 1). Train 2019-21 | test 2018+2022-26."""
import numpy as np, pandas as pd, pickle
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index(); rth=b.between_time('09:30','15:55'); D=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'o':D.open.first(),'h':D.high.max(),'l':D.low.min(),'c':D.close.last(),'n':D.size()}); dd=dd[dd.n>=70]; dd['rng']=dd.h-dd.l; dd['lr']=np.log(dd.rng)
ov=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); ov.index=ov.index.normalize()
k=ov.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1; k=np.clip(k,0,len(ov)-1); dd['imp']=np.log(ov.atm_iv.values[k]/np.sqrt(252)*dd.o.values)
dd['x1']=dd.lr.shift(1); dd['x5']=dd.lr.shift(1).rolling(5).mean(); dd['x22']=dd.lr.shift(1).rolling(22).mean(); dd=dd[dd.index>='2016-06-01'].dropna(subset=['x1','x5','x22'])
X=np.c_[np.ones(len(dd)),dd[['x1','x5','x22','imp']].values]; y=dd.lr.values; p0=np.full(len(dd),np.nan)
for i in range(500,len(dd)):
    if i==500 or (i-500)%20==0: Xt,yt=X[i-500:i],y[i-500:i]; be=np.linalg.lstsq(Xt,yt,rcond=None)[0]
    p0[i]=X[i]@be
dd['R0']=np.exp(p0); dd=dd[dd.index>='2018-01-01'].dropna(subset=['R0'])
CK=list(range(600,880,30)); CP=[]; Dd={}
for d,g in D:
    if d not in dd.index: continue
    o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;v=g.volume.values.astype(float); mm=(g.index.hour*60+g.index.minute).values
    lc=np.log(c); lo_=np.log(o); lh=np.log(h); ll_=np.log(l); rsb=(lh-lc)*(lh-lo_)+(ll_-lc)*(ll_-lo_); rvb=np.r_[0,np.diff(lc)]**2; hodv=np.maximum.accumulate(h); lodv=np.minimum.accumulate(l)
    Dd[d]=(o,h,l,c,v,mm,hodv,lodv)
    for ck in CK:
        ii=np.where(mm==ck)[0]
        if len(ii)==0: continue
        i=ii[0]-1
        CP.append((d,ck,hodv[i]-lodv[i],np.sqrt(rvb[:i+1].sum())*o[0],np.sqrt(max(rsb[:i+1].sum(),1e-12))*o[0],dd.R0[d],dd.rng[d],hodv[i],lodv[i],dd.h[d],dd.l[d]))
C=pd.DataFrame(CP,columns=['d','ck','rs','rv','rsv','R0','fr','H','L','FH','FL']); Rt={}
for ck,u in C.groupby('ck'):
    u=u.sort_values('d').reset_index(drop=True); Xc=np.c_[np.ones(len(u)),np.log(u.R0),np.log(u.rs),np.log(u.rv),np.log(u.rsv)]; yc=np.log(u.fr.values)
    for kx in range(250,len(u)):
        if kx==250 or (kx-250)%20==0: lo=max(0,kx-500); be=np.linalg.lstsq(Xc[lo:kx],yc[lo:kx],rcond=None)[0]; sm=np.mean(np.exp(yc[lo:kx]-Xc[lo:kx]@be))
        Rt[(u.d[kx],ck)]=np.exp(Xc[kx]@be)*sm
C['Rt']=[Rt.get((d,ck),np.nan) for d,ck in zip(C.d,C.ck)]; C=C.dropna(subset=['Rt']); C['split']=np.where((C.d.dt.year>=2019)&(C.d.dt.year<=2021),'train','test')
C['eh']=(C.FH-(C.L+C.Rt))/C.Rt; C['el']=(C.FL-(C.H-C.Rt))/C.Rt
print('--- calibration of the running zones (final extreme minus zone, in units of R^t): share within +/-0.15 | median error')
for ck,g in C.groupby('ck'):
    print(f'  {ck//60}:{ck%60:02d}',' | '.join(f'{sp} n{len(x)} hi {(x.eh.abs()<=.15).mean():.0%} ({x.eh.median():+.2f})  lo {(x.el.abs()<=.15).mean():.0%} ({x.el.median():+.2f})' for sp,x in g.groupby('split')))
# trades
out=[]
for d,(o,h,l,c,v,mm,hodv,lodv) in Dd.items():
    if not any((d,ck) in Rt for ck in CK): continue
    cnt={1:0,-1:0}
    for i in range(7,len(o)-2):
        if mm[i]<600 or mm[i]>=870: continue
        ck=(mm[i]//30)*30; Rh=Rt.get((d,ck))
        if Rh is None: continue
        H=hodv[i-1]; L=lodv[i-1]; Zhi=L+Rh; Zlo=H-Rh; rng=h[i]-l[i]
        if rng<=0: continue
        X_=(H-L)/Rh
        for dr in (-1,1):
            if cnt[dr]>=2: continue
            if dr==-1:
                if not (h[i]>H or h[i]>=Zhi-.10*Rh) or (h[i]-c[i])<.5*rng: continue   # candidate high: new session high or touching the zone
                inz=h[i]>=Zhi-.10*Rh; newx=h[i]>H
                if not newx and not inz: continue
                stop=max(h[i],Zhi)+.10*Rh; ext=h[i]
            else:
                if (c[i]-l[i])<.5*rng: continue
                inz=l[i]<=Zlo+.10*Rh; newx=l[i]<L
                if not newx and not inz: continue
                stop=min(l[i],Zlo)-.10*Rh; ext=l[i]
            # group: zone = in zone; placebo = new session extreme well outside zone
            if inz: grp='zone'
            elif newx and ((dr==-1 and h[i]<Zhi-.25*Rh) or (dr==1 and l[i]>Zlo+.25*Rh)): grp='placebo'
            else: continue
            e=o[i+1]; Rs=abs(stop-e)
            if not (10<=Rs<=250): continue
            tp=(h[:i+1]+l[:i+1]+c[:i+1])/3; vw=(tp*v[:i+1]).sum()/max(v[:i+1].sum(),1e-9); em=c[0]
            for q_ in c[1:i+1]: em+=2/21*(q_-em)
            tg={'vwap':vw,'ema20':em,'mid':(H+L)/2}; pn={}
            for tn,t_ in tg.items():
                pn[tn]=np.nan
                if (dr==-1 and t_>=e-3) or (dr==1 and t_<=e+3): continue
                r_=None
                for j in range(i+1,len(o)):
                    hs=(h[j]>=stop) if dr==-1 else (l[j]<=stop); ht=(l[j]<=t_) if dr==-1 else (h[j]>=t_)
                    if hs: r_=-Rs;break
                    if ht: r_=abs(t_-e);break
                    if mm[j]>=955: r_=(e-c[j]) if dr==-1 else (c[j]-e);break
                pn[tn]=(r_ if r_ is not None else ((e-c[-1]) if dr==-1 else (c[-1]-e)))-1.0
            cnt[dr]+=1; out.append((d,dr,grp,X_,Rs,pn['vwap'],pn['ema20'],pn['mid'],float(mm[i+1])))
T=pd.DataFrame(out,columns=['d','dr','grp','X','Rs','p_vwap','p_ema','p_mid','em']); T['split']=np.where((T.d.dt.year>=2019)&(T.d.dt.year<=2021),'train','test')
def tt(x): x=x.dropna(); return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>2 and x.std()>0 else np.nan
def show(nm,m):
    for col in ('p_vwap','p_ema','p_mid'):
        s=[]
        for sp in ('train','test'):
            q=T[m&(T.split==sp)][col].dropna(); s.append(f'n{len(q):4d} {q.mean():+6.1f} (t{tt(q):+.1f}) win {(q>0).mean():.0%}')
        print(f'  {nm:22s} {col:7s} {s[0]} | {s[1]}')
print('\\n--- running-zone fade (stop beyond the zone) vs placebo (new extreme, outside the zone); median stop',round(T.Rs.median()),'pts')
for g_ in ('zone','placebo'): show(g_,T.grp==g_)
print('--- zone fades by share of range spent X')
for nm,m in (('zone X<.8',(T.grp=='zone')&(T.X<.8)),('zone .8<=X<1',(T.grp=='zone')&(T.X>=.8)&(T.X<1)),('zone X>=1',(T.grp=='zone')&(T.X>=1))): show(nm,m)
pickle.dump((T,C),open(SP+'zones266.pkl','wb'))
