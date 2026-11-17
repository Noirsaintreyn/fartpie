"""Report 264: VOLATILITY ZONES + ORDERFLOW (Rey Nov 17: forecast the day's range, anchor the candidate high/low, target the rest of the range). DECLARED before results. Builds on xs/anchored_range.py (Nov 9: expected range anchored at VECTOR flips; TP at projected extreme did not beat the current exit); this is a standalone reversal at the candidate extreme.
STEP 1 range forecast R^ (RTH high-low, points) known at 09:30 from prior data only: (A) HAR on log range (lags 1, 5-mean, 22-mean); (B) HAR + log implied move (prior-close QQQ ATM IV/sqrt(252) x open); (C) HAR + log Yang-Zhang 5d daily vol. Rolling OLS on the trailing 500 sessions, refit every 20 sessions, bias-corrected (mean exp residual). Judge by out-of-sample R2 on log range, train 2019-21 | test 2018+2022-26; the model used downstream is chosen on TRAIN only.
STEP 2 candidate extreme: 5m bar 10:00-14:00 that makes a new RTH session low (high) and rejects (closes back with >=50% of its range as the wick); anchor A = bar extreme; entry next 5m open; stop = A -/+ 1; T = A +/- f x R^ with f in {0.618, 1.0} (the fib of the projected range); skip if stop distance <10 or >150 or reward < 1.0 x risk; flat 15:55; cost 1 pt; max 2 per day per direction. U = (open - A)/R^ = share of the projected range already used on the move to the candidate (Rey: used 5 of 20 -> U=.25, target the other 15).
STEP 3 report pts/trade by U bucket (<.25, .25-.5, >=.5) and by orderflow flags from report 255 (V1 volume building, V2 absorption, both), train | test. A pass = positive in BOTH splits with n >= 40 per split."""
import numpy as np, pandas as pd, pickle
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index()
rth=b.between_time('09:30','15:55'); D=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'o':D.open.first(),'h':D.high.max(),'l':D.low.min(),'c':D.close.last(),'n':D.size()}); dd=dd[dd.n>=70]
dd['rng']=dd.h-dd.l; dd['lr']=np.log(dd.rng)
ov=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); ov.index=ov.index.normalize()
k=ov.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1; k=np.clip(k,0,len(ov)-1); dd['imp']=np.log(ov.atm_iv.values[k]/np.sqrt(252)*dd.o.values)
# Yang-Zhang 5d (daily bars: overnight gap, open-close, Rogers-Satchell)
on=np.log(dd.o/dd.c.shift(1)); oc=np.log(dd.c/dd.o); hh=np.log(dd.h/dd.o); ll=np.log(dd.l/dd.o); rs=hh*(hh-oc)+ll*(ll-oc)
n=5; kk=.34/(1.34+(n+1)/(n-1)); yz=(on.rolling(n).var()+kk*oc.rolling(n).var()+(1-kk)*rs.rolling(n).mean()); dd['yz']=np.log(np.sqrt(yz.clip(lower=1e-12))*dd.o)
dd['x1']=dd.lr.shift(1); dd['x5']=dd.lr.shift(1).rolling(5).mean(); dd['x22']=dd.lr.shift(1).rolling(22).mean(); dd['yzp']=dd.yz.shift(1)
dd=dd[dd.index>='2016-06-01'].dropna(subset=['x1','x5','x22','yzp']).copy()
def forecast(cols):
    pred=np.full(len(dd),np.nan); sm=np.full(len(dd),np.nan); X=np.c_[np.ones(len(dd)),dd[cols].values]; y=dd.lr.values; beta=None
    for i in range(500,len(dd)):
        if beta is None or (i-500)%20==0:
            Xt,yt=X[i-500:i],y[i-500:i]; beta=np.linalg.lstsq(Xt,yt,rcond=None)[0]; sm_=np.mean(np.exp(yt-Xt@beta))
        pred[i]=X[i]@beta; sm[i]=sm_
    return pred,sm
models={'A HAR':['x1','x5','x22'],'B HAR+IV':['x1','x5','x22','imp'],'C HAR+YZ5':['x1','x5','x22','yzp']}
dd['yr']=dd.index.year; dd['split']=np.where((dd.yr>=2019)&(dd.yr<=2021),'train','test'); P={}
print('range forecast quality (out-of-sample R2 on log range | MAE of range in points), train | test')
for nm,cols in models.items():
    p,sm=forecast(cols); dd['p_'+nm[0]]=p; dd['R_'+nm[0]]=np.exp(p)*sm; P[nm]=p
    out=[]
    for sp in ('train','test'):
        g=dd[(dd.split==sp)&dd.yr.ge(2018)].dropna(subset=['p_'+nm[0]]); r2=1-((g.lr-g['p_'+nm[0]])**2).sum()/((g.lr-g.lr.mean())**2).sum(); mae=(g.rng-g['R_'+nm[0]]).abs().mean(); out.append(f'{r2:.3f} | {mae:.0f}')
    print(f'  {nm:10s} train {out[0]}   test {out[1]}')
dd=dd[dd.yr>=2018]
# choose model by TRAIN R2
best=max(models,key=lambda nm:1-((dd[(dd.split=="train")].lr-dd[(dd.split=="train")]['p_'+nm[0]])**2).sum()/((dd[(dd.split=="train")].lr-dd[(dd.split=="train")].lr.mean())**2).sum()); RC='R_'+best[0]; print('model chosen on train:',best)
out=[]
for d,g in D:
    if d not in dd.index or not np.isfinite(dd[RC][d]): continue
    Rh=dd[RC][d]; o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;v=g.volume.values.astype(float); mm=(g.index.hour*60+g.index.minute).values
    dl=v*(2*c-h-l)/np.maximum(h-l,1e-9); o0=o[0]; rh=-np.inf; rl=np.inf; ev={1:[],-1:[]}; cnt={1:0,-1:0}
    for i in range(len(g)-2):
        nh=h[i]>rh; nl=l[i]<rl; rh=max(rh,h[i]); rl=min(rl,l[i])
        for dr,new in ((-1,nh),(1,nl)):
            if not new: continue
            pv=ev[dr][:]; ev[dr].append(v[i])
            if not (600<=mm[i]<840) or cnt[dr]>=2: continue
            rng=h[i]-l[i]
            if rng<=0: continue
            if dr==-1:
                if (h[i]-c[i])<.5*rng: continue
                A=h[i]; stop=A+1
            else:
                if (c[i]-l[i])<.5*rng: continue
                A=l[i]; stop=A-1
            e=o[i+1]; Rs=abs(stop-e)
            if Rs<10 or Rs>150: continue
            U=(A-o0)/Rh if dr==-1 else (o0-A)/Rh
            pm=v[max(0,i-12):i].mean() if i>0 else np.nan; rel=v[i]/pm if pm>0 else np.nan
            vb=bool(len(pv)>=2 and v[i]>=np.mean(pv) and rel>=1.3); ab=bool(-dr*dl[max(0,i-2):i+1].sum()>0 and rel>=1.3)
            res={}
            for f in (.618,1.0):
                tgt=A-dr*-1*0+(0)  # placeholder
                tgt=(A-f*Rh) if dr==-1 else (A+f*Rh)
                if abs(tgt-e)<Rs or (dr==-1 and tgt>=e) or (dr==1 and tgt<=e): res[f]=np.nan; continue
                r=None
                for j in range(i+1,len(g)):
                    hs=(h[j]>=stop) if dr==-1 else (l[j]<=stop); ht=(l[j]<=tgt) if dr==-1 else (h[j]>=tgt)
                    if hs: r=-Rs;break
                    if ht: r=abs(tgt-e);break
                    if mm[j]>=955: r=(e-c[j]) if dr==-1 else (c[j]-e);break
                if r is None: r=(e-c[-1]) if dr==-1 else (c[-1]-e)
                res[f]=r-1
            cnt[dr]+=1; out.append((d,dr,Rs,Rh,U,vb,ab,res[.618],res[1.0],A,float(mm[i+1])))
T=pd.DataFrame(out,columns=['d','dr','R','Rh','U','vb','ab','p618','p100','A','em']); T['split']=np.where((T.d.dt.year>=2019)&(T.d.dt.year<=2021),'train','test')
def tt(x): x=x.dropna(); return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>2 and x.std()>0 else np.nan
def row(nm,m):
    for f,col in ((.618,'p618'),(1.0,'p100')):
        s=[]
        for sp in ('train','test'):
            x=T[m&(T.split==sp)][col].dropna(); s.append(f'n{len(x):4d} {x.mean():+6.1f} (t{tt(x):+.1f}) win {(x>0).mean():.0%}')
        print(f'  {nm:26s} T={f:5.3f}R^ | {s[0]} | {s[1]}')
print('signals',len(T),'| with reward>=risk at f=.618:',int(T.p618.notna().sum()),'f=1.0:',int(T.p100.notna().sum()))
print('--- all rejection candidates (train | test)'); row('all',np.ones(len(T),bool))
print('--- by share of projected range used U'); 
for nm,m in (('U<.25',T.U<.25),('.25<=U<.5',(T.U>=.25)&(T.U<.5)),('U>=.5',T.U>=.5)): row(nm,m)
print('--- orderflow flags'); 
for nm,m in (('V1 volume building',T.vb),('V2 absorption',T.ab),('V1 & V2',T.vb&T.ab),('V1 & U<.5',T.vb&(T.U<.5))): row(nm,m)
pickle.dump((T,dd[['R_A','R_B','R_C','rng','split']]),open(SP+'vz264.pkl','wb'))

T['dl']=T.d.map(dd.l); T['dh']=T.d.map(dd.h); T['held']=np.where(T.dr==1,T.dl>=T.A-0.25,T.dh<=T.A+0.25)
T['rest']=np.where(T.dr==1,T.dh-T.A,T.A-T.dl)/T.Rh   # realized remaining range from the anchor, in units of R^
print('--- is the rejected extreme the day extreme? P(held) and median remaining range / R^ (train | test)')
def pr(nm,m):
    print(f'  {nm:18s}',' | '.join(f'n{int((m&(T.split==sp)).sum()):4d} held {T[m&(T.split==sp)].held.mean():.0%} rest/R^ med {T[m&(T.split==sp)].rest.median():.2f} mean {T[m&(T.split==sp)].rest.mean():.2f}' for sp in ('train','test')))
pr('all',np.ones(len(T),bool))
for nm,m in (('U<.25',T.U<.25),('.25<=U<.5',(T.U>=.25)&(T.U<.5)),('U>=.5',T.U>=.5),('V1',T.vb),('V2',T.ab),('V1&V2',T.vb&T.ab),('time<12:00',T.em<720),('time>=12:00',T.em>=720)): pr(nm,m)
print('  calibration: realized RTH range / R^ (median) train',round((dd[dd.split=="train"].rng/dd[dd.split=="train"].R_B).median(),2),'test',round((dd[dd.split=="test"].rng/dd[dd.split=="test"].R_B).median(),2))
pickle.dump(T,open(SP+'vz264b.pkl','wb'))
