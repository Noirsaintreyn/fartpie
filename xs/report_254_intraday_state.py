"""Report 254: INTRADAY CAUSAL STATE -> does the rest of the day mean-revert? DECLARED before results (Rey Nov 17: vector flips, volume at extremes / absorption / tapering, options flow, ATRP/IV, overnight inventory; not VWAP-based, no Hurst/GMM/HMM).
Checkpoint t = 11:00 ET (also 10:30). Label (hindsight, about the REST of the day only): ER_after = |close 15:55 - close at t| / (high-low of bars after t); MR_after = bottom third, TREND_after = top third.
Features at t (all causal), hypothesis direction fixed: 
 F1 vector flips: count of 15m vector regime flips 09:30-t (more flips -> MR).
 F2 volume share at extremes: share of 09:30-t volume in bars whose high >= running high - .1 ATR or low <= running low + .1 ATR (more -> MR).
 F3 taper: mean volume of the last 2 new-session-extreme bars / first 2 (needs >=4) (lower -> MR).
 F4 absorption: -sign(move since open) * cum est. delta / volume (higher = flow against price -> MR).
 F5 range used: (high-low so far)/(prior-close QQQ ATM IV/sqrt(252) x open) (higher -> MR, range already spent).
 F6 ATRP: ATR14/close trailing-252 rank (lower -> MR).
 F7 options flow: prior-day QQQ call vol / put vol (higher -> MR); F7b net GEX > 0 (-> MR).
 F8 overnight inventory: |ON net move|/ON range (18:00-09:30) (higher -> MR) and F8b: open at ON extreme (|open-ON mid|/ON half range, higher -> MR).
Check 1: AUC MR_after vs TREND_after per feature, train 2019-21 vs test 2018+2022-26 (a feature counts if on the hypothesized side in BOTH). Check 2: combined score = count of counting features on the MR side at t; fade (VWAP-free: fade of a >=.75 ATR extension from the 09:30 open, stop 1 ATR beyond, target open, flat 15:55, cost 1) entered after t, by score."""
import sys, numpy as np, pandas as pd, pickle
sys.path.insert(0,'/home/claude/fartpie/xs'); from vector_pine import run_state_machine
from scipy.stats import rankdata
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.sort_values('dt').set_index('dt')
g=b.resample('15min',label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
res=run_state_machine(g.assign(datetime=g.index)); reg=pd.Series(np.asarray(res[0]),index=g.index)
flip=(reg!=reg.shift(1))&reg.notna()&reg.shift(1).notna()&(reg!=0)
rth=b.between_time('09:30','15:55').copy(); rth['d']=rth.index.normalize()
on=b.between_time('18:00','23:59'); on2=b.between_time('00:00','09:29'); 
Dg=rth.groupby('d'); dd=pd.DataFrame({'o':Dg.open.first(),'h':Dg.high.max(),'l':Dg.low.min(),'c':Dg.close.last(),'n':Dg.size()}); dd=dd[dd.n>=70]; dd['atr']=(dd.h-dd.l).rolling(14).mean().shift(1)
dd['atrp']=(dd.atr/dd.c.shift(1)); dd['atrp_r']=dd.atrp.rolling(252,min_periods=120).rank(pct=True)
ov=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); ov.index=ov.index.normalize()
gx=pd.read_csv('/home/claude/fartpie/xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').sort_index()
rows=[]; fadetr=[]
bars_by_day={d:g_ for d,g_ in Dg}
# overnight per day: 18:00 prev -> 09:29
bi=b.index
for d,gg in bars_by_day.items():
    if d not in dd.index or not np.isfinite(dd.atr[d]): continue
    a=dd.atr[d]; o_=gg.open.values; h=gg.high.values; l=gg.low.values; c=gg.close.values; v=gg.volume.values.astype(float); t=gg.index
    mm=t.hour*60+t.minute
    ons=b.loc[d-pd.Timedelta(hours=6):d+pd.Timedelta(hours=9,minutes=29)]
    if len(ons)<50: continue
    onr=ons.high.max()-ons.low.min(); onmv=ons.close.iloc[-1]-ons.open.iloc[0]
    for CK in (630,660):
        ix=np.where(mm<CK)[0]; ia=np.where(mm>=CK)[0]
        if len(ix)<6 or len(ia)<30: continue
        rh=np.maximum.accumulate(h[ix]); rl=np.minimum.accumulate(l[ix])
        ext=((h[ix]>=np.r_[-np.inf,rh[:-1]]-.1*a)&(h[ix]>=rh-.1*a))|((l[ix]<=rl+.1*a)&(l[ix]<=rl+.1*a))
        F2=v[ix][ext].sum()/v[ix].sum()
        newx=np.where((h[ix]>np.r_[-np.inf,rh[:-1]])|(l[ix]<np.r_[np.inf,rl[:-1]]))[0]
        F3=(v[ix][newx[-2:]].mean()/v[ix][newx[:2]].mean()) if len(newx)>=4 else np.nan
        dl=v[ix]*(2*c[ix]-h[ix]-l[ix])/np.maximum(h[ix]-l[ix],1e-9); mv=c[ix][-1]-o_[0]
        F4=-np.sign(mv)*dl.sum()/v[ix].sum()
        ks=ov.index.searchsorted(d)-1
        iv=ov.atm_iv.iloc[ks] if ks>=0 else np.nan; F5=(rh[-1]-rl[-1])/(iv/np.sqrt(252)*o_[0]) if np.isfinite(iv) else np.nan
        cv=ov.call_volume.iloc[ks]/max(ov.put_volume.iloc[ks],1) if ks>=0 else np.nan
        kg=gx.index.searchsorted(d)-1; gpos=float(gx.net_gex.iloc[kg]>0) if kg>=0 else np.nan
        F8=abs(onmv)/onr if onr>0 else np.nan; F8b=abs(o_[0]-(ons.high.max()+ons.low.min())/2)/(onr/2) if onr>0 else np.nan
        te=t[ix[-1]]; t0=t[ix[0]]
        F1=int(flip.loc[t0:te].sum())
        ca=c[ia]; erA=abs(ca[-1]-c[ix][-1])/max(h[ia].max()-l[ia].min(),1e-9)
        rows.append(dict(d=d,ck=CK,F1=F1,F2=F2,F3=F3,F4=F4,F5=F5,F6=dd.atrp_r[d],F7=cv,F7b=gpos,F8=F8,F8b=F8b,erA=erA))
R=pd.DataFrame(rows); R['yr']=R.d.dt.year; R['split']=np.where((R.yr>=2019)&(R.yr<=2021),'train','test')
HYP={'F1':1,'F2':1,'F3':-1,'F4':1,'F5':1,'F6':-1,'F7':1,'F7b':1,'F8':1,'F8b':1}
def auc(x,y):
    r=rankdata(x); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
cnt={}
for CK in (630,660):
    X=R[R.ck==CK].copy(); lo,hi=X.erA.quantile([1/3,2/3]); X['lab']=np.where(X.erA<=lo,1,np.where(X.erA>=hi,0,np.nan)); X=X.dropna(subset=['lab'])
    print(f'--- checkpoint {CK//60}:{CK%60:02d}  AUC of hypothesized-MR direction (0.5 none); n',len(X))
    for f,sg in HYP.items():
        out=[]
        for sp,gx_ in X.groupby('split'):
            z=gx_.dropna(subset=[f]); out.append(auc(sg*z[f].values,z.lab.values.astype(int)))
        ok=all(o>0.52 for o in out); cnt[(CK,f)]=ok
        print(f'  {f:4s} train {out[1]:.3f} test {out[0]:.3f} {"<- counts" if ok else ""}')
pickle.dump((R,cnt),open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/state254.pkl','wb'))
print('counting features:',{k for k,v in cnt.items() if v})
