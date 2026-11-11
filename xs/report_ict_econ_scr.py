"""ICT + ECON features on screener flips (9 tickers with 15m bars). Features use only data before entry bar."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
N=6; BRK=.08
T=pd.read_parquet('xs/ext/wide_T.parquet'); T=T[T.sp&(T.ratio>=1.5)].copy()
F=[]
for tk in sorted(T.tk.unique()):
    f=glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet')
    if not f: continue
    H=pd.read_parquet(f[0],columns=['datetime','open','high','low','close']); H['dt']=pd.to_datetime(H.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m=H.dt.dt.hour*60+H.dt.dt.minute; H=H[(m>=570)&(m<960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    H['d']=H.dt.dt.normalize()
    D=H.groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'))
    r=np.log(D.c).diff(); gap=D.o/D.c.shift()-1
    E=pd.DataFrame(index=D.index)
    E['gz']=(gap/gap.abs().rolling(20).mean()).shift(0)*1.0          # today's open is known at entry
    E['gz']=gap/gap.abs().rolling(20).mean().shift()
    E['volr']=(r.rolling(5).std()/r.rolling(20).std()).shift()
    E['vr']=((np.log(D.c).diff(5).rolling(60).var())/(5*r.rolling(60).var())).shift()
    E['ac']=r.rolling(60).apply(lambda z: pd.Series(z).autocorr(1),raw=True).shift()
    E['tt']=(r.rolling(20).mean()/(r.rolling(20).std()/20**.5)).shift()
    E['pdl']=D.l.shift(); E['pdh']=D.h.shift()
    atr=(H.high-H.low).rolling(26).mean().shift()
    H['atr']=atr
    # intraday ICT, per bar using only bars up to and including previous bar
    H['body']=(H.close-H.open)
    H['disp']=((H.body>1.5*H.atr)).astype(int)
    H['fvg']=((H.low>H.high.shift(2))).astype(int)   # bullish FVG completes on bar t
    rows=[]
    for d,g in H.groupby('d'):
        g=g.reset_index(drop=True); pdl=E.pdl.get(d); pdh=E.pdh.get(d)
        lo=g.low.cummin(); hi=g.high.cummax()
        sw_lo=((lo<pdl)&(g.close>pdl)).astype(int).cummax() if pdl==pdl else 0   # swept PDL & reclaimed
        sw_hi=((hi>pdh)&(g.close>pdh)).astype(int).cummax() if pdh==pdh else 0   # broke PDH & holding
        sh=g.high.rolling(4).max().shift(1)                                      # swing/recent high
        mss=((g.close>g.high.shift(1).rolling(6).max())).astype(int).cummax()    # close above last 6 bars high
        dsp=g.disp.rolling(8,min_periods=1).max(); fv=g.fvg.rolling(12,min_periods=1).max()
        x=pd.DataFrame({'dt':g.dt,'swlo':sw_lo,'swhi':sw_hi,'mss':mss,'dsp':dsp,'fvg':fv})
        x[['swlo','swhi','mss','dsp','fvg']]=x[['swlo','swhi','mss','dsp','fvg']].shift(1).fillna(0)  # only prior bars
        rows.append(x)
    I=pd.concat(rows).set_index('dt')
    t=T[T.tk==tk].copy(); t['d']=t.entry.dt.normalize()
    t=t.join(E[['gz','volr','vr','ac','tt']],on='d').join(I,on='entry'); F.append(t)
T=pd.concat(F).dropna(subset=['gz','volr','vr','ac','tt']); T['yr']=T.entry.dt.year; print(len(T),'trades')
def row(n,x):
    if len(x)<30: return f'{n:22s} n {len(x)}'
    return f'{n:22s} n {len(x):5d} {x.ret.mean()*1e4:+6.1f}bp win {(x.ret>0).mean():.0%} t {x.ret.mean()/(x.ret.std()/len(x)**.5):+.1f}'
print('=== ICT flags (1 vs 0), by era')
for c in ('swlo','swhi','mss','dsp','fvg'):
    for e,m in (('19-21',T.yr<=2021),('22-26',T.yr>=2022)):
        a=T[m&(T[c]>0)];b=T[m&(T[c]==0)]; print(c,e,'| ON',row('',a),'| OFF',row('',b))
print('=== ECON terciles (cut on 19-21), by era')
cut=T[T.yr<=2021]
for c in ('gz','volr','vr','ac','tt'):
    q1,q2=cut[c].quantile([1/3,2/3])
    for e,m in (('19-21',T.yr<=2021),('22-26',T.yr>=2022)):
        z=T[m]; print(c,e,'| low',row('',z[z[c]<=q1]),'| mid',row('',z[(z[c]>q1)&(z[c]<=q2)]),'| high',row('',z[z[c]>q2]))
# composite scores, thresholds from 19-21
ict=['swlo','mss','dsp','fvg']; T['ict']=T[ict].sum(axis=1)
sign={c:np.sign(cut[[c]].join(cut.ret).corr().iloc[0,1]) for c in ('gz','volr','vr','ac','tt')}; print('econ signs from 19-21:',sign)
def rk(c): return T[c].rank(pct=True)*sign[c]
T['econ']=sum((T[c]-cut[c].mean())/cut[c].std()*sign[c] for c in sign)
qe=cut.assign(econ=sum((cut[c]-cut[c].mean())/cut[c].std()*sign[c] for c in sign)).econ.quantile([1/3,2/3])
T['et']=np.where(T.econ<=qe.iloc[0],.5,np.where(T.econ>qe.iloc[1],1.5,1.))
T['it']=np.where(T.ict<=0,.75,np.where(T.ict>=2,1.25,1.))
T['ie']=T.et*T.it
print('mean ret by econ third / ict count:'); 
for e,m in (('19-21',T.yr<=2021),('22-26',T.yr>=2022)):
    z=T[m]; print(e,'econ',[round(z[z.et==k].ret.mean()*1e4,1) for k in (.5,1,1.5)],'ict0/1/2+',[round(z[(z.ict==k)|((k==2)&(z.ict>=2))].ret.mean()*1e4,1) for k in (0,1,2)])
def sim(Tx,w=None):
    Tx=Tx.sort_values(['entry','ratio'],ascending=[True,False],kind='stable')
    op=[];pend=[];real=0.;hist=[];win=deque();lvl0=0.;out=[]
    for e,x,r,wt in zip(Tx.entry.values,Tx.exit.values,Tx.ret.values,(Tx[w].values if w else np.ones(len(Tx)))):
        done=sorted([p for p in pend if p[0]<=e],key=lambda z:z[0]);pend=[p for p in pend if p[0]>e]
        for xt,rr in done: real+=np.log1p(rr/N);hist.append((xt,real));win.append((xt,real))
        op=[o for o in op if o>e]
        if len(op)>=N: continue
        if hist:
            t0=e-np.timedelta64(28,'D')
            while win and win[0][0]<t0: lvl0=win.popleft()[1]
            if real-max([lvl0]+[h for _,h in win])<np.log1p(-BRK): continue
        op.append(x);pend.append((x,r*wt));out.append((pd.Timestamp(e),r*wt))
    s=pd.DataFrame(out,columns=['e','r']);s['yr']=s.e.dt.year;return s
def rep(n,Tx,w=None):
    s=sim(Tx,w);o=[]
    for lab,m in (('19-21',s.yr<=2021),('22-26',s.yr>=2022),('all',s.yr>0)):
        z=s[m];y=max((z.e.max()-z.e.min()).days/365.25,.1);o.append(f'{lab}: n {len(z)} ann~{np.expm1(np.log1p(z.r/N).sum()/y):+.1%}')
    print(f'{n:30s} '+' | '.join(o))
print('=== slot sim (weights scale position size; slots unchanged)')
rep('base',T); rep('ECON tilt .5/1.5',T,'et'); rep('ICT tilt .75/1.25',T,'it'); rep('ICT x ECON',T,'ie')
rep('skip ECON bottom third',T[T.et>.5]); rep('skip ICT==0',T[T.ict>0]); rep('skip both (econ low & ict 0)',T[~((T.et==.5)&(T.ict==0))])
