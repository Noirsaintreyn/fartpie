"""Nov 14 2026: report_34 redone with a PROPER expected range known at the open. HAR-style OLS on log RTH range: features at day D = log(range D-1), log(mean range D-5..D-1), log(mean D-20..D-1), log(overnight range D, 18:00->09:29), log(1+|open gap|); target log(range D); expanding fit on days < D (>=250 obs, refit every 21 days), Duan smearing. ER_D = forecast RTH range. rho2 = ER_D / prior-day VA width; expanding-PIT terciles (>=120 prior days). Same models and splits as report_34 (strict eras + five regimes). Plus VA-80 GATE test: trade VA-80 only when rho2 = LOW. Forecast quality vs ADR20 is printed first."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/report_19_amt_papers.py').read().split("def sim(")[0].split('"""',2)[2])
nq=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); nq['dt']=pd.to_datetime(nq.dt); nq=nq.set_index('dt').sort_index()
eth=nq[(nq.index.hour>=18)|(nq.index.hour<9)|((nq.index.hour==9)&(nq.index.minute<30))].copy(); eth['td']=(eth.index+pd.Timedelta(hours=6)).normalize()
onr=eth.groupby('td').apply(lambda g:g.high.max()-g.low.min())
dl=[d for d in days if d in RTH]; rng=pd.Series({d:RTH[d].high.max()-RTH[d].low.min() for d in dl}); opn=pd.Series({d:RTH[d].open.values[0] for d in dl}); cls=pd.Series({d:RTH[d].close.values[-1] for d in dl})
F=pd.DataFrame({'rng':rng}); F['lr1']=np.log(rng.shift(1)); F['lr5']=np.log(rng.rolling(5).mean().shift(1)); F['lr20']=np.log(rng.rolling(20).mean().shift(1)); F['lon']=np.log(onr.reindex(F.index)); F['lgap']=np.log1p((opn-cls.shift(1)).abs()); F['y']=np.log(F.rng)
F['adr20']=rng.rolling(20).mean().shift(1); X=['lr1','lr5','lr20','lon','lgap']; F=F.dropna(subset=X+['y']).copy(); idx=F.index
pred=pd.Series(np.nan,index=idx); beta=None
for i,d in enumerate(idx):
    if i<250: continue
    if beta is None or (i-250)%21==0:
        tr=F.iloc[:i]; A_=np.c_[np.ones(len(tr)),tr[X].values]; beta=np.linalg.lstsq(A_,tr.y.values,rcond=None)[0]; res=tr.y.values-A_@beta; smear=np.mean(np.exp(res))
    pred[d]=np.exp(beta[0]+F.loc[d,X].values@beta[1:])*smear
F['er']=pred; V=F.dropna(subset=['er'])
print('forecast quality (days with forecast: %d): corr(log ER, log range) HAR %.3f vs ADR20 %.3f | MAE%% HAR %.1f vs ADR20 %.1f'%(len(V),np.corrcoef(np.log(V.er),V.y)[0,1],np.corrcoef(np.log(V.adr20),V.y)[0,1],100*np.mean(np.abs(V.er/V.rng-1)),100*np.mean(np.abs(V.adr20/V.rng-1))))
rows=[]
for d in V.index:
    j=dl.index(d)
    if j<1 or dl[j-1] not in PRO: continue
    pO,vah,val=PRO[dl[j-1]]; w=vah-val
    if w>0: rows.append(dict(day=d,rho=V.er[d]/w,rho_adr=V.adr20[d]/w,act=V.rng[d]/V.er[d]))
R=pd.DataFrame(rows).set_index('day'); R['rk']=[(R.rho.iloc[:k]<R.rho.iloc[k]).mean() if k>=120 else np.nan for k in range(len(R))]
R['b']=np.where(R.rk<=1/3,'LOW',np.where(R.rk>=2/3,'HIGH','MID')); R.loc[R.rk.isna(),'b']=None
print('days',len(R),'| mean actual/forecast range by bucket:',R.groupby('b').act.mean().round(2).to_dict())
RGS=[('R1','2018-01-01','2020-03-01'),('R2','2020-03-01','2022-01-01'),('R3','2022-01-01','2023-04-01'),('R4','2023-04-01','2025-04-01'),('R5','2025-04-01','2027-01-01')]
rg=lambda t: next((n for n,a,b in RGS if pd.Timestamp(a)<=t<pd.Timestamp(b)),None); ERA={'18-21':('2018-01-01','2022-01-01'),'22+':('2022-01-01','2027-01-01')}
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>5 and len(b)>5 else np.nan
def table(df,name,pred):
    print(f'\n{name}  [prediction: {pred}]')
    for e,(a,b) in ERA.items():
        s=df[(df.day>=a)&(df.day<b)&df.b.notna()]; lo=s[s.b=='LOW'].pts; mi=s[s.b=='MID'].pts; hi=s[s.b=='HIGH'].pts
        print(f'   {e:6s} LOW {lo.mean():+7.2f} (n{len(lo)}) | MID {mi.mean():+7.2f} (n{len(mi)}) | HIGH {hi.mean():+7.2f} (n{len(hi)}) | LOW-HIGH {lo.mean()-hi.mean():+6.2f} t {tt(lo,hi):+.1f}')
    line='   by regime LOW/HIGH: '
    for n,_,_ in RGS:
        s=df[(df.day.map(rg)==n)&df.b.notna()]; lo=s[s.b=='LOW'].pts; hi=s[s.b=='HIGH'].pts; line+=f'{n} {lo.mean():+.1f}/{hi.mean():+.1f} (n{len(lo)}/{len(hi)})  '
    print(line)
print('\n#### MECHANISM by rho (forecast): actual RTH range / forecast, close inside prior VA')
for e,(a,b) in ERA.items():
    s=R[(R.index>=a)&(R.index<b)&R.b.notna()]
    for bk in ('LOW','MID','HIGH'): x=s[s.b==bk]; print(f'  {e:6s} {bk:4s} n={len(x):4d} actual/forecast {x.act.mean():.2f}')
A=pd.read_pickle(SC+'amt_models_T.pkl'); A['day']=pd.to_datetime(A.d).dt.normalize(); A=A.join(R[['b']],on='day')
C=pd.read_pickle(SC+'icc_T.pkl'); C['day']=pd.to_datetime(C.d).dt.normalize(); C=C.join(R[['b']],on='day')
for m,pr in (('VA-80','LOW > HIGH (rotation works when range fits in value)'),('VA-F','LOW > HIGH'),('INV-F','LOW > HIGH'),('VA-B','HIGH > LOW'),('IB-B','HIGH > LOW'),('SHAPE','?')): table(A[A.model==m],f'NQ {m} (pts/trade)',pr)
table(C,'ICT chain (pts/trade)','?')
import pickle
S=pickle.load(open(SC+'s6.pkl','rb')); legs=S[0]['V2']; nm=S[1]; allp=[]
for (T,u,_),n in zip(legs,nm):
    x=T.copy(); x['day']=pd.to_datetime(x.day).dt.normalize(); x['pts']=x.pts*x.sz*u; x=x.join(R[['b']],on='day'); allp.append(x[['day','pts','b']]); table(x,f'main-algo leg {n}','?')
d=pd.concat(allp).groupby('day').agg(pts=('pts','sum'),b=('b','first')).reset_index(); table(d,'main algo all legs (pts/DAY)','?')
print('\n#### VA-80 GATE: trade only when rho2 = LOW (pts/trade, all LOW trades vs all trades)')
v=A[(A.model=='VA-80')&A.b.notna()]
for e,(a,b) in ERA.items():
    s=v[(v.day>=a)&(v.day<b)]; g=s[s.b=='LOW']; print(f'   {e:6s} ungated {s.pts.mean():+6.2f} (n{len(s)}, win {np.mean(s.pts>0):.0%}) | gated LOW {g.pts.mean():+6.2f} (n{len(g)}, win {np.mean(g.pts>0):.0%}, t {g.pts.mean()/(g.pts.std()/len(g)**.5):+.1f})')
g=v[v.b=='LOW']; print('   gated LOW by year:',' '.join(f'{y}:{x.pts.mean():+.1f}(n{len(x)})' for y,x in g.groupby(g.day.dt.year)))
R.to_pickle(SC+'rho2_R.pkl')
