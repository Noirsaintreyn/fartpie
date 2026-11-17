"""Report 268: (A) QUANTILE range forecasts, HAR vs Yang-Zhang vs implied (Rey: 'YZ was better for the zones, quantile something' - I could not find that result in the logs, so tested directly). Quantile regression (q10, q50, q90) of log RTH range, rolling 500 sessions refit every 20, out-of-sample; models: HAR | HAR+IV | HAR+YZ5 | HAR+YZ20 | HAR+IV+YZ20 | IV+YZ20 (no HAR). Metrics per split (train 2019-21 | test 2018+2022-26): average pinball loss (x100, lower better) over the three quantiles, and coverage of [q10,q90] (target 80%).
(B) The ONE mildly mean-reverting zone from report 266 (zone-touch rejection, X = range so far / R^_t in [0.8, 1.0)) on the ex-post day type (MR / MID / TREND tertiles of RTH efficiency = the 'cheating' view) and on prior-close QQQ net GEX sign (positive vs negative gamma); targets VWAP / EMA20 / mid. Not tradable by day type (hindsight) - it only shows where the signal lives; gamma is causal."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index(); rth=b.between_time('09:30','15:55'); D=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'o':D.open.first(),'h':D.high.max(),'l':D.low.min(),'c':D.close.last(),'n':D.size()}); dd=dd[dd.n>=70]; dd['rng']=dd.h-dd.l; dd['lr']=np.log(dd.rng)
ov=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); ov.index=ov.index.normalize()
k=ov.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1; k=np.clip(k,0,len(ov)-1); dd['imp']=np.log(ov.atm_iv.values[k]/np.sqrt(252)*dd.o.values)
on=np.log(dd.o/dd.c.shift(1)); oc=np.log(dd.c/dd.o); hh=np.log(dd.h/dd.o); ll=np.log(dd.l/dd.o); rs=hh*(hh-oc)+ll*(ll-oc)
for n in (5,20):
    kk=.34/(1.34+(n+1)/(n-1)); yz=on.rolling(n).var()+kk*oc.rolling(n).var()+(1-kk)*rs.rolling(n).mean(); dd[f'yz{n}']=np.log(np.sqrt(yz.clip(lower=1e-12))*dd.o).shift(1)
dd['x1']=dd.lr.shift(1); dd['x5']=dd.lr.shift(1).rolling(5).mean(); dd['x22']=dd.lr.shift(1).rolling(22).mean(); dd=dd[dd.index>='2016-06-01'].dropna(subset=['x1','x5','x22','yz5','yz20'])
M={'HAR':['x1','x5','x22'],'HAR+IV':['x1','x5','x22','imp'],'HAR+YZ5':['x1','x5','x22','yz5'],'HAR+YZ20':['x1','x5','x22','yz20'],'HAR+IV+YZ20':['x1','x5','x22','imp','yz20'],'IV+YZ20':['imp','yz20']}
QS=(.1,.5,.9); y=dd.lr.values; PRED={}
for nm,cols in M.items():
    X=sm.add_constant(dd[cols].values); P=np.full((len(dd),3),np.nan)
    for i in range(500,len(dd)):
        if i==500 or (i-500)%20==0:
            fits=[sm.QuantReg(y[i-500:i],X[i-500:i]).fit(q=q,max_iter=200).params for q in QS]
        P[i]=[X[i]@f for f in fits]
    PRED[nm]=P
dd['yr']=dd.index.year; dd['split']=np.where((dd.yr>=2019)&(dd.yr<=2021),'train','test'); ok=(dd.yr>=2018).values
def pin(yv,p,q): e=yv-p; return np.mean(np.maximum(q*e,(q-1)*e))
print('quantile range forecasts: avg pinball x100 | coverage of [q10,q90]  (train | test)')
for nm,P in PRED.items():
    s=[]
    for sp in ('train','test'):
        m=ok&(dd.split==sp).values&np.isfinite(P[:,0]); L=np.mean([pin(y[m],P[m,j],q) for j,q in enumerate(QS)])*100; cov=np.mean((y[m]>=P[m,0])&(y[m]<=P[m,2])); s.append(f'{L:.2f} | {cov:.0%}')
    print(f'  {nm:12s} train {s[0]}   test {s[1]}')
# B: zone fade by day type and gamma
T,C=pickle.load(open(SP+'zones266.pkl','rb')); E=pickle.load(open(SP+'ev256.pkl','rb')); gx=pd.read_csv('xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').sort_index()
kg=gx.index.searchsorted(pd.DatetimeIndex(T.d).values.astype('datetime64[ns]'))-1; kg=np.clip(kg,0,len(gx)-1); T['gpos']=(gx.net_gex.values[kg]>0)
T['type']=T.d.map(E.type); Z=T[(T.grp=='zone')&(T.X>=.8)&(T.X<1.0)].copy()
def tt(x): x=x.dropna(); return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>2 and x.std()>0 else np.nan
def show(nm,m,df=None):
    df=Z if df is None else df
    for col in ('p_vwap','p_ema','p_mid'):
        s=[]
        for sp in ('train','test'):
            q=df[m.reindex(df.index,fill_value=False)&(df.split==sp)][col].dropna() if hasattr(m,'reindex') else df[m&(df.split==sp)][col].dropna(); s.append(f'n{len(q):4d} {q.mean():+6.1f} (t{tt(q):+.1f})')
        print(f'  {nm:24s} {col:7s} {s[0]} | {s[1]}')
print('\n--- zone fade X in [.8,1.0): n',len(Z))
print('by ex-post day type (hindsight view):'); 
for ty in ('MR','MID','TREND'): show(f'type {ty}',Z.type==ty)
print('by prior-close QQQ gamma (causal):')
show('positive gamma',Z.gpos); show('negative gamma',~Z.gpos)
print('all zone fades (X any) by gamma:')
A=T[T.grp=='zone']; show('positive gamma',A.gpos,A); show('negative gamma',~A.gpos,A)
print('placebo (new extreme outside zone) by gamma:'); P_=T[T.grp=='placebo']; show('positive gamma',P_.gpos,P_); show('negative gamma',~P_.gpos,P_)
