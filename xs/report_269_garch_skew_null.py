"""Report 269 (declared before results). (A) Range forecast, quantile pinball as in 268: add GJR-GARCH(1,1) one-step vol (open-to-close + overnight daily returns, refit every 20d on trailing 500d, params frozen between refits) and QQQ 25d skew as features on HAR+IV. Keep only if both splits improve.
(B) Bachelier/reflection-principle null for 'extreme holds': at checkpoints 10:00..14:30 every 30m, driftless Brownian motion with remaining sigma from HAR+IV forecast (E[range]=1.596 sigma sqrt(T)), time profile of variance from train days. P(no new high in rest of day | d below high) = 2Phi(d/s_rem)-1 (same low). Compare empirical hold rate vs null by predicted-probability bin, split, and by gamma sign, X (range spent) bin. Excess over null = the part not explained by a random walk. Split train 2019-21 | test 2018+2022-26."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from scipy.stats import norm
from arch import arch_model
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index(); rth=b.between_time('09:30','15:55'); D=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'o':D.open.first(),'h':D.high.max(),'l':D.low.min(),'c':D.close.last(),'n':D.size()}); dd=dd[dd.n>=70]; dd['rng']=dd.h-dd.l; dd['lr']=np.log(dd.rng)
ov=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); ov.index=ov.index.normalize()
k=np.clip(ov.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1,0,len(ov)-1)
dd['imp']=np.log(ov.atm_iv.values[k]/np.sqrt(252)*dd.o.values); dd['skew']=ov.skew_25d.values[k]
gx=pd.read_csv('xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); kg=np.clip(gx.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1,0,len(gx)-1); dd['gpos']=gx.net_gex.values[kg]>0
ret=(np.log(dd.c/dd.c.shift(1))*100).values
dd['x1']=dd.lr.shift(1); dd['x5']=dd.lr.shift(1).rolling(5).mean(); dd['x22']=dd.lr.shift(1).rolling(22).mean()
# GJR vol (pct) one-step forecast, rolling refit
g=np.full(len(dd),np.nan)
for i in range(320,len(dd),20):
    r=ret[i-300:i]; r=r[np.isfinite(r)]
    try:
        res=arch_model(r,mean='Zero',vol='GARCH',p=1,o=1,q=1).fit(disp='off'); w,a,go,be=res.params[['omega','alpha[1]','gamma[1]','beta[1]']]; s2=res.conditional_volatility[-1]**2; rl=r[-1]
        for j in range(i,min(i+20,len(dd))):
            s2=w+(a+go*(rl<0))*rl**2+be*s2 if j>i else w+(a+go*(rl<0))*rl**2+be*s2
            g[j]=0.5*np.log(s2)+0  # forecast for day j uses info to j-1
            rl=ret[j] if np.isfinite(ret[j]) else rl
    except Exception: pass
dd['gjr']=g
dd=dd[dd.index>='2016-06-01'].dropna(subset=['x1','x5','x22','imp','gjr','skew'])
M={'HAR+IV':['x1','x5','x22','imp'],'HAR+IV+GJR':['x1','x5','x22','imp','gjr'],'HAR+IV+skew':['x1','x5','x22','imp','skew'],'HAR+GJR':['x1','x5','x22','gjr'],'HAR':['x1','x5','x22']}
QS=(.1,.5,.9); y=dd.lr.values; yr=dd.index.year.values; sp=np.where((yr>=2019)&(yr<=2021),'train','test'); ok=yr>=2018
def pin(yv,p,q): e=yv-p; return np.mean(np.maximum(q*e,(q-1)*e))
print('(A) pinball x100 train | test')
for nm,cols in M.items():
    X=sm.add_constant(dd[cols].values); P=np.full((len(dd),3),np.nan)
    for i in range(300,len(dd)):
        if (i-300)%20==0: fits=[sm.QuantReg(y[i-300:i],X[i-300:i]).fit(q=q,max_iter=60).params for q in QS]
        P[i]=[X[i]@f for f in fits]
    s=[]
    for t in ('train','test'):
        m=ok&(sp==t)&np.isfinite(P[:,0]); s.append(np.mean([pin(y[m],P[m,j],q) for j,q in enumerate(QS)])*100)
    print(f'  {nm:12s} {s[0]:.2f} | {s[1]:.2f}')
    if nm=='HAR+IV': Pf=P.copy()
# (B) null
bars=rth[['open','high','low','close']].copy(); bars['d']=bars.index.normalize(); bars['t']=bars.index.hour*60+bars.index.minute
# variance profile from train days: share of squared 5m returns after time t
tr=[d for d in dd.index if 2019<=d.year<=2021]
r5=np.log(bars.close).groupby(bars.d).diff()**2; r5=r5[bars.d.isin(tr)]; prof=r5.groupby(bars.t[r5.index]).sum(); prof=prof/prof.sum(); rem=prof[::-1].cumsum()[::-1].shift(-1).fillna(0)  # variance share strictly after bar t
CK=[600+30*i for i in range(10)]; rows=[]
med=Pf[:,1]; pos={d:i for i,d in enumerate(dd.index)}
G=bars.groupby('d')
for d,gdf in G:
    if d not in pos or not np.isfinite(med[pos[d]]) or d.year<2018: continue
    i=pos[d]; Rhat=np.exp(med[i]); sday=Rhat/1.596   # points, sigma*sqrt(T_day)
    hi=gdf.high.values; lo=gdf.low.values; cl=gdf.close.values; tt=gdf.t.values
    for c in CK:
        j=np.where(tt==c)[0]
        if len(j)==0: continue
        j=j[0]; H=hi[:j+1].max(); Lw=lo[:j+1].min(); srem=sday*np.sqrt(max(rem.get(c,0),1e-6))
        for side,dist,fut in (('hi',H-cl[j],hi[j+1:].max() if j+1<len(hi) else np.nan),('lo',cl[j]-Lw,lo[j+1:].min() if j+1<len(lo) else np.nan)):
            if not np.isfinite(fut): continue
            held=(fut<=H) if side=='hi' else (fut>=Lw)
            rows.append((d,c,side,dist/srem,2*norm.cdf(dist/srem)-1,held,(H-Lw)/Rhat,dd.gpos.iloc[i],'train' if 2019<=d.year<=2021 else 'test'))
R=pd.DataFrame(rows,columns=['d','c','side','z','pnull','held','X','gpos','split'])
print('\n(B) empirical hold vs null (P no new extreme in rest of day), by null-probability bin')
R['bin']=pd.cut(R.pnull,[0,.2,.4,.6,.8,.9,1.0001],include_lowest=True)
def tab(df,lbl):
    for s in ('train','test'):
        q=df[df.split==s]; g=q.groupby('bin').agg(n=('held','size'),nd=('d','nunique'),null=('pnull','mean'),emp=('held','mean'))
        print(f' {lbl} {s}:'); print('   '+g.assign(ex=g.emp-g.null).round(3).to_string().replace('\n','\n   '))
tab(R,'ALL')
print('\nexcess (emp-null) at pnull<.4 i.e. price at/near the extreme, by X bin and gamma')
q=R[R.pnull<.4].copy(); q['xb']=pd.cut(q.X,[0,.6,.8,1.0,9])
for s in ('train','test'):
    z=q[q.split==s]; print(' ',s); print('   '+z.assign(ex=z.held-z.pnull).groupby('xb').agg(n=('held','size'),emp=('held','mean'),null=('pnull','mean'),ex=('ex','mean')).round(3).to_string().replace('\n','\n   '))
    print('   '+z.assign(ex=z.held-z.pnull).groupby('gpos').agg(n=('held','size'),emp=('held','mean'),null=('pnull','mean'),ex=('ex','mean')).round(3).to_string().replace('\n','\n   '))
R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/null269.pkl')
