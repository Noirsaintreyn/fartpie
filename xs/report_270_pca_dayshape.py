"""Report 270 (declared before results). PCA fitted on TRAIN days only (2019-21): (A) shape of the first 2h (09:30-11:30, 24 5m closes, path from open divided by prior-20d mean range). Do the principal shapes (trend/reversal/V) predict the REST of day (11:30->15:55): continuation cont=sign(p1130-open)*(close-p1130)/R20 and rest efficiency erA? OLS on top-3 PC scores fitted on train, scored OOS on test (2018+2022-26) as R2 and corr; plus mean cont by PC1/PC2 tercile (terciles cut from train). (B) PCA of the decorrelated context block [HAR lags, IV, skew, gamma sign GEX, vanna, charm] -> full-day ER (efficiency); compare with raw regression."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index(); rth=b.between_time('09:30','15:55'); G=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'o':G.open.first(),'h':G.high.max(),'l':G.low.min(),'c':G.close.last(),'n':G.size()}); dd=dd[dd.n>=70]; dd['rng']=dd.h-dd.l; dd['R20']=dd.rng.shift(1).rolling(20).mean()
dd['er']=(dd.c-dd.o).abs()/dd.rng
P=[];Y=[]
for d,g in rth.groupby(rth.index.normalize()):
    if d not in dd.index: continue
    a=g.between_time('09:30','11:25'); rest=g.between_time('11:30','15:55')
    if len(a)<24 or len(rest)<50: continue
    o=dd.o[d]; R=dd.R20[d]; p=a.close.values[:24]
    pe=p[-1]; sgn=np.sign(pe-o) or 1; cont=sgn*(dd.c[d]-pe)/R if np.isfinite(R) else np.nan
    erA=abs(dd.c[d]-pe)/max(rest.high.max()-rest.low.min(),1e-9)
    P.append(pd.Series((p-o)/R,name=d)); Y.append((d,cont,erA))
Pm=pd.DataFrame(P); Y=pd.DataFrame(Y,columns=['d','cont','erA']).set_index('d'); ok=Pm.notna().all(1)&Y.cont.notna(); Pm=Pm[ok]; Y=Y[ok]
yr=Pm.index.year; tr=(yr>=2019)&(yr<=2021); te=(yr==2018)|(yr>=2022)
print('days train',tr.sum(),'test',te.sum())
pca=PCA(4).fit(Pm[tr]); S=pca.transform(Pm); print('(A) var explained',pca.explained_variance_ratio_.round(3))
print('   PC1 loads early/mid/late (idx 0,11,23):',pca.components_[0][[0,11,23]].round(2),' PC2:',pca.components_[1][[0,11,23]].round(2),' PC3:',pca.components_[2][[0,11,23]].round(2))
for tgt in ('cont','erA'):
    for nm,cols in (('PC1-3',[0,1,2]),('PC1',[0]),('PC2',[1]),('PC3',[2])):
        m=LinearRegression().fit(S[tr][:,cols],Y[tgt][tr]); pr=m.predict(S[:,cols]); out=[]
        for msk in (tr,te):
            yv=Y[tgt][msk].values; pp=pr[msk]; out.append(f'R2 {1-((yv-pp)**2).sum()/((yv-yv.mean())**2).sum():+.3f} corr {np.corrcoef(yv,pp)[0,1]:+.3f}')
        print(f'  {tgt:4s} ~ {nm:6s} train {out[0]} | test {out[1]}')
print('\nmean cont (rest-of-day move in direction of first 2h, units of R20) by train-cut tercile')
for j in (0,1,2):
    q=np.quantile(S[tr][:,j],[1/3,2/3]); lab=np.digitize(S[:,j],q)
    for t,lb in enumerate(('low','mid','high')):
        s=[]
        for msk in (tr,te):
            v=Y.cont[msk&(lab==t)]; s.append(f'n{len(v):4d} {v.mean():+.3f} (t{v.mean()/(v.std()/np.sqrt(len(v))):+.1f})')
        print(f'  PC{j+1} {lb:4s} {s[0]} | {s[1]}')
# B context PCA
ov=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); ov.index=ov.index.normalize()
v2=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_v2_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); v2.index=v2.index.normalize()
gx=pd.read_csv('xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); gx.index=gx.index.normalize()
F=pd.DataFrame(index=dd.index); lr=np.log(dd.rng)
F['x1']=lr.shift(1); F['x5']=lr.shift(1).rolling(5).mean(); F['x22']=lr.shift(1).rolling(22).mean()
def lag(s): k=np.clip(s.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1,0,len(s)-1); return s.values[k]
F['iv']=lag(ov.atm_iv); F['skew']=lag(ov.skew_25d); F['gex']=lag(gx.net_gex); F['vanna']=lag(v2.net_vanna); F['charm']=lag(v2.net_charm)
for c in ('gex','vanna','charm'): F[c]=F[c].rank(pct=True)  # crude; ranks use full sample (monotone, no sign leak) - acceptable for this exploratory comparison
F=F.dropna(); y=dd.er.reindex(F.index); yr=F.index.year; tr=(yr>=2019)&(yr<=2021); te=(yr==2018)|(yr>=2022)
mu=F[tr].mean(); sd=F[tr].std(); Z=(F-mu)/sd
pc=PCA(4).fit(Z[tr]); print('\n(B) context PCA var',pc.explained_variance_ratio_.round(3)); S2=pc.transform(Z)
for nm,Xm in (('raw 8',Z.values),('PCA 3',S2[:,:3]),('PCA 4',S2)):
    m=LinearRegression().fit(Xm[tr],y[tr]); pr=m.predict(Xm); o=[]
    for msk in (tr,te): yv=y[msk].values; o.append(f'R2 {1-((yv-pr[msk])**2).sum()/((yv-yv.mean())**2).sum():+.3f}')
    print(f'  day ER ~ {nm:6s} {o[0]} | {o[1]}')
