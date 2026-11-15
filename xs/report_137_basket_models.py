"""report 137: basket/structure models on small-cap flips. (1) random-matrix (Marchenko-Pastur) idiosyncrasy share; (2) empirical-Bayes partial pooling over sector x cap cells.
Strict split: fit on flips d<2022, judge 2022+. Target: BIG (net>=+30%) from combined_scores 'big'."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X='ext/'
O=pd.read_pickle(X+'combined_scores.pkl'); P=pd.read_pickle(X+'physics_flips.pkl')[['tk','d','PEAK']]
O=O.merge(P,on=['tk','d'],how='left').dropna(subset=['ATRP','EXT','LTURN','PEAK']).reset_index(drop=True)
O['tr']=O.d<'2022-01-01'; tr=O.tr
# OS = train ECDF ranks of ATRP+EXT+LTURN
def ecdf(c): 
    s=np.sort(O.loc[tr,c].values); return np.searchsorted(s,O[c].values)/len(s)
O['OS']=sum(ecdf(c) for c in ('ATRP','EXT','LTURN'))
# ---- (1) random matrix
W=pd.read_pickle(X+'wide_daily_ohlcv.pkl')
C=pd.DataFrame({k:v.close for k,v in W.items()}); C=C[~C.index.duplicated()]; R=np.log(C).diff().clip(-.5,.5)
mo=sorted({(d.year,d.month) for d in O.d})
idio={}; kk={}
for y,m in mo:
    t0=pd.Timestamp(y,m,1); win=R[(R.index<t0)].iloc[-250:]
    ok=win.columns[win.notna().sum()>=240]; Z=win[ok].fillna(0); Z=(Z-Z.mean())/(Z.std()+1e-9)
    T_,N_=Z.shape
    if T_<200 or N_<50: continue
    ev,V=np.linalg.eigh(np.corrcoef(Z.values.T)); lam=(1+np.sqrt(N_/T_))**2
    k=int((ev>lam).sum()); kk[(y,m)]=k
    Vk=V[:,-k:]*np.sqrt(ev[-k:]); share=1-(Vk**2).sum(1)           # idiosyncratic share of variance
    for tk,s in zip(ok,share): idio[(tk,y,m)]=s
O['IDIO']=[idio.get((r.tk,r.d.year,r.d.month),np.nan) for r in O.itertuples()]
print('MP signal factors per month: median',np.median(list(kk.values())),'range',min(kk.values()),max(kk.values()))
# ---- (2) partial pooling over sector x cap on residual log-peak after OS
sic=pd.read_csv(X+'sic.csv').set_index('tk').sic; O['sec']=(O.tk.map(sic)//100).fillna(-1).astype(int)
O['cap']=pd.qcut(O.MCAP,3,labels=False,duplicates='drop')
O['L']=np.log1p(O.PEAK)
b=np.polyfit(O.loc[tr,'OS'],O.loc[tr,'L'],1); O['res']=O.L-np.polyval(b,O.OS)
cell=O[tr].groupby(['sec','cap']).res.agg(['mean','count','var'])
s2w=O[tr].res.var(); tau2=max(cell['mean'].var()-s2w*(1/cell['count']).mean(),1e-6)
cell['shr']=cell['mean']*tau2/(tau2+s2w/cell['count']); print('pooling tau2 %.4f  within var %.4f  cells %d'%(tau2,s2w,len(cell)))
O['POOL']=[cell.shr.get((r.sec,r.cap),0.0) for r in O.itertuples()]
O['POOL_raw']=[cell['mean'].get((r.sec,r.cap),0.0) for r in O.itertuples()]
def auc(s,y,m): 
    return roc_auc_score(y[m],s[m]) if y[m].nunique()>1 else np.nan
print('\nAUC for BIG / HUGE  (train | test)')
for tgt in ('big','huge'):
    for c in ('OS','IDIO','POOL','POOL_raw'):
        x=O[c].fillna(O[c].median()); print(f'  {tgt:4s} {c:9s} {auc(x,O[tgt],tr):.3f} | {auc(x,O[tgt],~tr):.3f}')
print('\nlogistic add-on to OS  (BIG test AUC; fit on train)')
def fit(cols,tgt):
    A=O[cols].fillna(O[cols].median()); mu,sd=A[tr].mean(),A[tr].std(); A=(A-mu)/sd
    lr=LogisticRegression(C=1.0,max_iter=500).fit(A[tr],O.loc[tr,tgt]); p=lr.predict_proba(A)[:,1]
    return auc(p,O[tgt],~tr), lr.coef_[0]
for tgt in ('big','huge'):
    for cols in (['OS'],['OS','IDIO'],['OS','POOL'],['OS','IDIO','POOL']):
        a,cf=fit(cols,tgt); print(f'  {tgt:4s} {"+".join(cols):16s} test AUC {a:.3f}  coefs {np.round(cf,2)}')
# slot book on capped expectation proxy: top-20% by score, mean net return test
print('\ntest top-20% hit rates (BIG share, mean r%)')
te=O[~tr]
for c in ('OS','IDIO','POOL'):
    t=te[te[c]>=te[c].quantile(.8)]; print(f'  {c:5s} BIG {t.big.mean():.3f}  HUGE {t.huge.mean():.3f}  mean r {100*t.r.mean():+.1f}%  n {len(t)}')
print(f'  all   BIG {te.big.mean():.3f}  HUGE {te.huge.mean():.3f}  mean r {100*te.r.mean():+.1f}%')
O.to_pickle(X+'basket_models.pkl')
