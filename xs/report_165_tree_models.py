"""report 165: decision tree / boosted trees / logistic on the NQ data, strict split (fit on 2018-21 ONLY, scored on 2022+, thresholds frozen from the train fit). DECLARED BEFORE RESULTS, small fixed feature sets, shallow models (tree depth 3 min leaf 30; HGB depth 2 x 100 x lr .05; logistic L2).
M1 META-LABEL on IFVG trades (target pts>0): features = dir-signed distance of entry from 5m EMA200/15m EMA21/1H EMA50/4H EMA21/daily EMA21 (in ATR5), dir*(RSI5-50), dir*gap/ATRd, dir*prior-day body/ATRd, minute of day, risk, pd position, idle-day flag. Score = mean pts of the trades the model rates in the top half (cut = train median score) vs the bottom half, B only.
M2 DAILY DIRECTION: target = next RTH open->close sign; features known at the 09:30 open: gap/ATR, open vs prior VAH/VAL/POC in ATR, prior tightness, daily RSI14, EMA20/50 distance in ATR, 5d return, ATRp, weekday. Score = OOS AUC on B and the mean open->close pts of the model's long/short calls when |p-.5|>=0.05."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; BL=pd.Timestamp('2022-01-01')
src=open('xs/report_164_ema_grid.py').read(); exec(src.split('base=cell')[0])
exec(open('xs/report_156_va_tightness.py').read().split("A=dd[dd.index<BL]")[0].split('"""',2)[2])
atr5=pd.Series(np.maximum.reduce([b.high-b.low,(b.high-b.close.shift()).abs(),(b.low-b.close.shift()).abs()])).rolling(14).mean().values
rsi5=None
def wild(x,n): return x.ewm(alpha=1/n,adjust=False).mean()
dl=b.close.diff(); rs5=(100-100/(1+wild(dl.clip(lower=0),14)/wild((-dl).clip(lower=0),14).replace(0,np.nan))).values
Dd=b[(b.m>=570)&(b.m<=955)].groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last')); Dd['atr']=(Dd.h-Dd.l).rolling(14).mean(); Dp=Dd.shift(1)
# ---- M1
j=t.j.values; dr=t.dr.values
F=pd.DataFrame(index=t.index)
for nm,(tf,L) in {'e5_200':('5m',200),'e15_21':('15m',21),'e1h_50':('1H',50),'e4h_21':('4H',21),'ed_21':('D',21)}.items(): F[nm]=dr*(t.ent.values-EM[(tf,L)][j])/atr5[j]
F['rsi']=dr*(rs5[j-1]-50); dd_=t.d.map(Dd.atr.shift(1)).values; F['gap']=dr*(t.d.map(Dd.o).values-t.d.map(Dp.c).values)/dd_; F['pb']=dr*(t.d.map(Dp.c).values-t.d.map(Dp.o).values)/dd_
F['min']=t.m.values; F['risk']=t.risk.values; F['pd']=t.pd.values; F['idle']=t.idle.astype(float).values; ok=F.notna().all(1).values; F=F[ok]; y=(t.pts.values[ok]>0).astype(int); pts=t.pts.values[ok]; A=(t.d.values[ok]<np.datetime64(BL))
print('M1 meta-label IFVG: train n',A.sum(),'test n',(~A).sum(),'| base mean pts A',pts[A].mean().round(2),'B',pts[~A].mean().round(2))
mods={'tree3':DecisionTreeClassifier(max_depth=3,min_samples_leaf=30,random_state=0),'hgb':HistGradientBoostingClassifier(max_depth=2,max_iter=100,learning_rate=.05,random_state=0),'logit':make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=500))}
for nm,m in mods.items():
    m.fit(F[A],y[A]); pa=m.predict_proba(F[A])[:,1]; pb=m.predict_proba(F[~A])[:,1]; cut=np.median(pa); top=pb>=cut
    print(f'  {nm:6s} AUC train {roc_auc_score(y[A],pa):.2f} | test {roc_auc_score(y[~A],pb):.2f}  test top-half mean {pts[~A][top].mean():+.2f} (n{top.sum()}) vs bottom {pts[~A][~top].mean():+.2f} (n{(~top).sum()})')
    if nm=='tree3': print(export_text(m,feature_names=list(F.columns),max_depth=3))
# ---- M2
rows=[]; vp={}
for i,d in enumerate(days):
    if i<60 or d<pd.Timestamp('2018-02-01') or d>pd.Timestamp('2026-01-30'): continue
    p=P[days[i-1]]; x=by[d]
    if x.m.iloc[0]!=570 or x.m.iloc[-1]!=955: continue
    a=Dd.atr.get(days[i-1]); o=x.open.iloc[0]; pc=Dd.c.get(days[i-1])
    if not a>0: continue
    c=Dd.c; k=Dd.index.get_loc(days[i-1]); e20=c.iloc[:k+1].ewm(span=20,adjust=False).mean().iloc[-1]; e50=c.iloc[:k+1].ewm(span=50,adjust=False).mean().iloc[-1]
    dl_=c.iloc[:k+1].diff().tail(60); r14=100-100/(1+(dl_.clip(lower=0).tail(14).mean()/max((-dl_).clip(lower=0).tail(14).mean(),1e-9)))
    rows.append(dict(d=d,y=int(x.close.iloc[-1]>o),oc=x.close.iloc[-1]-o,gap=(o-pc)/a,vah=(o-p['vah'])/a,val=(o-p['val'])/a,poc=(o-p['poc'])/a,t=p['t'],rsi=r14,e20=(pc-e20)/a,e50=(pc-e50)/a,r5=(pc/c.iloc[k-5]-1)*100,atrp=a/pc*100,dow=pd.Timestamp(d).dayofweek))
Z=pd.DataFrame(rows); Za=Z[Z.d<BL]; Zb=Z[Z.d>=BL]; X_=[c for c in Z.columns if c not in ('d','y','oc')]
print('\nM2 daily direction: train n',len(Za),'test n',len(Zb),' base up-rate A',Za.y.mean().round(3),'B',Zb.y.mean().round(3))
for nm,m in mods.items():
    m.fit(Za[X_],Za.y); pa=m.predict_proba(Za[X_])[:,1]; pb=m.predict_proba(Zb[X_])[:,1]; call=np.where(pb>=.55,1,np.where(pb<=.45,-1,0)); sel=call!=0
    pl=(Zb.oc.values*call-.5)[sel]; print(f'  {nm:6s} AUC train {roc_auc_score(Za.y,pa):.2f} | test {roc_auc_score(Zb.y,pb):.2f}  calls n{sel.sum()} mean {pl.mean():+.1f} pts (t {pl.mean()/(pl.std()/np.sqrt(len(pl))) if len(pl)>2 else 0:+.1f})')
    if nm=='tree3': print(export_text(m,feature_names=X_,max_depth=3))
