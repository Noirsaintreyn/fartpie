"""Nov 12 2026: PATH-SIGNATURE META-LABEL on the vector's 1h bull flips + the standard 'MA-whipsaw' fixes re-expressed with the VECTOR instead of an EMA/SMA. DECLARED BEFORE RESULTS.
Literature fixes for MA trend systems that enter late/in chop (StockCharts/Hill whipsaw filters, Kaufman efficiency ratio, ATR-band hysteresis, Chandelier/SuperTrend ratchets, vol targeting, meta-labelling): the vector already contains the hysteresis (ATR-break + confirm bar) and the ratchet. What remains testable:
 (a) ER gate: Kaufman efficiency ratio of the pre-entry path (|net move| / path length) over 20 and 48 bars as a trend-quality filter (high = clean trend, low = chop);
 (b) band-distance gate: close - vector level in ATR (the 'quantifying filter' with the vector as the MA);
 (c) PATH-SIGNATURE meta-label (Rey's attachment): 4 channels over the last 48 bars before the flip close: X1 cumulative log return / ATR%, X2 (close - vector level)/ATR [vector replaces the trend estimate], X3 cumulative return minus the equal-weight universe return, X4 time. Depth-2 signature (4 + 16 features), standardised, L2 logistic (C=.05). Label = trade net return > 0 under the fixed exit (next open after bear flip), 40bp.
Three models per book on the SAME events: S0 simple inputs (ext, rv, atrp, lpx, er20, er48) | S1 simple + signature | SIG signature only. Train < 2025-07-01, test after; acceptance = test AUC gain of S1 over S0 and kept-trade mean/ex-top-1% beating random-skip (2000 draws). Books: $1-5 and BIG proxy. Everything is exploratory except the S1-vs-S0 comparison."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from vector_pine import run_state_machine
from vector_core import compute_atr_series
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); SPLIT=pd.Timestamp('2025-07-01'); W=48
TK={tk:Y.dropna() for tk,Y in data.items() if len(Y.dropna())>=1500}; MKT=pd.concat({tk:np.log(Y.close).diff() for tk,Y in TK.items()},axis=1).mean(axis=1)
dv={tk:(Y.close*Y.volume).groupby(Y.index.normalize()).sum().median() for tk,Y in TK.items()}; big={t for t,v in dv.items() if v>=5e7}
def sig2(X):
    d=np.diff(X,axis=0); X0=X[:-1]-X[0]; S1=X[-1]-X[0]; S2=(X0.T@d)+0.5*(d.T@d).T   # S2[i,j]=sum_k (Xi(k)-Xi(0)) dXj(k) + 0.5 dXi dXj
    return np.r_[S1,S2.ravel()]
rows=[]; sg=[]
for tk,Y in TK.items():
    idx=Y.index; o,hi,lo,c,vol=[Y[k].values for k in ('open','high','low','close','volume')]; atr=compute_atr_series(hi,lo,c,14); reg,lev,_,fl=run_state_machine(Y.assign(datetime=idx)); m=MKT.reindex(idx).fillna(0).values; lr=np.r_[0,np.diff(np.log(c))]; vm=pd.Series(vol).rolling(20).mean().shift().values; pos=None
    for i,s in fl:
        if s==1 and pos is None: pos=i
        elif s==-1 and pos is not None:
            fi=pos; xb=i+1; pos=None
            if xb>=len(Y) or fi<250 or not np.isfinite(atr[fi]) or not np.isfinite(vm[fi]): continue
            e=fi+1; px=o[e]; sm=1<=px<=5; bg=(tk in big and px>=10)
            if not(sm or bg): continue
            A=atr[fi]; ap=A/c[fi]; sl=slice(fi-W,fi+1); cr=np.cumsum(lr[sl])-lr[sl][0]; X=np.c_[cr/ap,(c[sl]-lev[sl])/A,(np.cumsum(lr[sl]-m[sl])-(lr[sl]-m[sl])[0])/ap,np.linspace(0,1,W+1)]
            if not np.isfinite(X).all(): continue
            er=lambda n:abs(c[fi]-c[fi-n])/max(np.abs(np.diff(c[fi-n:fi+1])).sum(),1e-9)
            rows.append(dict(tk=tk,t=idx[e],book='1-5' if sm else 'BIG',ret=o[xb]/px-1-(.004 if sm else .0015),ext=(c[fi]-lev[fi])/A,rv=vol[fi]/vm[fi],atrp=ap,lpx=np.log(px),er20=er(20),er48=er(48))); sg.append(sig2(X))
T=pd.DataFrame(rows); S=np.array(sg); T['tr']=(T.t<SPLIT).values; T['win']=(T.ret>0).astype(int); print('events',len(T),T.groupby('book').size().to_dict())
SIMPLE=['ext','rv','atrp','lpx','er20','er48']
def st(g):
    r=g.ret.sort_values(); return 'n%-4d win %2.0f%% mean %+5.2f%% ex-top1 %+5.2f%%'%(len(g),(g.ret>0).mean()*100,g.ret.mean()*100,r.iloc[:int(len(r)*.99)].mean()*100)
def filt(nm,p,te,keep=.7):
    K=te[p>=np.quantile(p,1-keep)]; null=np.mean([te.sample(len(K),random_state=i).ret.mean()>=K.ret.mean() for i in range(2000)]); print('   %-16s keep top %d%%: %s | all %+5.2f%% | p(random-skip) %.3f'%(nm,keep*100,st(K),te.ret.mean()*100,null))
for bk in ('1-5','BIG'):
    m=(T.book==bk).values; D=T[m].reset_index(drop=True); Sg=S[m]; tr=D.tr.values; print('\n================ %s train %d test %d | win%% train %.0f test %.0f | mean ret train %+.2f%% test %+.2f%%'%(bk,tr.sum(),(~tr).sum(),D[tr].win.mean()*100,D[~tr].win.mean()*100,D[tr].ret.mean()*100,D[~tr].ret.mean()*100))
    print('-- single feature test AUC for WIN (train/test): '+', '.join('%s %.3f/%.3f'%(f,roc_auc_score(D[tr].win,D[tr][f]),roc_auc_score(D[~tr].win,D[~tr][f])) for f in SIMPLE))
    te=D[~tr]
    for var,lab in (('er20','ER(20) trend-quality gate'),('er48','ER(48) trend-quality gate'),('ext','band-distance gate (close-vector in ATR)')):
        q=D[tr][var].quantile(1/3); print('  %s | low tercile: %s || upper two terciles: %s'%(lab,st(te[te[var]<=q]),st(te[te[var]>q])))
    def fit(cols,Sx):
        Xt=np.c_[D[cols].values if cols else np.zeros((len(D),0)),Sx] if Sx is not None else D[cols].values; sc=StandardScaler().fit(Xt[tr]); Z=np.clip(sc.transform(Xt),-5,5)
        lg=LogisticRegression(C=.05,max_iter=800).fit(Z[tr],D[tr].win); return lg.predict_proba(Z[~tr])[:,1],lg.predict_proba(Z[tr])[:,1]
    res={}
    for nm,cols,Sx in (('S0 simple',SIMPLE,None),('S1 simple+sig',SIMPLE,Sg),('SIG only',[],Sg)):
        p,ptr=fit(cols,Sx); res[nm]=p; print('-- %-14s train AUC %.3f  TEST AUC %.3f'%(nm,roc_auc_score(D[tr].win,ptr),roc_auc_score(te.win,p)))
    for nm,p in res.items(): filt(nm,p,te)
