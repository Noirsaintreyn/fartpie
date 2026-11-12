"""Nov 12 2026: ACCEPTANCE vs CANDLES - meta-labelling the vector's 1h bull flips. DECLARED BEFORE RESULTS.
Rey's thesis: a candle is a representation, it forecasts nothing; what carries information is whether the market ACCEPTS price (volume at price, time at price, effort vs result). Test: same trades, same split (train < 2025-07-01 / test after), feature families compared head-to-head:
  CANDLE  : flip-bar return/ATR, body/range, close location, range/ATR, previous 3 bar returns/ATR
  ACCEPT  : acc (share of last-20-bar volume above the prior 140-bar value-area high), above_poc, va_width/ATR, poc_dist/ATR, volume-profile skew, time_above_level (share of last 20 closes above the new vector level),
            eff (price change per unit of relative volume over 5 bars = effort vs result), clv20/clv5 (volume-weighted close location), upvol10/20 (share of volume on up-closes), rv (flip-bar rel. volume), rv5, range efficiency
  CONTEXT : extension from vector level/ATR, ATR%, log price, 4h bull, daily bull
Models (hyper-parameters fixed before looking): Gaussian NB, Categorical NB (train-quintile bins), logistic (L2), LightGBM (4 leaves, 150 trees, lr .03, strong regularisation), 1D CNN on the last 48 bars x 5 channels (ret/ATR, log rel vol, close location, range/ATR, (close-level)/ATR), 3-seed ensemble.
Labels: WIN (trade ret > 0) and BAD (ret <= -3%). Primary pre-declared test: LightGBM(ALL features), P(win) score, skip the lowest 30% of test candidates -> kept mean / win / ex-top-1% vs random-skip null (2000 draws).
Everything else is exploratory; 2 books x ~10 model/label combos, so ~1 in 20 'significant' by luck. Books: SMALL ($1-5) and BIG (median daily $vol >= $50M, px >= $10)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from scipy.stats import skew
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, torch, torch.nn as nn
from vector_pine import run_state_machine
from vector_core import compute_atr_series
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); SPLIT=pd.Timestamp('2025-07-01'); COST=.004; W=48
def htf(Y,tf):
    day=Y.index.normalize(); g=[day,(Y.index.hour>=13).astype(int)] if tf=='4h' else [day]
    G=Y.groupby(g); o=G.agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'))
    first=G.apply(lambda x:x.index[0]); last=G.apply(lambda x:x.index[-1]); o.index=pd.DatetimeIndex(first.values)
    end=pd.DatetimeIndex(last.values)+pd.to_timedelta(np.where(pd.DatetimeIndex(last.values).hour==15,30,60),unit='m')
    reg=run_state_machine(o.assign(datetime=o.index))[0]; return pd.Series(reg,index=end).sort_index()
def profile(tp,lo,hi,v,nb=30):
    if hi<=lo: return np.nan,np.nan,np.nan
    e=np.linspace(lo,hi,nb+1); b=np.clip(np.searchsorted(e,tp)-1,0,nb-1); h=np.bincount(b,weights=v,minlength=nb); p=h.argmax(); tot=h.sum(); a=b_=p; cum=h[p]
    while cum<.7*tot and (a>0 or b_<nb-1):
        up=h[b_+1] if b_<nb-1 else -1; dn=h[a-1] if a>0 else -1
        if up>=dn: b_+=1; cum+=h[b_]
        else: a-=1; cum+=h[a]
    return (e[p]+e[p+1])/2,e[a],e[b_+1]
dv={tk:(Y.close*Y.volume).groupby(Y.index.normalize()).sum().median() for tk,Y in data.items() if len(Y)>800}; big={t for t,v in dv.items() if v>=5e7}
rows=[]; seqs=[]
for tk,Y in data.items():
    Y=Y.dropna()
    if len(Y)<1500: continue
    idx=Y.index; o,hi,lo,c,vol=[Y[k].values for k in ('open','high','low','close','volume')]; tp=(hi+lo+c)/3; atr=compute_atr_series(hi,lo,c,14)
    reg,lev,_,fl=run_state_machine(Y.assign(datetime=idx)); r4,rd=htf(Y,'4h'),htf(Y,'1d'); vm=pd.Series(vol).rolling(20).mean().shift().values
    clv=np.where(hi>lo,(2*c-hi-lo)/np.maximum(hi-lo,1e-9),0); pos=None
    for i,s in fl:
        if s==1 and pos is None: pos=i
        elif s==-1 and pos is not None:
            fi=pos; e,x=fi+1,i+1; pos=None
            if x>=len(Y) or fi<150 or not np.isfinite(atr[fi]) or atr[fi]<=0 or not np.isfinite(vm[fi]): continue
            px=o[e]; sm=(1<=px<=5); bg=(tk in big and px>=10)
            if not(sm or bg): continue
            A=atr[fi]; poc,val,vah=profile(tp[fi-140:fi+1],lo[fi-140:fi+1].min(),hi[fi-140:fi+1].max(),vol[fi-140:fi+1]); _,pv,pvah=profile(tp[fi-140:fi-20],lo[fi-140:fi-20].min(),hi[fi-140:fi-20].max(),vol[fi-140:fi-20]); ppoc=profile(tp[fi-140:fi-20],lo[fi-140:fi-20].min(),hi[fi-140:fi-20].max(),vol[fi-140:fi-20])[0]
            if not np.isfinite(poc) or not np.isfinite(pvah) or vah<=val: continue
            v20=vol[fi-19:fi+1]; t20=tp[fi-19:fi+1]; v10=vol[fi-9:fi+1]; c10=c[fi-10:fi+1]; c20=c[fi-20:fi+1]; w=vol[fi-139:fi+1]; m_=np.average(tp[fi-139:fi+1],weights=w+1e-9)
            up20=(np.diff(c[fi-20:fi+1])>0); up10=up20[-10:]; end_t=idx[fi]+pd.Timedelta(minutes=30 if idx[fi].hour==15 else 60)
            r4v=r4[r4.index<=end_t]; rdv=rd[rd.index<=end_t]
            f=dict(tk=tk,t=idx[e],ret=o[x]/o[e]-1-COST,book='SMALL' if sm else 'BIG',
              c_ret=(c[fi]/o[fi]-1)*c[fi]/A,c_body=abs(c[fi]-o[fi])/max(hi[fi]-lo[fi],1e-9),c_clv=clv[fi],c_rng=(hi[fi]-lo[fi])/A,c_p1=(c[fi-1]-c[fi-2])/A,c_p2=(c[fi-2]-c[fi-3])/A,c_p3=(c[fi-3]-c[fi-4])/A,
              acc=v20[t20>pvah].sum()/max(v20.sum(),1),above_poc=v20[t20>ppoc].sum()/max(v20.sum(),1),va_width=(vah-val)/A,poc_dist=(c[fi]-poc)/A,vp_skew=float(np.sum(w*(tp[fi-139:fi+1]-m_)**3)/np.sum(w)/(np.sqrt(np.sum(w*(tp[fi-139:fi+1]-m_)**2)/np.sum(w))**3+1e-9)),
              time_above=(c[fi-19:fi+1]>lev[fi]).mean(),eff=(c[fi]-c[fi-5])/A/max(vol[fi-4:fi+1].sum()/vm[fi]/5,.05),clv20=np.sum(clv[fi-19:fi+1]*v20)/max(v20.sum(),1),clv5=np.sum(clv[fi-4:fi+1]*vol[fi-4:fi+1])/max(vol[fi-4:fi+1].sum(),1),
              upvol10=vol[fi-9:fi+1][up10].sum()/max(v10.sum(),1),upvol20=v20[up20].sum()/max(v20.sum(),1),rv=vol[fi]/vm[fi],rv5=vol[fi-4:fi+1].mean()/vm[fi],reff=abs(c20[-1]-c20[0])/max(np.abs(np.diff(c20)).sum(),1e-9),
              ext=(c[fi]-lev[fi])/A,atrp=A/c[fi],lpx=np.log(px),b4=float(r4v.iat[-1]==1) if len(r4v) else 0.,bd=float(rdv.iat[-1]==1) if len(rdv) else 0.)
            rows.append(f)
            sl=slice(fi-W+1,fi+1); S=np.c_[np.diff(np.log(c[fi-W:fi+1]))/(A/c[fi]),np.log(np.maximum(vol[sl],1)/max(vm[fi],1)),clv[sl],(hi[sl]-lo[sl])/A,(c[sl]-lev[fi])/A]; seqs.append(np.clip(np.nan_to_num(S),-6,6).astype(np.float32))
T=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan); Sq=np.stack(seqs); keep=T.notna().all(axis=1).values; T=T[keep].reset_index(drop=True); Sq=Sq[keep]; T['tr']=(T.t<SPLIT).values
T.to_pickle(SC+'accept_T.pkl'); np.save(SC+'accept_S.npy',Sq); print('events',len(T),T.groupby('book').size().to_dict(),'| train',T.tr.sum(),'test',(~T.tr).sum())
CAN=['c_ret','c_body','c_clv','c_rng','c_p1','c_p2','c_p3']; ACC=['acc','above_poc','va_width','poc_dist','vp_skew','time_above','eff','clv20','clv5','upvol10','upvol20','rv','rv5','reff']; CTX=['ext','atrp','lpx','b4','bd']
def gbm(): return lgb.LGBMClassifier(num_leaves=4,n_estimators=150,learning_rate=.03,min_child_samples=50,subsample=.7,subsample_freq=1,colsample_bytree=.7,reg_lambda=5,verbose=-1,random_state=0)
class CNN(nn.Module):
    def __init__(s): super().__init__(); s.c=nn.Sequential(nn.Conv1d(5,16,5,padding=2),nn.ReLU(),nn.Conv1d(16,16,5,padding=2),nn.ReLU()); s.o=nn.Sequential(nn.Dropout(.3),nn.Linear(32,1))
    def forward(s,x): h=s.c(x.transpose(1,2)); return s.o(torch.cat([h.mean(2),h.amax(2)],1)).squeeze(1)
def cnn_fit(Xt,yt,Xs,seeds=3,ep=25):
    out=np.zeros(len(Xs)); n=len(Xt); cut=int(n*.8); mu,sd=Xt.reshape(-1,5).mean(0),Xt.reshape(-1,5).std(0)+1e-6; f=lambda a:torch.tensor((a-mu)/sd)
    for sdd in range(seeds):
        torch.manual_seed(sdd); m=CNN(); opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=1e-2); xt,xv,ys,yv=f(Xt[:cut]),f(Xt[cut:]),torch.tensor(yt[:cut],dtype=torch.float32),torch.tensor(yt[cut:],dtype=torch.float32); best=(9,None); lf=nn.BCEWithLogitsLoss()
        for e_ in range(ep):
            m.train(); p=torch.randperm(cut)
            for j in range(0,cut,128): b=p[j:j+128]; opt.zero_grad(); lf(m(xt[b]),ys[b]).backward(); opt.step()
            m.eval()
            with torch.no_grad(): l=lf(m(xv),yv).item()
            if l<best[0]: best=(l,{k:v.clone() for k,v in m.state_dict().items()})
        m.load_state_dict(best[1]); m.eval()
        with torch.no_grad(): out+=torch.sigmoid(m(f(Xs))).numpy()/seeds
    return out
def report(nm,sc,S,rng):
    thr=np.quantile(sc,.3); K=S[sc>=thr]; null=np.array([S.sample(len(K),random_state=i).ret.mean() for i in range(2000)]); r=K.ret.sort_values()
    return '%-22s kept n%d win %2.0f%% mean %+.2f%% ex-top1 %+.2f%% | all mean %+.2f%% win %2.0f%% | p(random-skip) %.3f'%(nm,len(K),(K.ret>0).mean()*100,K.ret.mean()*100,r.iloc[:int(len(r)*.99)].mean()*100,S.ret.mean()*100,(S.ret>0).mean()*100,(null>=K.ret.mean()).mean())
rng=np.random.default_rng(0)
for bk in ('SMALL','BIG'):
    Bk=T[T.book==bk]; tr,te=Bk[Bk.tr],Bk[~Bk.tr]; itr,ite=tr.index.values,te.index.values; print('\n================ %s  train %d test %d | win%% train %.0f test %.0f | bad%% train %.0f test %.0f'%(bk,len(tr),len(te),(tr.ret>0).mean()*100,(te.ret>0).mean()*100,(tr.ret<=-.03).mean()*100,(te.ret<=-.03).mean()*100))
    for lab,fn,sgn in (('WIN',lambda d:(d.ret>0).astype(int),1),('BAD',lambda d:(d.ret<=-.03).astype(int),-1)):
        ytr,yte=fn(tr).values,fn(te).values; print('\n-- label',lab,' (AUC on TEST; 0.5=nothing)')
        for fs,cols in (('CANDLE',CAN),('ACCEPT',ACC),('CONTEXT',CTX),('ALL',CAN+ACC+CTX)):
            m=gbm().fit(tr[cols],ytr); print('  GBM %-8s train-AUC %.3f  test-AUC %.3f'%(fs,roc_auc_score(ytr,m.predict_proba(tr[cols])[:,1]),roc_auc_score(yte,m.predict_proba(te[cols])[:,1])))
        cols=CAN+ACC+CTX; sc_=StandardScaler().fit(tr[cols]); Xt,Xs=np.clip(sc_.transform(tr[cols]),-5,5),np.clip(sc_.transform(te[cols]),-5,5)
        P={'GaussNB':GaussianNB().fit(Xt,ytr).predict_proba(Xs)[:,1],'Logit':LogisticRegression(C=.05,max_iter=500).fit(Xt,ytr).predict_proba(Xs)[:,1],'LightGBM':gbm().fit(tr[cols],ytr).predict_proba(te[cols])[:,1]}
        q=np.unique(tr[cols].quantile([.2,.4,.6,.8],axis=0).values.ravel()) 
        Bt=np.column_stack([np.searchsorted(np.unique(tr[c_].quantile([.2,.4,.6,.8]).values),tr[c_]) for c_ in cols]); Bs=np.column_stack([np.searchsorted(np.unique(tr[c_].quantile([.2,.4,.6,.8]).values),te[c_]) for c_ in cols])
        P['CatNB']=CategoricalNB(min_categories=5).fit(Bt,ytr).predict_proba(Bs)[:,1]
        P['CNN']=cnn_fit(Sq[itr],ytr,Sq[ite])
        for k,p in P.items(): print('  %-9s test-AUC %.3f'%(k,roc_auc_score(yte,p)))
        if lab=='WIN':
            print('\n-- FILTER: skip lowest-30%% P(win) candidates in TEST (primary = LightGBM)')
            for k,p in P.items(): print('  '+report(k+('  <PRIMARY' if k=='LightGBM' else ''),p,te,rng))
        else:
            print('\n-- FILTER: skip highest-30%% P(bad)')
            for k,p in P.items(): print('  '+report(k,-p,te,rng))
    m=gbm().fit(tr[CAN+ACC+CTX],(tr.ret>0).astype(int)); imp=pd.Series(m.booster_.feature_importance('gain'),index=CAN+ACC+CTX); imp=imp/imp.sum(); print('\nLightGBM(WIN) gain share:',', '.join('%s %.0f%%'%(k,v*100) for k,v in imp.sort_values(ascending=False).head(8).items()),'| family totals: CANDLE %.0f%% ACCEPT %.0f%% CONTEXT %.0f%%'%(imp[CAN].sum()*100,imp[ACC].sum()*100,imp[CTX].sum()*100))
