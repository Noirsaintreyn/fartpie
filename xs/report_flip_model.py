"""FLIP MODEL, strict split: train <=2021 (screener flips start 2019), test 2022+ untouched. Fixed hyperparameters (no test tuning).
Features = multi-timeframe maximin-pivot ALIGNMENT at entry (completed bars only): per TF distance to the pivot on the trade side (ATR), room to opposite pivot, slope, bars since cross, plus
confluence across TFs (how many TFs have price on the trade side, spread of pivots, how many pivots sit within 1 ATR), and context (ATR%, time, daily trend/extension, gap, flip history).
Datasets: (1) screener flips, 9 tickers with 15m data (wide_T base+sticky, S&P, ratio>=1.5); (2) NQ V15a/RA entries (mix_tiers, bars 5m RTH).
Models: logistic L2, shallow gradient boosting, random forest. Null: train-label permutation. Also univariate confluence tables (does 'together' matter?)."""
import glob, numpy as np, pandas as pd, warnings, pickle; warnings.filterwarnings('ignore')
from collections import deque
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
rng=np.random.default_rng(0)
def lbf(low,L): return low.rolling(L).min().rolling(L).max()
def hbf(high,L): return high.rolling(L).max().rolling(L).min()
def mkTF(S, minutes):
    S=S.copy().reset_index(drop=True)
    pc=S.close.shift(); tr=np.maximum(S.high-S.low,np.maximum((S.high-pc).abs(),(S.low-pc).abs())); S['atr']=tr.rolling(14).mean()
    for L in (10,20): S[f'lb{L}']=lbf(S.low,L); S[f'hb{L}']=hbf(S.high,L)
    S['lb20s']=S.lb20-S.lb20.shift(3); S['hb20s']=S.hb20-S.hb20.shift(3)
    for nm,cond in (('l',S.close>S.lb20),('s',S.close<S.hb20)):
        g=(cond!=cond.shift()).cumsum(); S['age_'+nm]=cond.groupby(g).cumcount().clip(upper=60)
    S['known']=S.dt+pd.Timedelta(minutes=minutes); return S.dropna(subset=['atr','lb20','hb20']).reset_index(drop=True)
def tf_set(Hb, base_min, tfs):
    out={}
    for nm,mins in tfs:
        if mins==base_min: S=Hb[['dt','open','high','low','close']]
        elif mins=='1d':
            d=Hb.groupby(Hb.dt.dt.normalize()).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last')).reset_index().rename(columns={'dt':'dt'}); d.columns=['dt','open','high','low','close']; S=d
        else:
            S=Hb.set_index('dt').resample(f'{mins}min',origin='start_day',offset='30min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        out[nm]=mkTF(S,(24*60+0 if mins=='1d' else mins) if mins!='1d' else 0)
        if mins=='1d': out[nm]['known']=out[nm].dt+pd.Timedelta(hours=16)
    return out
def feats(trades, Hb, base_min, tfs, dirs):
    """trades: DataFrame with entry (Timestamp), dir (+1/-1). returns feature DataFrame aligned to trades.index"""
    T=trades.sort_values('entry'); X=pd.DataFrame(index=T.index); TF=tf_set(Hb,base_min,tfs); P={};A={}
    s=T.dir.values.astype(float)
    for nm,S in TF.items():
        x=pd.merge_asof(T[['entry']],S[['known','close','atr','lb10','hb10','lb20','hb20','lb20s','hb20s','age_l','age_s']],left_on='entry',right_on='known',direction='backward'); x.index=T.index
        L=s>0; atr=x.atr.values
        pw20=np.where(L,x.lb20,x.hb20); pg20=np.where(L,x.hb20,x.lb20); pw10=np.where(L,x.lb10,x.hb10)
        X[f'dw_{nm}']=s*(x.close-pw20)/atr; X[f'dw10_{nm}']=s*(x.close-pw10)/atr; X[f'room_{nm}']=s*(pg20-x.close)/atr
        X[f'slope_{nm}']=s*np.where(L,x.lb20s,x.hb20s)/atr; X[f'age_{nm}']=np.where(L,x.age_l,x.age_s)
        P[nm]=pw20; A[nm]=atr; X[f'atrp_{nm}']=atr/x.close
    names=list(TF); base=names[0]; Pm=np.vstack([P[n] for n in names]); a0=A[base]
    X['n_side']=sum((X[f'dw_{n}']>0).astype(int) for n in names); X['n_slope']=sum((X[f'slope_{n}']>0).astype(int) for n in names)
    X['spread']=(np.nanmax(Pm,0)-np.nanmin(Pm,0))/a0; X['near1']=sum((X[f'dw_{n}'].abs()*A[n]/a0<1).astype(int) for n in names)
    X['min_dw']=np.nanmin(np.vstack([X[f'dw_{n}'].values for n in names]),0); X['max_dw']=np.nanmax(np.vstack([X[f'dw_{n}'].values for n in names]),0)
    X['n_age_fresh']=sum((X[f'age_{n}']<=4).astype(int) for n in names)
    return X
def daily_ctx(Hb, T):
    d=Hb.groupby(Hb.dt.dt.normalize()).agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last')); pc=d.c.shift(); tr=np.maximum(d.h-d.l,np.maximum((d.h-pc).abs(),(d.l-pc).abs())); atr=tr.rolling(14).mean().shift()
    D=pd.DataFrame({'o':d.o,'atr':atr,'ret5':d.c.shift()/d.c.shift(6)-1,'ret20':d.c.shift()/d.c.shift(21)-1,'pdret':d.c.shift()/d.c.shift(2)-1,'gap':(d.o-pc)/atr,'hi20':(d.c.shift()-d.h.shift().rolling(20).max())/atr,'lo20':(d.c.shift()-d.l.shift().rolling(20).min())/atr})
    day=T.entry.dt.normalize(); x=D.reindex(day.values); x.index=T.index; s=T.dir.values
    return pd.DataFrame({'ret5':s*x.ret5,'ret20':s*x.ret20,'pdret':s*x.pdret,'gap':s*x.gap,'ext':np.where(s>0,x.hi20,-x.lo20)},index=T.index)
def models():
    return {'logit': make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=2000)),
            'gboost': HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,max_iter=150,min_samples_leaf=40,l2_regularization=1.0,random_state=0),
            'rforest': RandomForestClassifier(300,min_samples_leaf=30,max_features='sqrt',n_jobs=-1,random_state=0)}
def evaluate(name, X, y_ret, yr, unit):
    X=X.copy(); tr=(yr<=2021).values; te=(yr>=2022).values; y=(y_ret>0).astype(int).values
    med=X[tr].median(); Xf=X.fillna(med)
    print(f'\n######## {name}: train n {tr.sum()} (<=2021) | test n {te.sum()} (2022+) | base win train {y[tr].mean():.1%} test {y[te].mean():.1%} | avg {unit} train {y_ret[tr].mean():+.2f} test {y_ret[te].mean():+.2f} | features {X.shape[1]}', flush=True)
    res={}
    for mn,m in models().items():
        m.fit(Xf[tr],y[tr]); sc=m.predict_proba(Xf[te])[:,1]; auc=roc_auc_score(y[te],sc); rho=spearmanr(sc,y_ret[te].values)[0]
        nul=[]
        for _ in range(25):
            yp=rng.permutation(y[tr]); m2=type(m)(**m.get_params()) if mn!='logit' else make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=2000)); m2.fit(Xf[tr],yp); nul.append(roc_auc_score(y[te],m2.predict_proba(Xf[te])[:,1]))
        q=pd.qcut(pd.Series(sc).rank(method='first'),5,labels=False).values; r=y_ret[te].values; top=r[q==4]; bot=r[q==0]
        bs=[rng.choice(top,len(top)).mean()-rng.choice(bot,len(bot)).mean() for _ in range(1000)]
        print(f'  {mn:8s} OOS AUC {auc:.3f} (perm-null mean {np.mean(nul):.3f}, 95th {np.quantile(nul,.95):.3f}) | Spearman(score,ret) {rho:+.3f} | quintile mean {unit}: '+' '.join(f'{r[q==k].mean():+.1f}' for k in range(5))+f' | top-bottom {top.mean()-bot.mean():+.1f} [{np.quantile(bs,.025):+.1f},{np.quantile(bs,.975):+.1f}]', flush=True)
        res[mn]=(m,sc)
    return res,Xf,tr,te,y
def confluence_table(name,X,y_ret,yr,unit):
    print(f'--- {name}: does confluence ("together") matter? mean {unit} by # of TFs with price on the trade side of its pivot (n_side) and # fresh crosses', flush=True)
    for e,mm in (('<=2021',yr<=2021),('2022+',yr>=2022)):
        z=pd.DataFrame({'k':X.n_side[mm].values,'f':X.n_fresh[mm].values if 'n_fresh' in X else X.n_age_fresh[mm].values,'r':y_ret[mm].values})
        print(f'   {e} n_side: '+' | '.join(f'{int(k)}: n{len(g)} {g.r.mean():+.1f}' for k,g in z.groupby('k') if len(g)>=25)+'   || fresh-cross TFs: '+' | '.join(f'{int(k)}: n{len(g)} {g.r.mean():+.1f}' for k,g in z.groupby('f') if len(g)>=25), flush=True)
def importances(res,Xf,te,y,top=10):
    m=res['rforest'][0]; pi=permutation_importance(m,Xf[te],y[te],scoring='roc_auc',n_repeats=10,random_state=0,n_jobs=-1)
    s=pd.Series(pi.importances_mean,index=Xf.columns).sort_values(ascending=False).head(top); print('  top permutation importances (rforest, test AUC drop): '+', '.join(f'{k} {v:.3f}' for k,v in s.items()),flush=True)
def yearly(name,X,y_ret,yr):
    y=(y_ret>0).astype(int).values; out=[]
    for Y in sorted(set(yr[yr>=2022])):
        tr=(yr<Y).values; te=(yr==Y).values
        if te.sum()<40: continue
        Xf=X.fillna(X[tr].median()); m=models()['logit']; m.fit(Xf[tr],y[tr]); a1=roc_auc_score(y[te],m.predict_proba(Xf[te])[:,1]); g=models()['gboost']; g.fit(Xf[tr],y[tr]); a2=roc_auc_score(y[te],g.predict_proba(Xf[te])[:,1]); out.append(f'{Y}: logit {a1:.2f} gboost {a2:.2f} (n{te.sum()})')
    print(f'  expanding walk-forward AUC by year ({name}): '+' | '.join(out),flush=True)
# ================= (1) SCREENER FLIPS =================
import os; os.chdir('/mnt/user-data/uploads/fartpie')
T=pd.read_parquet('xs/ext/wide_T.parquet'); T=T[T.sp&(T.ratio>=1.5)].copy(); T['dir']=1
rows=[];FX=[]
for tk in sorted(T.tk.unique()):
    f=glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet')
    if not f: continue
    H=pd.read_parquet(f[0],columns=['datetime','open','high','low','close']); H['dt']=pd.to_datetime(H.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m=H.dt.dt.hour*60+H.dt.dt.minute; H=H[(m>=570)&(m<960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    t=T[T.tk==tk].sort_values('entry').copy()
    X=feats(t,H,15,[('15m',15),('1h',60),('1d','1d')],None); C=daily_ctx(H,t); X=X.join(C)
    X['hour']=t.entry.dt.hour+t.entry.dt.minute/60; X['dow']=t.entry.dt.dayofweek; X['ratio']=t.ratio.values
    ex=t.exit.values; en=t.entry.values; X['flips10']=[((en<en[i])&(en>=en[i]-np.timedelta64(14,'D'))).sum() for i in range(len(t))]
    last=[];
    for i in range(len(t)):
        pr=[j for j in range(i) if ex[j]<=en[i]]; last.append(t.ret.values[pr[-1]] if pr else np.nan)
    X['last_ret']=last; FX.append(X.join(t[['entry','exit','ret','ratio']].rename(columns={'ratio':'r2'}))); 
F=pd.concat(FX).dropna(subset=['dw_15m','dw_1h','dw_1d']); F['yr']=F.entry.dt.year; print('screener flips with features:',len(F),flush=True)
featcols=[c for c in F.columns if c not in ('entry','exit','ret','r2','yr')]
Xs=F[featcols]; ys=pd.Series(F.ret.values*1e4,index=F.index); yrs=F.yr
confluence_table('SCREENER',Xs,ys,yrs,'bp')
res,Xf,tr,te,y=evaluate('SCREENER 9 tickers, long flips',Xs,ys,yrs,'bp'); importances(res,Xf,te,y); yearly('screener',Xs,ys,yrs)
# slot sim on 2022+: base vs model-filtered (threshold = train-score quantile, frozen)
N=6;BRK=.08
def sim(Tx):
    Tx=Tx.sort_values(['entry','ratio'],ascending=[True,False],kind='stable'); op=[];pend=[];real=0.;hist=[];win=deque();lvl0=0.;out=[]
    for e,x,r in zip(Tx.entry.values,Tx.exit.values,Tx.ret.values):
        done=sorted([p for p in pend if p[0]<=e],key=lambda z:z[0]);pend=[p for p in pend if p[0]>e]
        for xt,rr in done: real+=np.log1p(rr/N);hist.append((xt,real));win.append((xt,real))
        op=[o for o in op if o>e]
        if len(op)>=N: continue
        if hist:
            t0=e-np.timedelta64(28,'D')
            while win and win[0][0]<t0: lvl0=win.popleft()[1]
            if real-max([lvl0]+[h for _,h in win])<np.log1p(-BRK): continue
        op.append(x);pend.append((x,r));out.append((pd.Timestamp(e),r))
    s=pd.DataFrame(out,columns=['e','r']);s['yr']=s.e.dt.year;return s
def ann(Tx):
    s=sim(Tx); z=s[s.yr>=2022]; y_=max((z.e.max()-z.e.min()).days/365.25,.1); return len(z), np.expm1(np.log1p(z.r/N).sum()/y_)
Tt=F[F.yr>=2022].copy(); Tt['ratio']=Tt.r2; print('--- slot sim 2022+ (6 slots, breaker as live): base vs skip flips whose frozen train-fit score is below a train-score quantile',flush=True)
n0,a0=ann(Tt); print(f'   base: n {n0} ann~{a0:+.1%}',flush=True)
for mn,(m,sc) in res.items():
    trsc=m.predict_proba(Xf[tr])[:,1]; Tt2=Tt.copy(); Tt2['sc']=sc
    for qn in (.2,.35,.5):
        th=np.quantile(trsc,qn); n1,a1=ann(Tt2[Tt2.sc>=th]); print(f'   {mn:8s} skip below train-pct {qn:.0%}: n {n1} ann~{a1:+.1%}',flush=True)
# ================= (2) NQ V15a / RA entries =================
MT=pickle.load(open(SC+'mix_tiers.pkl','rb'))['rows']; MT=MT[MT.day>='2014-03-01'].copy(); MT['entry']=MT.day+pd.to_timedelta(MT.m,unit='m')
nb=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt'); mn_=nb.dt.dt.hour*60+nb.dt.dt.minute; nb=nb[(mn_>=570)&(mn_<960)].reset_index(drop=True)
XN=feats(MT,nb,5,[('5m',5),('15m',15),('1h',60),('1d','1d')],None); XN=XN.join(daily_ctx(nb,MT)); XN['hour']=MT.m/60; XN['dow']=MT.entry.dt.dayofweek; XN['is_ra']=(MT.leg=='RA').astype(int); XN['short']=(MT.dir<0).astype(int)
ok=XN.dw_5m.notna()&XN.dw_1d.notna(); XN=XN[ok]; MTq=MT.loc[XN.index]; yrn=MTq.entry.dt.year
sel=(yrn>=2018); XN2=XN[sel]; yN=MTq.pts[sel]; yN2=yrn[sel]
print(f'\nNQ entries with features 2018+: {len(XN2)} (train <=2021: {(yN2<=2021).sum()}, test 2022+: {(yN2>=2022).sum()})',flush=True)
confluence_table('NQ V15a/RA',XN2,yN,yN2,'pts')
resn,Xfn,trn,ten,yn=evaluate('NQ V15a+RA entries',XN2,yN,yN2,'pts'); importances(resn,Xfn,ten,yn); yearly('NQ',XN2,yN,yN2)
pickle.dump(dict(F=F,XN=XN2,yN=yN),open(SC+'flip_model_data.pkl','wb'))
