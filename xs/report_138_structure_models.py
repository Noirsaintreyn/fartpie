"""report 138: remaining structure/basket models on all three engines. Declared before results.
Models: (3) contagion on correlation graph (top-5 corr peers flipped in prior 5d), (4) critical slowing down (AC1, rising variance ratio, 10d changes),
(5) price impact / absorption (Amihud, dollar-volume per unit range), (6) OU / quantum-harmonic displacement (z in stationary-std units, OU half-life).
Hypotheses: CSD up and low absorption = system near a transition -> big move; contagion high = flips cluster in a risk-on phase; OUZ high (stretched) = worse for continuation.
Strict split: thresholds/fits on train (<2022), judge 2022+. PASS = same sign both halves and |t|>=2 both."""
import sys, numpy as np, pandas as pd, glob, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/fartpie/xs'); from feats138 import dfeat, contagion
from sklearn.linear_model import LogisticRegression; from sklearn.metrics import roc_auc_score
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X=SP+'ext/'; RU='/mnt/user-data/uploads/fartpie/'
FE=['AC1','dAC','LVR','dLVR','LAMIH','LABS','LHL','OUZ']
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>8 and len(b)>8 else np.nan
def part(which):
  if which=='C':    # ---------------- small caps
    O=pd.read_pickle(X+'basket_models.pkl'); W=pd.read_pickle(X+'wide_daily_ohlcv.pkl')
    F={tk:dfeat(pd.DataFrame({'h':v.high,'l':v.low,'c':v.close,'v':v.volume})[~v.index.duplicated()]) for tk,v in W.items()}
    for c in FE: O[c]=[F[r.tk][c].asof(r.d) if r.tk in F else np.nan for r in O.itertuples()]
    C=pd.DataFrame({k:v.close[~v.index.duplicated()] for k,v in W.items()}); R=np.log(C).diff().clip(-.5,.5)
    fd={tk:np.sort(g.d.values) for tk,g in O.groupby('tk')}; cf=contagion(R,fd); O['CONT']=[cf(r.tk,r.d) for r in O.itertuples()]
    tr=O.tr; print('SMALL CAP flips',len(O))
    allf=FE+['CONT']
    print('AUC BIG / HUGE (train | test); sign must match'); 
    for tgt in ('big','huge'):
        for c in allf:
            x=O[c].fillna(O[c].median()); a=[roc_auc_score(O[tgt][m],x[m]) for m in (tr,~tr)]; print(f'  {tgt:4s} {c:6s} {a[0]:.3f} | {a[1]:.3f}{"  <== same side, |dev|>.03 both" if (a[0]-.5)*(a[1]-.5)>0 and min(abs(a[0]-.5),abs(a[1]-.5))>.03 else ""}')
    print('add-on to OS (fit train, test AUC)')
    for tgt in ('big','huge'):
        def fit(cols):
            A=O[cols].fillna(O[cols].median()); A=(A-A[tr].mean())/A[tr].std(); l=LogisticRegression(max_iter=500).fit(A[tr],O.loc[tr,tgt]); p=l.predict_proba(A)[:,1]; return roc_auc_score(O[tgt][~tr],p[~tr]),roc_auc_score(O[tgt][tr],p[tr])
        print(f'  {tgt} OS {fit(["OS"])[0]:.3f}')
        for c in allf: a=fit(['OS',c]); print(f'     OS+{c:6s} test {a[0]:.3f} (train {a[1]:.3f})')
        a=fit(['OS']+allf); print(f'     OS+ALL     test {a[0]:.3f} (train {a[1]:.3f})')
    O.to_pickle(X+'structure_small.pkl')
  if which=='B':    # ---------------- large-cap screener
    T=pd.read_pickle(X+'screener_T_feat.pkl'); TT=pd.read_parquet(RU+'xs/ext/wide_T.parquet')
    D={}
    for f in glob.glob(RU+'lse_clean/*_15m_*.parquet')+glob.glob(RU+'lse_ohlcv_cache/*_15m_*.parquet'):
        tk=os.path.basename(f).split('_')[0]
        if tk in D: continue
        H=pd.read_parquet(f,columns=['datetime','high','low','close','volume']); H['dt']=pd.to_datetime(H.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        m=H.dt.dt.hour*60+H.dt.dt.minute; H=H[(m>=570)&(m<960)].drop_duplicates('dt').set_index('dt').sort_index(); g=H.groupby(H.index.normalize())
        D[tk]=pd.DataFrame({'h':g.high.max(),'l':g.low.min(),'c':g.close.last(),'v':g.volume.sum()})
    F={k:dfeat(v) for k,v in D.items()}
    for c in FE: T[c]=[F[r.tk][c].asof(r.entry.normalize()) for r in T.itertuples()]
    R=np.log(pd.DataFrame({k:v.c for k,v in D.items()})).diff(); fd={tk:np.sort(g.entry.dt.normalize().values) for tk,g in TT[TT.tk.isin(D)].groupby('tk')}
    cf=contagion(R,fd,k=5,win=3); T['CONT']=[cf(r.tk,r.entry.normalize()) for r in T.itertuples()]
    tr=T.tr; print('LARGE CAP screener trades',len(T)); allf=FE+['CONT']
    print('trade return by tercile (low|mid|high) train // test, spread high-low and t; PASS if same sign |t|>=2 both')
    keep=[]
    for c in allf:
        q=T[tr][c].quantile([1/3,2/3]).values if T[c].nunique()>3 else [0.5,1.5]; res=[];txt=[]
        for m in (tr,~tr):
            lo,hi=T[m&(T[c]<=q[0])].ret,T[m&(T[c]>q[1])].ret; mid=T[m&(T[c]>q[0])&(T[c]<=q[1])].ret; txt.append(f'{100*lo.mean():+.2f}|{100*mid.mean():+.2f}|{100*hi.mean():+.2f}'); res.append((hi.mean()-lo.mean(),tt(hi,lo)))
        ps=np.sign(res[0][0])==np.sign(res[1][0]) and all(abs(r[1])>=2 for r in res); print(f'  {c:6s} {txt[0]} // {txt[1]} | t {res[0][1]:+.1f} // {res[1][1]:+.1f}{"  <== PASS" if ps else ""}'); 
        if ps: keep.append((c,q,np.sign(res[0][0])))
    T.to_pickle(X+'structure_large.pkl'); print('PASS list',[k[0] for k in keep])
  if which=='A':    # ---------------- prop algo
    import pickle, nz_core as Z
    from nz_core import A, ALLD, DAYS, GRID
    d=pd.DataFrame({dd:{'h':np.nanmax(A[dd][0][1]),'l':np.nanmin(A[dd][0][2]),'c':A[dd][0][3][~np.isnan(A[dd][0][3])][-1],'v':np.nansum(A[dd][0][4])} for dd in ALLD}).T; d.index=pd.DatetimeIndex(d.index)
    MD=dfeat(d.astype(float))
    OUT,NAMES=pickle.load(open(SP+'s6.pkl','rb')); BM=dict(pickle.load(open(SP+'base_mv30.pkl','rb'))); TR={}
    for (t,u,y),n in zip(OUT['V2'],NAMES):
        if n in ('ON+','ON-'): continue
        t=t.copy(); t['day']=pd.to_datetime(t.day).dt.normalize(); B_=BM[n].copy(); B_['day']=pd.to_datetime(B_.day).dt.normalize(); B_=B_[B_.sz>0][['day','m','sz']].rename(columns={'sz':'szb'}); t=t.merge(B_,on=['day','m']); t['sz']=t.szb
        TR[n]=t[(t.day>='2014-02-01')&(t.day<='2026-01-30')].reset_index(drop=True)
    allp=pd.concat([t[['day','pts']].assign(n=n) for n,t in TR.items()]); daily=allp.groupby('day').pts.sum().reindex(MD.index).fillna(0)
    CONT=(daily.rolling(3).sum().shift(1)); MD['CONT']=CONT       # leg-loss contagion: system P&L over prior 3 days
    FE2=FE+['CONT']; E=lambda x:x<'2022-01-01'
    rng=np.random.default_rng(5)
    def flags(t,pts):
        out={}; e=E(t.day).values; l=~e
        for c in FE2:
            col=t[c]; q=col[e].quantile([1/3,2/3]).values
            for nm,sel in (('low',col<=q[0]),('high',col>q[1])):
                res=[]
                for mm in (e,l):
                    a,b=pts[mm&sel.values],pts[mm&~sel.values&col.notna().values]; res.append((a.mean()-b.mean(),tt(a,b),len(a)))
                out[f'{c} {nm}']=tuple(res)
        return out
    def passed(r): return np.sign(r[0][0])==np.sign(r[1][0]) and all(np.isfinite(x[1]) and abs(x[1])>=2 and x[2]>=30 for x in r)
    for n,t in TR.items():
        t=t.join(MD,on='day'); t=t[t.AC1.notna()]; fl=flags(t,t.pts); ps={k:v for k,v in fl.items() if passed(v)}; e=E(t.day).values; null=[]
        for _ in range(100):
            sh=t.pts.copy()
            for mm in (e,~e): sh[mm]=rng.permutation(sh[mm].values)
            null.append(sum(passed(v) for v in flags(t,sh).values()))
        print(f'  {n:6s} {len(t):4d} trades {len(fl)} tests  real PASS {len(ps)} vs chance {np.mean(null):.1f} (95th {np.percentile(null,95):.0f})')
        for k,v in sorted(ps.items(),key=lambda kv:-abs(kv[1][0][1])): print(f'       {k:12s} {v[0][0]:+6.1f} (t {v[0][1]:+.2f}, n{v[0][2]}) // {v[1][0]:+6.1f} (t {v[1][1]:+.2f}, n{v[1][2]})')
if __name__=='__main__': part(sys.argv[1])
