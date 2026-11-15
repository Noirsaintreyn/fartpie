"""report 140: TWAP footprint. DECLARED BEFORE RESULTS.
Mechanism: a large institution working a parent order with a TWAP/VWAP slicer leaves (a) unusually regular bar volume (low coefficient of variation after removing the intraday U-shape),
(b) price that rides above the running time-weighted average price (close>TWAP in most bars), (c) positive serial correlation of bar returns. Over the prior 5 sessions this = steady accumulation -> flips continue.
Features (prior 5 completed sessions, mean): VCV (de-seasonalised bar-volume CV), TWAPAB (share of bars closing above running TWAP), ACS (lag-1 autocorr of bar returns), plus 1-day versions.
Tested where intraday history exists: large-cap screener (15m, 2019-26) and NQ prop legs (5m, 2014-26). Small caps have only 60d of intraday history: not testable.
PASS = same sign both halves, |t|>=2 (trade return by train-frozen thirds)."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/fartpie/xs')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; RU='/mnt/user-data/uploads/fartpie/'
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>8 and len(b)>8 else np.nan
def dayfeat(o,h,l,c,v):
    ok=~np.isnan(c)&(v>0); 
    if ok.sum()<20: return (np.nan,np.nan,np.nan)
    c,h,l,o,v=c[ok],h[ok],l[ok],o[ok],v[ok]; tp=(h+l+c)/3; tw=np.cumsum(tp)/np.arange(1,len(tp)+1)
    r=np.diff(np.log(c)); ac=np.corrcoef(r[1:],r[:-1])[0,1] if len(r)>5 else np.nan
    return (v,(c[1:]>tw[1:]).mean(),ac)
def build(days):
    """days: iterable of (day,o,h,l,c,v) arrays on a common grid -> DataFrame VCV,TWAPAB,ACS (lagged, 5d mean + 1d)"""
    rows={}; prof=[]
    for d,o,h,l,c,v in days:
        rows[d]=dayfeat(o,h,l,c,v)
    return rows
def finish(rows,grid_prof):
    out={}
    for d,(v,tw,ac) in rows.items():
        if isinstance(v,float): out[d]=(np.nan,tw,ac); continue
        v=np.asarray(v,float); w=v/ v.mean(); out[d]=(np.nanstd(w)/np.nanmean(w),tw,ac)
    F=pd.DataFrame(out,index=['VCV','TWAPAB','ACS']).T; F.index=pd.DatetimeIndex(F.index); F=F.sort_index()
    G=pd.concat([F.shift(1).add_suffix('_1'),F.shift(1).rolling(5,min_periods=4).mean().add_suffix('_5')],axis=1); return G
def report(T,cols,tr,y,lab):
    print(lab); keep=[]
    for c in cols:
        q=T[tr][c].quantile([1/3,2/3]).values; res=[];txt=[]
        for m in (tr,~tr):
            lo,hi=T[m&(T[c]<=q[0])][y],T[m&(T[c]>q[1])][y]; mid=T[m&(T[c]>q[0])&(T[c]<=q[1])][y]; txt.append(f'{lo.mean():+.2f}|{mid.mean():+.2f}|{hi.mean():+.2f}'); res.append((hi.mean()-lo.mean(),tt(hi,lo)))
        ps=np.sign(res[0][0])==np.sign(res[1][0]) and all(abs(r[1])>=2 for r in res); print(f'  {c:9s} {txt[0]} // {txt[1]} | t {res[0][1]:+.1f} // {res[1][1]:+.1f}{"  <== PASS" if ps else ""}')
        if ps: keep.append(c)
    return keep
def partB():
    T=pd.read_pickle(SP+'ext/screener_T_feat.pkl'); G={}
    for f in glob.glob(RU+'lse_clean/*_15m_*.parquet')+glob.glob(RU+'lse_ohlcv_cache/*_15m_*.parquet'):
        tk=os.path.basename(f).split('_')[0]
        if tk in G or tk not in set(T.tk): continue
        H=pd.read_parquet(f,columns=['datetime','open','high','low','close','volume']); H['dt']=pd.to_datetime(H.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        m=H.dt.dt.hour*60+H.dt.dt.minute; H=H[(m>=570)&(m<960)].drop_duplicates('dt').set_index('dt').sort_index()
        # de-seasonalise: divide volume by ticker median per time-of-day
        tod=H.index.hour*60+H.index.minute; prof=H.volume.groupby(tod).transform('median'); H['v']=H.volume/prof.replace(0,np.nan)
        rows={}
        for d,g in H.groupby(H.index.normalize()):
            if len(g)<20: continue
            rows[d]=dayfeat(g.open.values,g.high.values,g.low.values,g.close.values,g.v.fillna(1).values)
        G[tk]=finish(rows,None)
    for c in ('VCV_1','VCV_5','TWAPAB_1','TWAPAB_5','ACS_1','ACS_5'): T[c]=[G[r.tk][c].asof(r.entry.normalize()) if r.tk in G else np.nan for r in T.itertuples()]
    T=T.dropna(subset=['VCV_5','TWAPAB_5']); tr=T.tr; print('LARGE CAP',len(T))
    k=report(T,['VCV_1','VCV_5','TWAPAB_1','TWAPAB_5','ACS_1','ACS_5'],tr,'ret','trade ret% (low|mid|high thirds) train // test'); T.to_pickle(SP+'ext/twap_large.pkl'); print('PASS',k)
def partA():
    import pickle, nz_core as Z
    from nz_core import A, ALLD
    rows={}
    for d in ALLD:
        o,h,l,c,v,vw=A[d][0]; rows[d]=dayfeat(o,h,l,c,np.nan_to_num(v)) 
    MD=finish(rows,None); OUT,NAMES=pickle.load(open(SP+'s6.pkl','rb')); BM=dict(pickle.load(open(SP+'base_mv30.pkl','rb')))
    rng=np.random.default_rng(3); cols=['VCV_1','VCV_5','TWAPAB_1','TWAPAB_5','ACS_1','ACS_5']
    for (t,u,y),n in zip(OUT['V2'],NAMES):
        if n in ('ON+','ON-'): continue
        t=t.copy(); t['day']=pd.to_datetime(t.day).dt.normalize(); B_=BM[n].copy(); B_['day']=pd.to_datetime(B_.day).dt.normalize(); B_=B_[B_.sz>0][['day','m','sz']].rename(columns={'sz':'szb'}); t=t.merge(B_,on=['day','m']); t=t[(t.day>='2014-02-01')&(t.day<='2026-01-30')].join(MD,on='day').dropna(subset=['VCV_5']); t['tr']=t.day<'2022-01-01'
        print(f'{n} n={len(t)}'); report(t,cols,t.tr,'pts','   pts/trade (low|mid|high) early // late')
if __name__=='__main__': {'A':partA,'B':partB}[sys.argv[1]]()
