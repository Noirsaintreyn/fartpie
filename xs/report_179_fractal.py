"""Report 179: fractal / roughness / vol-of-vol features (Rey, Nov 15). DECLARED BEFORE RESULTS.
Hypothesis (mechanism): returns have long memory -> Hurst H>0.5 = persistent (trend legs should work, fades fail), H<0.5 = anti-persistent (fades work). Fluctuations feeding fluctuations = volatility clustering; vol-of-vol / roughness (path length / range) measures how choppy the last days were.
Features (all from prior days only, NQ RTH 5m, trailing windows): H20 = Hurst from variance scaling of 5m returns aggregated k=1,2,4,8,16 (pooled 20 d, H = slope/2); RUF5 = log(path length / high-low range) of the RTH session, 5 d mean (roughness, a Katz-style fractal dimension); VOV20 = std of log daily range over 20 d; dH = H20 - H20 10d ago.
Legs tested: the 7 variant-I legs, IFVG, S1 premium fade (regime-ON). Lower / upper tercile cut fit on 2018-21 trades (<2022), judged on 2022+. PASS = same sign, |t|>=2, n>=30 both eras. Also: does H20 predict next-day close-to-close autocorrelation sign (day-direction persistence), and does VOV20 predict next-day range.
"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/fartpie/xs'); from feats138 import dfeat
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>8 and len(b)>8 else np.nan
def prep():
    import pickle, nz_core as Z
    from nz_core import A, ALLD, DAYS, GRID
    d=pd.DataFrame({dd:{'h':np.nanmax(A[dd][0][1]),'l':np.nanmin(A[dd][0][2]),'c':A[dd][0][3][~np.isnan(A[dd][0][3])][-1],'v':np.nansum(A[dd][0][4])} for dd in ALLD}).T; d.index=pd.DatetimeIndex(d.index)
    MD=dfeat(d.astype(float))
    OUT,NAMES=pickle.load(open(SP+'s6.pkl','rb')); BM=dict(pickle.load(open(SP+'base_mv30.pkl','rb'))); TR={}
    for (t,u,y),n in zip(OUT['V2'],NAMES):
        if n in ('ON+','ON-'): continue
        t=t.copy(); t['day']=pd.to_datetime(t.day).dt.normalize(); B_=BM[n].copy(); B_['day']=pd.to_datetime(B_.day).dt.normalize(); B_=B_[B_.sz>0][['day','m','sz']].rename(columns={'sz':'szb'}); t=t.merge(B_,on=['day','m']); t['sz']=t.szb
        TR[n]=t[(t.day>='2014-02-01')&(t.day<='2026-01-30')].reset_index(drop=True)
    return TR
def fractal_feats():
    b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt'); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute; b=b[(b.m>=570)&(b.m<=955)]
    rows={}
    for d,g in b.groupby('d'):
        c=np.log(g.close.values)
        if len(c)<60: continue
        v={k:np.mean((c[k:]-c[:-k])**2) for k in (1,2,4,8,16)}; s=np.sum(np.abs(np.diff(c))); rg=g.high.max()/g.low.min(); rows[d]=dict(**{f'v{k}':v[k] for k in v},ruf=np.log(s/np.log(rg)),lr=np.log(np.log(rg)))
    D=pd.DataFrame(rows).T; D.index=pd.DatetimeIndex(D.index)
    ks=np.array([1,2,4,8,16]); lk=np.log(ks)
    V=np.log(D[[f'v{k}' for k in ks]].rolling(20).mean().values); h=np.array([np.polyfit(lk,r,1)[0]/2 if np.isfinite(r).all() else np.nan for r in V])
    F=pd.DataFrame({'H20':h,'RUF5':D.ruf.rolling(5).mean().values,'VOV20':D.lr.rolling(20).std().values,'LR':D.lr.values},index=D.index); F['dH']=F.H20-F.H20.shift(10)
    return F, D
if __name__=='__main__':
    TR=prep(); F,D=fractal_feats(); Fs=F.shift(1)
    print('H20 distribution (train 2018-21 | test 2022+): mean %.3f | %.3f ; sd %.3f | %.3f'%(F.H20['2018':'2021'].mean(),F.H20['2022':].mean(),F.H20['2018':'2021'].std(),F.H20['2022':].std()))
    # day-level predictions
    c=np.log(pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').assign(d=lambda x:x.dt.dt.normalize()).groupby('d').close.last()); r=c.diff(); fw=r.shift(-1)
    for lab,sl in (('2018-21',slice('2018','2021')),('2022+',slice('2022',None))):
        X=pd.DataFrame({'H':F.H20,'r':r,'fw':fw,'lr':F.LR.shift(-1),'vov':F.VOV20,'lrn':F.LR}).loc[sl].dropna(); q=X.H.quantile([1/3,2/3]).values; q2=pd.read_pickle if False else None
        hi=X[X.H>q[1]]; lo=X[X.H<=q[0]]
        print(f'  [{lab}] continuation corr(r_t, r_t+1): H high {np.corrcoef(hi.r,hi.fw)[0,1]:+.3f} (n{len(hi)}) | H low {np.corrcoef(lo.r,lo.fw)[0,1]:+.3f} (n{len(lo)}) ; next-day range corr with today range {np.corrcoef(X.lrn,X.lr)[0,1]:+.3f}; with VOV20 (partial via resid) {np.corrcoef(X.vov,X.lr-np.polyval(np.polyfit(X.lrn,X.lr,1),X.lrn))[0,1]:+.3f}')
    FE=['H20','RUF5','VOV20','dH']; E=lambda x:x<'2022-01-01'
    def passed(r): return np.sign(r[0][0])==np.sign(r[1][0]) and all(np.isfinite(x[1]) and abs(x[1])>=2 and x[2]>=30 for x in r)
    ex={}
    I=pd.read_pickle(SP+'ext/uni_ifvg.pkl'); I=I[I.model=='IFVG']; ex['IFVG']=pd.DataFrame({'day':pd.to_datetime(I.d).dt.normalize(),'pts':I.pts.values})
    S1=pd.read_pickle(SP+'s1_trades162.pkl'); S1=S1[S1.reg>0]; ex['S1on']=pd.DataFrame({'day':pd.to_datetime(S1.day).dt.normalize(),'pts':S1.pts.values})
    allT=dict(TR); allT.update(ex); print('\nleg tercile tests (spread = leg pts in tercile minus rest) 2018-21 // 2022+')
    for n,t in allT.items():
        t=t[(t.day>='2018-01-01')].copy().reset_index(drop=True); 
        for c_ in FE: t[c_]=Fs[c_].reindex(t.day).values
        t=t.dropna(subset=FE); e=E(t.day).values
        for c_ in FE:
            q=t[c_][e].quantile([1/3,2/3]).values
            for nm,sel in (('low',t[c_]<=q[0]),('high',t[c_]>q[1])):
                res=[]
                for mm in (e,~e):
                    a,bb=t.pts[mm&sel.values],t.pts[mm&~sel.values]; res.append((a.mean()-bb.mean(),tt(a,bb),len(a)))
                tag='  <== PASS' if passed(res) else ''
                if tag or abs(res[0][1])>=1.5 and abs(res[1][1])>=1.5 and np.sign(res[0][0])==np.sign(res[1][0]): print(f'  {n:6s} {c_:6s} {nm:4s} {res[0][0]:+6.1f} (t{res[0][1]:+.1f},n{res[0][2]}) // {res[1][0]:+6.1f} (t{res[1][1]:+.1f},n{res[1][2]}){tag}')
        print(f'  -- {n}: {len(t)} trades tested', flush=True)
