"""Nov 12 2026: WHERE the flip happens - location vs volatility, multi-day z/range position, and SUPPLY/DEMAND ZONE MEMORY (does a flip at an area that previously rejected hold or fail?). DECLARED BEFORE RESULTS.
Rey's idea (AMT/S-R): a flip at 3.70 that rejected at 3.90 and fell to 3.50, then flips again at 3.70 - is it more likely to break the same place or hold?  Two opposite mechanisms, so no pre-declared sign: (A) 'failed before' = supply sits overhead / sellers defended that area -> worse;  (B) 'level tested again' = absorption / weaker supply each test -> better.
Per 1h bull flip (V0: next open entry, exit next open after bear flip, 40bp $1-5 | 15bp BIG), zone = flip close +/- 1.0 ATR(1h), memory = prior flips of the same ticker in the last 60 sessions (420 bars) that already EXITED before this flip:
  n_zone (count of prior bull flips in the zone), fail_zone (share of them with trade ret < 0), last_fail (most recent one failed?), zone_rej (mean max high reached after them, in ATR),
  sup_n (confirmed 1h pivot highs [8-bar strength] within +1.5 ATR above the flip = overhead supply), sup_d (distance to nearest overhead pivot high, ATR, cap 10), dem_n (pivot lows within -1.5 ATR below = demand under it).
Location: dist_open ((close - session open)/(px*sigma_HAR)), gap, z5/z20/z60 (close vs mean of last 5/20/60 completed daily closes over px*sigma), pr20/pr60 (position in the 20/60-day high-low range), dist_hi60 ((60d high - close)/ATR: room to the highs).
Labels: HOLD12 (no bear flip within 12 bars), WIN (ret>0). Report: single-feature AUC (train/test), zone-memory table, quintile tables. Train < 2025-07-01 / test after. ~20 cells per book, so ~1 chance hit per book is expected."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from sklearn.metrics import roc_auc_score
from vector_pine import run_state_machine
from vector_core import compute_atr_series
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); SPLIT=pd.Timestamp('2025-07-01'); LB=420
TK={tk:Y.dropna() for tk,Y in data.items() if len(Y.dropna())>=1500}; dv={tk:(Y.close*Y.volume).groupby(Y.index.normalize()).sum().median() for tk,Y in TK.items()}; big={t for t,v in dv.items() if v>=5e7}
def daily(Y):
    d=Y.index.normalize(); G=Y.groupby(d); o=G.agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last')); o['rv']=(np.log(Y.close).diff()**2).groupby(d).sum(); return o.iloc[1:]
HD={}; X=[]
for tk,Y in TK.items():
    D=daily(Y); D=D[D.rv>0]; h=pd.DataFrame({'d':np.log(D.rv),'w':np.log(D.rv.rolling(5).mean()),'m':np.log(D.rv.rolling(22).mean())}); h['y']=np.log(D.rv.rolling(5).mean().shift(-5)); HD[tk]=(D,h); X.append(h[h.index<SPLIT-pd.Timedelta(days=7)].dropna())
X=pd.concat(X); beta=np.linalg.lstsq(np.c_[np.ones(len(X)),X[['d','w','m']].values],X.y.values,rcond=None)[0]
rows=[]
for tk,Y in TK.items():
    D,h=HD[tk]; idx=Y.index; days=idx.normalize(); o,hi,lo,c=[Y[k].values for k in ('open','high','low','close')]; atr=compute_atr_series(hi,lo,c,14); reg,lev,_,fl=run_state_machine(Y.assign(datetime=idx))
    sgd=pd.Series(np.sqrt(np.exp(beta[0]+h[['d','w','m']].values@beta[1:])),index=h.index); ix=np.searchsorted(sgd.index.values,days.values)-1; sg=np.where(ix>=0,sgd.values[np.clip(ix,0,None)],np.nan)
    dstart=pd.Series(np.arange(len(idx)),index=idx).groupby(days).transform('min').values; dopen=o[dstart]
    # confirmed pivots (strength 8)
    ph=[(k,hi[k]) for k in range(8,len(Y)-8) if hi[k]==hi[k-8:k+9].max()]; pl=[(k,lo[k]) for k in range(8,len(Y)-8) if lo[k]==lo[k-8:k+9].min()]
    PHk=np.array([k for k,_ in ph]); PHp=np.array([p for _,p in ph]); PLk=np.array([k for k,_ in pl]); PLp=np.array([p for _,p in pl])
    dcl=D.close.values; dhi=D.high.values; dlo=D.low.values; dpos={d:i for i,d in enumerate(D.index)}
    prior=[]; pos=None
    for i,s in fl:
        if s==1 and pos is None: pos=i
        elif s==-1 and pos is not None:
            fi=pos; xb=i+1; pos=None
            if xb>=len(Y) or fi<300 or not np.isfinite(atr[fi]) or not np.isfinite(sg[fi]): continue
            e=fi+1; px=o[e]; sm=1<=px<=5; bg=(tk in big and px>=10); cost=.004 if sm else .0015; A=atr[fi]; P=c[fi]; ret=o[xb]/px-1-cost
            mx=(hi[e:xb+1].max()-P)/A
            if sm or bg:
                near=[p_ for p_ in prior if fi-p_[0]<=LB and abs(P-p_[1])<=1.0*A]
                d0=dpos.get(days[fi],None)
                if d0 is not None and d0>=61:
                    m=(PHk+8<=fi)&(fi-PHk<=LB); above=PHp[m][(PHp[m]>P)]; m2=(PLk+8<=fi)&(fi-PLk<=LB); below=PLp[m2][(PLp[m2]<P)&(PLp[m2]>=P-1.5*A)]
                    f=dict(tk=tk,t=idx[e],book='1-5' if sm else 'BIG',ret=ret,hold=float(not any(j>fi and j<=fi+12 and t_==-1 for j,t_ in fl)) if fi+12<len(Y) else np.nan,
                      n_zone=len(near),fail_zone=np.mean([q[2]<0 for q in near]) if near else np.nan,last_fail=float(near[-1][2]<0) if near else np.nan,zone_rej=np.mean([q[3] for q in near]) if near else np.nan,
                      sup_n=float(((above<=P+1.5*A)).sum()),sup_d=min((above.min()-P)/A,10) if len(above) else 10.,dem_n=float(len(below)),
                      dist_open=(P-dopen[fi])/(P*sg[fi]),gap=(dopen[fi]-dcl[d0-1])/(dcl[d0-1]*sg[fi]),
                      z5=(P-dcl[d0-5:d0].mean())/(P*sg[fi]),z20=(P-dcl[d0-20:d0].mean())/(P*sg[fi]),z60=(P-dcl[d0-60:d0].mean())/(P*sg[fi]),
                      pr20=(P-dlo[d0-20:d0].min())/max(dhi[d0-20:d0].max()-dlo[d0-20:d0].min(),1e-9),pr60=(P-dlo[d0-60:d0].min())/max(dhi[d0-60:d0].max()-dlo[d0-60:d0].min(),1e-9),dist_hi60=(dhi[d0-60:d0].max()-P)/A)
                    rows.append(f)
            prior.append((fi,P,ret,mx))
T=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan); T['tr']=T.t<SPLIT; T['win']=(T.ret>0).astype(int); T.to_pickle(SC+'zones_T.pkl'); print('flips',len(T),T.groupby('book').size().to_dict())
FE=['n_zone','fail_zone','last_fail','zone_rej','sup_n','sup_d','dem_n','dist_open','gap','z5','z20','z60','pr20','pr60','dist_hi60']
def st(g):
    if len(g)<8: return 'n%d'%len(g)
    r=g.ret.sort_values(); return 'n%-4d hold %2.0f%% win %2.0f%% mean %+5.2f%% ex-top1 %+5.2f%%'%(len(g),g.hold.mean()*100,(g.ret>0).mean()*100,g.ret.mean()*100,r.iloc[:int(len(r)*.99)].mean()*100)
for bk in ('1-5','BIG'):
    D=T[T.book==bk]; tr,te=D[D.tr],D[~D.tr]; print('\n================ %s train %d test %d | hold12 train %.0f%% test %.0f%% | win train %.0f%% test %.0f%%'%(bk,len(tr),len(te),tr.hold.mean()*100,te.hold.mean()*100,tr.win.mean()*100,te.win.mean()*100))
    print('-- single-feature AUC train/test  [WIN]  and [HOLD12]  (* = same side of .5 and |AUC-.5|>=.02 in both halves)')
    out=[]
    for f in FE:
        a=tr.dropna(subset=[f]); b=te.dropna(subset=[f]); r=[]
        for lab in ('win','hold'):
            aa=a.dropna(subset=[lab]); bb=b.dropna(subset=[lab])
            if aa[f].nunique()<3 or len(aa)<100 or len(bb)<100: r.append('n/a'); continue
            x,y=roc_auc_score(aa[lab].astype(int),aa[f]),roc_auc_score(bb[lab].astype(int),bb[f]); r.append('%.3f/%.3f%s'%(x,y,'*' if (x-.5)*(y-.5)>0 and min(abs(x-.5),abs(y-.5))>=.02 else ' '))
        out.append('  %-10s WIN %-14s HOLD %-14s'%(f,r[0],r[1]))
    print('\n'.join(out))
    print('-- ZONE MEMORY: flips at an area where a prior bull flip (same ticker, last 60 sessions) already exited')
    for nm,m in (('train',D.tr),('test ',~D.tr)):
        d=D[m]; print('  %s no prior flip in zone : %s'%(nm,st(d[d.n_zone==0])))
        print('  %s prior flip(s), LAST FAILED: %s'%(nm,st(d[d.last_fail==1]))); print('  %s prior flip(s), LAST WORKED: %s'%(nm,st(d[d.last_fail==0])))
        print('  %s 2+ prior in zone       : %s'%(nm,st(d[d.n_zone>=2])))
    print('-- OVERHEAD SUPPLY (confirmed pivot highs within +1.5 ATR above the flip)')
    for nm,m in (('train',D.tr),('test ',~D.tr)):
        d=D[m]; print('  %s sup_n=0: %s | sup_n>=1: %s | sup_n>=3: %s'%(nm,st(d[d.sup_n==0]),st(d[d.sup_n>=1]),st(d[d.sup_n>=3])))
    print('-- QUINTILES (train cut points)')
    for f in ('dist_open','z20','z60','pr60','dist_hi60','gap'):
        q=np.unique(tr[f].quantile([.2,.4,.6,.8]).values); D2=D.assign(q=np.searchsorted(q,D[f]))
        for nm,m in (('train',D2.tr),('test ',~D2.tr)): print('  %-9s %s '%(f,nm)+' | '.join('Q%d %s'%(k,st(g).replace(' ex-top1','x1').replace('hold ','h').replace('win ','w').replace('mean ','m')) for k,g in D2[m].groupby('q')))
