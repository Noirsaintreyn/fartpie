"""Nov 12 2026: ENTRY TOOL for the vector 1h bull flip ("floor trader" build). DECLARED BEFORE RESULTS. Everything here has a stated mechanism and fixed rules; nothing is fitted on test (train < 2025-07-01 / test after).
Evidence base that motivates each piece (our own runs):  (i) edge lives at $1-2/$1-5; big-cap flips lose money after costs (report_12_zema/accept);  (ii) flip-bar participation (volume>=1.5x), daily bull, Kalman up, price above 1h+4h EMA200 stacked gave the best trades (report_12_mtf/smooth);
 (iii) 'CONF-H' (entry bar closes above the flip-bar high) was +12% but is KNOWN ONLY AFTER the bar, so it was not tradable as an open entry -> TRIGGER below turns it into a resting buy-stop;  (iv) z-score/acceptance features added nothing (report_12_zall/zfloor/zema/accept) so they are not used.
TRIGGER (mechanism: breakout acceptance - make the market prove the flip, instead of paying the open): V0 next open (baseline) | V1 buy-stop at the flip-bar high, live for the next 1 bar | V2 same, live 2 bars. Fill = max(open, flip high) if the bar trades through it; cancelled if not filled or if the bear flip prints first. Exit unchanged (next open after bear flip). 40 bp round trip.
GATES (each = 1 point): G1 flip-bar volume >= 1.5x prior-20 mean (informed participation) | G2 4h AND daily vector bull on last completed bars (timeframe agreement) | G3 close above 1h EMA200 and 4h EMA200 (institutional anchor) | G4 causal Kalman local-linear-trend slope > 0 (smoothed drift)
 | G5 NOT a last-session-bar flip (15:30 bar: fills at next open with overnight gap risk).  Grade A >= 4 points, B = 3, C <= 2.  Context tags reported but NOT gated: market breadth (share of universe with 1h vector bull at flip), hour-of-day bucket.
SIZING (mechanism: equal risk per name): book sim with 8 slots, 1/8 equity each, priority = flip volume ratio; also a vol-scaled variant (weight ~ 1/ATR%, capped) reported as sensitivity.
PRIMARY pre-declared test: V1 trigger + grade A (>=4) vs V0 baseline (all $1-5 flips) on TEST: kept mean / win / ex-top-1% / random-skip p, and book return / max drawdown. Everything else is descriptive.  Books: $1-5 (main) and $1-2 (screener band); BIG proxy shown once for completeness."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from vector_pine import run_state_machine
from vector_core import compute_atr_series
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); SPLIT=pd.Timestamp('2025-07-01'); COST=.004
def htf(Y,tf):
    day=Y.index.normalize(); g=[day,(Y.index.hour>=13).astype(int)] if tf=='4h' else [day]
    G=Y.groupby(g); o=G.agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'))
    first=G.apply(lambda x:x.index[0]); last=G.apply(lambda x:x.index[-1]); o.index=pd.DatetimeIndex(first.values)
    end=pd.DatetimeIndex(last.values)+pd.to_timedelta(np.where(pd.DatetimeIndex(last.values).hour==15,30,60),unit='m')
    reg=run_state_machine(o.assign(datetime=o.index))[0]; return pd.Series(reg,index=end).sort_index(), pd.Series(o.close.ewm(span=200,adjust=False).mean().values,index=end).sort_index()
def kslope(c):
    l=np.log(c); r=np.diff(l); R=np.var(r) if len(r)>3 else 1e-4; q1,q2=.05*R,.001*R; x0=l[0]; x1=0.; P00,P01,P11=R,0.,R; out=np.zeros(len(l))
    for i in range(len(l)):
        # predict
        x0p=x0+x1; x1p=x1; P00p=P00+2*P01+P11+q1; P01p=P01+P11; P11p=P11+q2
        S=P00p+R; K0,K1=P00p/S,P01p/S; y=l[i]-x0p; x0=x0p+K0*y; x1=x1p+K1*y; P00=(1-K0)*P00p; P01=(1-K0)*P01p; P11=P11p-K1*P01p; out[i]=x1
    return out
REG={}; TK={}
for tk,Y in data.items():
    Y=Y.dropna()
    if len(Y)<1500: continue
    TK[tk]=Y; REG[tk]=pd.Series(run_state_machine(Y.assign(datetime=Y.index))[0]==1,index=Y.index)
BR=pd.DataFrame(REG).mean(axis=1)   # breadth: share of universe 1h bull at each timestamp
dv={tk:(Y.close*Y.volume).groupby(Y.index.normalize()).sum().median() for tk,Y in TK.items()}; big={t for t,v in dv.items() if v>=5e7}
rows=[]
for tk,Y in TK.items():
    idx=Y.index; o,hi,lo,c,vol=[Y[k].values for k in ('open','high','low','close','volume')]; atr=compute_atr_series(hi,lo,c,14); reg,lev,_,fl=run_state_machine(Y.assign(datetime=idx))
    ks=kslope(c); e1=pd.Series(c).ewm(span=200,adjust=False).mean().values; r4,e4=htf(Y,'4h'); rd,_=htf(Y,'1d'); vm=pd.Series(vol).rolling(20).mean().shift().values; pos=None
    bears=[i for i,s in fl if s==-1]
    for i,s in fl:
        if s==1 and pos is None: pos=i
        elif s==-1 and pos is not None:
            fi=pos; xb=i+1; pos=None
            if xb>=len(Y) or fi<250 or not np.isfinite(atr[fi]) or not np.isfinite(vm[fi]): continue
            e=fi+1; px=o[e]
            sm12=1<=px<2; sm15=1<=px<=5; bg=(tk in big and px>=10)
            if not(sm15 or bg): continue
            end_t=idx[fi]+pd.Timedelta(minutes=30 if idx[fi].hour==15 else 60); a4=r4[r4.index<=end_t]; ad=rd[rd.index<=end_t]; v4=e4[e4.index<=end_t]
            if not len(a4) or not len(ad): continue
            g1=vol[fi]/vm[fi]>=1.5; g2=(a4.iat[-1]==1) and (ad.iat[-1]==1); g3=(c[fi]>e1[fi]) and (c[fi]>v4.iat[-1]); g4=ks[fi]>0; g5=idx[fi].hour!=15
            fh=hi[fi]; res={}
            for nm,L in (('V0',0),('V1',1),('V2',2)):
                if L==0: res[nm]=(e,px)
                else:
                    ent=None
                    for k in range(1,L+1):
                        b=fi+k
                        if b>=xb: break      # bear flip printed (exit bar) before fill
                        if hi[b]>=fh: ent=(b,max(o[b],fh)); break
                    res[nm]=ent
            for nm,en in res.items():
                if en is None: continue
                eb,ep=en
                if eb>=xb: continue
                rows.append(dict(tk=tk,var=nm,t=idx[eb],tx=idx[xb],ret=o[xb]/ep-1-COST,book='BIG' if bg and not sm15 else '1-5',b12=sm12,pts=int(g1)+int(g2)+int(g3)+int(g4)+int(g5),g1=g1,g2=g2,g3=g3,g4=g4,g5=g5,
                    rv=vol[fi]/vm[fi],atrp=atr[fi]/c[fi],breadth=BR.get(idx[fi],np.nan),hour=idx[fi].hour))
T=pd.DataFrame(rows); T['tr']=T.t<SPLIT; T['grade']=np.where(T.pts>=4,'A',np.where(T.pts==3,'B','C')); T.to_pickle(SC+'entrytool_T.pkl')
print('rows',len(T),T.groupby(['book','var']).size().to_dict())
def st(g):
    if len(g)<5: return 'n%d'%len(g)
    r=g.ret.sort_values(); return 'n%-4d win %2.0f%% mean %+5.1f%% med %+5.1f%% ex-top1 %+5.1f%%'%(len(g),(g.ret>0).mean()*100,g.ret.mean()*100,g.ret.median()*100,r.iloc[:int(len(r)*.99)].mean()*100)
def book(G,K=8):
    G=G.sort_values('t'); eq=1.; open_=[]; curve=[]; n=0
    for _,r in G.iterrows():
        open_=[(tx,w) for tx,w in open_ if tx>r.t] ; 
        if len(open_)>=K: continue
        eq*=1+r.ret/K; open_.append((r.tx,1)); n+=1; curve.append((r.tx,eq))
    if not curve: return None
    cv=pd.Series([e for _,e in curve],index=[t for t,_ in curve]).sort_index(); dd=(cv/cv.cummax()-1).min(); yrs=max((cv.index[-1]-G.t.min()).days/365,.25); return eq**(1/yrs)-1,dd,n
def null(S,K,n=2000): return float(np.mean([S.sample(len(K),random_state=i).ret.mean()>=K.ret.mean() for i in range(n)]))
for bk,fil in (('$1-5',lambda d:(d.book=='1-5')),('$1-2',lambda d:(d.book=='1-5')&d.b12),('BIG proxy',lambda d:d.book=='BIG')):
    D=T[fil(T)]; print('\n==================== BOOK %s'%bk)
    for sp,m in (('TRAIN',D.tr),('TEST ',~D.tr)):
        print('-- %s'%sp)
        for v in ('V0','V1','V2'):
            S=D[m&(D['var']==v)]; print('  %s all      %s'%(v,st(S)))
            for gname in ('A','B','C'): print('  %s grade %s  %s'%(v,gname,st(S[S.grade==gname])))
        for v in ('V0','V1'):
            S=D[m&(D['var']==v)]; b=book(S); b2=book(S[S.grade=='A']); b3=book(S[S.pts>=3])
            print('  book(8 slots) %s all: %s | grade A: %s | grade A+B: %s'%(v,'cagr %+.0f%% dd %.0f%% n%d'%(b[0]*100,b[1]*100,b[2]) if b else '-','cagr %+.0f%% dd %.0f%% n%d'%(b2[0]*100,b2[1]*100,b2[2]) if b2 else '-','cagr %+.0f%% dd %.0f%% n%d'%(b3[0]*100,b3[1]*100,b3[2]) if b3 else '-'))
    base=D[(~D.tr)&(D['var']=='V0')]; K=D[(~D.tr)&(D['var']=='V1')&(D.grade=='A')]
    if len(K)>10: print('  PRIMARY (test): V1+gradeA kept %s | V0 all %s | random-skip p vs V0 pool %.3f'%(st(K),st(base),null(base,K)))
    # per-gate marginal (V1, both halves)
    print('  gate marginal on V1 (mean when gate on vs off):'+'; '.join('%s tr %+.1f/%+.1f te %+.1f/%+.1f'%(g,D[(D['var']=='V1')&D.tr&D[g]].ret.mean()*100,D[(D['var']=='V1')&D.tr&~D[g]].ret.mean()*100,D[(D['var']=='V1')&~D.tr&D[g]].ret.mean()*100,D[(D['var']=='V1')&~D.tr&~D[g]].ret.mean()*100) for g in ('g1','g2','g3','g4','g5')))
    V=D[D['var']=='V1']; V=V.assign(bq=pd.qcut(V.breadth,3,labels=['lowB','midB','highB'],duplicates='drop'),hb=pd.cut(V.hour,[0,9,12,14,16],labels=['9:30','10-12','13-14','15:30']))
    for tag in ('bq','hb'):
        for sp,m in (('tr',V.tr),('te',~V.tr)): print('  %s %s: '%(tag,sp)+' | '.join('%s %s'%(k,st(g)) for k,g in V[m].groupby(tag)))
