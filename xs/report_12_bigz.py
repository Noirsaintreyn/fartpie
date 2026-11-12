"""Nov 12 2026: LARGER CAPS - tradable test of the EMA z-score tilt. DECLARED BEFORE RESULTS. report_12_zema found, in BIG-book vector-bull bars with rel. volume >= 1.5, that stretch above the daily/4h EMAs (daily 50-200 spread, daily/4h z vs EMA200) was followed by slightly HIGHER returns (IC +0.02..+0.05 both halves), i.e. continuation, the opposite of the small book.
Here: make it a rule and charge costs. SIGNAL (BIG tickers = median daily $vol >= $50M, px >= $10): on a 1h bar close with 1h vector bull AND flip-or-pullback-agnostic rv >= 1.5 AND feature >= TRAIN tercile cut (cuts from train bars only). Enter next open, hold H bars (24 / 48 / 120) or until the 1h vector turns bear, whichever first; one position per ticker at a time; 15 bp round trip.
Features tried (fixed list, all causal): sp_d (daily EMA50-EMA200 spread / (px*sigma)), zd_200, z4h_200, sp_4h, plus 'any2' = at least two of those four above their train upper-tercile.  Compared with the same rule without the z filter (VB & rv>=1.5) and with random-skip of equal n. Train < 2025-07-01 | test after."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from vector_pine import run_state_machine
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); SPLIT=pd.Timestamp('2025-07-01'); COST=.0015
def daily(Y):
    d=Y.index.normalize(); o=Y.groupby(d).agg(close=('close','last')); o['rv']=(np.log(Y.close).diff()**2).groupby(d).sum(); return o.iloc[1:]
TK={tk:Y.dropna() for tk,Y in data.items() if len(Y.dropna())>=1500}; dv={tk:(Y.close*Y.volume).groupby(Y.index.normalize()).sum().median() for tk,Y in TK.items()}; big=[t for t,v in dv.items() if v>=5e7]
HD={}; X=[]
for tk,Y in TK.items():
    D=daily(Y); D=D[D.rv>0]; h=pd.DataFrame({'d':np.log(D.rv),'w':np.log(D.rv.rolling(5).mean()),'m':np.log(D.rv.rolling(22).mean())}); h['y']=np.log(D.rv.rolling(5).mean().shift(-5)); HD[tk]=h; X.append(h[h.index<SPLIT-pd.Timedelta(days=7)].dropna())
X=pd.concat(X); beta=np.linalg.lstsq(np.c_[np.ones(len(X)),X[['d','w','m']].values],X.y.values,rcond=None)[0]
B=[]; PX={}
for tk in big:
    Y=TK[tk]; h=HD[tk]; idx=Y.index; days=idx.normalize(); sgd=pd.Series(np.sqrt(np.exp(beta[0]+h[['d','w','m']].values@beta[1:])),index=h.index)
    ix=np.searchsorted(sgd.index.values,days.values)-1; sg=np.where(ix>=0,sgd.values[np.clip(ix,0,None)],np.nan); c=Y.close; k4=pd.MultiIndex.from_arrays([days,(idx.hour>=13).astype(int)])
    F=pd.DataFrame(index=idx); F['px']=c.values
    def hz(keys,n): return pd.Series(c.groupby(keys).last().ewm(span=n,adjust=False).mean().shift(1).reindex(keys).values,index=idx)
    ed200,ed50,e4200,e450=hz(days,200),hz(days,50),hz(k4,200),hz(k4,50)
    F['sp_d']=((ed50-ed200)/(c*sg)).values; F['zd_200']=((c-ed200)/(c*sg)).values; F['z4h_200']=((c-e4200)/(c*sg)).values; F['sp_4h']=((e450-e4200)/(c*sg)).values
    F['rv']=(Y.volume/Y.volume.rolling(20).mean().shift()).values; reg=run_state_machine(Y.assign(datetime=idx))[0]; F['vb']=(reg==1); F['tk']=tk; B.append(F); PX[tk]=Y
B=pd.concat(B); B['tr']=B.index<SPLIT; FE=['sp_d','zd_200','z4h_200','sp_4h']; base=B[B.vb&(B.rv>=1.5)&(B.px>=10)].dropna(subset=FE)
cuts={f:base[base.tr][f].quantile(2/3) for f in FE}; print('train upper-tercile cuts',{k:round(v,2) for k,v in cuts.items()}); base['any2']=sum((base[f]>=cuts[f]).astype(int) for f in FE)>=2
def run(sig,H):
    out=[]
    for tk,g in sig.groupby('tk'):
        Y=PX[tk]; o,c=Y.open.values,Y.close.values; reg=run_state_machine(Y.assign(datetime=Y.index))[0]; ip={t:i for i,t in enumerate(Y.index)}; free=-1
        for t in g.index:
            i=ip[t]
            if i<=free or i+1>=len(Y): continue
            e=i+1; x=min(e+H,len(Y)-1); 
            for j in range(e,x):
                if reg[j]==-1: x=j+1 if j+1<len(Y) else j; break
            out.append((Y.index[e],o[x]/o[e]-1-COST,tk)); free=x
    return pd.DataFrame(out,columns=['t','ret','tk'])
def st(r):
    if len(r)<8: return 'n%d'%len(r)
    s=r.sort_values(); return 'n%-4d win %2.0f%% mean %+5.2f%% ex-top1 %+5.2f%%'%(len(r),(r>0).mean()*100,r.mean()*100,s.iloc[:int(len(s)*.99)].mean()*100)
for H in (24,48,120):
    print('\n=== hold up to %d bars (or vector bear flip)'%H)
    ref=run(base,H); ref['tr']=ref.t<SPLIT
    for nm,m in (('train',ref.tr),('test ',~ref.tr)): print('  %s  VB & rv>=1.5, no z filter: %s'%(nm,st(ref[m].ret)))
    for f in FE+['any2']:
        sig=base[(base[f]>=cuts[f])] if f!='any2' else base[base.any2]; r=run(sig,H); r['tr']=r.t<SPLIT; line=[]
        for nm,m in (('train',r.tr),('test ',~r.tr)):
            rr=ref[ref.tr==(nm=='train')].ret; k=r[m].ret; null=np.mean([rr.sample(min(len(k),len(rr)),random_state=i).mean()>=k.mean() for i in range(1000)]) if len(k)>8 else np.nan
            line.append('%s %s p(skip) %.3f'%(nm,st(k),null))
        print('  %-8s '%f+' || '.join(line))
