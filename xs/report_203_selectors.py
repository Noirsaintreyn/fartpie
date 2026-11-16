"""Report 203: selectors for which sub-$5 flip becomes big. DECLARED before results.
Universe: outlier_features.pkl flips with px<5 (2019-01..2026-02; train entry<2022, test 2022+; r = trade return, BIG = r>=+50%) + late-2026 check = report 199 clean flips entered 2026-03..11 (ret).
Earnings-based z-score is NOT testable: the earnings files cover 0% of sub-$5 names (data/fund/earn holds split files for 108 large names). Quarterly self-history z-scores substitute: each feature vs the same name's own values 63/126/189/252 trading days earlier (4 prior quarters).
Features at the signal-day close: BASE (60d range/close), PRIOR_DD (130d drawdown), ZBASE (BASE z vs own prior 4 quarters), ZATR (ATRP z), ZVOL (20d dollar volume z), COIL (ATR10/ATR60; low = compressed 'spring'), DLOW (days since 60d low /60; small = just left the low), VX (flip-day volume / 20d mean), P1 = ATRP*sqrt(VX) (report 99 physics pressure), OS2 = ATRP+EXT train-ECDF rank-sum (report 118 minus turnover), GATE = index drawdown <= -10% at the signal day (market_state.idx_dd; April-2026 hypothesis: flips right after a broad selloff), plus TIME-IN-GATE none.
Selectors, cuts frozen on TRAIN (top quintile by train 80th pct; COIL and DLOW bottom quintile): S_OS2, S_BASE, S_PDD, S_ZBASE, S_ZATR, S_COIL, S_DLOW, S_P1, S_VX, S_GATE, combos S_BASE&GATE, S_BASE&COIL, S_BASE&VX, S_OS2&BASE, S_BASE&ZBASE. 15 selectors tried (multiple-testing bar: |t|>=2.9 at Bonferroni 15). Report per era: n, mean r, mean ex best 3, median, win, P(BIG); PASS lead = test mean > all-flips test mean AND ex-best3 > all AND same direction in train AND n>=30 both eras."""
import pandas as pd, numpy as np
from scipy import stats
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; E=X+'ext/'
Of=pd.read_pickle(E+'outlier_features.pkl'); Of=Of[Of.px<5][['tk','d','r','ATRP','EXT','VOLX']].copy(); Of['src']='of'
R=pd.read_pickle(X+'penny2026_199.pkl'); R=R[(~R.disc)&(R.entry>='2026-03-01')].copy(); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); MS=pd.read_pickle(E+'market_state.pkl')
R['sd']=R.entry; ev=[(tk,d,r,'of') for tk,d,r in zip(Of.tk,Of.d,Of.r)]+[(tk,e,r,'late') for tk,e,r in zip(R.tk,R.entry,R.ret)]
def lagz(s,lags=(63,126,189,252)):
    M=np.column_stack([s.shift(l).values for l in lags]); return (s.values-np.nanmean(M,1))/(np.nanstd(M,1)+1e-9)
FE={}
for tk in {e[0] for e in ev}:
    B=D.get(tk)
    if B is None: continue
    B=B[~B.index.duplicated()]; B=B[B.index>='2017-06-01']; c,h,l,v=B.close,B.high,B.low,B.volume; pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr14=tr.rolling(14).mean(); atrp=atr14/c; b60=(h.rolling(60).max()-l.rolling(60).min())/c; pdd=1-c.rolling(131).min()/c.rolling(131).max(); dv=np.log((v*c).rolling(20).mean()+1)
    dlow=l.rolling(60).apply(lambda x:(59-np.argmin(x))/59.,raw=True)
    F=pd.DataFrame(dict(BASE=b60,PRIOR_DD=pdd,ZBASE=lagz(b60),ZATR=lagz(atrp),ZVOL=lagz(dv),COIL=tr.rolling(10).mean()/tr.rolling(60).mean(),DLOW=dlow,VXr=v/v.rolling(20).mean(),ATRP=atrp,EXT=c/l.rolling(60).min()-1),index=B.index); FE[tk]=F
rows=[]
for tk,d,r,src in ev:
    F=FE.get(tk)
    if F is None: continue
    d=pd.Timestamp(d); 
    if src=='late': d=F.index[F.index.searchsorted(d)-1]
    if d not in F.index: continue
    f=F.loc[d]; 
    if f.isna().any(): continue
    ix=MS.index.searchsorted(d); dd=MS.idx_dd.iloc[min(ix,len(MS)-1)]
    rows.append(dict(tk=tk,d=d,r=r,src=src,**f.to_dict(),IDXDD=dd))
Z=pd.DataFrame(rows); Z['era']=np.where(Z.d<'2022-01-01','train',np.where(Z.src=='late','late26','test')); Z['big']=Z.r>=.5; Z['P1']=Z.ATRP*np.sqrt(Z.VXr); Z['GATE']=Z.IDXDD<=-.10
print('flips',len(Z),Z.groupby('era').size().to_dict())
tr=Z[Z.era=='train']; ec=lambda col:(lambda s:(lambda x:np.searchsorted(s,x,side='right')/len(s)))(np.sort(tr[col].values)); Z['OS2']=ec('ATRP')(Z.ATRP.values)+ec('EXT')(Z.EXT.values); tr=Z[Z.era=='train']
print('\nAUC for BIG (r>=+50%), same sign needed: train | test (2022-Feb26) | late-2026')
for f in ('BASE','PRIOR_DD','ZBASE','ZATR','ZVOL','COIL','DLOW','VXr','P1','OS2','IDXDD'):
    out=[]
    for e in ('train','test','late26'):
        s=Z[Z.era==e]; a=s[s.big][f]; b=s[~s.big][f]; out.append(f'{stats.mannwhitneyu(a,b).statistic/(len(a)*len(b)):.3f}(nBIG{len(a)})' if len(a)>3 else 'n/a')
    print(f'  {f:9s}',' | '.join(out))
q=lambda f,p:tr[f].quantile(p)
S={'ALL':np.ones(len(Z),bool),'S_OS2':Z.OS2>=q('OS2',.8),'S_BASE':Z.BASE>=q('BASE',.8),'S_PDD':Z.PRIOR_DD>=q('PRIOR_DD',.8),'S_ZBASE':Z.ZBASE>=q('ZBASE',.8),'S_ZATR':Z.ZATR>=q('ZATR',.8),'S_COIL(low)':Z.COIL<=q('COIL',.2),'S_DLOW(low)':Z.DLOW<=q('DLOW',.2),'S_P1':Z.P1>=q('P1',.8),'S_VX':Z.VXr>=q('VXr',.8),'S_GATE':Z.GATE}
S['S_BASE&GATE']=S['S_BASE']&Z.GATE; S['S_BASE&COIL']=S['S_BASE']&(Z.COIL<=q('COIL',.5)); S['S_BASE&VX']=S['S_BASE']&(Z.VXr>=q('VXr',.5)); S['S_OS2&BASE']=S['S_OS2']&S['S_BASE']; S['S_BASE&ZBASE']=S['S_BASE']&(Z.ZBASE>=q('ZBASE',.5))
print('\nselector results (cuts frozen on train): n | mean | ex-best3 | median | win | P(BIG)')
for k,m in S.items():
    cells=[]
    for e in ('train','test','late26'):
        s=Z[m&(Z.era==e)]
        if len(s)<5: cells.append('n<5'); continue
        r=s.r.sort_values(); cells.append(f'n{len(s):4d} {s.r.mean()*100:6.1f}% x3 {r.iloc[:-3].mean()*100:6.1f}% med {s.r.median()*100:6.1f}% win {np.mean(s.r>0)*100:3.0f}% BIG {s.big.mean():.0%}')
    print(f'  {k:14s}',' | '.join(cells))
Z.to_pickle(X+'selectors203.pkl')
