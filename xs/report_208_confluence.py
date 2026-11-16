"""Report 208: confluence count for BIG sub-$5 flips (Rey: rank by how many probability-raising factors line up). DECLARED before results. Equal-weight binary factors (no fitted weights), cuts from TRAIN 2019-21 only: F1 BASE (60d range/close) >= train 80th pct; F2 PRIOR_DD (130d drawdown) >= 80th; F3 P1 = ATRP*sqrt(flip-day volume ratio) >= 80th; F4 name's 252d-median ATRP (lagged) >= 80th; F5 an 8-K filed in the prior 7 days. CAVEAT stated up front: these factors were picked from reports 201-207 after seeing test-era AUCs, so test is NOT clean; F1-F4 are strongly collinear (rank corr ~.8-.9), so the count mostly measures one thing plus the 8-K flag. Score = count 0-5. Report per era (train | test | late 2026): n, P(BIG>=+50%), mean, mean ex-best3, median, win by count; Spearman(count, r); then the same count applied to the 3,436 1H flips Aug-Nov 2026 (r net 100bp; P1 uses the 1H flip-bar VOLX), lead only."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; E=SP+'ext/'
Z=pd.read_pickle(SP+'selectors203.pkl'); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); e8=pd.read_csv(E+'8k_wide.csv',parse_dates=['filed']); G8={t:g.filed.values for t,g in e8.groupby('ticker')}
def namefeat(tk,d):
    B=D.get(tk)
    if B is None: return None
    B=B[~B.index.duplicated()]; i=B.index.searchsorted(d)-1
    return B,i
cache={}
def atrn(tk):
    if tk in cache: return cache[tk]
    B=D[tk]; B=B[~B.index.duplicated()]; c,h,l=B.close,B.high,B.low; pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); cache[tk]=(tr.rolling(14).mean()/c).rolling(252,min_periods=120).median().shift(1); return cache[tk]
def k8(tk,d): a=G8.get(tk); return bool(a is not None and ((a>=np.datetime64(d-pd.Timedelta(days=7)))&(a<=np.datetime64(d))).any())
Z['ATRn']=[atrn(t).get(d,np.nan) if t in D else np.nan for t,d in zip(Z.tk,Z.d)]; Z['K8']=[k8(t,d) for t,d in zip(Z.tk,Z.d)]; Z=Z.dropna(subset=['ATRn'])
tr=Z[Z.era=='train']; cut={f:tr[f].quantile(.8) for f in ('BASE','PRIOR_DD','P1','ATRn')}
def score(df,pcol='P1'): return (df.BASE>=cut['BASE']).astype(int)+(df.PRIOR_DD>=cut['PRIOR_DD']).astype(int)+(df[pcol]>=cut['P1']).astype(int)+(df.ATRn>=cut['ATRn']).astype(int)+df.K8.astype(int)
Z['C']=score(Z); print('flips',len(Z),'| rank corr among F1-F4:',Z[['BASE','PRIOR_DD','P1','ATRn']].corr(method='spearman').round(2).values[np.triu_indices(4,1)].tolist())
def show(df,tag):
    print(f'\n{tag}: count -> n | P(BIG) | mean | ex-best3 | median | win')
    for e in ('train','test','late26'):
        s=df[df.era==e]; rho=stats.spearmanr(s.C,s.r)[0]; rhoB=stats.spearmanr(s.C,s.big)[0]; print(f'  {e}: Spearman(count, r) {rho:+.3f}, (count, BIG) {rhoB:+.3f}')
        for k in range(0,6):
            x=s[s.C==k] if k<5 else s[s.C>=5]; 
            if len(x)<5: continue
            r=x.r.sort_values(); print(f'     C={k}{"+" if k==5 else " "} n{len(x):4d} BIG {x.big.mean()*100:5.1f}% mean {x.r.mean()*100:6.1f}% x3 {r.iloc[:-3].mean()*100 if len(r)>3 else np.nan:6.1f}% med {x.r.median()*100:6.1f}% win {np.mean(x.r>0)*100:3.0f}%')
        hi=s[s.C>=3]; lo=s[s.C<3]; print(f'     C>=3 n{len(hi):4d} BIG {hi.big.mean()*100:5.1f}% mean {hi.r.mean()*100:6.1f}% | C<3 n{len(lo):4d} BIG {lo.big.mean()*100:5.1f}% mean {lo.r.mean()*100:6.1f}%')
show(Z,'(1) daily sub-$5 flips')
print('\n(2) 1H flips Aug-Nov 2026 (net r after 100bp, lead only)')
Fh=pd.read_pickle(E+'flips_1h_score.pkl'); rows=[]
for ix,tk,d,vx in zip(Fh.index,Fh.tk,Fh.d,Fh.VOLX):
    B=D.get(tk)
    if B is None: continue
    B=B[~B.index.duplicated()]; i=B.index.searchsorted(d)-1
    if i<131: continue
    c,h,l=B.close.values,B.high.values,B.low.values; pc=np.r_[c[0],c[:-1]]; t_=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); atr=t_[i-13:i+1].mean()/c[i]; an=atrn(tk).iloc[i] if tk in cache or True else np.nan
    rows.append(dict(ix=ix,BASE=(h[i-59:i+1].max()-l[i-59:i+1].min())/c[i],PRIOR_DD=1-c[i-130:i+1].min()/c[i-130:i+1].max(),P1=atr*np.sqrt(vx),ATRn=an,K8=k8(tk,B.index[i])))
H=pd.DataFrame(rows).set_index('ix'); H=Fh.join(H,how='inner').dropna(subset=['ATRn']); H['C']=score(H); H['era']='1H'; H['big']=H.r>=.1
print('n',len(H),'; BIG here = net r >= +10% (the 1H flips are shorter holds)')
for k in range(0,6):
    x=H[H.C==k] if k<5 else H[H.C>=5]
    if len(x)<5: continue
    print(f'  C={k}{"+" if k==5 else " "} n{len(x):4d} >=+10% {x.big.mean()*100:5.1f}% mean {x.r.mean()*100:6.2f}% med {x.r.median()*100:6.2f}% win {np.mean(x.r>0)*100:3.0f}%')
print('  Spearman(count, r) %+.3f, (count, >=+10%%) %+.3f'%(stats.spearmanr(H.C,H.r)[0],stats.spearmanr(H.C,H.big)[0]))
