"""Report 207: universe selection (Rey: 'like my options ratio, we need a way to pick a universe, test 1h only'). DECLARED before results.
DATA LIMIT: 1H bars for sub-$5 names exist only for Aug 20-Nov 12 2026 (60 days of 5m, one regime, the drought), so a split test of a 1H-only sub-$5 universe is impossible. Plan: (1) TEMPLATE: the large-cap options-ratio universe on 1H-only vs 15m-only flips (wide_T, 2019-26, train<2022 | test): mean, median, win, BIG(>=+5% large caps), and the dose-response of trade return to the ratio bucket on 1H flips (is the universe variable graded?). (2) SMALL-CAP analogue: name-level, lagged (252d median shifted 1 day) variables on the 2019-26 daily sub-$5 flips (report 203 table: train 2019-21 | test 2022-Feb26 | late 2026): SVR = FINRA short-volume ratio (ShortVolume/TotalVolume), DVL = log dollar volume, ATR = ATRP. One hypothesis each, sign +: higher level -> higher BIG rate and higher mean ex-best-3 (squeeze fuel / attention / volatility). Quintile table + AUC. (3) Apply the same frozen top-tercile universe cuts to the 3,436 1H flips of Aug-Nov 2026 (flips_1h_score.pkl, r net 100bp) as a lead-only check."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; E=SP+'ext/'
W=pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); T=W[W.sp&(W.ratio>=1.5)].copy(); T['era']=np.where(T.entry<'2022-01-01','train','test')
print('(1) large-cap flips (ratio>=1.5): per-trade return by track and era (ret, before cost)')
for nm,m in (('15m',~T.h1),('1H',T.h1)):
    for e in ('train','test'):
        s=T[m&(T.era==e)]; r=s.ret.sort_values(); print(f'  {nm:4s}{e:6s} n{len(s):6d} mean {s.ret.mean()*100:5.2f}% x3 {r.iloc[:-3].mean()*100:5.2f}% med {s.ret.median()*100:5.2f}% win {np.mean(s.ret>0)*100:3.0f}% >=+5%: {np.mean(s.ret>=.05)*100:4.1f}%  >=+10%: {np.mean(s.ret>=.1)*100:4.1f}%')
print('  dose-response on 1H flips: ratio bucket -> mean ret, win (train | test)')
H=T[T.h1].copy(); H['rb']=pd.cut(H.ratio,[1.5,1.75,2,2.5,3,100],right=False)
for b,g in H.groupby('rb'): print(f'    ratio {str(b):12s}',' | '.join(f'n{len(g[g.era==e]):5d} {g[g.era==e].ret.mean()*100:5.2f}% win {np.mean(g[g.era==e].ret>0)*100:3.0f}%' for e in ('train','test')))
Z=pd.read_pickle(SP+'selectors203.pkl'); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); sv=pd.read_pickle(E+'finra_sv_wide.pkl'); sv['Date']=pd.to_datetime(sv.Date); sv=sv[sv.Symbol.isin(set(Z.tk))]; sv['r']=sv.ShortVolume/sv.TotalVolume.clip(lower=1)
SV={k:g.set_index('Date').r.sort_index() for k,g in sv.groupby('Symbol')}; rows=[]
for tk,g in Z.groupby('tk'):
    B=D.get(tk); 
    if B is None: continue
    B=B[~B.index.duplicated()]; c,h,l,v=B.close,B.high,B.low,B.volume; pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); atrp=(tr.rolling(14).mean()/c).rolling(252,min_periods=120).median().shift(1); dvl=np.log((v*c).rolling(20).mean()+1).rolling(252,min_periods=120).median().shift(1)
    s=SV.get(tk); svr=(s.rolling(252,min_periods=120).median().shift(1).reindex(B.index,method='ffill') if s is not None else pd.Series(np.nan,index=B.index))
    for ix,d in zip(g.index,g.d):
        if d in B.index: rows.append((ix,svr.loc[d],dvl.loc[d],atrp.loc[d]))
F=pd.DataFrame(rows,columns=['ix','SVR','DVL','ATRn']).set_index('ix'); Zf=Z.join(F,how='inner'); print('\n(2) sub-$5 name-level universe variables, flips with data:',len(Zf),'| SVR available',Zf.SVR.notna().sum())
for f in ('SVR','DVL','ATRn'):
    s_=Zf.dropna(subset=[f]); print(f'  {f}: quintile (cuts on train) -> n | mean | ex-best3 | median | BIG   (train | test | late26); AUC for BIG')
    tr=s_[s_.era=='train']; q=tr[f].quantile([.2,.4,.6,.8]).values
    for e in ('train','test','late26'):
        s=s_[s_.era==e]; a=s[s.big][f]; b=s[~s.big][f]; print(f'     {e:6s} AUC {stats.mannwhitneyu(a,b).statistic/(len(a)*len(b)):.3f} (nBIG{len(a)})  ', end='')
        for k,(lo,hi) in enumerate(zip([-np.inf,*q],[*q,np.inf])):
            x=s[(s[f]>lo)&(s[f]<=hi)]; r=x.r.sort_values(); print(f'Q{k+1}: n{len(x):3d} {x.r.mean()*100:6.1f}% x3 {r.iloc[:-3].mean()*100 if len(r)>3 else np.nan:6.1f}% BIG {x.big.mean()*100:3.0f}% | ',end='')
        print()
print('\n(3) 1H flips Aug-Nov 2026 (flips_1h_score.pkl, net r after 100bp): top-tercile vs rest on the same name-level variables (cuts from the 2019-21 train table)')
Fh=pd.read_pickle(E+'flips_1h_score.pkl'); rows=[]
for tk,g in Fh.groupby('tk'):
    B=D.get(tk)
    if B is None: continue
    B=B[~B.index.duplicated()]; c,v=B.close,B.volume; dvl=np.log((v*c).rolling(20).mean()+1).rolling(252,min_periods=120).median().shift(1); pc=c.shift(1); atrp=((B.high-B.low).rolling(14).mean()/c).rolling(252,min_periods=120).median().shift(1)
    s=SV.get(tk); svr=s.rolling(252,min_periods=120).median().shift(1).reindex(B.index,method='ffill') if s is not None else None
    for ix,d in zip(g.index,g.d):
        i=B.index.searchsorted(d)-1
        if i>=0: rows.append((ix,svr.iloc[i] if svr is not None else np.nan,dvl.iloc[i],atrp.iloc[i]))
Fx=pd.DataFrame(rows,columns=['ix','SVR','DVL','ATRn']).set_index('ix'); Hh=Fh.join(Fx,how='inner'); trn=Zf[Zf.era=='train']
for f in ('SVR','DVL','ATRn'):
    c3=trn[f].quantile(2/3); s=Hh.dropna(subset=[f]); a=s[s[f]>=c3]; b=s[s[f]<c3]; print(f'  {f}: top-tercile n{len(a):4d} mean {a.r.mean()*100:6.2f}% med {a.r.median()*100:6.2f}% >=+10% {np.mean(a.r>=.1)*100:4.1f}% | rest n{len(b):4d} mean {b.r.mean()*100:6.2f}% med {b.r.median()*100:6.2f}% >=+10% {np.mean(b.r>=.1)*100:4.1f}%')
