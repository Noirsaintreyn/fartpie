"""Report 202: checking report 201's candidate patterns (deep prior drawdown PRIOR_DD, wide 60d BASE, negative R60) and the April-2026 timing cluster. (a) 2026 flips by entry month. (b) the same features on the 2019-Feb 2026 sub-$5 flips (outlier_features.pkl, px<5; r = trade return; features recomputed from daily bars at the signal day, 130d window for PRIOR_DD). PASS = same sign AUC vs BIG(r>=.5) on both eras (train 2019-21 / test 2022+), |t|>=2 on the Spearman IC, n>=30."""
import pandas as pd, numpy as np
from scipy import stats
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
R=pd.read_pickle(SP+'big_flip_feats201.pkl'); R['mo']=R.entry.dt.to_period('M')
g=R.groupby('mo').agg(n=('ret','size'),mean=('ret','mean'),median=('ret','median'),big=('big','sum'),huge=('huge','sum')); g['mean']=(g['mean']*100).round(1); g['median']=(g['median']*100).round(1); print('(a) 2026 flips by entry month'); print(g.to_string())
Of=pd.read_pickle(X+'outlier_features.pkl'); D=pd.read_pickle(X+'wide_daily_ohlcv.pkl'); Of=Of[Of.px<5].copy(); rows=[]
for tk,z in Of.groupby('tk'):
    B=D.get(tk)
    if B is None: continue
    B=B[~B.index.duplicated()]; idx=B.index; h,l,c=B.high.values,B.low.values,B.close.values
    for ix,d in zip(z.index,z.d):
        i=idx.searchsorted(d)
        if i<130 or i>=len(idx) or idx[i]!=d: continue
        w=c[i-130:i+1]; rows.append((ix,1-w.min()/w.max(),(h[i-59:i+1].max()-l[i-59:i+1].min())/c[i],c[i]/c[i-60]-1))
F=pd.DataFrame(rows,columns=['ix','PRIOR_DD','BASE','R60c']).set_index('ix'); Z=Of.join(F,how='inner'); Z['big']=Z.r>=.5; Z['era']=np.where(Z.d<'2022-01-01','train','test')
print('\n(b) 2019-Feb2026 sub-$5 flips with features:',len(Z),Z.groupby('era').size().to_dict())
for f in ('PRIOR_DD','BASE','R60c'):
    for e in ('train','test'):
        s=Z[Z.era==e]; a=s[s.big][f]; b=s[~s.big][f]; auc=stats.mannwhitneyu(a,b).statistic/(len(a)*len(b)); ic=s[f].corr(s.r,method='spearman'); t=ic*np.sqrt((len(s)-2)/(1-ic**2))
        q=s[f].quantile([.2,.8]).values; lo=s[s[f]<=q[0]].r; hi=s[s[f]>=q[1]].r
        print(f'  {f:9s}{e:6s} nBIG{len(a):5d}/{len(s):6d} AUC {auc:.3f} IC {ic:+.3f} t{t:+5.1f} | bottom-quintile mean {lo.mean()*100:5.2f}% top-quintile {hi.mean()*100:5.2f}%  P(BIG) {lo.gt(.5).mean():.1%} -> {hi.gt(.5).mean():.1%}')
print('\n(c) robustness of the top-quintile mean (PRIOR_DD, BASE): drop the best 3 trades, median, win rate, and controlling for ATRP+EXT (OS2 rank-sum, train ECDF)')
tr=Z[Z.era=='train'].dropna(subset=['ATRP','EXT']); ec=lambda col:(lambda s:(lambda x:np.searchsorted(s,x,side='right')/len(s)))(np.sort(tr[col].values)); fa,fe=ec('ATRP'),ec('EXT'); Z['OS2']=fa(Z.ATRP.values)+fe(Z.EXT.values)
for f in ('PRIOR_DD','BASE'):
    for e in ('train','test'):
        s=Z[Z.era==e]; q=s[f].quantile(.8); top=s[s[f]>=q]; rest=s[s[f]<q]; t3=top.r.sort_values().iloc[:-3]
        print(f'  {f:9s}{e:6s} top-quintile n{len(top):4d} mean {top.r.mean()*100:6.2f}% ex-best3 {t3.mean()*100:6.2f}% median {top.r.median()*100:6.2f}% win {np.mean(top.r>0)*100:3.0f}% | rest mean {rest.r.mean()*100:5.2f}% median {rest.r.median()*100:6.2f}% | corr with OS2 {s[f].corr(s.OS2,method="spearman"):+.2f}')
        hi_os=s[s.OS2>=s.OS2.quantile(.5)]; a=hi_os[hi_os[f]>=hi_os[f].quantile(.8)]; b=hi_os[hi_os[f]<hi_os[f].quantile(.8)]; print(f'      within high-OS2 half: top-quintile {f} mean {a.r.mean()*100:6.2f}% (n{len(a)}) vs rest {b.r.mean()*100:6.2f}% (n{len(b)}); P(BIG) {a.r.gt(.5).mean():.1%} vs {b.r.gt(.5).mean():.1%}')
