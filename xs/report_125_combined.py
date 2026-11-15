"""report_125 (Nov 15 2026): ONE FROZEN CLASSIFIER from the pieces that survived (ATRP, EXT, turnover, mcap, price, flip volume, break size F2/F11, RS F7, location F4/F1, base F9, VR, PEERS). Ridge logistic on TRAIN-ECDF ranks, target BIG (net >= +30%) and HUGE (>= +100%), trained 2019-21 only, frozen. Missing shares -> train median. Compared with OS (3 ranks) and frozen lr on test: AUC, capture of BIG/HUGE in top 20% / top 10%, precision, mean/median trade. Universes: all flips, px<5."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; O=pd.read_pickle(X+'outlier_features.pkl'); P=pd.read_pickle(X+'physics_flips.pkl')[['tk','d','TURN','MCAP','ATRP']].rename(columns={'TURN':'T2','MCAP':'M2','ATRP':'A2'})
O=O.merge(P,on=['tk','d'],how='left'); O['LTURN']=np.log(O.TURN.clip(lower=1e-9)); O['LMC']=np.log(O.MCAP.clip(lower=1)); O=O[O.d>='2019-01-01'].copy()
O['n']=O.r-.003; O['BIG']=O.n>=.3; O['HUGE']=O.n>=1.0
feats=['ATRP','EXT','LTURN','LMC','LPX','VOLX','F2','F11','F7','F4','F1','F5','R60','BLOW','PEERS','VR','F9']
tr=O[~O.test]
def ecdf(c): s=np.sort(tr[c].dropna().values); return lambda x: np.searchsorted(s,x,side='right')/len(s)
R=pd.DataFrame(index=O.index)
for f in feats:
    e=ecdf(f); r=pd.Series(e(O[f].fillna(tr[f].median()).values),index=O.index); R[f]=r
Xtr=R[~O.test.values]
res={}
for tgt in ('BIG','HUGE'):
    for C in (.05,.5):
        m=LogisticRegression(C=C,max_iter=2000).fit(Xtr,tr[tgt]); O[f'm_{tgt}_{C}']=m.decision_function(R)
        res[(tgt,C)]=dict(zip(feats,m.coef_[0].round(2)))
for k,v in res.items(): print(k,v)
O['OS']=R.ATRP+R.EXT+R.LTURN
def rep(name,col,mask):
    U=O[mask]; line=f'  {name:26s}'
    for era,e in (('TR',~U.test),('TE',U.test)):
        z=U[e]; 
        if len(z)<50: continue
        a1=roc_auc_score(z.BIG,z[col]); a2=roc_auc_score(z.HUGE,z[col]); cut=U[~U.test][col].quantile(.8); top=z[z[col]>=cut]; cut9=U[~U.test][col].quantile(.9); t9=z[z[col]>=cut9]
        line+=f' | {era} AUC big {a1:.3f} huge {a2:.3f} | top20: {len(top)/len(z)*100:4.1f}% of flips catch big {top.BIG.sum()/max(1,z.BIG.sum())*100:4.1f}% huge {top.HUGE.sum()/max(1,z.HUGE.sum())*100:4.1f}% mean {top.n.mean()*100:+6.1f}% med {top.n.median()*100:+5.1f}% | top10 catch big {t9.BIG.sum()/max(1,z.BIG.sum())*100:4.1f}% huge {t9.HUGE.sum()/max(1,z.HUGE.sum())*100:4.1f}%'
    print(line)
for un,mk in (('ALL flips',O.px>0),('px<5',O.px<5)):
    print('\n###',un)
    for col in ('lr','OS','m_BIG_0.05','m_BIG_0.5','m_HUGE_0.05','m_HUGE_0.5'): rep(col,col,mk)
O.to_pickle(X+'combined_scores.pkl')
