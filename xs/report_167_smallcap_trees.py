"""report 167: tree / boosted tree / logistic on the sub-$5 small-cap vector flips (25,519 flips 2019-2026, ext/structure_small.pkl). DECLARED BEFORE RESULTS. Fit on train (tr, <2022) ONLY, score 2022+.
Features (all entry-time): OS, ATRP, LTURN, LMC, VR, EXT, R60, BLOW, VOLX, PEERS, LDV, LPX, IDIO, POOL_raw, TS12, TS1, MAC, TTR, ENR, SKEW, KURT, PANIC, MKT126, MVR, CONT. Targets: big (net >= +30%) and huge (>= +100%).
Models: logistic on OS only (the live ranking), logistic on all, tree depth 3 (min leaf 200), HGB depth 3 x 200 x lr .05 (min leaf 200). Metrics: test AUC, and lift = huge/big rate in the TOP DECILE of the model score in test vs top decile of OS (decile cut frozen from the train score distribution)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
O=pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/structure_small.pkl'); tr=O.tr.values.astype(bool)
FE=['OS','ATRP','LTURN','LMC','VR','EXT','R60','BLOW','VOLX','PEERS','LDV','LPX','IDIO','POOL_raw','TS12','TS1','MAC','TTR','ENR','SKEW','KURT','PANIC','MKT126','MVR','CONT']
FE=[c for c in FE if c in O.columns]; X=O[FE].astype(float); X=X.fillna(X[tr].median())
print('flips',len(O),'train',tr.sum(),'test',(~tr).sum(),'| big rate train/test',O.big[tr].mean().round(3),O.big[~tr].mean().round(3),'| huge',O.huge[tr].mean().round(4),O.huge[~tr].mean().round(4)); print('features',FE)
for tgt in ('big','huge'):
    y=O[tgt].values.astype(int); print(f'\n=== target {tgt}')
    mods={'logit OS only':(['OS'],make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))),'logit ALL':(FE,make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=800))),'tree3':(FE,DecisionTreeClassifier(max_depth=3,min_samples_leaf=200,random_state=0)),'HGB':(FE,HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=.05,min_samples_leaf=200,random_state=0))}
    base=None
    for nm,(cols,m) in mods.items():
        m.fit(X.loc[tr,cols],y[tr]); pa=m.predict_proba(X.loc[tr,cols])[:,1]; pb=m.predict_proba(X.loc[~tr,cols])[:,1]; cut=np.quantile(pa,.9); top=pb>=cut
        print(f'  {nm:14s} AUC train {roc_auc_score(y[tr],pa):.3f} | test {roc_auc_score(y[~tr],pb):.3f}   top-decile (train cut) n{top.sum():5d} {tgt}-rate {y[~tr][top].mean():.3f}  vs test base {y[~tr].mean():.3f}  lift {y[~tr][top].mean()/y[~tr].mean():.2f}x')
        if nm=='tree3' and tgt=='huge': print(export_text(m,feature_names=FE,max_depth=3))
