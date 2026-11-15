"""report 141: daily-bar TWAP/accumulation proxy on the sub-$5 small-cap flips (where the ideology came from). DECLARED BEFORE RESULTS.
Proxy for regular institutional slicing without intraday data: VCVd = CV of dollar volume over the prior 10 sessions (excluding the flip day) relative to 60d level;
VSK = volume-weighted share of up-closes (prior 10d); RNG = prior-10d mean daily range/ATR (quiet accumulation = small). Low VCVd + high VSK + low RNG = steady buying.
Test on all flips and on px<5; AUC BIG/HUGE train|test, add-on to OS."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression; from sklearn.metrics import roc_auc_score
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
O=pd.read_pickle(X+'basket_models.pkl'); W=pd.read_pickle(X+'wide_daily_ohlcv.pkl'); tr=O.tr
def f(v):
    v=v[~v.index.duplicated()]; dv=v.close*v.volume; r=v.close.diff(); up=(r>0)
    tr_=pd.concat([v.high-v.low,(v.high-v.close.shift()).abs(),(v.low-v.close.shift()).abs()],axis=1).max(axis=1); atr=tr_.rolling(14).mean()
    g=pd.DataFrame({'VCVd':(dv.rolling(10).std()/dv.rolling(10).mean())/(dv.rolling(60).std()/dv.rolling(60).mean()),
      'VSK':(v.volume*up).rolling(10).sum()/v.volume.rolling(10).sum(),'RNG':(tr_/atr).rolling(10).mean()}); return g.shift(1)
F={k:f(v) for k,v in W.items()}
for c in ('VCVd','VSK','RNG'): O[c]=[F[r.tk][c].asof(r.d) if r.tk in F else np.nan for r in O.itertuples()]
for lab,M in (('ALL flips',O.px>=0),('px<5',O.px<5)):
    print(lab,int(M.sum()))
    for tgt in ('big','huge'):
        for c in ('VCVd','VSK','RNG'):
            x=O[c].fillna(O[c].median()); a=[roc_auc_score(O[tgt][m&M],x[m&M]) for m in (tr,~tr)]; print(f'  {tgt:4s} {c:5s} AUC {a[0]:.3f} | {a[1]:.3f}{"  <== same side" if (a[0]-.5)*(a[1]-.5)>0 and min(abs(a[0]-.5),abs(a[1]-.5))>.025 else ""}')
def fit(cols,tgt,M):
    A=O[cols].fillna(O[cols].median()); A=(A-A[tr].mean())/A[tr].std(); l=LogisticRegression(max_iter=500).fit(A[tr&M],O.loc[tr&M,tgt]); p=l.predict_proba(A)[:,1]; return roc_auc_score(O[tgt][~tr&M],p[~tr&M])
for lab,M in (('ALL',O.px>=0),('px<5',O.px<5)):
    for tgt in ('big','huge'): print(f'add-on {lab} {tgt}: OS {fit(["OS"],tgt,M):.3f}  +VCVd {fit(["OS","VCVd"],tgt,M):.3f}  +VSK {fit(["OS","VSK"],tgt,M):.3f}  +RNG {fit(["OS","RNG"],tgt,M):.3f}  +all3 {fit(["OS","VCVd","VSK","RNG"],tgt,M):.3f}')
