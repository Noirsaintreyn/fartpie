"""Report 204: physics add-on to report 203. Critical-slowing-down early-warning signals before a regime shift (Scheffer et al.): rising lag-1 autocorrelation and rising variance in the window before the flip. Features at the signal-day close from daily log returns: AR1 = lag-1 autocorr over 40d; VARR = variance(last 20d)/variance(prior 60d); SKEWN = skew of last 40d. Declared sign: higher AR1 and higher VARR -> BIG more likely (system near a transition). Same universe/eras as report 203 (selectors203.pkl); AUC for BIG; train 2019-21 | test 2022-Feb26 | late 2026."""
import pandas as pd, numpy as np
from scipy import stats
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; Z=pd.read_pickle(X+'selectors203.pkl'); D=pd.read_pickle(X+'ext/wide_daily_ohlcv.pkl'); rows=[]
for tk,g in Z.groupby('tk'):
    B=D[tk]; B=B[~B.index.duplicated()]; lr=np.log(B.close).diff(); idx=B.index
    for ix,d in zip(g.index,g.d):
        i=idx.searchsorted(d)
        if i<110 or idx[i]!=d: continue
        w=lr.values[i-39:i+1]; a=w[1:]; b=w[:-1]; ar=np.corrcoef(a,b)[0,1]; v20=np.var(lr.values[i-19:i+1]); v60=np.var(lr.values[i-79:i-19]); rows.append((ix,ar,v20/(v60+1e-12),stats.skew(w)))
F=pd.DataFrame(rows,columns=['ix','AR1','VARR','SKEWN']).set_index('ix'); W=Z.join(F,how='inner'); print('n',len(W))
for f in ('AR1','VARR','SKEWN'):
    out=[]
    for e in ('train','test','late26'):
        s=W[W.era==e]; a=s[s.big][f]; b=s[~s.big][f]; out.append(f'{stats.mannwhitneyu(a,b).statistic/(len(a)*len(b)):.3f}(nBIG{len(a)})')
    print(f'  {f:6s}',' | '.join(out))
