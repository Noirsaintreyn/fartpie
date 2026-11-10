"""Nov 10 2026: SETUP SCORECARD - log every feature of the market at 10:00, reference each to its historical hit rate, add the evidence into ONE probability, size by it. DECLARED BEFORE RESULTS.
Target: S6 day (non-overnight legs) pts > 0 [y_pos] and top-third day [y_top]. Features known at 10:00: gap/|r20|, gap x dir30, first-30m range/r20, efficiency, volume ratio, open location vs prior VA, balance (ov5), VXN pct, QQQ 20d z, NQ 20d-range pct.
Two combiners: (1) NAIVE BAYES scorecard = prior log-odds + sum over features of ln[P(bin|win)/P(bin|loss)] on tercile bins (transparent, double-counts correlated features), (2) L2 LOGISTIC (weights shrink correlated features).
Walk-forward: refit each January on all prior data (2016+), predict that year. Report OOS AUC, calibration by predicted-probability quintile (hit rate and pts/day, halves), then sizing tiers from trailing quantiles: Sharpe/pts both periods. PASS: top-minus-bottom quintile same sign |t|>=2 both periods AND sizing helps both."""
import numpy as np, pandas as pd, warnings, yfinance as yf; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import roc_auc_score
src=open('idea1_daytype.py').read(); exec(src[:src.index("X=['f_gap'")])
Y2=yf.download(['QQQ','^VXN'],start='2015-01-01',interval='1d',progress=False,auto_adjust=True)['Close'].ffill(); Y2.index=pd.to_datetime(Y2.index).normalize()
def prior(s_,days): return s_.reindex(pd.DatetimeIndex(days)-pd.Timedelta(days=1),method='ffill').values
vx=Y2['^VXN']; qq=Y2['QQQ']; F['f_vxn']=prior(vx.rolling(252,min_periods=120).rank(pct=True),F.index); F['f_qz']=prior((qq-qq.rolling(20).mean())/qq.rolling(20).std(),F.index)
F['f_r20p']=pd.Series(F.rng.values,index=F.index).rolling(20).mean().rolling(252,min_periods=120).rank(pct=True).shift(1).values
X=['f_gap','f_gapdir','f_r30','f_eff','f_v30','f_ol','f_ov','f_ov1','f_vxn','f_qz','f_r20p']
y=S6.set_index('day').w.reindex(F.index); F['pts']=y; F=F.dropna(subset=X+['pts']); F=F[F.index>='2016-06-01']
F['y_pos']=(F.pts>0).astype(int); F['y_top']=(F.pts>=F.pts.expanding(100).quantile(.67).shift(1)).astype(int); F=F.dropna(subset=['y_top'])
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=15 and len(b)>=15 else np.nan
# --- naive Bayes on tercile bins
def nb_fit(tr,tgt):
    cuts={c:np.nanquantile(tr[c],[1/3,2/3]) for c in X}; L={}; p1=tr[tgt].mean()
    for c in X:
        b=np.digitize(tr[c],cuts[c]); L[c]=np.array([np.log(((tr[tgt][b==k].sum()+1)/(tr[tgt].sum()+3))/(((1-tr[tgt])[b==k].sum()+1)/((1-tr[tgt]).sum()+3))) for k in range(3)])
    return cuts,L,np.log(p1/(1-p1))
def nb_pred(m,te):
    cuts,L,prior0=m; s=np.full(len(te),prior0)
    for c in X: s=s+L[c][np.digitize(te[c],cuts[c])]
    return 1/(1+np.exp(-s)),s
res={}
for nm in ('NB','LOGIT'):
    for tgt in ('y_pos','y_top'):
        pr=pd.Series(np.nan,index=F.index)
        for yr in range(2018,2027):
            tr=F[F.index<f'{yr}-01-01']; te=F[(F.index>=f'{yr}-01-01')&(F.index<f'{yr+1}-01-01')]
            if len(te)==0: continue
            if nm=='NB': pr[te.index]=nb_pred(nb_fit(tr,tgt),te)[0]
            else:
                ss=StandardScaler().fit(tr[X]); m=LogisticRegression(C=0.05,max_iter=500).fit(ss.transform(tr[X]),tr[tgt]); pr[te.index]=m.predict_proba(ss.transform(te[X]))[:,1]
        res[nm+tgt]=pr
        oos=F.join(pr.rename('p')).dropna(subset=['p']); a=[roc_auc_score(oos[tgt][m],oos.p[m]) for m in (oos.index.year<2022,oos.index.year>=2022)]
        print(f'\n##### {nm} target {tgt}: OOS AUC: 2018-21 {a[0]:.3f} | 2022-26 {a[1]:.3f} | base hit rate {oos[tgt].mean():.0%}')
        print('  calibration by predicted-probability quintile (cuts from each period): mean predicted p | actual hit | S6 pts/day | halves')
        for lab,m in (('18-21',oos.index.year<2022),('22-26',oos.index.year>=2022)):
            x=oos[m]; q=pd.qcut(x.p,5,labels=False)
            print('   '+lab+': '+' | '.join(f'Q{k+1} p{x.p[q==k].mean():.2f} hit{x[tgt][q==k].mean():.2f} {x.pts[q==k].mean():+5.1f}' for k in range(5))+f' | Q5-Q1 pts t {tt(x.pts[q==4],x.pts[q==0]):+.2f}')
        # sizing: trailing 252-day quantiles of p
        lo=pr.rolling(252,min_periods=100).quantile(.2).shift(1); hi=pr.rolling(252,min_periods=100).quantile(.8).shift(1)
        S=lambda x: x.mean()/x.std()*np.sqrt(252)
        for up,dn in ((1.25,.75),(1.5,.5),(1.0,.5),(1.5,1.0)):
            mult=pd.Series(1.0,index=pr.index); mult[pr>=hi]=up; mult[pr<=lo]=dn; o=[]
            for lab,mm in (('18-21',(pr.index.year>=2018)&(pr.index.year<2022)),('22-26',pr.index.year>=2022)):
                b=F.pts[mm]; s2=b*mult[mm]; o.append(f'{lab}: Sharpe {S(b):.2f}->{S(s2):.2f} pts {b.sum():+.0f}->{s2.sum():+.0f}')
            print(f'   sizing top20% x{up}/bottom20% x{dn}: '+' | '.join(o))
# --- "log of today's market": present the scorecard for the latest available features (yesterday close) and the learned evidence per feature
tr=F; m=nb_fit(tr,'y_pos'); print('\n##### Scorecard evidence per feature (full history; ln-LR per tercile bin: low/mid/high). Positive = favors an up day (S6day>0)')
for c in X: print(f'   {c:9} cuts {m[0][c][0]:+.2f}/{m[0][c][1]:+.2f}  evidence low {m[1][c][0]:+.2f} mid {m[1][c][1]:+.2f} high {m[1][c][2]:+.2f}')
