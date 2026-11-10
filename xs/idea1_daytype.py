"""Nov 10 2026 IDEA 1: DAY TYPE from the first 30 min (09:30-10:00), SVM walk-forward, used to SIZE the noise engine. DECLARED BEFORE RESULTS.
Features (all known at 10:00): |gap|/r20, signed gap*dir30, first-30m range/r20, first-30m efficiency |c-o|/range, first-30m volume / trailing-20 same, open location vs prior VA, ov5 (balance), 5-day mean overlap.
Targets: T_range = day range / r20 >= 1.15 ; T_trend = |close-open|/range >= 0.55. RBF SVM (C=1), expanding train from 2014-17, refit each year, score = decision_function out-of-sample.
Report: OOS AUC per target; then NZ pts/day (all noise entries are >= 10:00) by tercile of the T_range score, and sizing Sharpe x1.25/x0.75 by trailing terciles, 2018-21 vs 2022-26. PASS: top-bottom same sign, |t|>=2 both."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.svm import SVC; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import roc_auc_score
src=open('idea5_balance.py').read(); exec(src[:src.index("def rep(")])
rows=[]
for d in Z.ALLD:
    a=Z.A[d][0]; o,h,l,c,v=a[0],a[1],a[2],a[3],a[4]; ok=~np.isnan(c)
    if ok.sum()<70: continue
    rows.append((d,np.nanmax(h[ok])-np.nanmin(l[ok]),o[ok][0],c[ok][-1],np.nanmax(h[:6]),np.nanmin(l[:6]),c[5],np.nansum(v[:6])))
F=pd.DataFrame(rows,columns=['day','rng','o','c','h30','l30','c30','v30']).set_index('day')
F['r20']=F.rng.rolling(20).mean().shift(1); F['pc']=Z.PC.reindex(F.index); F['gap']=(F.o-F.pc)
F['dir30']=np.sign(F.c30-F.o); F['f_gap']=F.gap.abs()/F.r20; F['f_gapdir']=np.sign(F.gap)*F.dir30
F['f_r30']=(F.h30-F.l30)/F.r20; F['f_eff']=(F.c30-F.o).abs()/(F.h30-F.l30+1e-9); F['f_v30']=F.v30/F.v30.rolling(20).mean().shift(1)
F['f_ol']=oloc.reindex(F.index); F['f_ov']=ov5.reindex(F.index); F['f_ov1']=ov.reindex(F.index).shift(1)
X=['f_gap','f_gapdir','f_r30','f_eff','f_v30','f_ol','f_ov','f_ov1']
F['T_range']=(F.rng/F.r20>=1.15).astype(int); F['T_trend']=((F.c-F.o).abs()/F.rng>=0.55).astype(int)
F=F.dropna(subset=X+['r20']); F=F[F.index>='2014-03-01']
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=10 and len(b)>=10 else np.nan
for tgt in ('T_range','T_trend'):
    sc=pd.Series(np.nan,index=F.index)
    for yr in range(2018,2027):
        tr=F[F.index<f'{yr}-01-01']; te=F[(F.index>=f'{yr}-01-01')&(F.index<f'{yr+1}-01-01')]
        if len(te)==0: continue
        ss=StandardScaler().fit(tr[X]); m=SVC(C=1.0,kernel='rbf',gamma='scale').fit(ss.transform(tr[X]),tr[tgt]); sc[te.index]=m.decision_function(ss.transform(te[X]))
    F['s_'+tgt]=sc; oos=F.dropna(subset=['s_'+tgt]); print(f'{tgt}: base rate {F[tgt].mean():.2f} | OOS AUC 2018-21 {roc_auc_score(oos[tgt][oos.index.year<2022],oos["s_"+tgt][oos.index.year<2022]):.3f}  2022-26 {roc_auc_score(oos[tgt][oos.index.year>=2022],oos["s_"+tgt][oos.index.year>=2022]):.3f}')
NZd2=NZd.set_index('day').w; S=lambda x: x.mean()/x.std()*np.sqrt(252)
for tgt in ('T_range','T_trend'):
    s=F['s_'+tgt].dropna(); o=[]
    for lab,m in (('18-21',s.index.year<2022),('22-26',s.index.year>=2022)):
        x=s[m]; q=np.quantile(x,[1/3,2/3]); p=NZd2.reindex(x.index).fillna(0); lo=p[x<=q[0]]; hi=p[x>=q[1]]
        o.append(f'{lab}: NZ pts/day low {lo.mean():+5.1f} high {hi.mean():+5.1f} t {tt(hi,lo):+.2f}')
    print(f' {tgt} score terciles:',' | '.join(o))
    lo_=s.rolling(252,min_periods=100).quantile(1/3).shift(1); hi_=s.rolling(252,min_periods=100).quantile(2/3).shift(1)
    for up,dn in ((1.25,.75),(1.5,.5)):
        m=pd.Series(1.0,index=s.index); m[s>=hi_]=up; m[s<=lo_]=dn; o=[]
        for lab,mm in (('18-21',s.index.year<2022),('22-26',s.index.year>=2022)):
            b=NZd2.reindex(s.index).fillna(0)[mm]; o.append(f'{lab}: Sharpe {S(b):.2f} -> {S(b*m[mm]):.2f} pts {b.sum():+.0f} -> {(b*m[mm]).sum():+.0f}')
        print(f'   size x{up}/x{dn}:',' | '.join(o))
