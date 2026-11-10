"""Nov 10 2026 IDEA 6: day-type Markov. DECLARED BEFORE RESULTS.
Day type from the finished session: range ratio (range/r20 >= 1) x efficiency (|c-o|/range >= .5): TREND-WIDE, TREND-NARROW, ROTATION-WIDE, ROTATION-NARROW (4 states).
P(type_t | type_t-1, and 2-day history) estimated walk-forward (expanding, before each year). Tests: (a) does yesterday's type predict today's efficiency / NZ pts/day (group means, both periods);
(b) 1-step chain log-score vs unconditional base rate OOS. PASS: group means differ with same sign and |t|>=2 in both periods."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src=open('idea1_daytype.py').read(); exec(src[:src.index("X=['f_gap'")])
F['wide']=(F.rng/F.r20>=1.0); F['eff']=((F.c-F.o).abs()/F.rng>=0.5)
F['typ']=np.where(F.eff,np.where(F.wide,'TW','TN'),np.where(F.wide,'RW','RN')); F=F.dropna(subset=['r20'])
F['prev']=F.typ.shift(1); F['prev2']=F.typ.shift(2)
NZd2=NZd.set_index('day').w.reindex(F.index).fillna(0)
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=10 and len(b)>=10 else np.nan
print('unconditional type shares:',F.typ.value_counts(normalize=True).round(2).to_dict())
print('transition P(next|prev) full-sample:'); print(pd.crosstab(F.prev,F.typ,normalize='index').round(2).to_string())
for lab,m in (('2018-21',(F.index>='2018-02-01')&(F.index.year<2022)),('2022-26',F.index.year>=2022)):
    x=F[m]; p=NZd2[m]
    print(f'\n{lab}: NZ pts/day by PREVIOUS day type:',{k:(round(p[x.prev==k].mean(),1),int((x.prev==k).sum())) for k in ('TW','TN','RW','RN')})
    print(f'   P(today trend-efficient) by prev type:',{k:round(x.eff[x.prev==k].mean(),2) for k in ('TW','TN','RW','RN')},' wide:',{k:round(x.wide[x.prev==k].mean(),2) for k in ('TW','TN','RW','RN')})
    print(f'   after RN (quiet rotation) vs others NZ t {tt(p[x.prev=="RN"],p[x.prev!="RN"]):+.2f} | after TW vs others t {tt(p[x.prev=="TW"],p[x.prev!="TW"]):+.2f}')
ll=[];lb=[]
for yr in range(2018,2027):
    tr=F[F.index<f'{yr}-01-01']; te=F[(F.index>=f'{yr}-01-01')&(F.index<f'{yr+1}-01-01')].dropna(subset=['prev'])
    T=pd.crosstab(tr.prev,tr.typ,normalize='index'); B=tr.typ.value_counts(normalize=True)
    for r in te.itertuples(): ll.append(np.log(max(T.loc[r.prev,r.typ],1e-3)) if r.prev in T.index and r.typ in T.columns else np.nan); lb.append(np.log(max(B.get(r.typ,1e-3),1e-3)))
ll=np.array(ll); lb=np.array(lb); print(f'\nOOS mean log-score 1-step chain {np.nanmean(ll):.4f} vs unconditional {np.nanmean(lb):.4f}')
