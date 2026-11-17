"""Report 247: explicit level 'stories' (pairs of level groups at the same price) with order flow, on the report-246 candidates (Rey Nov 17: call wall at the 1-day HVN, fib at the 5-day POC; the story, not single levels). DECLARED before results. Candidates = the 10,250 stage-2 rejections of report 246 (cand246.pkl); outcome 2R after 1 pt cost. For every pair of the 10 level groups (45 pairs) present together at the touched level: n, net pts and t per era for ALL such rejections and for the absorbed subset (stage 3); then L1 logistic with all group flags + the 45 pair products + order flow, trained 2019-21, tested 2018+2022-26 and expanding walk-forward. Read the pair table as exploratory (45 pairs x 2 sets x 2 eras): only same-sign-both-eras cells with n >= 25 train are listed; bar ~3."""
import sys, warnings; warnings.filterwarnings('ignore'); src=open('/home/claude/fartpie/xs/report_232_vx_exits.py').read(); exec(src[:src.index("MODES=")])
import itertools
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
C=pd.read_pickle(SP+'cand246.pkl'); GR=['VPd','VP3','VPw','VPm','EXT','SWG','FIB','OPT','POOR','EQ']
def tstat(x):
    d=x.groupby('day').pts.sum(); return d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>5 else np.nan
out=[]
for i_,r in C.iterrows():
    e=int(r.e)
    if e>=n-2 or not (575<=m[e]<=930): out.append(np.nan); continue
    gp,mfe,j=trade(e,int(r.s),r.R,'fix',2.0); out.append(gp-1.0)
C['pts']=out; C=C.dropna(subset=['pts']).reset_index(drop=True); C['win']=(C.pts>0).astype(int); C['y']=C.day.dt.year
tr=(C.y>=2019)&(C.y<=2021); te=(C.y==2018)|(C.y>=2022)
rows=[]
for A,B in itertools.combinations(GR,2):
    for nm,ex in (('all',None),('absorbed',C.ab)):
        mk=C[A]&C[B]
        if ex is not None: mk=mk&ex
        a_=C[mk&tr]; b_=C[mk&te]
        if len(a_)>=25 and len(b_)>=25: rows.append((A,B,nm,len(a_),a_.pts.mean(),tstat(a_),len(b_),b_.pts.mean(),tstat(b_)))
R=pd.DataFrame(rows,columns=['A','B','set','n_tr','net_tr','t_tr','n_te','net_te','t_te'])
print('pairs with n>=25 in both eras:',len(R))
G_=R[(R.net_tr>0)&(R.net_te>0)].sort_values('net_te',ascending=False)
print('\nSame-sign POSITIVE in both eras (train 2019-21 | test 2018+2022-26), 2R net pts (rejection candidates, not sequenced):')
for _,r in G_.head(14).iterrows(): print(f'  {r.A:5s}+{r.B:5s} {r.set:8s} train n{int(r.n_tr):4d} {r.net_tr:+6.1f} (t{r.t_tr:+4.1f}) | test n{int(r.n_te):4d} {r.net_te:+6.1f} (t{r.t_te:+4.1f})')
print(f'\npairs x sets positive in both eras: {len(G_)} of {len(R)}; negative in both: {int(((R.net_tr<0)&(R.net_te<0)).sum())}; mixed: {int(((R.net_tr<0)&(R.net_te>0)).sum()+((R.net_tr>0)&(R.net_te<0)).sum())}')
# L1 with pair products
from sklearn.linear_model import LogisticRegression
F=GR+['ng','dpush','rvr','neg','iv','chop','er','vect','hr','Rn','ab']; X=C[F].astype(float).copy(); X['rvr']=X.rvr.clip(upper=6); X['dpush']=X.dpush.clip(-1,1); X['Rn']=X.Rn.clip(upper=X.Rn.quantile(.99))
for A,B in itertools.combinations(GR,2): X[f'{A}*{B}']=(C[A]&C[B]).astype(float)
for A,B in itertools.combinations(GR,2): X[f'{A}*{B}*ab']=(C[A]&C[B]&C.ab).astype(float)
def fitpred(trm,tem,Cc=.03):
    mu=X[trm].mean(); sd=X[trm].std().replace(0,1); Z=((X-mu)/sd)
    mdl=LogisticRegression(penalty='l1',solver='liblinear',C=Cc,max_iter=500).fit(Z[trm].values,C.win[trm].values); return mdl,mdl.decision_function(Z[trm].values),mdl.decision_function(Z[tem].values)
def auc(p,y):
    r=pd.Series(p).rank().values; n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def run(cs):
    o=[]; last=-1
    for _,r in cs.sort_values('e').iterrows():
        e=int(r.e)
        if e<=last or e>=n-2 or not (575<=m[e]<=930): continue
        gp,mfe,j=trade(e,int(r.s),r.R,'fix',2.0); o.append(dict(day=r.day,pts=gp-1.0)); last=j
    return pd.DataFrame(o)
mdl,ptr,pte=fitpred(tr,te); kept=[(f,w) for f,w in zip(X.columns,mdl.coef_[0]) if abs(w)>1e-6]
print('\nL1 (pairs + flow) kept',len(kept),'of',X.shape[1],':',', '.join(f'{f} {w:+.2f}' for f,w in sorted(kept,key=lambda t:-abs(t[1]))[:10]))
print(f'AUC train {auc(ptr,C.win[tr].values):.3f} | test {auc(pte,C.win[te].values):.3f}')
for q in (.8,.9):
    thr=np.quantile(ptr,q); S=C[te].copy(); S['sc']=pte; D=run(S[S.sc>=thr]); print(f'   (a) test top {int((1-q)*100)}%: n{len(D)} ({len(D)/6.55:.0f}/yr) win{np.mean(D.pts>0)*100:.0f}% net {D.pts.mean():+.1f} t{tstat(D):+.1f}')
Sa=[]
for yv in (2018,2022,2023,2024,2025,2026):
    trm=(C.y>=2019)&(C.y<=(2021 if yv==2018 else yv-1)); tem=(C.y==yv)
    _,ptr_,pte_=fitpred(trm,tem); thr=np.quantile(ptr_,.9); S=C[tem].copy(); S['sc']=pte_; Sa.append(S[S.sc>=thr])
D=run(pd.concat(Sa)); D['y']=D.day.dt.year; print(f'   (b) walk-forward top 10%: n{len(D)} ({len(D)/6.55:.0f}/yr) win{np.mean(D.pts>0)*100:.0f}% net {D.pts.mean():+.1f} t{tstat(D):+.1f} | by year',' '.join(f'{y}:{g.pts.mean():+.0f}({len(g)})' for y,g in D.groupby('y')))
