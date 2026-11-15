"""report_135 (Nov 15 2026): TWO PHYSICS LEVERS. (1) SIZING: with tail index alpha ~1-2 the payoff has (near) infinite variance, so equal-dollar slots hand risk to whichever name has the biggest tail. Sizing rules (weights normalised on the TRAIN selection): w0 equal; wV vol-target (1/ATRP, risk-equal diffusion); wP conviction (M1 P(+100%)); wVP both; all clipped [0.33,3]. Metrics: book total (% of start capital), ex-top-3 trades, maxDD, monthly Sharpe, and the geometric-growth-optimal leverage scan G(s)=mean log(1+s*r_m) (a sizing rule is better if its optimum growth is higher).
(2) REGIME from physics, known at the START of each month, using only flips already EXITED: (a) rolling 12-month Hill tail index of the excursion L=log(1+PEAK) over u = train 70th pct (px<5 flips); (b) discrete-time Hawkes (Poisson, lambda_t = mu + a*sum_k e^{-b k} N_{t-k}) fitted on the trailing 252 days of daily FLIP COUNTS of the whole universe: branching ratio n = a/(e^b-1) and current excitation x_t/mu. Declared [-] for all three (hot tails / clustering = crowded flip waves = worse or thinner returns); tested as month-level Spearman of regime variable vs that month's mean net / BIG rate of px<5 top-20% flips (train months vs test months), then as a GATE (size x0.5 or skip in the bad tercile, cut from train)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from scipy.stats import spearmanr
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
O=pd.read_pickle(X+'combined_scores.pkl'); S=pd.read_pickle(X+'physics_scores.pkl'); P=pd.read_pickle(X+'physics_flips.pkl')[['tk','d','PEAK','DUR']]
A=O.merge(S,on=['tk','d']).merge(P,on=['tk','d']); A=A[A.d>='2019-01-01'].copy(); A['n']=A.r-.003; A['exit']=[d+pd.offsets.BDay(int(k)) for d,k in zip(A.d,A.DUR.clip(lower=1))]; A['L']=np.log1p(A.PEAK)
A['mo']=A.d.dt.to_period('M'); A=A.sort_values('d').reset_index(drop=True)
U=A[A.px<5]; cut=U[~U.test].M1P.quantile(.8); sel=U[(U.M1P>=cut)].copy(); tr_sel=sel[~sel.test]
cl=lambda x:np.clip(x,.33,3.)
sel['wV']=cl(tr_sel.ATRP.median()/sel.ATRP); sel['wP']=cl(sel.M1P/tr_sel.M1P.mean()); 
for w in ('wV','wP'): sel[w]=sel[w]/sel[~sel.test][w].mean()
sel['wVP']=cl(sel.wV*sel.wP); sel['wVP']=sel.wVP/sel[~sel.test].wVP.mean(); sel['w0']=1.
def book(df,wcol,N=20,f=.05,skip=None):
    z=df.sort_values(['d','M1P'],ascending=[True,False]); open_=[]; rows=[]
    for r in z.itertuples():
        open_=[e for e in open_ if e>r.d]
        w=getattr(r,wcol)*(0 if (skip is not None and skip.get(r.mo,False)) else 1)
        if len(open_)<N and w>0: open_.append(r.exit); rows.append((r.exit,f*w*r.n))
    p=pd.DataFrame(rows,columns=['exit','p']).sort_values('exit'); return p
def stats(p):
    eq=p.p.cumsum(); dd=(eq.cummax()-eq).max(); m=p.groupby(p.exit.dt.to_period('M')).p.sum(); m=m.reindex(pd.period_range(m.index.min(),m.index.max(),freq='M'),fill_value=0)
    sh=m.mean()/m.std()*np.sqrt(12) if m.std()>0 else np.nan; t=p.p.sort_values(ascending=False); ex3=p.p.sum()-t.iloc[:3].sum()
    best=max(((np.log1p(s*m).mean(),s) for s in np.arange(.1,6.01,.1) if (1+s*m).min()>0),default=(np.nan,np.nan)); return p.p.sum(),ex3,dd,sh,best[0],best[1],len(p)
print('PART 1 SIZING (px<5, top-20% M1P, N=20, test 2022+; total as % of START capital)')
te=sel[sel.test]
for w,lab in (('w0','equal'),('wV','vol-target 1/ATRP'),('wP','conviction P(+100%)'),('wVP','vol x conviction')):
    tot,ex3,dd,sh,g,s,n=stats(book(te,w)); print(f'  {lab:22s} total {tot*100:+6.0f}% ex-top3 {ex3*100:+5.0f}% maxDD {dd*100:4.0f}pts monthly Sharpe {sh:4.2f} | optimal-growth leverage x{s:.1f} (max log-growth/month {g*100:5.2f}%) n{n}')
print('  (same on TRAIN selection for reference)')
tr=sel[~sel.test]
for w,lab in (('w0','equal'),('wV','vol-target 1/ATRP'),('wP','conviction P(+100%)'),('wVP','vol x conviction')):
    tot,ex3,dd,sh,g,s,n=stats(book(tr,w)); print(f'  {lab:22s} total {tot*100:+6.0f}% ex-top3 {ex3*100:+5.0f}% maxDD {dd*100:4.0f}pts monthly Sharpe {sh:4.2f} | opt lev x{s:.1f} G {g*100:5.2f}%/mo n{n}')
# ---------- PART 2: regime variables at month start
months=pd.period_range('2019-07','2026-02',freq='M'); u=np.quantile(U[~U.test].L,.7); rows=[]
FL=A.groupby('d').size(); idx=pd.bdate_range(A.d.min(),A.d.max()); N=FL.reindex(idx).fillna(0).values
def hawkes_fit(y):
    T=len(y)
    def nll(th):
        mu,a,b=np.exp(th); x=np.zeros(T); 
        for t in range(1,T): x[t]=np.exp(-b)*(x[t-1]+y[t-1])
        lam=mu+a*x; return float(np.sum(lam-y*np.log(lam+1e-9)))
    best=None
    for th0 in ([np.log(y.mean()*.5+.1),np.log(.1),np.log(.3)],[np.log(y.mean()*.8+.1),np.log(.02),np.log(1.)]):
        r=minimize(nll,th0,method='Nelder-Mead',options=dict(maxiter=300,xatol=1e-3,fatol=1e-3)); 
        if best is None or r.fun<best.fun: best=r
    mu,a,b=np.exp(best.x); x=0.; 
    for t in range(T): x=np.exp(-b)*(x+y[t-1] if t>0 else 0)
    return mu,a,b,a/(np.exp(b)-1),(mu+a*x)/mu
for m in months:
    ms=m.to_timestamp(); mat=U[(U.exit<ms)&(U.d>=ms-pd.DateOffset(months=12))]; ex=mat.L[mat.L>u]
    alpha=1/np.mean(np.log(ex/u)) if len(ex)>=40 else np.nan
    pos=idx.searchsorted(ms); y=N[max(0,pos-252):pos]
    if len(y)>=200: mu,a,b,br,heat=hawkes_fit(y)
    else: br=heat=np.nan
    cur=sel[sel.mo==m]; rows.append(dict(mo=m,alpha=alpha,branch=br,heat=heat,n=len(cur),mean=cur.n.mean() if len(cur) else np.nan,big=np.mean(cur.n>=.3) if len(cur) else np.nan,test=ms>=pd.Timestamp('2022-01-01')))
R=pd.DataFrame(rows); R.to_pickle(X+'regime_monthly_physics.pkl')
print('\nPART 2 REGIME (month-level; mean net and BIG rate of selected px<5 flips that month; Spearman with regime var at month start)')
for v,lab,sg in (('alpha','Hill tail index (rolling 12m)','[-]'),('branch','Hawkes branching ratio','[-]'),('heat','Hawkes excitation lambda/mu','[-]')):
    out=[]
    for era,e in (('TR',~R.test),('TE',R.test)):
        z=R[e].dropna(subset=[v,'mean']); z=z[z.n>=3]; out.append(f'{era}: n_mo {len(z)} rho(mean) {spearmanr(z[v],z["mean"])[0]:+.2f} rho(big) {spearmanr(z[v],z.big)[0]:+.2f}')
    print(f'  {lab:32s} {sg} '+' | '.join(out))
print(R[['alpha','branch','heat']].describe().round(3).T.to_string())
# gate: bad tercile (declared direction [-] => high value bad) cut from train months
for v in ('alpha','branch','heat'):
    q=R[~R.test][v].quantile(2/3); bad=dict(zip(R.mo,R[v]>=q))
    for mode,wk in (('skip',0.),):
        for wcol in ('w0','wVP'):
            pg=book(te,wcol,skip=bad); pb=book(te,wcol); a=stats(pg); b=stats(pb)
            print(f'  GATE {v:7s} (skip when >= train 67th pct {q:.2f}) sizing {wcol:4s}: gated total {a[0]*100:+5.0f}% ex3 {a[1]*100:+4.0f}% maxDD {a[2]*100:3.0f} Sharpe {a[3]:.2f} | ungated {b[0]*100:+5.0f}% ex3 {b[1]*100:+4.0f}% maxDD {b[2]*100:3.0f} Sharpe {b[3]:.2f}')
