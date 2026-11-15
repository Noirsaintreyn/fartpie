"""report_133 (Nov 15 2026): PHYSICS / EXTREME-VALUE MODELS of the flip payoff. Theory: a vector flip is a position with a trailing (drawdown) stop on a diffusion. For Brownian motion with drift mu, vol sigma and a drawdown stop b the running maximum at the stop is EXPONENTIAL (Taylor 1975): P(M>m)=exp(-m*2mu/(sigma^2(exp(2 mu b/sigma^2)-1))); with unobserved drift heterogeneity (some names carry information drift, most carry none) the unconditional peak is a MIXTURE of exponentials = heavy (power-law-like) tail. So: excursion L=log(1+PEAK) (max high during hold) is modelled, not the noisy net return.
M0 exponential, scale exp(t0 + ts*log ATRP + t.x)  [diffusive scaling predicts ts ~ 1]. M1 two-state mixture of exponentials, logistic mixing pi(x) ('real move' vs 'noise'), separate scale heads. M2 generalized Pareto (shape xi constant), scale exp(eta(x)) [extreme-value, heavy tail]. M3 GPD with covariate-dependent shape xi(x). Baseline M-: same-family models with constant parameters (no covariates); and the logistic ridge on BIG (report_125) for ranking.
Features x (standardized with TRAIN moments): EXT, LTURN, LMC, LPX, F11, VOLX, F2, F7, F4. Fit on 2019-21 ONLY (MLE, torch), judged on 2022+: held-out log-likelihood per flip, AUC for PEAK>=+30% / +100%, calibration by decile, top-20% net return. Also: tail-index stability (refit M2 on each era separately), scaling exponent ts by era, Hill tail index by supply bucket."""
import numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings('ignore'); torch.manual_seed(0)
from sklearn.metrics import roc_auc_score
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
O=pd.read_pickle(X+'combined_scores.pkl'); P=pd.read_pickle(X+'physics_flips.pkl')[['tk','d','PEAK','DUR']]; O=O.merge(P,on=['tk','d'],how='left'); O=O[(O.d>='2019-01-01')&O.PEAK.notna()&(O.ATRP>0)].copy()
O['L']=np.log1p(O.PEAK).clip(lower=1e-4); O['lA']=np.log(O.ATRP); feats=['EXT','LTURN','LMC','LPX','F11','VOLX','F2','F7','F4']; tr=O[~O.test]
Xm=pd.DataFrame({f:(O[f].fillna(tr[f].median())-tr[f].median())/tr[f].std() for f in feats}); Xm['lA']=(O.lA-tr.lA.mean())/tr.lA.std(); cols=['lA']+feats; Xm=Xm[cols]
T=lambda a:torch.tensor(np.asarray(a),dtype=torch.float64); XA=T(Xm.values); LA=T(O.L.values); trm=torch.tensor((~O.test).values); tem=~trm; Ltr,Xtr=LA[trm],XA[trm]
k=len(cols); sp=torch.nn.functional.softplus
def fit(nll,params,steps=400):
    opt=torch.optim.LBFGS(params,lr=.5,max_iter=steps,line_search_fn='strong_wolfe'); 
    def cl(): opt.zero_grad(); l=nll(); l.backward(); return l
    opt.step(cl); return float(nll())
def mk(n): return torch.zeros(n,dtype=torch.float64,requires_grad=True)
# ---- model definitions: return nll(X,L,params) and survival function S(m|X)
class M:
    def __init__(s,kind,cov=True): s.kind=kind; s.cov=cov; s.p={}; s.init()
    def init(s):
        z=lambda n:torch.zeros(n,dtype=torch.float64,requires_grad=True); s.p=dict(b0=z(1),w=z(k)); 
        if s.kind=='mix': s.p.update(b1=z(1),w1=z(k),g0=z(1),g=z(k)); s.p['b0'].data+=-2.; s.p['b1'].data+=-.5
        if s.kind in('gpd','gpdx'): s.p['xi']=z(1); s.p['xi'].data+=-1.; 
        if s.kind=='gpdx': s.p['xw']=z(k)
        if s.kind=='exp': s.p['b0'].data+=-2.
        if s.kind in('gpd','gpdx'): s.p['b0'].data+=-2.
        if not s.cov:
            for kk in ('w','w1','g','xw'): 
                if kk in s.p: s.p[kk]=torch.zeros_like(s.p[kk])
    def params(s): return [v for kk,v in s.p.items() if not(not s.cov and kk in('w','w1','g','xw'))]
    def eta(s,Xx,b,w): return b+(Xx@w if s.cov else 0.)
    def nll_vec(s,Xx,L):
        p=s.p
        if s.kind=='exp': e=s.eta(Xx,p['b0'],p['w']); return e+L*torch.exp(-e)
        if s.kind=='mix':
            e0=s.eta(Xx,p['b0'],p['w']); e1=s.eta(Xx,p['b1'],p['w1']); pi=torch.sigmoid(p['g0']+(Xx@p['g'] if s.cov else 0.))
            f=(1-pi)*torch.exp(-e0-L*torch.exp(-e0))+pi*torch.exp(-e1-L*torch.exp(-e1)); return -torch.log(f+1e-300)
        e=s.eta(Xx,p['b0'],p['w']); xi=sp(p['xi']+(Xx@p['xw'] if s.kind=='gpdx' else 0.))+1e-3; beta=torch.exp(e)
        return e+(1/xi+1)*torch.log1p(xi*L/beta)
    def surv(s,Xx,thr):
        p=s.p; thr=torch.as_tensor(thr,dtype=torch.float64)
        if s.kind=='exp': return torch.exp(-thr*torch.exp(-s.eta(Xx,p['b0'],p['w'])))
        if s.kind=='mix':
            e0=s.eta(Xx,p['b0'],p['w']); e1=s.eta(Xx,p['b1'],p['w1']); pi=torch.sigmoid(p['g0']+(Xx@p['g'] if s.cov else 0.)); return (1-pi)*torch.exp(-thr*torch.exp(-e0))+pi*torch.exp(-thr*torch.exp(-e1))
        e=s.eta(Xx,p['b0'],p['w']); xi=sp(p['xi']+(Xx@p['xw'] if s.kind=='gpdx' else 0.))+1e-3; return (1+xi*thr/torch.exp(e))**(-1/xi)
res={}
for nm,kind,cov in (('const exp','exp',False),('M0 exp(x)','exp',True),('const mix2','mix',False),('M1 mix2(x)','mix',True),('const GPD','gpd',False),('M2 GPD(x)','gpd',True),('M3 GPD xi(x)','gpdx',True)):
    m=M(kind,cov); f=fit(lambda:m.nll_vec(Xtr,Ltr).mean(),m.params()); 
    with torch.no_grad(): te=float(m.nll_vec(XA[tem],LA[tem]).mean()); trn=float(m.nll_vec(Xtr,Ltr).mean())
    res[nm]=(m,trn,te); print(f'{nm:14s} train NLL {trn:.4f} | TEST NLL {te:.4f}  (lower better; test-const gap shown below)')
base_te=res['const GPD'][2]; print('test NLL improvement vs const GPD (nats/flip):',{k:round(base_te-v[2],4) for k,v in res.items()})
# scaling exponent & tail index
m0=res['M0 exp(x)'][0]; print('\nM0 coefficients (standardized):',dict(zip(cols,m0.p['w'].detach().numpy().round(3))),'| ts(log ATRP) per 1 sd of logATRP =',round(float(m0.p['w'][0]),3),'; sd of log ATRP =',round(float(tr.lA.std()),3),'=> elasticity ts =',round(float(m0.p['w'][0])/float(tr.lA.std()),3),'(diffusive theory ~1)')
m2=res['M2 GPD(x)'][0]; print('M2 xi (train fit) =',round(float(sp(m2.p['xi'])+1e-3),3),' -> tail index alpha=1/xi =',round(1/float(sp(m2.p['xi'])+1e-3),2))
for era,mask in (('2019-21',~O.test.values),('2022+',O.test.values)):
    mm=M('gpd',False); Lm=LA[torch.tensor(mask)]; Xm_=XA[torch.tensor(mask)]; fit(lambda:mm.nll_vec(Xm_,Lm).mean(),mm.params()); xi=float(sp(mm.p['xi'])+1e-3); print(f'  const-GPD refit on {era}: xi {xi:.3f} (alpha {1/xi:.2f}), scale {float(torch.exp(mm.p["b0"])):.4f}')
# Hill by supply bucket (flips with L above train 70th pct)
for col,lab in (('LTURN','turnover'),('LMC','market cap'),('LPX','price')):
    out=[]
    for era,e in (('TR',~O.test),('TE',O.test)):
        z=O[e]; q1,q2=O[~O.test][col].quantile([1/3,2/3]); 
        for b,mkk in (('low',z[col]<=q1),('high',z[col]>=q2)):
            y=z[mkk].L.values; u=np.quantile(O[~O.test].L,.7); ex=y[y>u]; hill=1/np.mean(np.log(ex/u)) if len(ex)>20 else np.nan; out.append(f'{era}-{b} alpha {hill:.2f} (n{len(ex)})')
    print(f'Hill tail index by {lab}:',' | '.join(out))
# evaluation: tail-probability ranking
thr30,thr100=np.log(1.3),np.log(2.0)
print('\nRANKING (test): AUC for PEAK>=+30% / +100% and for NET >=+30% / >=+100% (net = after 30bp), by predicted tail prob; baseline = logistic ridge m_BIG_0.5')
O['n']=O.r-.003
for un,mk in (('ALL',np.ones(len(O),bool)),('px<5',(O.px<5).values)):
    tmk=O.test.values&mk; print(' ',un)
    cand={'logistic m_BIG':O['m_BIG_0.5'].values}
    for nm in ('M0 exp(x)','M1 mix2(x)','M2 GPD(x)','M3 GPD xi(x)'):
        with torch.no_grad(): cand[nm]=res[nm][0].surv(XA,thr100).numpy()
    for nm,sc in cand.items():
        z=O[tmk]; s=sc[tmk]; cut=np.quantile(sc[(~O.test.values)&mk],.8); top=z[s>=cut]
        print(f'    {nm:15s} AUC peak30 {roc_auc_score(z.PEAK>=.3,s):.3f} peak100 {roc_auc_score(z.PEAK>=1,s):.3f} | net30 {roc_auc_score(z.n>=.3,s):.3f} net100 {roc_auc_score(z.n>=1,s):.3f} | top20 n{len(top):4d} mean {top.n.mean()*100:+6.1f}% med {top.n.median()*100:+5.1f}% catch net100 {top[top.n>=1].shape[0]/max(1,(z.n>=1).sum())*100:4.1f}%')
# calibration of M1/M2 for P(peak>=+100%) by decile, test, all
for nm in ('M1 mix2(x)','M2 GPD(x)'):
    with torch.no_grad(): p=res[nm][0].surv(XA,thr100).numpy()
    z=O[O.test].copy(); z['p']=p[O.test.values]; z['dc']=pd.qcut(z.p,10,labels=False,duplicates='drop'); c=z.groupby('dc').agg(pred=('p','mean'),act=('PEAK',lambda s:np.mean(s>=1)),n=('p','size')).round(4); print(f'\ncalibration {nm}: test P(peak>=+100%) by predicted decile\n',c.T.to_string())
import pickle; pickle.dump({k:(v[1],v[2]) for k,v in res.items()},open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/physics_models_nll.pkl','wb'))

# ---------- part 2: capped-expectation ranking and slot book (px<5), models vs logistic
print('\nPART 2: ranking by predicted CAPPED expected excursion E[min(L,cap)] and slot books (px<5, test 2022+, 5% of start capital per slot, FCFS with score tie-break within a day, net of 30bp)')
cap=np.log(3.0)
def capE(m,Xx):
    p=m.p
    with torch.no_grad():
        if m.kind=='mix':
            e0=m.eta(Xx,p['b0'],p['w']); e1=m.eta(Xx,p['b1'],p['w1']); pi=torch.sigmoid(p['g0']+(Xx@p['g'])); m0=torch.exp(e0); m1=torch.exp(e1); return ((1-pi)*m0*(1-torch.exp(-cap/m0))+pi*m1*(1-torch.exp(-cap/m1))).numpy()
        if m.kind=='exp': m0=torch.exp(m.eta(Xx,p['b0'],p['w'])); return (m0*(1-torch.exp(-cap/m0))).numpy()
        e=m.eta(Xx,p['b0'],p['w']); xi=(sp(p['xi']+(Xx@p['xw'] if m.kind=='gpdx' else 0.))+1e-3); beta=torch.exp(e); 
        g=torch.linspace(0,cap,400,dtype=torch.float64); S=((1+xi[:,None]*g[None,:]/beta[:,None])**(-1/xi[:,None])); return torch.trapz(S,g,dim=1).numpy()
sc={'lr (frozen score)':O.lr.values,'logistic m_BIG':O['m_BIG_0.5'].values,'OS':O.OS.values}
for nm in ('M0 exp(x)','M1 mix2(x)','M2 GPD(x)'):
    with torch.no_grad(): sc[nm+' P(+100%)']=res[nm][0].surv(XA,thr100).numpy()
    sc[nm+' E[min(L,3x)]']=capE(res[nm][0],XA)
O['exit']=[d+pd.offsets.BDay(int(k)) for d,k in zip(O.d,O.DUR.fillna(20).clip(lower=1))]
def book(sel,score,N,f=.05):
    z=O[sel].copy(); z['s']=score[sel]; z=z.sort_values(['d','s'],ascending=[True,False]); open_=[]; pnl=[]; 
    for r in z.itertuples():
        open_=[e for e in open_ if e>r.d]
        if len(open_)<N: open_.append(r.exit); pnl.append((r.exit,f*r.n,r.d))
    pnl=pd.DataFrame(pnl,columns=['exit','p','d']).sort_values('exit'); eq=pnl.p.cumsum(); dd=(eq.cummax()-eq).max(); yr=pnl.groupby(pnl.exit.dt.year).p.sum()
    t=pnl.p.sort_values(ascending=False); return pnl.p.sum(),dd,len(pnl),(pnl.p.sum()-t.iloc[:3].sum()),yr
print('  selection = top-20% by score (cut = train 80th pct within px<5); total = % of start capital over 2022-01..2026-02')
mk=(O.px<5).values
for nm,s in sc.items():
    cut=np.quantile(s[mk&(~O.test.values)],.8); sel=mk&O.test.values&(s>=cut)
    out=[]
    for N in (10,20):
        tot,dd,n,ex3,yr=book(sel,s,N); out.append(f'N={N}: {tot*100:+6.0f}% (ex top-3 trades {ex3*100:+5.0f}%) maxDD {dd*100:4.0f}pts n{n}')
    print(f'  {nm:26s} sel n{int(sel.sum()):4d} | '+' | '.join(out))
