"""Report 213: tests from the outside critique Rey attached. DECLARED before results.
(1) TARGET: 'capturable move' label instead of the tail label. From the entry (next open after the daily flip), on daily highs/lows within 40 trading days: L30 = +30% reached before -15% (a bar touching both counts as the stop; no resolution = 0); LATR = +3 ATR14 before -1.5 ATR14 (volatility-scaled, same rules). Both fixed now, no tuning. Higher base rate than BIG (>= +50% realized) = more power.
(2) FEATURES, signal-day close, bars up to the signal only: ASY = log( up-day return per dollar traded / down-day return per dollar traded ) over the last 20 days (OHLCV proxy for asymmetric price response; not true order flow); dASY = ASY(last 20d) - ASY(days -80..-20) (is buying becoming more effective vs the name's own history); RETN = share of the largest 3-day advance (>= 2 ATR) of the last 20 days still held at the signal (impulse retention). Declared sign + for all three.
Not re-run: 'failed bearish pressure' (features of the preceding bear state) = reports 123/124 (short bear leg, shallow depth, whipsaws): all failed.
Tests on the report-203 sub-$5 flips (train 2019-21 | test 2022-Feb26 | late 2026): raw AUC for each label; incremental value over controls (ATRP, EXT, R20, log dollar volume): logistic fit on train, test AUC with vs without the feature; matched cells (train terciles of ATRP x R20): above- vs below-cell-median feature, hit-rate difference with z. PASS = same sign train and test, test |z| >= 2 in the matched comparison, and a test AUC gain >= +0.01 over controls. 3 features x 2 labels = 6 tests (Bonferroni |z| >= 2.64). Also: do OS2 and BASE predict the capturable labels?"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; E=SP+'ext/'
Z=pd.read_pickle(SP+'selectors203.pkl'); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); rows=[]
def label(o,h,l,e,pe,up,dn,H=40):
    n=len(o)
    for j in range(e,min(e+H,n)):
        if l[j]<=dn: return 0.
        if h[j]>=up: return 1.
    return 0. if e+H<=n else np.nan
for tk,g in Z.groupby('tk'):
    B=D[tk]; B=B[~B.index.duplicated()]; idx=B.index; o,h,l,c,v=[B[k].values.astype(float) for k in ('open','high','low','close','volume')]; pc=np.r_[c[0],c[:-1]]; tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); r=c/pc-1; dv=c*v
    for ix,d,src in zip(g.index,g.d,g.src):
        i=idx.searchsorted(d)
        if src=='late': i=idx.searchsorted(d)-1 if (i>=len(idx) or idx[i]!=d) else i
        if i<85 or i+1>=len(idx): continue
        e=i+1; pe=o[e]; atr=tr[i-13:i+1].mean()
        def asy(a,b):
            rr=r[a:b+1]; dd=dv[a:b+1]; u=rr>0; dn=rr<0
            if u.sum()<4 or dn.sum()<4 or dd[u].sum()<=0 or dd[dn].sum()<=0: return np.nan
            return np.log((rr[u].sum()/dd[u].sum())/(abs(rr[dn]).sum()/dd[dn].sum()))
        A=asy(i-19,i); Bq=asy(i-79,i-20)
        best=None
        for s in range(i-19,i-4):
            gsz=c[s+3]-c[s]
            if gsz>=2*tr[s-13:s+1].mean() and (best is None or gsz>best[0]): best=(gsz,s)
        retn=np.clip((c[i]-c[best[1]])/best[0],-1,2) if best else np.nan
        rows.append(dict(ix=ix,L30=label(o,h,l,e,pe,pe*1.3,pe*.85),LATR=label(o,h,l,e,pe,pe+3*atr,pe-1.5*atr),ASY=A,dASY=A-Bq if A==A and Bq==Bq else np.nan,RETN=retn,R20=c[i]/c[i-20]-1,DVL=np.log(dv[i-19:i+1].mean()+1)))
X=Z.join(pd.DataFrame(rows).set_index('ix'),how='inner'); print('flips',len(X),X.groupby('era').size().to_dict())
print('\nbase rates (train | test | late26) and mean realized r when label=1 vs 0:')
for L in ('L30','LATR'):
    print(f'  {L}: '+' | '.join(f'{X[X.era==e][L].mean()*100:4.1f}% (r {X[(X.era==e)&(X[L]==1)].r.mean()*100:5.1f}% vs {X[(X.era==e)&(X[L]==0)].r.mean()*100:5.1f}%, n{X[X.era==e][L].notna().sum()})' for e in ('train','test','late26')), '| BIG base', ' | '.join(f'{X[X.era==e].big.mean()*100:4.1f}%' for e in ('train','test','late26')))
CT=['ATRP','EXT','R20','DVL']
def auc(s,f,L):
    s=s.dropna(subset=[f,L]); 
    return roc_auc_score(s[L],s[f]) if s[L].nunique()>1 and len(s)>30 else np.nan
print('\nraw AUC (train | test | late26)')
for f in ('ASY','dASY','RETN','OS2','BASE','ATRP','R20'):
    print(f'  {f:5s} '+'  '.join(f'{L}: '+' | '.join(f'{auc(X[X.era==e],f,L):.3f}' for e in ('train','test','late26')) for L in ('L30','LATR')), f' n(test) {X[X.era=="test"][f].notna().sum()}')
print('\nincremental over controls (logistic on train ranks -> test AUC) and matched cells (train terciles ATRP x R20; above vs below cell-median feature, hit rate diff)')
tr=X[X.era=='train']; qa=tr.ATRP.quantile([1/3,2/3]).values; qr=tr.R20.quantile([1/3,2/3]).values
for f in ('ASY','dASY','RETN'):
    for L in ('L30','LATR'):
        s=X.dropna(subset=[f,L]+CT).copy(); a=s[s.era=='train']; b=s[s.era=='test']
        rk=lambda col,ref,x: np.searchsorted(np.sort(ref[col].values),x[col].values)/len(ref)
        def fit(cols):
            m=LogisticRegression(C=1.0,max_iter=500).fit(np.column_stack([rk(k,a,a) for k in cols]),a[L]); return roc_auc_score(b[L],m.predict_proba(np.column_stack([rk(k,a,b) for k in cols]))[:,1]), m.coef_[0][-1]
        a0,_=fit(CT); a1,co=fit(CT+[f]); out=[]
        for e in ('train','test'):
            q=s[s.era==e].copy(); q['cell']=np.digitize(q.ATRP,qa)*3+np.digitize(q.R20,qr); q['hi']=q[f]>q.groupby('cell')[f].transform('median'); p1,p0=q[q.hi][L].mean(),q[~q.hi][L].mean(); n1,n0=q.hi.sum(),(~q.hi).sum(); p=q[L].mean(); z=(p1-p0)/np.sqrt(p*(1-p)*(1/n1+1/n0)); out.append(f'{e} {p1*100:4.1f}% vs {p0*100:4.1f}% z{z:+.1f} (n{len(q)})')
        print(f'  {f:5s} {L:4s}: test AUC controls {a0:.3f} -> +feature {a1:.3f} (coef {co:+.2f}) | matched: '+' ; '.join(out))
X.to_pickle(SP+'capturable213.pkl')
