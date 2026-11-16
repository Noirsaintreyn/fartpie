"""Report 214: audits from the second outside critique + a placebo for the vector itself. DECLARED before results.
(A) NOMINAL-PRICE AUDIT. Yahoo prices are split-adjusted; nominal price on the signal date = adjusted price x product of the split ratios dated after it. Cohorts on ALL daily flips (outlier_features.pkl, 30,080): adj<5 & nominal<5 (stay), adj>=5 & nominal<5 (newly eligible), adj<5 & nominal>=5 (drop out): n, mean r, median, win, BIG, by era (train 2019-21 | test 2022+). Returns stay on the adjusted series. No direction assumed.
(B) SURVIVORSHIP: share of the 1,656 names whose data ends before Oct 2026.
(C) COMPLETE WINDOWS: L30 / LATR recomputed only for flips with a full 40-trading-day window after entry (unresolved-in-progress flips dropped rather than counted); outcome split target / stop / timeout.
(D) ASY decomposition (above vs below the matched-cell median, report 213 cells): target-first, stop-first, timeout shares, actual-exit mean and median r; full-window flips only.
(E) PLACEBO for the vector (Rey: 'the vector has so much edge'): each sub-$5 flip is matched with 5 random signal days of the SAME ticker within +-250 trading days (>= 10 days from the flip, adjusted close < $5, full window). Same entry (next open), same labels, plus the 40-day open-to-open return. If flips do not beat matched random days, the edge is the universe's tail, not the flip. Also split the random days by the vector state on that day (bull / bear) = the value of the state classifier. z-test on rates, Welch t on returns winsorized at the 99th pct."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from vector_pine import run_state_machine
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; E=SP+'ext/'; rng=np.random.default_rng(7)
Of=pd.read_pickle(E+'outlier_features.pkl'); S=pickle.load(open(E+'splits_wide.pkl','rb')); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl')
def fac(tk,d):
    s=S.get(tk)
    if s is None or not len(s): return 1.0
    ix=s.index.tz_localize(None) if s.index.tz is not None else s.index; return float(np.prod(s.values[ix>d]))
Of['nom']=Of.px*[fac(t,d) for t,d in zip(Of.tk,Of.d)]; Of['era']=np.where(Of.d<'2022-01-01','train','test'); Of['big']=Of.r>=.5
print('(A) nominal-price audit, all daily flips',len(Of),'| flips with a later split:',int((Of.nom!=Of.px).sum()))
for nm,m in (('stay (adj<5 & nominal<5)',(Of.px<5)&(Of.nom<5)),('NEWLY ELIGIBLE (adj>=5, nominal<5)',(Of.px>=5)&(Of.nom<5)),('DROP OUT (adj<5, nominal>=5)',(Of.px<5)&(Of.nom>=5)),('corrected universe (nominal<5)',Of.nom<5),('old universe (adj<5)',Of.px<5)):
    print(f'  {nm:36s}'+' | '.join(f'{e}: n{len(s):5d} mean {s.r.mean()*100:6.1f}% med {s.r.median()*100:6.1f}% win {np.mean(s.r>0)*100:3.0f}% BIG {s.big.mean()*100:4.1f}%' if len(s)>=5 else f'{e}: n{len(s)}' for e in ('train','test') for s in [Of[m&(Of.era==e)]]))
last=pd.Series({t:b.index[-1] for t,b in D.items()}); print('\n(B) names whose data ends before 2026-10:',int((last<'2026-10-01').sum()),'of',len(last),'-> the universe holds survivors only')
X=pd.read_pickle(SP+'capturable213.pkl')
def lab3(o,h,l,e,up,dn,H=40):
    if e+H>len(o): return np.nan
    for j in range(e,e+H):
        if l[j]<=dn: return 0
        if h[j]>=up: return 1
    return 2
REG={}; rows=[]; plc=[]
for tk,g in X.groupby('tk'):
    B=D[tk]; B=B[~B.index.duplicated()]; idx=B.index; o,h,l,c=[B[k].values.astype(float) for k in ('open','high','low','close')]; pc=np.r_[c[0],c[:-1]]; tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); reg=np.asarray(run_state_machine(B.assign(datetime=B.index))[0]); n=len(c)
    for ix,d,src,era in zip(g.index,g.d,g.src,g.era):
        i=idx.searchsorted(d)
        if i>=n or idx[i]!=d: i=idx.searchsorted(d)-1
        e=i+1
        if e>=n: continue
        atr=tr[i-13:i+1].mean(); pe=o[e]; a=lab3(o,h,l,e,pe*1.3,pe*.85); b=lab3(o,h,l,e,pe+3*atr,pe-1.5*atr); f40=o[e+40]/pe-1 if e+40<n else np.nan
        rows.append(dict(ix=ix,T30=a,TATR=b,F40=f40))
        cand=[k for k in range(max(90,i-250),min(n-42,i+250)) if abs(k-i)>=10 and c[k]<5]
        if len(cand)>=5 and a==a:
            for k in rng.choice(cand,5,replace=False):
                e2=k+1; at=tr[k-13:k+1].mean(); p2=o[e2]; plc.append(dict(era=era,T30=lab3(o,h,l,e2,p2*1.3,p2*.85),TATR=lab3(o,h,l,e2,p2+3*at,p2-1.5*at),F40=o[e2+40]/p2-1,state=reg[k]))
Y=X.join(pd.DataFrame(rows).set_index('ix'),how='inner'); F=Y.dropna(subset=['T30']); PL=pd.DataFrame(plc)
print('\n(C) full 40-day windows only: flips',F.groupby('era').size().to_dict(),'(dropped as incomplete:',int(Y.T30.isna().sum()),')')
for L in ('T30','TATR'):
    print(f'  {L}: target / stop / timeout  '+' | '.join(f'{e}: {np.mean(s[L]==1)*100:4.1f} / {np.mean(s[L]==0)*100:4.1f} / {np.mean(s[L]==2)*100:4.1f}%' for e in ('train','test','late26') for s in [F[F.era==e]]))
print('\n(D) ASY above vs below matched-cell median (cells = train terciles ATRP x R20), full windows')
tr_=F[F.era=='train']; qa=tr_.ATRP.quantile([1/3,2/3]).values; qr=tr_.R20.quantile([1/3,2/3]).values
for e in ('train','test','late26'):
    q=F[(F.era==e)&F.ASY.notna()].copy(); q['cell']=np.digitize(q.ATRP,qa)*3+np.digitize(q.R20,qr); q['hi']=q.ASY>q.groupby('cell').ASY.transform('median')
    for nm,s in (('high ASY',q[q.hi]),('low ASY ',q[~q.hi])):
        print(f'  {e:6s} {nm} n{len(s):4d} | +30/-15: target {np.mean(s.T30==1)*100:4.1f}% stop {np.mean(s.T30==0)*100:4.1f}% timeout {np.mean(s.T30==2)*100:4.1f}% | ATR label: target {np.mean(s.TATR==1)*100:4.1f}% stop {np.mean(s.TATR==0)*100:4.1f}% | actual exit mean {s.r.mean()*100:6.1f}% x3 {s.r.sort_values().iloc[:-3].mean()*100:6.1f}% med {s.r.median()*100:6.1f}%')
    a,b=q[q.hi].r.values,q[~q.hi].r.values; w=lambda x:np.minimum(x,np.quantile(np.r_[a,b],.99)); print(f'         actual-exit return, high minus low: t {stats.ttest_ind(w(a),w(b),equal_var=False).statistic:+.1f}')
print('\n(E) placebo: flips vs matched random days of the same names')
def zt(p1,n1,p0,n0): p=(p1*n1+p0*n0)/(n1+n0); return (p1-p0)/np.sqrt(p*(1-p)*(1/n1+1/n0))
for e in ('train','test','late26'):
    f=F[F.era==e]; p=PL[PL.era==e]
    if not len(p): continue
    cap=np.nanquantile(np.r_[f.F40.dropna().values,p.F40.values],.99); w=lambda x:np.minimum(x[~np.isnan(x)],cap)
    print(f'  {e:6s} flips n{len(f):5d}: +30/-15 target {np.mean(f.T30==1)*100:4.1f}% stop {np.mean(f.T30==0)*100:4.1f}% | ATR target {np.mean(f.TATR==1)*100:4.1f}% | 40d return mean {np.nanmean(f.F40)*100:5.1f}% med {np.nanmedian(f.F40)*100:5.1f}%')
    for nm,s in (('random days (all)',p),('random, vector BULL',p[p.state==1]),('random, vector BEAR',p[p.state==-1])):
        print(f'         {nm:20s} n{len(s):5d}: +30/-15 target {np.mean(s.T30==1)*100:4.1f}% stop {np.mean(s.T30==0)*100:4.1f}% | ATR target {np.mean(s.TATR==1)*100:4.1f}% | 40d return mean {s.F40.mean()*100:5.1f}% med {s.F40.median()*100:5.1f}% | flips minus this: z(+30) {zt(np.mean(f.T30==1),len(f),np.mean(s.T30==1),len(s)):+.1f} z(ATR) {zt(np.mean(f.TATR==1),len(f),np.mean(s.TATR==1),len(s)):+.1f} t(40d) {stats.ttest_ind(w(f.F40.values),w(s.F40.values),equal_var=False).statistic:+.1f}')
