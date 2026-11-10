"""Nov 10 2026: IV SQUEEZE TEST on noise trades. DECLARED BEFORE RESULTS. Features at prior close: ivr (252d rank of QQQ ATM IV), iv5 (5d change in ATM IV), ivrv (ATM IV minus 20d realized), rc (NQ 5d/20d mean range), squeeze = ivr<.2 & rc<median, release = iv5 top tercile. Per-day pts of noise trades by tercile/flag, 2018-21 vs 2022-26; PASS = same sign & |t|>=2 in both for top-minus-bottom."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
exec(open(f'{SP}/ctxhead.py').read())
iv=G1.atm_iv; ivr=iv.rolling(252,min_periods=120).rank(pct=True)
rv20=np.log(Dn.DC.astype(float)).diff().rolling(20).std()*np.sqrt(252)
F=pd.DataFrame(index=ix)
sh=lambda s: s.shift(1).reindex(ix,method='ffill')
F['ivr']=sh(ivr).values; F['iv5']=sh(iv.diff(5)).values; F['ivrv']=(sh(iv)-rv20.shift(1)).values
F['rc']=(R.rolling(5).mean()/R.rolling(20).mean()).shift(1).values
F['sq']=((F.ivr<.2)&(F.rc<F.rc.median())).astype(float).where(F.ivr.notna())
G=pd.read_pickle(f'{SP}/G_noise.pkl'); G['day']=pd.to_datetime(G.day).dt.normalize(); G=G[G.day>='2018-02-01']
days=pd.DatetimeIndex(sorted(G.day.unique())); P=G.groupby('day').pts.sum().reindex(days).fillna(0)
def t(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>8 and len(b)>8 else np.nan
for nm in ['ivr','iv5','ivrv','rc']:
    f=F[nm].reindex(days); out=[]
    for lab,m in (('18-21',days.year<2022),('22-26',days.year>=2022)):
        x=f[m]; p=P[m]; lo=x<=x.quantile(1/3); hi=x>=x.quantile(2/3)
        out.append(f'{lab}: low {p[lo].mean():+5.1f} mid {p[~lo&~hi].mean():+5.1f} high {p[hi].mean():+5.1f} (hi-lo t {t(p[hi],p[lo]):+.2f})')
    print(f'{nm:5}', ' | '.join(out))
s=F.sq.reindex(days); out=[]
for lab,m in (('18-21',days.year<2022),('22-26',days.year>=2022)):
    p=P[m]; a=s[m]==1; out.append(f'{lab}: squeeze {p[a].mean():+5.1f} (n{a.sum()}) other {p[~a].mean():+5.1f} (t {t(p[a],p[~a]):+.2f})')
print('squeeze', ' | '.join(out))

# ---- S6 (V2) legs: sz-weighted pts on squeeze vs other days
import pickle
O,N=pickle.load(open(f'{SP}/s6.pkl','rb')); legs=O['V2']
sqd=set(F.index[F.sq==1])
print('\nS6 legs, sz-weighted pts/trade on squeeze days vs other (18-21 | 22-26)')
tot={}
for nm,(df,_,_) in zip(N,legs):
    df=df.copy(); df['day']=pd.to_datetime(df.day).dt.normalize(); df['w']=df.pts*df.sz; df['sq']=df.day.isin(sqd)
    o=[]
    for lab,m in (('18-21',df.day.dt.year<2022),('22-26',df.day.dt.year>=2022)):
        d=df[m]; o.append(f'{lab}: sq {d[d.sq].w.mean():+6.1f} (n{d.sq.sum()}) other {d[~d.sq].w.mean():+6.1f}')
    print(f'  {nm:6}',' | '.join(o))
