"""Nov 13 2026: WHY does the chain do well against overnight inventory in era B (+14.2, n=50) but not A (+1.4, n=44)? Rey's call: not a skip, look for a mechanism. Descriptive splits only on the H2 trades; no new filter is adopted without both eras agreeing. Splits: year, inventory side, ON range/ADR, gap, sweep level, hour, chain count/day, ON-high/low taken before the entry."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/report_19_amt_papers.py').read().split("def sim(")[0].split('"""',2)[2])
C=pd.read_pickle(SC+'icc_T.pkl'); dl=[d for d in days if d in RTH]; ctx={}
for i,D in enumerate(dl):
    if i<21 or (D-dl[i-1]).days>5: continue
    P=dl[i-1]; settle=RTH[P].close.values[-1]; on=by[P]; on=on[on.m>=1080]; on=pd.concat([on,by[D][by[D].m<570]])
    if len(on)<60: continue
    adr=np.mean([RTH[d].high.max()-RTH[d].low.min() for d in dl[i-20:i]])
    inv=float((on.close.values>settle).mean()); o=RTH[D].open.values[0]
    ctx[D]=dict(inv=inv,onr=(on.high.max()-on.low.min())/adr,gap=(o-settle)/adr,onret=(on.close.values[-1]-settle)/adr)
rows=[]
for r in C.itertuples():
    if r.d in ctx: x=ctx[r.d]; rows.append(dict(d=r.d,dr=r.dr,pts=r.pts,m=r.m,era=r.era,yr=r.yr,lvl=r.lvl,kz=r.kz,risk=r.risk,**x))
T=pd.DataFrame(rows); T['side']=np.where(T.inv>=.9,1,np.where(T.inv<=.1,-1,0)); T['ctr']=(T.side!=0)&(T.dr==-T.side); T['wth']=(T.side!=0)&(T.dr==T.side); T['mid']=T.side==0
def st(x): return f'n={len(x):3d} mean {x.pts.mean():+6.1f} win {np.mean(x.pts>0):.0%}' if len(x) else 'n=0'
print('chain by relation to inventory, era A | era B')
for k in ['ctr','wth','mid']: print(f'  {k}: A {st(T[(T[k])&(T.era=="A")])} | B {st(T[(T[k])&(T.era=="B")])}')
c=T[T.ctr]; print('\nCOUNTER by year:'); [print(f'  {y}: {st(g)}') for y,g in c.groupby('yr')]
print('\nCOUNTER by chain direction (long vs short):'); [print(f'  dr={d:+d}: A {st(g[g.era=="A"])} | B {st(g[g.era=="B"])}') for d,g in c.groupby('dr')]
print('\nCOUNTER by ON range/ADR (tercile on all days):'); q=T.onr.quantile([1/3,2/3]).values
c=c.assign(onb=np.digitize(c.onr,q)); [print(f'  ON range tercile {b}: A {st(g[g.era=="A"])} | B {st(g[g.era=="B"])}') for b,g in c.groupby('onb')]
print('\nCOUNTER by |ON return|/ADR (size of the overnight push) <0.15 / 0.15-0.4 / >0.4:')
c=c.assign(ob=np.digitize(c.onret.abs(),[.15,.4])); [print(f'  {b}: A {st(g[g.era=="A"])} | B {st(g[g.era=="B"])}') for b,g in c.groupby('ob')]
print('\nCOUNTER by entry hour:'); c=c.assign(h=(c.m//60)); [print(f'  {h}h: A {st(g[g.era=="A"])} | B {st(g[g.era=="B"])}') for h,g in c.groupby('h')]
print('\nCOUNTER by swept level:'); [print(f'  {l}: A {st(g[g.era=="A"])} | B {st(g[g.era=="B"])}') for l,g in c.groupby('lvl')]
print('\nShare of days with strong inventory (>=.9 or <=.1) by year, and counter-auction rate (day closes against inventory):')
rr=[]
for D,x in ctx.items():
    if D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30'): continue
    s=1 if x['inv']>=.9 else (-1 if x['inv']<=.1 else 0)
    if s==0: rr.append((D.year,0,np.nan)); continue
    g=RTH[D]; rr.append((D.year,1,float((g.close.values[-1]-g.open.values[0])*s<0)))
R=pd.DataFrame(rr,columns=['y','strong','ctr']); print(R.groupby('y').agg(strong=('strong','mean'),counter_rate=('ctr','mean')).round(2).T.to_string())

print('\n=== REGIME SWITCH (hypothesis from the above: inventory relation flips sign between regimes, so follow the TRAILING counter-auction rate; no lookahead, prior days only) ===')
print('rule: trailing rate r = share of the last N strong-inventory days that closed against inventory (open->close). r>0.5 -> take chain trades AGAINST inventory; r<0.5 -> take chain trades WITH inventory; strong-inv days only. N in {20,40,60} all shown, none picked.')
sd=sorted([(D,s,float((RTH[D].close.values[-1]-RTH[D].open.values[0])*s<0)) for D,x in ctx.items() if D in RTH for s in [1 if x['inv']>=.9 else (-1 if x['inv']<=.1 else 0)] if s!=0])
sdD=[a for a,_,_ in sd]; sdv=np.array([c for _,_,c in sd])
def tr(D,N):
    i=np.searchsorted(sdD,D)  # strictly prior days
    return sdv[max(0,i-N):i].mean() if i>=N//2 else np.nan
for N in (20,40,60):
    T['r']=[tr(d,N) for d in T.d]; s=T[(T.side!=0)&T.r.notna()].copy()
    s['tk']=np.where(s.r>.5,s.ctr,s.wth); k=s[s.tk]
    print(f'N={N}: switch  A {st(k[k.era=="A"])} | B {st(k[k.era=="B"])}   || all strong-inv chain (no switch) A {st(s[s.era=="A"])} | B {st(s[s.era=="B"])}')
    y=k.groupby('yr').apply(lambda g: f'{g.pts.mean():+.1f}(n{len(g)})'); print('     by year:',' '.join(f'{a}:{b}' for a,b in y.items()))

print('\n=== t-stats and saved filtered sets ===')
def tt(x): return x.pts.mean()/(x.pts.std()/len(x)**.5)
S=T[T.side!=0]
for e in 'AB': print(f'strong-inventory chain (either relation) era {e}: n={len(S[S.era==e])} mean {S[S.era==e].pts.mean():+.2f} t {tt(S[S.era==e]):+.2f} | balanced-inventory: n={len(T[(T.side==0)&(T.era==e)])} mean {T[(T.side==0)&(T.era==e)].pts.mean():+.2f} t {tt(T[(T.side==0)&(T.era==e)]):+.2f}')
T['r']=[tr(d,60) for d in T.d]; s=T[(T.side!=0)&T.r.notna()].copy(); s['tk']=np.where(s.r>.5,s.ctr,s.wth); K=s[s.tk]
for e in 'AB': print(f'switch N=60 era {e}: n={len(K[K.era==e])} mean {K[K.era==e].pts.mean():+.2f} t {tt(K[K.era==e]):+.2f}')
C2=C.set_index(['d','dr','m'])
for nm,X in (('strong',S),('switch60',K)):
    idx=set(zip(X.d,X.dr,X.m)); C[[ (a,b,c) in idx for a,b,c in zip(C.d,C.dr,C.m)]].to_pickle(SC+f'icc_{nm}.pkl')
