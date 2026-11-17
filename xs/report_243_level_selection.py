"""Report 243: WHICH LEVEL IS WORTH TRADING TODAY - confluence ranking of candidate levels (Rey Nov 17: use options flow + fib + order flow to decide which level to trade: POC one day, VAL the next, a ledge, an HVN, an LVN). DECLARED before results.
Candidate levels each day (all known at the open, from the prior sessions): VP family = POC, VAH, VAL, HVN (up to 3 local maxima of the smoothed 2-pt profile >= 1.25 x median, not within 10 pts of POC), LVN (up to 3 local minima <= 0.6 x median); FIB family = 38.2 / 50 / 61.8 of the prior RTH range; EXT family = prior-day high/low, overnight (prior 18:00 -> 09:25) high/low ('ledges'); OPT family = call wall and put wall (QQQ dealer gamma walls mapped to NQ, prior close).
CONFLUENCE of a level = number of OTHER families that have a level within 0.1 x ATR14 (14-day mean RTH range, ~25 pts) of it (0-3). Entry/exit exactly as report 242 (pullback to the level inside the 1H vector regime, stop = 1H vector level, target 1R / 2R, one trade at a time, 1 pt cost, first touch per level per day). Order-flow ABS = the 6 bars before the touch have delta against the trade.
Cells: by confluence 0 / 1 / 2+ (pooled); by family and by type (HVN, LVN, PDH/PDL, ONH/ONL, walls separately); confluence >= 2 & ABS; TOP-1 per day and side = the touch with the highest (confluence + ABS) score. About 25 cells -> ~3 is the bar; the question that matters is whether confluence rescues TRAIN 2019-21 (242: pooled -4 train, +8 test, trend-year effect). Train 2019-21 | test 2018+2022-26."""
import sys; src=open('/home/claude/fartpie/xs/report_242_level_pullbacks.py').read(); exec(src[:src.index("# daily levels")])
rth=(m>=570)&(m<=955); tpx=(h+l)/2; nbar=n
LV={}
for d,ix in pd.Series(np.arange(n)[rth]).groupby(day[rth]):
    ix=ix.values; H=h[ix].max(); L=l[ix].min(); px=np.round(tpx[ix]/2)*2; vol=pd.Series(v[ix]).groupby(px).sum().sort_index(); idx_=vol.index.values; pocp=vol.idxmax(); tot=vol.sum(); lo=hi_=int(np.where(idx_==pocp)[0][0]); acc=vol.iloc[lo]
    while acc<.7*tot and (lo>0 or hi_<len(idx_)-1):
        up=vol.iloc[hi_+1] if hi_<len(idx_)-1 else -1; dn=vol.iloc[lo-1] if lo>0 else -1
        if up>=dn: hi_+=1; acc+=up
        else: lo-=1; acc+=dn
    full=vol.reindex(np.arange(idx_.min(),idx_.max()+2,2)).fillna(0); sm=full.rolling(5,center=True,min_periods=1).mean(); med=sm[sm>0].median(); a=sm.values; ps=sm.index.values
    hv=[(a[i],ps[i]) for i in range(2,len(a)-2) if a[i]>=a[i-1] and a[i]>=a[i+1] and a[i]>=a[i-2] and a[i]>=a[i+2] and a[i]>=1.25*med and abs(ps[i]-pocp)>10]; hv=[p for _,p in sorted(hv,reverse=True)[:3]]
    lv=[(a[i],ps[i]) for i in range(2,len(a)-2) if a[i]<=a[i-1] and a[i]<=a[i+1] and a[i]<=a[i-2] and a[i]<=a[i+2] and 0<a[i]<=.6*med]; lv=[p for _,p in sorted(lv)[:3]]
    LV[pd.Timestamp(d)]=dict(POC=pocp,VAH=idx_[hi_],VAL=idx_[lo],HVN=hv,LVN=lv,F382=L+.382*(H-L),F50=L+.5*(H-L),F618=L+.618*(H-L),H=H,L=L,rng=H-L,cl=c[ix[-1]])
days=sorted(LV); nxt={days[i]:days[i+1] for i in range(len(days)-1)}
rr=pd.Series({d:LV[d]['rng'] for d in days}); atr14=rr.rolling(14).mean(); ATRD={nxt[d]:atr14[d] for d in nxt}
cl_=pd.Series({d:LV[d]['cl'] for d in days}); er_=((cl_-cl_.shift(10)).abs()/cl_.diff().abs().rolling(10).sum()); ERD={nxt[d]:er_[d] for d in nxt}
dayix={}
for i in range(n): dayix.setdefault(pd.Timestamp(day[i]),[]).append(i)
FAM={'POC':'VP','VAH':'VP','VAL':'VP','HVN':'VP','LVN':'VP','F382':'FIB','F50':'FIB','F618':'FIB','PDH':'EXT','PDL':'EXT','ONH':'EXT','ONL':'EXT','CW':'OPT','PW':'OPT'}
def levels_for(td):
    p=LV[days[days.index(td)-1]] if td in days and days.index(td)>0 else None
    pr=[d for d in days if d<td]
    if not pr: return None
    p=LV[pr[-1]]; L_=[('POC',p['POC']),('VAH',p['VAH']),('VAL',p['VAL']),('F382',p['F382']),('F50',p['F50']),('F618',p['F618']),('PDH',p['H']),('PDL',p['L'])]+[('HVN',x) for x in p['HVN']]+[('LVN',x) for x in p['LVN']]
    pd_=pr[-1]; ix_=[i for i in dayix.get(pd_,[]) if False]; on=[i for i in dayix.get(pd_,[]) if m[i]>=1080]+[i for i in dayix.get(td,[]) if m[i]<570]
    if len(on)>20: L_+=[('ONH',h[on].max()),('ONL',l[on].min())]
    f=prior(td)
    if f is not None:
        if np.isfinite(f.call_wall_nq) and abs(f.call_wall_nq-p['cl'])<3*p['rng']: L_.append(('CW',float(f.call_wall_nq)))
        if np.isfinite(f.put_wall_nq) and abs(f.put_wall_nq-p['cl'])<3*p['rng']: L_.append(('PW',float(f.put_wall_nq)))
    return L_
C=[]
for d in pd.unique(day):
    td=pd.Timestamp(d)
    if td<pd.Timestamp('2018-01-01') or td not in ATRD or not np.isfinite(ATRD[td]): continue
    L_=levels_for(td)
    if L_ is None: continue
    tol=.1*ATRD[td]; fams=[(FAM[t],p) for t,p in L_]
    conf=[len({f for f,q in fams if f!=FAM[t] and abs(q-p)<=tol}) for t,p in L_]
    seen=set(); ix=dayix[td]; f=prior(td); a=G.iloc[G.index.searchsorted(td)-1] if G.index.searchsorted(td)>0 else np.nan
    for j in ix:
        if m[j]<600 or m[j]>900 or j+1>=n: continue
        k=hi1[j]
        if k<0 or not np.isfinite(lev[k]) or reg[k]==0: continue
        s=int(reg[k]); e=j+1
        for q,(t,p) in enumerate(L_):
            if (q,s) in seen: continue
            if s==1 and l[j]<=p<c[j] and p>lev[k]: ok=True
            elif s==-1 and h[j]>=p>c[j] and p<lev[k]: ok=True
            else: ok=False
            if not ok: continue
            R=abs(o[e]-lev[k])
            if R<15: continue
            seen.add((q,s)); pre=cd[j-1]-cd[j-7]
            C.append(dict(e=e,s=s,R=R,T=t,F=FAM[t],conf=conf[q],day=td,absb=bool(np.sign(pre)==-s),neg=bool(np.isfinite(a) and a<0)))
C=pd.DataFrame(C); C['score']=C.conf+C.absb.astype(int); print('touch candidates',len(C),C.groupby('T').size().to_dict(),'| confluence counts',C.conf.value_counts().sort_index().to_dict())
def run(cs,k):
    out=[]; last=-1
    for _,r in cs.sort_values('e').iterrows():
        e=int(r.e)
        if e<=last or e>=n-2: continue
        gp,mfe,j=trade(e,int(r.s),r.R,'fix',k); out.append(dict(day=r.day,pts=gp-1.0)); last=j
    return pd.DataFrame(out)
def cell(nm,D):
    y=D.day.dt.year; res=[]
    for era,mk in (('train',(y>=2019)&(y<=2021)),('test',(y==2018)|(y>=2022))):
        x=D[mk]; res.append(f'{era}: n{len(x):4d} win{np.mean(x.pts>0)*100:3.0f}% net {x.pts.mean():+6.1f} t{tstat(x):+5.1f}' if len(x)>5 else f'{era}: n{len(x)}')
    print(f'{nm:30s}',' || '.join(res),flush=True)
TOP=C.sort_values(['score','e'],ascending=[False,True]).drop_duplicates(['day','s'])
for k in (1.0,2.0):
    print(f'\n=== target {k}R, stop = 1H vector level ===')
    for nm,mk in (('conf 0',C.conf==0),('conf 1',C.conf==1),('conf 2+',C.conf>=2),('conf 2+ & ABS',(C.conf>=2)&C.absb),('conf 1+ & ABS',(C.conf>=1)&C.absb),('conf 0 & ABS',(C.conf==0)&C.absb)): cell(nm,run(C[mk],k))
    for F_ in ('VP','FIB','EXT','OPT'): cell(f'family {F_}',run(C[C.F==F_],k))
    for T in ('HVN','LVN','PDH','PDL','ONH','ONL','CW','PW','POC','VAH','VAL'): cell(f'type {T}',run(C[C['T']==T],k))
    cell('TOP-1 per day/side by score',run(TOP,k)); cell('TOP-1 with score >= 2',run(TOP[TOP.score>=2],k)); cell('pooled first touch/day/side',run(C.sort_values('e').drop_duplicates(['day','s']),k))
print('\n=== CAUSAL CHECK: TOP-1 uses the best score of the WHOLE day (hindsight: a better touch may come later). Causal version = first touch of the day/side whose score reaches the threshold ===')
Cs=C.sort_values('e')
for k in (1.0,2.0):
    for thr_nm,mk in (('first touch score >= 1',Cs.score>=1),('first touch score >= 2',Cs.score>=2),('first touch score >= 3',Cs.score>=3),('first touch conf >= 2',Cs.conf>=2),('first touch conf >= 3',Cs.conf>=3)):
        cell(f'{k}R {thr_nm}',run(Cs[mk].drop_duplicates(['day','s']),k))
# how much of TOP-1's edge is hindsight: share of TOP-1 trades that are NOT the first touch of the day/side
first=Cs.drop_duplicates(['day','s']); print('TOP-1 trades that are not the first touch of the day/side:',round(1-TOP.merge(first[['day','s','e']],on=['day','s'],suffixes=('','_f')).eval('e==e_f').mean(),2))
