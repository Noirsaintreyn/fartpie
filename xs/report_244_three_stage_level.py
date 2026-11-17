"""Report 244: THREE-STAGE LEVEL REACTION, a side model for the OTHER days (Rey Nov 17: not the main alpha; stage 1 a level with a reason, stage 2 a reason to reject, stage 3 absorption to confirm; gate it if anything). DECLARED before results.
STAGE 1 LEVEL: the report-243 menu (POC, VAH, VAL, HVN, LVN; fib 38.2/50/61.8; prior-day H/L; overnight H/L; call / put wall) with confluence = number of OTHER families within 0.1 x ATR14. Resistance = level above the prior close, support = below. NOT tied to the vector regime.
STAGE 2 REJECTION: a 5m bar after 10:00 (before 15:00) whose high reaches the resistance (low reaches the support) while the prior bar closed on the near side, and which CLOSES back on the near side with a >= 50% wick through the level. Fade it at the next open; stop = the bar's extreme + 1 pt (R >= 10 pts), target 1R / 2R, flat 15:55, cost 1 pt, one trade at a time, first touch per level per day-side.
STAGE 3 ABSORPTION: (a) the cumulative estimated delta of the 3 bars into the level is in the push direction (aggressors lifting into resistance / hitting into support) AND (b) the touch bar's volume >= 1.3 x the mean of the prior 12 bars (effort without result).
FUNNEL (each stage cumulative): S2 = rejection only at conf >= 2 levels; S2+S3 = + absorption; then S2+S3 at conf 0 levels (does the level matter?), at wall / fib levels only, and GATED by state known at the open: positive gamma, chop (Kaufman ER bottom third of 2019-21), IVHI.
Read per era (n, win, net, day-clustered t) and BY YEAR, because the point is the days the vector algo does not pay (2019, 2021, 2023, 2024). Train 2019-21 | test 2018+2022-26; about 20 cells -> ~3 is the bar."""
import sys; src=open('/home/claude/fartpie/xs/report_243_level_selection.py').read(); exec(src[:src.index("C=[]")])
rv=pd.Series(v).rolling(12).mean().shift(1).values
tr_er=pd.Series({d:ERD[d] for d in ERD if pd.Timestamp('2019-01-01')<=d<pd.Timestamp('2022-01-01') and np.isfinite(ERD[d])}); q1=tr_er.quantile(1/3); print(f'ER chop cut {q1:.2f}')
C=[]
for d in pd.unique(day):
    td=pd.Timestamp(d)
    if td<pd.Timestamp('2018-01-01') or td not in ATRD or not np.isfinite(ATRD[td]): continue
    L_=levels_for(td)
    if L_ is None: continue
    tol=.1*ATRD[td]; fams=[(FAM[t],p) for t,p in L_]; conf=[len({f for f,q in fams if f!=FAM[t] and abs(q-p)<=tol}) for t,p in L_]
    ix=dayix[td]; f=prior(td); a=G.iloc[G.index.searchsorted(td)-1] if G.index.searchsorted(td)>0 else np.nan; seen=set()
    for j in ix:
        if m[j]<600 or m[j]>900 or j+1>=n or j<14: continue
        rng_=max(h[j]-l[j],1e-9); e=j+1
        for q,(t,p) in enumerate(L_):
            if (q,1) in seen or (q,-1) in seen: continue
            if c[j-1]<p<=h[j] and c[j]<p and (h[j]-max(o[j],c[j]))/rng_>=.5: s=-1; ext=h[j]; wick=True      # resistance rejected
            elif c[j-1]>p>=l[j] and c[j]>p and (min(o[j],c[j])-l[j])/rng_>=.5: s=1; ext=l[j]; wick=True    # support rejected
            else: continue
            R=abs(o[e]-ext)+1
            if R<10: continue
            seen.add((q,s)); dl3=cd[j-1]-cd[j-4]; ab=bool(np.sign(dl3)==-s) and bool(np.isfinite(rv[j]) and v[j]>=1.3*rv[j])
            C.append(dict(e=e,s=s,R=R,T=t,F=FAM[t],conf=conf[q],day=td,ab=ab,ad=bool(np.sign(dl3)==-s),av=bool(np.isfinite(rv[j]) and v[j]>=1.3*rv[j]),pos=bool(np.isfinite(a) and a>=0),chop=bool(np.isfinite(ERD.get(td,np.nan)) and ERD[td]<q1),iv=bool(f is not None and np.isfinite(f.ivp) and f.ivp>=.5)))
C=pd.DataFrame(C); print('rejections',len(C),'| absorbed',int(C.ab.sum()),'| conf>=2',int((C.conf>=2).sum()))
def run(cs,k):
    out=[]; last=-1
    for _,r in cs.sort_values('e').iterrows():
        e=int(r.e)
        if e<=last or e>=n-2 or not (575<=m[e]<=930): continue
        gp,mfe,j=trade(e,int(r.s),r.R,'fix',k); out.append(dict(day=r.day,pts=gp-1.0)); last=j
    return pd.DataFrame(out)
def cell(nm,D):
    y=D.day.dt.year; res=[]
    for era,mk in (('train',(y>=2019)&(y<=2021)),('test',(y==2018)|(y>=2022))):
        x=D[mk]; res.append(f'{era}: n{len(x):4d} win{np.mean(x.pts>0)*100:3.0f}% net {x.pts.mean():+6.1f} t{tstat(x):+5.1f}' if len(x)>5 else f'{era}: n{len(x)}')
    print(f'{nm:34s}',' || '.join(res),flush=True)
def byyear(nm,D): D=D.assign(y=D.day.dt.year); print(f'   {nm} by year:',' '.join(f'{y}:{g.pts.mean():+.0f}({len(g)})' for y,g in D.groupby('y')))
for k in (1.0,2.0):
    print(f'\n=== target {k}R ===')
    S2=C[C.conf>=2]; S3=S2[S2.ab]
    cell('S2 rejection @ conf>=2',run(S2,k)); cell('S2+S3 absorption @ conf>=2',run(S3,k))
    cell('S2+S3 @ conf 0 (placebo level)',run(C[(C.conf==0)&C.ab],k)); cell('S2 @ conf 0',run(C[C.conf==0],k))
    cell('S2+S3 @ wall/fib levels',run(C[(C.F.isin(['OPT','FIB']))&C.ab],k)); cell('S2+S3 @ conf>=2 & wall/fib',run(C[(C.F.isin(['OPT','FIB']))&C.ab&(C.conf>=2)],k))
    cell('S2+S3 @ conf>=2 & pos gamma',run(S3[S3.pos],k)); cell('S2+S3 @ conf>=2 & chop',run(S3[S3.chop],k)); cell('S2+S3 @ conf>=2 & IVHI',run(S3[S3.iv],k)); cell('S2+S3 @ conf>=2 & pos & chop',run(S3[S3.pos&S3.chop],k))
    cell('S2+S3 all levels (any conf)',run(C[C.ab],k))
    byyear('S2+S3 @ conf>=2',run(S3,k)); byyear('S2+S3 all levels',run(C[C.ab],k))

print('\n=== LAZY VERSION (Rey: fib + options walls as the reason), stage 3 decomposed ===')
W=C[C.F.isin(['OPT','FIB'])]
for k in (1.0,2.0):
    print(f'-- target {k}R')
    cell('fib/wall rejection only',run(W,k)); cell('+ delta into level (a)',run(W[W.ad],k)); cell('+ volume effort (b)',run(W[W.av],k)); cell('+ both (a)&(b) = stage 3',run(W[W.ab],k))
    cell('fib only & stage 3',run(W[(W.F=='FIB')&W.ab],k)); cell('walls only & stage 3',run(W[(W.F=='OPT')&W.ab],k)); cell('fib&wall confluence & stage 3',run(W[(W.conf>=1)&W.ab&(W['T'].isin(['F382','F50','F618','CW','PW']))],k))
    byyear('fib/wall stage 3',run(W[W.ab],k))
