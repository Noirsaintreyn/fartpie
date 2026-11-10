"""Nov 10 2026: SCORE-DRIVEN EXIT STYLE + WHICH VWAP BAND TO TAKE PROFIT AT. DECLARED BEFORE RESULTS.
Noise engine as before (market entry, stop 40, cost .5, 30-min decisions). At entry compute static VWAP bands: VWAP + k*sigma (long) / VWAP - k*sigma (short), sigma = volume-weighted std of typical price around VWAP, k in {1,1.5,2,2.5,3}.
TP level = the n-th band beyond the nearest band that is >= 10 pts in the profit direction (n = 0,1,2) or none (baseline exit). Fill at the band if price trades through by 1 tick (stop first if both in a bar).
Score = OOS P(big | win) from magnitude.py (walk-forward from 2020). Groups: low (bottom 40% of trailing score), mid, high (top 40%).
Question: does a different TP choice per group beat baseline? Choose the best n per group on 2020-22, then test on 2023-26 (and the reverse). PASS: chosen rule beats baseline in both directions."""
import sys; sys.path.insert(0,'/home/claude/fartpie/xs')
src=open('/home/claude/fartpie/xs/magnitude.py').read(); exec(src[:src.index("# ---- use the SIZE model")])
import nz_core as Z
from nz_core import A, OPEN, PC, sg, GRID, NB
STOP=40.0; COST=0.5
sc=pd.Series(np.nan,index=Gm.index)
for yr in range(2020,2027):
    tr=Gm[(Gm.day<pd.Timestamp(f'{yr}-01-01'))&(Gm.pts>0)]; te=Gm[(Gm.day>=pd.Timestamp(f'{yr}-01-01'))&(Gm.day<pd.Timestamp(f'{yr+1}-01-01'))]
    if len(te)<20: continue
    ss=StandardScaler().fit(tr[use]); m=LogisticRegression(C=0.05,max_iter=500).fit(ss.transform(tr[use]),tr['big']); sc[te.index]=m.predict_proba(ss.transform(te[use]))[:,1]
O=Gm.join(sc.rename('sc')).dropna(subset=['sc']); SCORE={(r.day,int(r.tod)):r.sc for r in O.itertuples()}
GRIDK=[1,1.5,2,2.5,3]
def bands(d,j,pos):                       # static bands at entry bar j using bars < j
    a=A[d][0]; h,l,c,v=a[1],a[2],a[3],np.nan_to_num(a[4]); tp=(h[:j]+l[:j]+c[:j])/3; ok=~np.isnan(tp)&(v[:j]>0)
    if ok.sum()<3: return None
    w=v[:j][ok]; x=tp[ok]; vw=(w*x).sum()/w.sum(); sd=np.sqrt(max((w*x*x).sum()/w.sum()-vw*vw,1e-9)); return [vw+pos*k*sd for k in GRIDK]
def run(d,n=None):
    a,_=A[d]; o,h,l,c,v,vw=a; s=sg(d); op=OPEN[d]; pc=PC[d] if np.isfinite(PC[d]) else op; R=[]; pos=0; pend=None; e=None; tp=None; ej=None
    for j in range(NB):
        if np.isnan(c[j]): continue
        if pend is not None:
            pos,e,ej=pend,o[j],j; pend=None; tp=None
            if n is not None:
                B=bands(d,j,pos)
                if B is not None:
                    cand=[b for b in B if (b-e)*pos>=10]
                    if len(cand)>n: tp=cand[n]
        if pos!=0:
            lo=(l[j]-e) if pos>0 else (e-h[j])
            if lo<=-STOP: R.append((d,GRID[ej-1],-STOP-COST)); pos=0; continue
            if tp is not None and ((pos>0 and h[j]>=tp+.25) or (pos<0 and l[j]<=tp-.25)): R.append((d,GRID[ej-1],(tp-e)*pos-COST)); pos=0; continue
        if j==NB-1 or np.isnan(o[j+1]):
            if j==NB-1 and pos!=0: R.append((d,GRID[ej-1],(c[j]-e)*pos-COST)); pos=0
            continue
        t=GRID[j]
        if (t+5)%30!=0 or t+5<600 or t+5>930 or not np.isfinite(s[j]): continue
        ub,lb=max(op,pc)*(1+s[j]),min(op,pc)*(1-s[j]); nx=o[j+1]
        if pos==1 and c[j]<max(ub,vw[j]): R.append((d,GRID[ej-1],(nx-e)*1-COST)); pos=0
        elif pos==-1 and c[j]>min(lb,vw[j]): R.append((d,GRID[ej-1],(e-nx)-COST)); pos=0
        if pos==0 and c[j]>ub: pend=1
        elif pos==0 and c[j]<lb: pend=-1
    return R
days=[d for d in Z.DAYS if d>=pd.Timestamp('2020-01-01')]
V={}
for nm,n in (('base',None),('n0',0),('n1',1),('n2',2)):
    V[nm]=pd.DataFrame([r for d in days for r in run(d,n)],columns=['day','tod','pts'])
    V[nm]['sc']=[SCORE.get((r.day,int(r.tod)),np.nan) for r in V[nm].itertuples()]; V[nm]=V[nm].dropna(subset=['sc'])
b=V['base']; print('baseline trades',len(b),'| check vs engine pts (mean)',round(b.pts.mean(),2),'(magnitude.py set mean',round(O[O.day>='2020-01-01'].pts.mean(),2),')')
# trailing thresholds from the score history (expanding), common to all variants
allsc=O.sort_values(['day','tod']); lo40=allsc.sc.expanding(100).quantile(.4).shift(1); hi60=allsc.sc.expanding(100).quantile(.6).shift(1)
thr=pd.DataFrame({'lo':lo40.values,'hi':hi60.values},index=pd.MultiIndex.from_frame(allsc[['day','tod']].astype({'tod':int})))
def grp(df): 
    t=thr.reindex(pd.MultiIndex.from_frame(df[['day','tod']].astype({'tod':int}))); g=np.where(df.sc.values<=t.lo.values,'low',np.where(df.sc.values>=t.hi.values,'high','mid')); g[np.isnan(t.lo.values)]='na'; return g
for k in V: V[k]['g']=grp(V[k])
S=lambda x: x.mean()/x.std()*np.sqrt(252)
print('\nmean pts per trade / total pts by group x TP choice (2020-22 | 2023-26)')
for gname in ('low','mid','high'):
    for nm in ('base','n0','n1','n2'):
        x=V[nm][V[nm].g==gname]; a=x[x.day.dt.year<2023]; bb=x[x.day.dt.year>=2023]
        print(f'  {gname:4} {nm:4} n {len(a):3}/{len(bb):3} | mean {a.pts.mean():+6.1f} / {bb.pts.mean():+6.1f} | total {a.pts.sum():+6.0f} / {bb.pts.sum():+6.0f} | win {np.mean(a.pts>0):.0%}/{np.mean(bb.pts>0):.0%}')
# rule selection: best n per group on one period, test on the other
def total(df,mk): return df[mk].pts.sum()
for train,test,lab in (((lambda d: d.day.dt.year<2023),(lambda d: d.day.dt.year>=2023),'choose on 2020-22 -> test 2023-26'),((lambda d: d.day.dt.year>=2023),(lambda d: d.day.dt.year<2023),'choose on 2023-26 -> test 2020-22')):
    pick={}
    for gname in ('low','mid','high'):
        sc_={nm:V[nm][(V[nm].g==gname)&train(V[nm])].pts.sum() for nm in V}; pick[gname]=max(sc_,key=sc_.get)
    tot_rule=sum(V[pick[g]][(V[pick[g]].g==g)&test(V[pick[g]])].pts.sum() for g in pick); tot_base=sum(V['base'][(V['base'].g==g)&test(V['base'])].pts.sum() for g in pick)
    print(f'\n{lab}: picked {pick} | test total with rule {tot_rule:+.0f} vs baseline {tot_base:+.0f}')
print('\nall-trades same TP for everyone: total pts 2020-22 | 2023-26'); [print(f'  {nm:4}',f'{V[nm][V[nm].day.dt.year<2023].pts.sum():+6.0f} | {V[nm][V[nm].day.dt.year>=2023].pts.sum():+6.0f}',f'| mean {V[nm].pts.mean():+.2f} | worst day {V[nm].groupby("day").pts.sum().min():+.0f}') for nm in V]
