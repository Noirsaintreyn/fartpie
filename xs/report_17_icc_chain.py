"""Nov 13 2026: ICT/ICC CHAIN on NQ 5m as a STANDALONE model (not a filter on vector trades). DECLARED BEFORE RESULTS.
Chain, in order (liquidity -> structure -> pullback entry): 1) SWEEP: an RTH 5m bar (09:35-14:00) wicks through a liquidity level (PDH, PDL, ONH, ONL, Asia H/L 18:00-03:00, London H/L 03:00-08:00) and CLOSES back inside.
2) MSS+DISPLACEMENT within 12 bars: a bar closes beyond the last confirmed 3-bar swing (high for longs after a low sweep) with body >= 0.6 x ATR14(5m) AND leaves a fair-value gap (long: low[j] > high[j-2]).
3) ENTRY: limit at the FVG midpoint, must fill within 12 bars (price trades 0.25 through it) -> ICC 'correction'. STOP = sweep extreme -/+2 pts (skip risk <6 or >40). TARGET 2R, else flat 15:55. cost 0.5 pt. One trade per day (first complete chain). 1 contract, NQ points.
CONTROL: the opposite trade from the same limit price (stop/target mirrored) - tells whether direction carries information or it is just the 2R/stop geometry.
ALIGNMENT tags (known at entry): bias = sign(09:30 open - prior RTH close) agrees; discount/premium = long below / short above the prior-day 50% line; KZ = entry 09:35-11:00; level type (PD vs session vs overnight); trend5 = prior close vs close 5 days ago agrees.
Eras A 2018-02..2021-12 / B 2022-01..2026-01; PASS = same sign, |t|>=2 in both eras, n>=30 per era. Slices are exploratory: counts how many aligned tags (0-4)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
by=dict(tuple(b.groupby('d'))); days=sorted(by); COST=.5
RT=b[(b.m>=570)&(b.m<960)].groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'))
tr=np.maximum(b.high-b.low,0); atr_all=tr.rolling(14).mean()
def pivots(h,l,k):   # last confirmed 3-bar swing high/low strictly before bar k (needs 3 bars after)
    sh=sl=np.nan
    for p in range(k-4,2,-1):
        if np.isnan(sh) and h[p]==h[p-3:p+4].max(): sh=h[p]
        if np.isnan(sl) and l[p]==l[p-3:p+4].min(): sl=l[p]
        if not np.isnan(sh) and not np.isnan(sl): break
    return sh,sl
def sim(h,l,c,o,f,ent,stop,dr,xend,Rm=2.0):
    risk=abs(ent-stop); tgt=ent+dr*Rm*risk
    for q in range(f,xend+1):
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -risk-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): return Rm*risk-COST
    return (c[xend]-ent)*dr-COST
R=[]
for i,D in enumerate(days):
    if i<21 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30') or days[i-1] not in RT.index or D not in RT.index: continue
    p=by[days[i-1]]; p=p[p.m>=1080]; cD=by[D]; cD=cD[cD.m<960]
    if len(p)<30 or len(cD)<150: continue
    x=pd.concat([p,cD]).reset_index(drop=True); m=x.m.values; msx=np.where(m>=1080,m-1080,m+360)
    h,l,c,o=[x[k].values.astype(float) for k in ('high','low','close','open')]; tr5=np.maximum.reduce([h-l,np.abs(h-np.r_[c[0],c[:-1]]),np.abs(l-np.r_[c[0],c[:-1]])]); atr=pd.Series(tr5).rolling(14).mean().values
    i930=int(np.searchsorted(msx,570+360)); i03=int(np.searchsorted(msx,540)); i08=int(np.searchsorted(msx,840)); i1400=int(np.searchsorted(msx,840+360+0)) if False else int(np.where(m==840)[0][-1]) if (m==840).any() else None
    xend=int(np.where(m==955)[0][-1]) if (m==955).any() else len(x)-1
    iend14=int(np.where(m==840)[0][-1]) if (m==840).any() else None
    if iend14 is None: continue
    pd_=RT.loc[days[i-1]]; PDH,PDL,PDC=pd_.h,pd_.l,pd_.c; mid=(PDH+PDL)/2; ONH,ONL=h[:i930].max(),l[:i930].min(); AH,AL=h[:i03].max(),l[:i03].min(); LH,LL=h[i03:i08].max(),l[i03:i08].min()
    lv_lo={'PDL':PDL,'ONL':ONL,'AsiaL':AL,'LonL':LL}; lv_hi={'PDH':PDH,'ONH':ONH,'AsiaH':AH,'LonH':LH}
    o930=o[i930]; bias=np.sign(o930-PDC); tr5d=np.sign(PDC-RT.c.iloc[RT.index.get_loc(days[i-1])-5]) if RT.index.get_loc(days[i-1])>=5 else 0
    done=False
    for k in range(i930+1,iend14+1):
        if done: break
        for dr,lv in ((1,lv_lo),(-1,lv_hi)):
            hit=[n for n,v in lv.items() if (l[k]<v and c[k]>v) if dr>0] if dr>0 else [n for n,v in lv.items() if (h[k]>v and c[k]<v)]
            if not hit: continue
            ext=l[k] if dr>0 else h[k]; sh,sl=pivots(h,l,k); piv=sh if dr>0 else sl
            if np.isnan(piv): continue
            for j in range(k+1,min(k+13,len(x)-1)):
                body=abs(c[j]-o[j]); ok=(c[j]>piv and c[j]>o[j]) if dr>0 else (c[j]<piv and c[j]<o[j])
                if not(ok and body>=.6*atr[j] and (l[j]>h[j-2] if dr>0 else h[j]<l[j-2])): continue
                ent=((l[j]+h[j-2])/2) if dr>0 else ((h[j]+l[j-2])/2)
                if (dr>0 and ent>=c[j]) or (dr<0 and ent<=c[j]): break
                f=None
                for q in range(j+1,min(j+13,len(x))):
                    if (l[q]<=ent-.25) if dr>0 else (h[q]>=ent+.25): f=q; break
                if f is None: break
                stop=ext-2*dr; risk=abs(ent-stop)
                if not(6<=risk<=40): break
                r=sim(h,l,c,o,f,ent,stop,dr,xend); rc=sim(h,l,c,o,f,ent,ent+dr*risk,-dr,xend)
                tm=m[f]; R.append(dict(d=D,dr=dr,lvl=hit[0],pts=r,ctl=rc,risk=risk,bias=float(bias==dr),disc=float((ent<mid) if dr>0 else (ent>mid)),kz=float(570<=tm<=660),pdlv=float(hit[0] in('PDH','PDL')),tr5=float(tr5d==dr)))
                done=True; break
            if done: break
T=pd.DataFrame(R); T['era']=np.where(T.d<pd.Timestamp('2022-01-01'),'A','B'); T['yr']=T.d.dt.year; T['n_al']=T.bias+T.disc+T.kz+T.tr5
T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/icc_T.pkl')
def t_(a): return a.mean()/(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>5 and a.std()>0 else np.nan
def line(nm,g,col='pts'):
    a,bb=g[g.era=='A'][col],g[g.era=='B'][col]
    if len(a)<8 or len(bb)<8: return '%-34s n%d/%d'%(nm,len(a),len(bb))
    ok=' <== PASS' if (a.mean()*bb.mean()>0 and abs(t_(a))>=2 and abs(t_(bb))>=2 and min(len(a),len(bb))>=30) else ''
    return '%-34s A n %3d mean %+6.2f win %2.0f%% t %+5.1f | B n %3d mean %+6.2f win %2.0f%% t %+5.1f | yrs>0 %2.0f%%%s'%(nm,len(a),a.mean(),(a>0).mean()*100,t_(a),len(bb),bb.mean(),(bb>0).mean()*100,t_(bb),(g.groupby('yr')[col].mean()>0).mean()*100,ok)
print('chain trades',len(T),'days',T.d.nunique(),'| long',int((T.dr>0).sum()),'short',int((T.dr<0).sum()))
print(line('ALL chain (model)',T)); print(line('ALL control (opposite)',T,'ctl'))
for nm,f in (('long',T.dr>0),('short',T.dr<0),('PD-level sweeps',T.pdlv==1),('session/overnight level',T.pdlv==0),('bias agrees',T.bias==1),('bias against',T.bias==0),('discount/premium ok',T.disc==1),('wrong side of 50%',T.disc==0),('kill zone 9:35-11',T.kz==1),('outside KZ',T.kz==0),('5d trend agrees',T.tr5==1)): print(line(nm,T[f])); print(line('   control',T[f],'ctl'))
print('-- count of aligned tags (bias, disc, KZ, trend5)')
for k in range(5): print(line('aligned=%d'%k,T[T.n_al==k])); 
print(line('aligned>=3',T[T.n_al>=3])); print(line('   control aligned>=3',T[T.n_al>=3],'ctl')); print(line('aligned>=3 & PD level',T[(T.n_al>=3)&(T.pdlv==1)]))
