"""report 144: ICT UNICORN (breaker block overlapping a FVG) and IFVG (inverse FVG) on NQ 5m. DECLARED BEFORE RESULTS (Rey's screenshot; definitions fixed here, nothing tuned).
UNICORN long: swing high p (3-bar pivot, p in j-40..j-4) followed by a sell-off (a low <= h[p] - 1.5 ATR before j); breaker zone = [min low, max high] of bars p and p+1 (the up candle and the down candle that starts the drop);
 displacement bar j (09:35-14:00, body >= 0.6 ATR14, green, 3-bar bull FVG low[j] > high[j-2]) closes ABOVE the zone top. Overlap of FVG and zone >= 2 pts. Entry = limit at overlap midpoint (12 bars, 0.25 pt through); stop = zone low - 2; risk 6..40. Short = mirror.
IFVG long: an earlier BEAR FVG (gap >= 3 pts) formed at g (j-30..j-3) is inverted by displacement bar j (green, body >= 0.6 ATR) that closes above the gap top and no earlier bar closed above it; entry = limit at gap midpoint (12 bars); stop = gap bottom - 2; risk 6..40. Short = mirror (bull FVG closed through downward).
Both: target 2R, flat 15:55, cost 0.5 pt, first signal per day per model, NQ points, 1 contract. A 2018-02..2021-12 | B 2022-01..2026-01. PASS = same sign, |t|>=2, n>=30 both.
Extras: PD alignment (range since 18:00 ET: long below 0.5 / short above), core-legs idle days. Prediction: unicorn > plain FVG (report 142: -0.1/-1.7) because the overlap = two reasons; IFVG fades as a failed-gap reversal."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; COST=.5
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
by=dict(tuple(b.groupby('d'))); days=sorted(by); RT=b[(b.m>=570)&(b.m<960)].groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'))
MAE={}
def sim(h,l,c,f,ent,stop,dr,xend,tgt):
    worst=0.0
    for q in range(f,xend+1):
        worst=min(worst,((l[q] if dr>0 else h[q])-ent)*dr)
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): MAE['w']=worst-COST; return -abs(ent-stop)-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): MAE['w']=worst-COST; return abs(tgt-ent)-COST
    MAE['w']=worst-COST; return (c[xend]-ent)*dr-COST
def fill(h,l,j,ent,dr,n,ie):
    for q in range(j+1,min(j+13,n)):
        if (l[q]<=ent-.25) if dr>0 else (h[q]>=ent+.25): return q if q<=ie+12 else None
    return None
R=[]
for i,D in enumerate(days):
    if i<21 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30') or days[i-1] not in RT.index or D not in RT.index: continue
    p_=by[days[i-1]]; p_=p_[p_.m>=1080]; cD=by[D]; cD=cD[cD.m<960]
    if len(p_)<30 or len(cD)<150: continue
    x=pd.concat([p_,cD]).reset_index(drop=True); m=x.m.values; msx=np.where(m>=1080,m-1080,m+360)
    h,l,c,o=[x[k].values.astype(float) for k in ('high','low','close','open')]; n=len(x); tr5=np.maximum.reduce([h-l,np.abs(h-np.r_[c[0],c[:-1]]),np.abs(l-np.r_[c[0],c[:-1]])]); atr=pd.Series(tr5).rolling(14).mean().values
    i930=int(np.searchsorted(msx,930)); ie=np.where(m==840)[0]; xe=np.where(m==955)[0]
    if not len(ie) or not len(xe): continue
    ie=int(ie[-1]); xend=int(xe[-1]); got={'UNI':False,'IFVG':False}
    for j in range(i930+3,ie+1):
        if all(got.values()): break
        body=abs(c[j]-o[j])
        if body<.6*atr[j]: continue
        for dr in (1,-1):
            green=c[j]>o[j]
            if (dr>0)!=green: continue
            # ---- UNICORN
            if not got['UNI'] and ((l[j]>h[j-2]) if dr>0 else (h[j]<l[j-2])):
                fvg=(h[j-2],l[j]) if dr>0 else (h[j],l[j-2])
                for p in range(j-4,max(j-41,4),-1):
                    pv=(h[p]==h[p-3:p+4].max()) if dr>0 else (l[p]==l[p-3:p+4].min())
                    if not pv: continue
                    zl=min(l[p],l[p+1]); zh=max(h[p],h[p+1])
                    drop=(h[p]-l[p+1:j-1].min()) if dr>0 else (h[p+1:j-1].max()-l[p]) if j-1>p+1 else 0
                    if drop<1.5*atr[j]: continue
                    if (dr>0 and c[j]<=zh) or (dr<0 and c[j]>=zl): continue
                    lo_,hi_=max(zl,fvg[0]),min(zh,fvg[1])
                    if hi_-lo_<2: continue
                    ent=(lo_+hi_)/2; stop=(zl-2) if dr>0 else (zh+2); risk=abs(ent-stop)
                    if risk<6 or risk>40: continue
                    f=fill(h,l,j,ent,dr,n,ie)
                    if f is None: continue
                    hi,lo=h[:j+1].max(),l[:j+1].min(); pdv=(ent-lo)/(hi-lo) if hi>lo else .5
                    R.append(dict(d=D,model='UNI',m=int(m[f]),ent=ent,zlo=lo_,zhi=hi_,dr=dr,pts=sim(h,l,c,f,ent,stop,dr,xend,ent+2*risk*dr),mae=MAE['w'],pd=pdv,al=(pdv<.5) if dr>0 else (pdv>.5),risk=risk)); got['UNI']=True; break
            # ---- IFVG
            if not got['IFVG']:
                for g in range(j-3,max(j-31,3),-1):
                    gap=(l[g-2]-h[g]) if dr>0 else (l[g]-h[g-2])    # dr>0: earlier BEAR gap (top l[g-2], bottom h[g]); dr<0: earlier BULL gap (bottom h[g-2], top l[g])
                    if gap<3: continue
                    top,bot=(l[g-2],h[g]) if dr>0 else (l[g],h[g-2])
                    if dr>0 and not(c[g]<o[g]): continue
                    if dr<0 and not(c[g]>o[g]): continue
                    seg=c[g+1:j]
                    if len(seg) and ((seg>top).any() if dr>0 else (seg<bot).any()): continue
                    if not ((c[j]>top) if dr>0 else (c[j]<bot)): continue
                    ent=(top+bot)/2; stop=(bot-2) if dr>0 else (top+2); risk=abs(ent-stop)
                    if risk<6 or risk>40: continue
                    if (dr>0 and ent>=c[j]) or (dr<0 and ent<=c[j]): continue
                    f=fill(h,l,j,ent,dr,n,ie)
                    if f is None: continue
                    hi,lo=h[:j+1].max(),l[:j+1].min(); pdv=(ent-lo)/(hi-lo) if hi>lo else .5
                    R.append(dict(d=D,model='IFVG',m=int(m[f]),ent=ent,zlo=bot,zhi=top,dr=dr,pts=sim(h,l,c,f,ent,stop,dr,xend,ent+2*risk*dr),mae=MAE['w'],pd=pdv,al=(pdv<.5) if dr>0 else (pdv>.5),risk=risk)); got['IFVG']=True; break
T=pd.DataFrame(R); T['A']=T.d<'2022-01-01'; idl=~pd.read_pickle(SC+'idle_state.pkl')['trade'].astype(bool); idl.index=pd.to_datetime(idl.index); T['idle']=T.d.map(idl).fillna(False).astype(bool); T.to_pickle(SC+'ext/uni_ifvg.pkl')
def t_(a): return a.mean()/(a.std()/np.sqrt(len(a))) if len(a)>8 else np.nan
def line(lab,m):
    a,b_=T[m&T.A].pts,T[m&~T.A].pts; ps=len(a)>=30 and len(b_)>=30 and np.sign(a.mean())==np.sign(b_.mean()) and abs(t_(a))>=2 and abs(t_(b_))>=2
    return f'  {lab:30s} A n{len(a):4d} {a.mean():+6.2f} win{100*(a>0).mean():3.0f}% t{t_(a):+5.1f} | B n{len(b_):4d} {b_.mean():+6.2f} win{100*(b_>0).mean():3.0f}% t{t_(b_):+5.1f}{"  <== PASS" if ps else ""}'
for mdl in ('UNI','IFVG'):
    M=T.model==mdl; print(f'\n== {mdl}: {int(M.sum())} trades on {T[M].d.nunique()} days')
    for nm,m in (('all',M),('long',M&(T.dr>0)),('short',M&(T.dr<0)),('PD-aligned',M&T.al),('PD-aligned long',M&T.al&(T.dr>0)),('NOT aligned',M&~T.al),('core legs idle',M&T.idle),('core legs active',M&~T.idle),('idle & aligned',M&T.idle&T.al)): print(line(nm,m))
