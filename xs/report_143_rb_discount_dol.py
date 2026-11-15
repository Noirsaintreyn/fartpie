"""report 143: THE ICT TRADER'S SETUP on NQ 1m: 1-minute rejection block (RB) limit at 50% of the wick + entry in the discount half of the current range + low-resistance buy-side (DOL) target. DECLARED BEFORE RESULTS.
(Rey: 'surprised RB or OB aren't better: 0.5 of current range + 1m RB + buyside low-resistance targets above'.) Report 142/18 tested FVG, OB, breaker on 5m only; RB and 1m were never tested.
RB (long): 1m bar with lower wick >= 60% of its range, range >= 0.7 x ATR14(1m), low = lowest low of the last 15 bars (a swept low), close in the top 40% of the bar. Entry = limit at the wick midpoint
 (low .. min(open,close)) placed at the bar's close, valid 15 bars (0.25 pt through). Stop = wick low -2 pts. Risk 4..40 pts. Short = mirror (upper wick, swept high).
CURRENT RANGE = high/low since 18:00 ET. PD = (entry - low)/(high - low); ALIGNED = long with PD<0.5 (discount) / short with PD>0.5 (premium).
POOLS = PDH PDL ONH ONL AsiaH/L LonH/L and the running day high/low. DOL target = nearest same-side pool beyond entry at >= 1.5R (cap 6R; none -> no DOL trade). LOWRES = no other pool/round-50 level between entry and target.
Trades: first signal per day 09:35-14:00 ET; flat 15:55; cost 0.5 pt; 1 contract; NQ points. Variants: 2R target vs DOL target. A 2018-02..2021-12 | B 2022-01..2026-01 (nothing tuned: thresholds fixed above).
PASS = same sign, |t|>=2 both eras, n>=30. Prediction (trader): ALIGNED & LOWRES & DOL best; RB50 alone ~ flat."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; COST=.5
d=pd.read_csv('/home/claude/nqdrive/NQ_1m_ETH.csv',parse_dates=['timestamp']); d['dt']=d.timestamp-pd.Timedelta(hours=6); d=d.sort_values('dt').drop_duplicates('dt').set_index('dt')
m_=d.index.hour*60+d.index.minute; days=sorted(set(d.index[(m_>=570)&(m_<960)].normalize())); RT=d[(m_>=570)&(m_<960)].groupby(d[(m_>=570)&(m_<960)].index.normalize()).agg(h=('high','max'),l=('low','min'),c=('close','last'))
def sim(h,l,c,f,ent,stop,dr,xend,tgt):
    for q in range(f,xend+1):
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -abs(ent-stop)-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): return abs(tgt-ent)-COST
    return (c[xend]-ent)*dr-COST
R=[]
for i,D in enumerate(days):
    if i<5 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30') or days[i-1] not in RT.index: continue
    x=d.loc[days[i-1]+pd.Timedelta(hours=18):D+pd.Timedelta(hours=15,minutes=56)]
    if len(x)<800: continue
    t=x.index; mm=(t.hour*60+t.minute).values; msx=np.where(mm>=1080,mm-1080,mm+360)
    o,h,l,c=[x[k].values.astype(float) for k in ('open','high','low','close')]; n=len(x)
    tr=np.maximum.reduce([h-l,np.abs(h-np.r_[c[0],c[:-1]]),np.abs(l-np.r_[c[0],c[:-1]])]); atr=pd.Series(tr).rolling(14).mean().values
    i930=int(np.searchsorted(msx,930)); i03=int(np.searchsorted(msx,540)); i08=int(np.searchsorted(msx,840)); ie=int(np.searchsorted(msx,1320)); xend=n-1
    if i930>=n or i930<60: continue
    pdh,pdl=RT.loc[days[i-1]].h,RT.loc[days[i-1]].l
    fixed={'PDH':pdh,'PDL':pdl,'ONH':h[:i930].max(),'ONL':l[:i930].min(),'AH':h[:i03].max(),'AL':l[:i03].min(),'LH':h[i03:i08].max(),'LL':l[i03:i08].min()}
    done=False
    for k in range(max(i930+5,20),min(ie,n-20)):
        if done: break
        rg=h[k]-l[k]
        if not(rg>=.7*atr[k] and rg>0): continue
        for dr in (1,-1):
            if dr>0: wick=(min(o[k],c[k])-l[k]); ok=wick>=.6*rg and c[k]>=l[k]+.6*rg and l[k]<=l[k-15:k].min()
            else: wick=(h[k]-max(o[k],c[k])); ok=wick>=.6*rg and c[k]<=h[k]-.6*rg and h[k]>=h[k-15:k].max()
            if not ok: continue
            ent=(l[k]+min(o[k],c[k]))/2 if dr>0 else (h[k]+max(o[k],c[k]))/2; stop=l[k]-2 if dr>0 else h[k]+2; risk=abs(ent-stop)
            if risk<4 or risk>40: continue
            f=None
            for q in range(k+1,min(k+16,n)):
                if (l[q]<=ent-.25) if dr>0 else (h[q]>=ent+.25): f=q; break
            if f is None: continue
            hi,lo=h[:k+1].max(),l[:k+1].min(); pd_=(ent-lo)/(hi-lo) if hi>lo else .5; aligned=(pd_<.5) if dr>0 else (pd_>.5)
            pools=dict(fixed); pools['DH']=h[i930:k+1].max() if k>i930 else h[k]; pools['DL']=l[i930:k+1].min() if k>i930 else l[k]
            side=[v for v in pools.values() if (v>ent+.5*risk if dr>0 else v<ent-.5*risk)]; side=sorted(side) if dr>0 else sorted(side,reverse=True)
            cand=[v for v in side if abs(v-ent)>=1.5*risk and abs(v-ent)<=6*risk]; dol=cand[0] if cand else np.nan
            def lowres(tg):
                lv=[v for v in list(pools.values())+list(np.arange(np.floor(min(ent,tg)/50)*50,max(ent,tg)+51,50)) if min(ent,tg)+.25*risk<v<max(ent,tg)-.25*risk and abs(v-tg)>1]
                return len(lv)==0
            t2=ent+2*risk*dr; p2=sim(h,l,c,f,ent,stop,dr,xend,t2); pD=sim(h,l,c,f,ent,stop,dr,xend,dol) if dol==dol else np.nan
            R.append(dict(d=D,dr=dr,pts2=p2,ptsD=pD,pd=pd_,al=aligned,lr2=lowres(t2),lrD=(lowres(dol) if dol==dol else False),hasD=dol==dol,risk=risk,nside=len(side))); done=True; break
T=pd.DataFrame(R); T['A']=T.d<'2022-01-01'; T.to_pickle(SC+'ext/rb_setup.pkl')
def t_(a): return a.mean()/(a.std()/np.sqrt(len(a))) if len(a)>8 else np.nan
def line(lab,m,col):
    a,b=T[m&T.A][col].dropna(),T[m&~T.A][col].dropna(); ps=len(a)>=30 and len(b)>=30 and np.sign(a.mean())==np.sign(b.mean()) and abs(t_(a))>=2 and abs(t_(b))>=2
    return f'  {lab:38s} A n{len(a):4d} {a.mean():+6.2f} win{100*(a>0).mean():3.0f}% t{t_(a):+5.1f} | B n{len(b):4d} {b.mean():+6.2f} win{100*(b>0).mean():3.0f}% t{t_(b):+5.1f}{"  <== PASS" if ps else ""}'
print(len(T),'RB trades, one per day, days',T.d.nunique())
for col,lab in (('pts2','2R target'),('ptsD','DOL target')):
    print('\n==',lab); an=T[col].notna()
    for nm,m in (('RB50 all',an),('  long',an&(T.dr>0)),('  short',an&(T.dr<0)),('ALIGNED (long discount/short premium)',an&T.al),('NOT aligned',an&~T.al),('ALIGNED long',an&T.al&(T.dr>0)),('ALIGNED & low-resistance path',an&T.al&(T.lr2 if col=='pts2' else T.lrD)),('ALIGNED & NOT low-res',an&T.al&~(T.lr2 if col=='pts2' else T.lrD)),('low-res path only',an&(T.lr2 if col=='pts2' else T.lrD))): print(line(nm,m,col))
