"""report 156: AUCTION TIGHTNESS MEAN-REVERSION (Rey's screenshot, MNQ/MES/M2K trader). DECLARED BEFORE RESULTS; rules copied from the screenshot, cuts frozen on 2018-21.
Prior RTH day (09:30-16:00) volume profile from NQ 5m bars (bar volume spread uniformly over its high-low, 1-pt bins); POC = max bin; VA = 70% of volume expanded from POC; tightness = (VAH-VAL)/day range.
Setup1: tightness <= median(2018-21) AND open > VAH -> SHORT, target VAL. Setup2: tightness <= 25th pct AND open < VAL -> LONG, target VAH. Setup4: tightness >= 75th pct AND open > VAH -> SHORT, target VAL.
Entry 09:30 open; stop = 0.5*ATR14 (daily RTH true range, simple mean); target = level; when price touches prior POC the stop ratchets to +0.25 ATR locked profit; flat 15:55; cost 0.5; NQ points 1 ct.
RESCUE variant: if the 09:30 bar's adverse excursion exceeds the stop, enter at that bar's CLOSE instead and re-stop 0.5 ATR from there. A <2022 | B 2022+."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
COST=.5; BL=pd.Timestamp('2022-01-01')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
r=b[(b.m>=570)&(b.m<=955)]; by=dict(tuple(r.groupby('d'))); days=[d for d in sorted(by) if len(by[d])>=75]
def prof(x):
    lo,hi=int(np.floor(x.low.min())),int(np.ceil(x.high.max())); bins=np.zeros(hi-lo+1)
    for a,bb,v in zip(x.low.values,x.high.values,x.volume.values):
        i0,i1=int(np.floor(a))-lo,int(np.ceil(bb))-lo; i1=max(i1,i0+1); bins[i0:min(i1,len(bins))]+=v/(min(i1,len(bins))-i0)
    poc=int(bins.argmax()); tot=bins.sum(); a_=bz=poc; cur=bins[poc]
    while cur<.7*tot and (a_>0 or bz<len(bins)-1):
        up=bins[bz+1] if bz<len(bins)-1 else -1; dn=bins[a_-1] if a_>0 else -1
        if up>=dn: bz+=1; cur+=up
        else: a_-=1; cur+=dn
    return lo+poc,lo+a_,lo+bz+1  # poc, val, vah
P={}
for d in days:
    x=by[d]; poc,val,vah=prof(x); rg=x.high.max()-x.low.min(); P[d]=dict(poc=poc,val=val,vah=vah,t=(vah-val)/rg if rg>0 else np.nan,tr=rg)
dd=pd.DataFrame(P).T; dd.index=pd.to_datetime(dd.index); dd['atr']=dd.tr.rolling(14).mean()
A=dd[dd.index<BL].t.dropna(); q25,q50,q75=A.quantile([.25,.5,.75]); print(f'tightness cuts (2018-21): p25 {q25:.2f} med {q50:.2f} p75 {q75:.2f}')
def run(ent_i,x,dr,ent,stop,tgt,poc,atr,xend):
    c=x.close.values; h=x.high.values; l=x.low.values; lock=False; st=stop
    for q in range(ent_i,xend+1):
        if (l[q]<=st) if dr>0 else (h[q]>=st): return (st-ent)*dr-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): return (tgt-ent)*dr-COST
        if not lock and ((h[q]>=poc and l[q]<=poc)): lock=True; st=ent+dr*.25*atr
        # lock applies from the next bar
    return (c[xend]-ent)*dr-COST
R=[]
for i,D in enumerate(days):
    if i<15 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30'): continue
    p=P[days[i-1]]; atr=dd.atr[days[i-1]]; tt=p['t']; x=by[D].reset_index(drop=True)
    if x.m.iloc[0]!=570 or x.m.iloc[-1]!=955 or not (atr>0): continue
    o=x.open[0]; setups=[]
    if tt<=q50 and o>p['vah']: setups.append(('S1_premium',-1,p['val']))
    if tt<=q25 and o<p['val']: setups.append(('S2_discount',1,p['vah']))
    if tt>=q75 and o>p['vah']: setups.append(('S4_loose_prem',-1,p['val']))
    for nm,dr,tgt in setups:
        xend=len(x)-1; st0=o-dr*.5*atr
        for resc in (False,True):
            ent_i=0; ent=o; stop=st0
            if resc and (((x.high[0]>=st0) if dr<0 else (x.low[0]<=st0))):
                ent=x.close[0]; stop=ent-dr*.5*atr; ent_i=1
                if (tgt-ent)*dr<=0: continue
            R.append(dict(d=D,cfg=nm+('_rescue' if resc else ''),pts=run(ent_i,x,dr,ent,stop,tgt,p['poc'],atr,xend),era='A' if D<BL else 'B'))
T=pd.DataFrame(R); T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/va156.pkl')
def stt(s):
    n=len(s); 
    if n==0: return 'n   0'
    pf=s[s>0].sum()/max(-s[s<0].sum(),1e-9); return f'n{n:4d} {s.mean():+6.2f} t{(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0):+5.1f} w{(s>0).mean()*100:3.0f}% PF{pf:4.2f} tot{s.sum():+7.0f}'
print('A=2018-21 | B=2022+  (NQ pts, 1 ct)')
for cfg,g in T.groupby('cfg'): print(f'{cfg:22s} A {stt(g[g.era=="A"].pts)} | B {stt(g[g.era=="B"].pts)}')
for rs in ('','_rescue'):
    g=T[T.cfg.isin(['S1_premium'+rs,'S2_discount'+rs,'S4_loose_prem'+rs])].sort_values('d').drop_duplicates(['d']); print(f'COMBINED{rs:8s} A {stt(g[g.era=="A"].pts)} | B {stt(g[g.era=="B"].pts)}')
