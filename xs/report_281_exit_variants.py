"""Report 281 (declared before results). Exits that do not depend on the extreme holding. NQ 5m RTH. Events as in 255/264: bar 10:00-15:00 making a new session extreme, >= .5 R20 from the open, rejecting (>=50% of its range as the wick on the push side), max 2 per side per day. Fade entry = next bar open. Flags: V1 (>=2 earlier new-extreme bars that day, volume >= their mean and >= 1.3x prior-12 mean), V2 (est. delta over bars i-2..i in the push direction > 0 and rel >= 1.3). Stops: TIGHT (extreme +-1), .25 R20 beyond the extreme, .5 R20 beyond, NONE (catastrophic stop 1.0 R20). Exits: time 30 min, 60 min, 120 min, flat at 15:55, VWAP (session VWAP at entry, else time 120). Cost 1 pt. Cells: all rejected extremes, V1, V2, V1&V2. Output: n, mean net pts per trade, day-clustered t, train 2019-21 | test 2018+2022-26. A cell passes if positive in both splits with n>=60 per split and |t|>=2 in test."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index(); rth=b.between_time('09:30','15:55'); G=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'h':G.high.max(),'l':G.low.min(),'n':G.size()}); dd=dd[dd.n>=70]; dd['R20']=(dd.h-dd.l).shift(1).rolling(20).mean()
STOPS=('tight','q25','q50','none'); EXITS=('t30','t60','t120','eod','vwap'); rows=[]
for d,g in G:
    if d not in dd.index or not np.isfinite(dd.R20[d]): continue
    R=dd.R20[d]; o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;v=g.volume.values.astype(float); mm=g.index.hour*60+g.index.minute; dl=v*(2*c-h-l)/np.maximum(h-l,1e-9); tp=(h+l+c)/3; vw=np.cumsum(tp*v)/np.maximum(np.cumsum(v),1)
    o0=o[0]; rh=-np.inf; rl=np.inf; ev={1:[],-1:[]}; cnt={1:0,-1:0}; n=len(g)
    for i in range(n-2):
        for dr,new in ((-1,h[i]>rh),(1,l[i]<rl)):
            if not new: continue
            pv=list(ev[dr]); ev[dr].append(v[i])
            if not(600<=mm[i]<900) or cnt[dr]>=2: continue
            rng=h[i]-l[i]
            if rng<=0: continue
            if dr==-1:
                if h[i]-o0<.5*R or (h[i]-c[i])<.5*rng: continue
                ext=h[i]
            else:
                if o0-l[i]<.5*R or (c[i]-l[i])<.5*rng: continue
                ext=l[i]
            cnt[dr]+=1; pm=v[max(0,i-12):i].mean() if i>0 else np.nan; rel=v[i]/pm if pm>0 else np.nan
            v1=bool(len(pv)>=2 and v[i]>=np.mean(pv) and rel>=1.3); v2=bool(-dr*dl[max(0,i-2):i+1].sum()>0 and rel>=1.3)
            e=o[i+1]; sd=-dr  # dr=-1 short -> position sign -1 ... position = dr (short at new high)
            pos=dr
            res={}
            for sn,sdist in (('tight',1.0),('q25',.25*R),('q50',.5*R),('none',1.0*R)):
                stop=ext-pos*sdist*(-1) if False else (ext+(-pos)*0)  # placeholder
                stop = ext + sdist if pos==-1 else ext - sdist
                for en,nb in (('t30',6),('t60',12),('t120',24),('eod',999),('vwap',24)):
                    tgt=vw[i] if en=='vwap' else None
                    r=None
                    for j in range(i+1,min(n,i+1+nb)):
                        hs=(h[j]>=stop) if pos==-1 else (l[j]<=stop)
                        if hs: r=-(abs(stop-e)) if True else 0; r=(e-stop) if pos==-1 else (stop-e); break
                        if tgt is not None:
                            ht=(l[j]<=tgt) if pos==-1 else (h[j]>=tgt)
                            if ht and ((e-tgt)*pos>0): r=(e-tgt) if pos==-1 else (tgt-e); break
                        if mm[j]>=955: r=(e-c[j]) if pos==-1 else (c[j]-e); break
                    if r is None:
                        j=min(n-1,i+nb) if nb<999 else n-1; r=(e-c[j]) if pos==-1 else (c[j]-e)
                    res[(sn,en)]=r-1
            rows.append((d,v1,v2,res))
E=pd.DataFrame([(d,v1,v2) for d,v1,v2,_ in rows],columns=['d','v1','v2']); E['sp']=np.where((E.d.dt.year>=2019)&(E.d.dt.year<=2021),'train',np.where((E.d.dt.year==2018)|(E.d.dt.year>=2022),'test','x'))
for sn in STOPS:
    for en in EXITS: E[f'{sn}_{en}']=[r[2][(sn,en)] for r in rows] if False else [r[3][(sn,en)] for r in rows]
print('events',len(E),' V1',int(E.v1.sum()),' V2',int(E.v2.sum()),' V1&V2',int((E.v1&E.v2).sum()))
def cell(m,col):
    s=[]
    for sp in ('train','test'):
        z=E[m&(E.sp==sp)]
        if len(z)<20: s.append(f'n{len(z):4d} -'); continue
        ds=z.groupby('d')[col].agg(['sum','size']); mean=z[col].mean(); se=np.sqrt(((ds['sum']-mean*ds['size'])**2).sum())/len(z); s.append(f'n{len(z):4d} {mean:+6.1f} (t{mean/se:+.1f})')
    return ' | '.join(s)
for nm,m in (('V1&V2',E.v1&E.v2),('V1',E.v1),('V2',E.v2),('all rejected',E.v1.notna())):
    print(f'\n== {nm} ==  rows=stop, cols=exit (train | test)')
    for sn in STOPS:
        for en in EXITS: print(f'  {sn:5s} {en:5s}',cell(m,f'{sn}_{en}'))
