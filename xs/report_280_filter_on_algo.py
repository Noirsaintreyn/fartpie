"""Report 280 (declared before results). Do volume-building / absorption extremes mark bad entries for the main algo? Legs with a known direction: RA (329 trades) and V15a (1,058); other weighted legs have no stored direction (M1, R1, RULES) or are overnight/09:30 entries (ON+, ON-, S1), so not testable here. For each trade, last completed 5m bar before entry m (bar i). ALIGNED = bar i made a new session high and dir=+1, or new session low and dir=-1 (entering with the push at a fresh extreme). F = rejection bar (closes back with >=50% of its range as the wick on the push side) + V1 volume building (>= mean volume of earlier new-extreme bars that day, >=2 earlier, and >= 1.3x prior-12-bar mean) + V2 absorption (est. delta of bars i-2..i in the push direction > 0 and rel>=1.3). Also looser flags: ALIGNED alone, ALIGNED & V1, ALIGNED & V2. Compare mean net pts per trade flagged vs unflagged, per leg and pooled (pts x sz so contracts-weighted), train 2019-21 | test 2018+2022-26, day-clustered t on the difference (flagged - rest). If flagged trades are worse in both splits, skipping them is a filter to test in the Lucid sim."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
L=pickle.load(open(SP+'legs280.pkl','rb'))
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index(); rth=b.between_time('09:30','15:55'); G={d:g for d,g in rth.groupby(rth.index.normalize())}
cache={}
def feats(day):
    if day in cache: return cache[day]
    g=G.get(day)
    if g is None: cache[day]=None; return None
    h=g.high.values;l=g.low.values;c=g.close.values;v=g.volume.values.astype(float); mm=g.index.hour*60+g.index.minute; dl=v*(2*c-h-l)/np.maximum(h-l,1e-9)
    rh=-np.inf; rl=np.inf; evh=[]; evl=[]; out={}
    for i in range(len(g)):
        nh=h[i]>rh; nl=l[i]<rl; rng=h[i]-l[i]
        for dr,new,evs in ((1,nh,evh),(-1,nl,evl)):
            if not new: continue
            pv=list(evs); evs.append(v[i]); pm=v[max(0,i-12):i].mean() if i>0 else np.nan; rel=v[i]/pm if pm and pm>0 else np.nan
            rej=(rng>0 and ((h[i]-c[i])>=.5*rng if dr==1 else (c[i]-l[i])>=.5*rng))
            v1=bool(len(pv)>=2 and v[i]>=np.mean(pv) and rel>=1.3); ds=dl[max(0,i-2):i+1].sum(); v2=bool(dr*ds>0 and rel>=1.3)
            out[(mm[i],dr)]=(rej,v1,v2)
        rh=max(rh,h[i]); rl=min(rl,l[i])
    cache[day]=out; return out
rows=[]
for leg in ('RA','V15a'):
    T=L[leg].copy(); T['day']=pd.to_datetime(T.day)
    for r in T.itertuples():
        f=feats(r.day)
        if f is None: continue
        k=(int(r.m)-5,int(r.dir)); 
        if k in f: rej,v1,v2=f[k]; al=True
        else: rej=v1=v2=False; al=False
        rows.append((leg,r.day,int(r.dir),r.pts*max(r.sz,0.0001) if False else r.pts,al,al and v1,al and v2,al and rej and v1 and v2))
E=pd.DataFrame(rows,columns=['leg','d','dir','pts','al','al_v1','al_v2','F']); E['sp']=np.where((E.d.dt.year>=2019)&(E.d.dt.year<=2021),'train',np.where((E.d.dt.year==2018)|(E.d.dt.year>=2022),'test','x'))
print('trades',len(E),E.groupby('leg').size().to_dict(),' aligned share',E.groupby('leg').al.mean().round(2).to_dict())
def cmp(m,lab,legs=('RA','V15a')):
    s=[]
    for sp in ('train','test'):
        z=E[E.leg.isin(legs)&(E.sp==sp)]; a=z[m.reindex(z.index,fill_value=False)]; r=z[~m.reindex(z.index,fill_value=False)]
        if len(a)<8: s.append(f'flag n{len(a):4d} -'); continue
        diff=a.pts.mean()-r.pts.mean(); se=np.sqrt(a.pts.var()/len(a)+r.pts.var()/len(r)); s.append(f'flag n{len(a):4d} {a.pts.mean():+6.1f} vs rest n{len(r):4d} {r.pts.mean():+6.1f} diff {diff:+6.1f} (t{diff/se:+.1f})')
    print(f' {lab:22s}',' | '.join(s))
for nm,legs in (('RA+V15a',('RA','V15a')),('RA',('RA',)),('V15a',('V15a',))):
    print(f'\n{nm}: pts per trade (net, per 1 contract of the leg)')
    cmp(E.al,'aligned at extreme',legs); cmp(E.al_v1,'aligned & V1',legs); cmp(E.al_v2,'aligned & V2',legs); cmp(E.F,'F (rej&V1&V2)',legs)
