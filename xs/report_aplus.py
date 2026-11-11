"""A+ bracket test: top-P(win) core-leg signals, fixed NQ bracket (T target, S stop pts) on 5m bars. Plus Markov for a few-trade eval."""
import pickle,numpy as np,pandas as pd
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
r=pickle.load(open(SC+'mix_tiers.pkl','rb'))['rows'].reset_index(drop=True); p=pickle.load(open(SC+'pwin.pkl','rb')); r['P']=p['P']
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['d']=b.dt.dt.normalize(); b['mn']=b.dt.dt.hour*60+b.dt.dt.minute
G={d:g.reset_index(drop=True) for d,g in b.groupby('d')}
def br(day,m,dr,T,S):
    g=G.get(pd.Timestamp(day)); 
    if g is None: return np.nan
    g=g[(g.mn>=m)&(g.mn<955)]
    if len(g)<2: return np.nan
    e=g.open.iloc[0]
    for h,l,c in zip(g.high.values,g.low.values,g.close.values):
        if dr>0:
            if l<=e-S: return -S      # stop first if both (conservative)
            if h>=e+T: return T
        else:
            if h>=e+S: return -S
            if l<=e-T: return T
    return (g.close.iloc[-1]-e)*dr
r=r.dropna(subset=['P']); r['yr']=pd.to_datetime(r.day).dt.year
print(len(r),'signals with P; by leg',r.leg.value_counts().to_dict())
for T,S in ((20,20),(20,15),(30,20),(15,15)):
    r['o']=[br(d,m,dr,T,S) for d,m,dr in zip(r.day,r.m,r.dir)]
    x=r.dropna(subset=['o']); x['win']=x.o>0
    print(f'--- bracket +{T}/-{S}: n {len(x)} base win {x.win.mean():.0%} (breakeven {S/(T+S):.0%}) avg {x.o.mean():+.1f}')
    for lab,q in (('all',0),('top50%',.5),('top25%',.75),('top10%',.9)):
        for e,mm in (('<=2021',x.yr<=2021),('2022+',x.yr>=2022)):
            z=x[mm]; z=z[z.P>=z.P.quantile(q)] if q else z
            print(f'   {lab:7s} {e:6s} n {len(z):4d} win {z.win.mean():.0%} avg {z.o.mean():+.1f} pts  days/yr {len(z)/(7.5 if e=="<=2021" else 4.8):.0f}')
