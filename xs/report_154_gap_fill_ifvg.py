"""report 154: RTH OPENING GAP fill as mean-reversion context for IFVG/FVG (Rey: 'gaps pointed away do better, ~81% retrace, look for fvg to take it back'). DECLARED BEFORE RESULTS.
gap = 09:30 open - prior RTH close (15:55 bar close). Fill dir = sign(prevclose-open). TOWARD = trade dr==fill dir; AWAY = opposite.
T1 base rate of fill (prev close touched by 15:55) by |gap| bucket. T2 IFVG (ext/uni_ifvg.pkl) and all FVG (ext/fvg_all_vp.pkl) pts split TOWARD/AWAY, gap >= 5 pts. T3 re-sim IFVG TOWARD with target = prev close (stop = entry -+ risk, flat 15:55, cost .5), needs target >= 1R away.
A <2022 | B 2022+. PASS = same sign, |t|>=2, n>=30 both."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; COST=.5
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
r=b[(b.m>=570)&(b.m<=955)]; by=dict(tuple(r.groupby('d'))); days=sorted(by); G={}
for i,D in enumerate(days):
    if i==0: continue
    x=by[D]; p=by[days[i-1]]
    if x.m.iloc[0]!=570 or p.m.iloc[-1]!=955 or x.m.iloc[-1]!=955: continue
    pc=p.close.iloc[-1]; o=x.open.iloc[0]; gap=o-pc; fd=np.sign(pc-o)
    filled=(x.high.max()>=pc) if gap<0 else (x.low.min()<=pc)
    G[D]=dict(gap=gap,pc=pc,fd=fd,filled=filled,x=x)
g=pd.DataFrame({k:{a:b_ for a,b_ in v.items() if a!='x'} for k,v in G.items()}).T; g.index=pd.to_datetime(g.index); g['gap']=g.gap.astype(float); g['filled']=g.filled.astype(bool)
g['era']=np.where(g.index<'2022-01-01','A','B'); g['ag']=g.gap.abs()
g['bk']=pd.cut(g.ag,[0,5,15,30,60,1e9])
print('T1 gap-fill base rate by day end (|gap| pts bucket)'); print(g.groupby(['bk','era']).filled.agg(['mean','size']).unstack().round(2))
def st(s):
    n=len(s); return f'n{n:4d} {s.mean():+6.2f} t{(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0):+5.1f} w{(s>0).mean()*100:3.0f}%' if n else 'n   0'
for nm,f in (('IFVG','uni_ifvg.pkl'),('FVGall','fvg_all_vp.pkl')):
    t=pd.read_pickle(S+'ext/'+f); t=pd.DataFrame(t); 
    if nm=='IFVG': t=t[t.model=='IFVG']
    t['d']=pd.to_datetime(t.d); t=t[t.d.isin(g.index)].copy()
    if 'dr' not in t: print(nm,'no dr col',t.columns.tolist()); continue
    t['gap']=t.d.map(g.gap); t['fd']=t.d.map(g.fd); t['tw']=(t.dr==t.fd); t['era']=np.where(t.d<'2022-01-01','A','B')
    print(f'\nT2 {nm}: all {st(t[t.era=="A"].pts)} | {st(t[t.era=="B"].pts)}')
    for gm in (0,5,15):
        u=t[t.gap.abs()>=gm]
        for lab,msk in (('TOWARD',u.tw),('AWAY',~u.tw)):
            print(f'  gap>={gm:2d} {lab:6s} A {st(u[msk&(u.era=="A")].pts)} | B {st(u[msk&(u.era=="B")].pts)}')
    if nm=='IFVG':
        out=[]
        for _,q in t[t.tw&(t.gap.abs()>=5)].iterrows():
            x=G[q.d]['x']; x=x[x.m>=q.m].reset_index(drop=True)
            if not len(x): continue
            ent,dr,risk,tgt=q.ent,q.dr,q.risk,G[q.d]['pc']
            if abs(tgt-ent)<risk: continue
            stop=ent-dr*risk; res=None
            for _,y in x.iterrows():
                if (y.low<=stop) if dr>0 else (y.high>=stop): res=-risk-COST;break
                if (y.high>=tgt) if dr>0 else (y.low<=tgt): res=abs(tgt-ent)-COST;break
            if res is None: res=(x.close.iloc[-1]-ent)*dr-COST
            out.append((q.d,res,abs(tgt-ent)/risk))
        o=pd.DataFrame(out,columns=['d','pts','rr']); o['era']=np.where(o.d<'2022-01-01','A','B')
        print(f'\nT3 IFVG TOWARD gap>=5, target = gap fill (>=1R): A {st(o[o.era=="A"].pts)} | B {st(o[o.era=="B"].pts)}  mean RR {o.rr.mean():.1f}')
        for lo,hi in ((1,2),(2,4),(4,99)):
            u=o[(o.rr>=lo)&(o.rr<hi)]; print(f'   RR {lo}-{hi}: A {st(u[u.era=="A"].pts)} | B {st(u[u.era=="B"].pts)}')
