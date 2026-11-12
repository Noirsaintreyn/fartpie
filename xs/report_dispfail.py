"""DISPLACEMENT FAILURE (NQ 5m RTH, declared before results). Stage 1 displacement: leg = origin(low)->peak(high) inside the last 24 bars, size >= M x ADR20, path efficiency E=|net|/sum|dc| >= Ethr.
Stage 2 deterioration: progress of the last third of the leg <= DEC x progress of the first third. Stage 3 structural failure: a close below the lowest low of the prior 3 bars, within 8 bars of the peak (mirror for down-legs).
Enter next open against the leg. Stop peak+2. Exits: ORIG (leg origin), HALF (50% retrace), 1R. Cost .5pt, 24-bar time stop, entries 10:00-14:30. First signal per day. Grid M in {.25,.35}, Ethr {.35,.5}, DEC {.35,.6}."""
import itertools
_s=open('xs/report_daytype.py').read(); exec(_s[:_s.index('# ---- gated fades')])
def disp(g,adr,M,Ethr,DEC,W=24):
    hi=g.high.values;lo=g.low.values;cl=g.close.values;N=len(g);m_=(g.dt.dt.hour*60+g.dt.dt.minute-570).values;dc=np.abs(np.diff(cl,prepend=cl[0]))
    for i in range(W,N-2):
        if m_[i]<30 or m_[i]>300: continue
        for up in (1,0):
            if up:
                p=i-W+1+int(np.argmax(hi[i-W+1:i+1])); 
                if i-p>8 or p<i-W+7: continue
                o=i-W+1+int(np.argmin(lo[i-W+1:p+1])); 
                if p-o<6: continue
                mv=hi[p]-lo[o]; net=cl[p]-cl[o]
            else:
                p=i-W+1+int(np.argmin(lo[i-W+1:i+1]))
                if i-p>8 or p<i-W+7: continue
                o=i-W+1+int(np.argmax(hi[i-W+1:p+1]))
                if p-o<6: continue
                mv=hi[o]-lo[p]; net=cl[o]-cl[p]
            if mv<M*adr: continue
            E=abs(cl[p]-cl[o])/max(dc[o+1:p+1].sum(),1e-9)
            if E<Ethr: continue
            k=max((p-o)//3,1); g1=abs(cl[o+k]-cl[o]); g3=abs(cl[p]-cl[p-k])
            if g3>DEC*g1: continue
            fail=(cl[i]<lo[i-3:i].min()) if up else (cl[i]>hi[i-3:i].max())
            if not fail: continue
            dr=-1 if up else 1; e=g.open.iloc[i+1]; ext=hi[p] if up else lo[p]; st=ext+2*(1 if up else -1); R=abs(e-st)
            if not (5<=R<=60): continue
            org=lo[o] if up else hi[o]; half=ext-dr*(-1)*0.5*mv*(-1) if False else (ext-0.5*mv if up else ext+0.5*mv)
            return dict(i=i,t=g.dt.iloc[i+1],dr=dr,e=e,st=st,R=R,org=org,half=half,mv=mv,E=E)
    return None
def res(g,i,dr,st,tgt):
    hi=g.high.values;lo=g.low.values;e=g.open.iloc[i];n=min(len(g),i+24)
    for j in range(i,n):
        if dr>0:
            if lo[j]<=st: return st-e-.5
            if hi[j]>=tgt: return tgt-e-.5
        else:
            if hi[j]>=st: return e-st-.5
            if lo[j]<=tgt: return e-tgt-.5
    return (g.close.iloc[n-1]-e)*dr-.5
OUT=[]
for M,Ethr,DEC in itertools.product((.25,.35),(.35,.5),(.35,.6)):
    for d in days:
        if d not in G or not D.adr.get(d)==D.adr.get(d) or d.year<2018: continue
        g=G[d]; s=disp(g,D.adr[d],M,Ethr,DEC)
        if s is None: continue
        for ex,t in (('ORIG',s['org']),('HALF',s['half']),('1R',s['e']+s['dr']*s['R'])):
            if (s['dr']<0 and t<s['e']) or (s['dr']>0 and t>s['e']): OUT.append(dict(M=M,E=Ethr,DEC=DEC,ex=ex,day=d,pts=res(g,s['i']+1,s['dr'],s['st'],t),R=s['R'],dr=s['dr']))
O=pd.DataFrame(OUT); O['era']=np.where(O.day<'2022-01-01','19-21','22-26'); O=O[O.day>='2019-01-01']
O['idle']=O.day.map(idle).fillna(False); O['pi']=O.day.map(PROBI[30]); O['pr']=O.day.map(PROBI[60])
def ln(x): return f'n {len(x):4d}' if len(x)<15 else f'n {len(x):4d} avg {x.pts.mean():+6.2f} win {(x.pts>0).mean():.0%} t {x.pts.mean()/(x.pts.std()/len(x)**.5):+.1f}'
print(f'{len(O)} trades; ADR20-scaled legs on NQ 5m')
for (M,E,DEC,ex),x in O.groupby(['M','E','DEC','ex']):
    print(f'M{M} E{E} DEC{DEC} {ex:4s} ALL: '+' || '.join(f'{e} {ln(x[x.era==e])}' for e in ('19-21','22-26'))+'  | IDLE: '+' || '.join(f'{e} {ln(x[(x.era==e)&x.idle])}' for e in ('19-21','22-26')))
print('--- median across grid cells, by exit (avg pts per trade)')
for ex,x in O.groupby('ex'):
    c=x.groupby(['M','E','DEC','era']).pts.mean().unstack(); print(ex, 'cells positive in both eras:', int(((c>0).all(axis=1)).sum()),'of',len(c),' median 19-21 %+.2f 22-26 %+.2f'%(c['19-21'].median(),c['22-26'].median()))
b_=O[(O.M==.25)&(O.E==.35)&(O.DEC==.6)]
print('--- loosest cell, gated by P(idle) at the 10:00 call (rank cuts) ; note entries before ~10:00 excluded by m>=30')
for ex,x in b_.groupby('ex'):
    q=x.pi.quantile([.5,.67]).tolist()
    for nm,cut in (('all',0),('top50%',q[0]),('top33%',q[1])): print(f'{ex:4s} {nm:7s}: '+' || '.join(f'{e} {ln(x[(x.era==e)&(x.pi>=cut)])}' for e in ('19-21','22-26')))
