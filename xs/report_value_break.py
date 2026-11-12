"""BREAKOUT through cluster edges (the mirror of report_value_shape, declared because OPTICS/TDA edge FADES lost in both eras): close beyond the edge by 0.05 ADR (10:00-14:00) -> trade WITH the break at next open, stop edge -0.15 ADR (inside), exits 1R / 2R / 24-bar time stop, cost .5. Edges: OPTICS, TDA, Spectral, VP5 (same pivot cloud, prior 5 sessions)."""
_s=open('xs/report_value_shape.py').read(); exec(_s[:_s.index("METH={'TDA'")]) ; exec(_s[_s.index("METH={'TDA'"):_s.index("rows=[]; SH=[]")])
import pickle
rows=[]
for i,d in enumerate(days):
    if i<25 or d not in G or not D.adr[d]==D.adr[d] or d.year<2019: continue
    adr=D.adr[d]; g=G[d]; prior=[days[j] for j in range(i-5,i) if days[j] in G]; x=np.array(sum((pivots(G[q]) for q in prior),[])); g5=pd.concat([G[q] for q in prior])
    E={}
    for nm,f in {'TDA':m_tda,'OPTICS':m_optics,'Spectral':m_spectral}.items():
        try: E[nm]=f(x,adr)
        except Exception: E[nm]=None
    E['VP5']=m_vp(g5,adr)
    cl=g.close.values; m_=(g.dt.dt.hour*60+g.dt.dt.minute-570).values
    for nm,e_ in E.items():
        if e_ is None: continue
        Lw,Up,ctr=e_
        if not (Up-Lw)>.15*adr: continue
        for i2 in range(6,len(g)-3):
            if m_[i2]<30 or m_[i2]>270: continue
            up=cl[i2]>=Up+.05*adr and cl[i2-1]<Up+.05*adr; dn=cl[i2]<=Lw-.05*adr and cl[i2-1]>Lw-.05*adr
            if not (up or dn): continue
            dr=1 if up else -1; e=g.open.iloc[i2+1]; edge=Up if up else Lw; st=edge-(.15*adr if up else -.15*adr)
            if not ((dr>0 and st<e) or (dr<0 and st>e)): break
            R=abs(e-st)
            if R<5 or R>80: break
            for ex,mult in (('1R',1),('2R',2)):
                rows.append(dict(day=d,meth=nm,ex=ex,dr=dr,pts=resolve(g,i2+1,dr,st,e+dr*R*mult),R=R))
            break
T=pd.DataFrame(rows); T['era']=np.where(T.day<'2022-01-01','19-21','22-26'); T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/value_break.pkl')
def ln(x): return f'n {len(x):4d}' if len(x)<15 else f'n {len(x):4d} avg {x.pts.mean():+6.2f} win {(x.pts>0).mean():.0%} t {x.pts.mean()/(x.pts.std()/len(x)**.5):+.1f}'
for (meth,ex),x in T.groupby(['meth','ex']): print(f'{meth:9s} {ex:3s} BREAK: '+' || '.join(f'{e} {ln(x[x.era==e])}' for e in ('19-21','22-26'))+'  | long '+ln(x[x.dr>0])+' | short '+ln(x[x.dr<0]))
