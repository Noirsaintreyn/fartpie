"""report 153: TURTLE SOUP + SMT + OPENING-PRICE PREMIUM/DISCOUNT (Rey's 3 PDFs: Ts model, SQT, time x price). DECLARED BEFORE RESULTS, nothing tuned.
Soup: NQ 5m bar j whose high takes a reference high (low takes ref low) and CLOSES back inside -> fade. Entry = close of j; stop = sweep extreme +-1; risk 6..40; target 2R; flat 15:55; cost 0.5; first signal per day per config.
Refs: CYC = high/low of NYAM cycle 2 (08:13-09:26), signal in cycle 3 (09:27-10:40), no earlier cycle-3 bar beyond ref; PDHL = prior RTH day high/low; ON = 18:00..09:26 high/low.
Windows: STRICT 09:27-09:33 (the PDF's ts window), BROAD 09:27-10:40.
SMT = ES did NOT take its own ref in cycle 3 up to bar j (NQ swept, ES failed). PD = short only if price above BOTH 00:00 and 08:30 opens (premium), long only if below both (deep discount); PD1 = beyond either one.
A 2018-02..2021-12 | B 2022-01..2026-01. PASS = same sign, |t|>=2, n>=30 both eras. Baseline = all soups (no filter)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
COST=.5
nq=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); es=pd.read_parquet('/tmp/claude-0/ml/es5_real.parquet')[['dt','open','high','low','close']]
b=nq.merge(es,on='dt',suffixes=('','_es')).sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute; by=dict(tuple(b.groupby('d'))); days=sorted(by)
RT=b[(b.m>=570)&(b.m<960)].groupby('d').agg(h=('high','max'),l=('low','min'))
def sim(h,l,c,f,ent,stop,dr,xend,tgt):
    for q in range(f,xend+1):
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -abs(ent-stop)-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): return abs(tgt-ent)-COST
    return (c[xend]-ent)*dr-COST
R=[]
for i,D in enumerate(days):
    if i<2 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30') or days[i-1] not in RT.index: continue
    x=by[D]; x=x[x.m<=955].reset_index(drop=True); p_=by[days[i-1]]; p_=p_[p_.m>=1080]
    x=pd.concat([p_,x]).reset_index(drop=True); m=x.m.values; ms=np.where(m>=1080,m-1440,m)
    if (ms==955).sum()!=1 or (ms==0).sum()!=1 or (ms==510).sum()!=1 or (ms==565).sum()!=1 or (ms==495).sum()!=1 or (ms==570).sum()!=1 or (m>=1080).sum()<20: continue
    h,l,c,o=[x[k].values.astype(float) for k in ('high','low','close','open')]; he,le=x.high_es.values.astype(float),x.low_es.values.astype(float)
    i0,i830,i927,xend=[int(np.where(ms==t)[0][0]) for t in (0,510,565,955)]; o0,o8=o[i0],o[i830]
    iON=int(np.where(m>=1080)[0][0]); iC2=int(np.where(ms==495)[0][0]); i930=int(np.where(ms==570)[0][0])
    refs={'CYC':(slice(iC2,i927)),'ON':(slice(iON,i927)),'PDHL':None}
    for rn in ('CYC','ON','PDHL'):
        if rn=='PDHL': rh,rl,reh,rel=RT.h[days[i-1]],RT.l[days[i-1]],np.nan,np.nan
        else: sl=refs[rn]; rh,rl,reh,rel=h[sl].max(),l[sl].min(),he[sl].max(),le[sl].min()
        if rn=='PDHL':
            # ES prior-day RTH extremes
            pe=by[days[i-1]]; pe=pe[(pe.m>=570)&(pe.m<960)]; reh,rel=pe.high_es.max(),pe.low_es.min()
        for wn,wend in (('STRICT',570),('BROAD',640)):
            done=False
            for j in range(i927,xend):
                if ms[j]>wend: break
                if done: break
                hh=h[i927:j].max() if j>i927 else -1e9; ll=l[i927:j].min() if j>i927 else 1e9
                for dr in (-1,1):
                    sw=(h[j]>rh and c[j]<rh and hh<=rh) if dr<0 else (l[j]<rl and c[j]>rl and ll>=rl)
                    if not sw: continue
                    ent=c[j]; stop=(h[j]+1) if dr<0 else (l[j]-1); risk=abs(ent-stop)
                    if risk<6 or risk>40: continue
                    ehh=he[i927:j+1].max(); ell=le[i927:j+1].min()
                    smt=(ehh<=reh) if dr<0 else (ell>=rel)
                    prem=(ent>o0 and ent>o8) if dr<0 else (ent<o0 and ent<o8); prem1=(ent>o0 or ent>o8) if dr<0 else (ent<o0 or ent<o8)
                    R.append(dict(d=D,ref=rn,win=wn,dr=dr,smt=smt,pd2=prem,pd1=prem1,pts=sim(h,l,c,j+1,ent,stop,dr,xend,ent+2*risk*dr),risk=risk)); done=True; break
T=pd.DataFrame(R); T['era']=np.where(T.d<'2022-01-01','A','B'); T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ts153.pkl')
def st(s):
    n=len(s); 
    return f'n{n:4d} {s.mean():+6.2f} t{(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0):+5.1f} w{(s>0).mean()*100:3.0f}%' if n else 'n   0'
flt={'ALL':lambda t:t.pts==t.pts,'SMT':lambda t:t.smt,'PD2':lambda t:t.pd2,'PD1':lambda t:t.pd1,'SMT+PD2':lambda t:t.smt&t.pd2,'SMT+PD1':lambda t:t.smt&t.pd1,'noSMT':lambda t:~t.smt}
print('A=2018-21 | B=2022+  (pts/trade, 1 NQ ct, cost .5)')
for (rn,wn),g in T.groupby(['ref','win']):
    for fn,f in flt.items():
        a,bb=g[(g.era=='A')&f(g)].pts,g[(g.era=='B')&f(g)].pts
        print(f'{rn:5s}{wn:7s}{fn:8s} A {st(a)} | B {st(bb)}')
    print()
