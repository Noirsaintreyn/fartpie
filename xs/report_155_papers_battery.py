"""report 155: BATTERY of everything in Rey's 3 PDFs not yet tested. DECLARED BEFORE RESULTS; each rule fixed here. NQ 5m + ES 5m; 1 ct; cost .5; 2R unless stated; flat 15:55; risk 6..40 where a stop is structural; first signal/day/config. A <2022 | B 2022+. PASS = same sign, |t|>=2, n>=30 both.
GB_ON/GB_PD (Goldbach, my reading of the indicator): dealing range R = overnight (18:00-09:24) or prior-RTH-day high-low; extension levels lo-0.111R / hi+0.111R; first bar 09:25-10:40 that trades beyond the extension and closes back inside -> fade; stop = extreme +-1.
CONT_{CYC,ON,PDHL}: post-hoc inverse of report 153 (flagged): bar takes ref and CLOSES beyond it (first, window 09:25-10:40) -> trade WITH it; stop = ref -+1 (needs risk 6..40).
SMTC: cycle-3 SMT: NQ extends past cycle-2 (08:15-09:24) high/low in 09:25-10:40 while ES (since 09:25) has not taken its own cycle-2 extreme -> fade at close, stop = extreme +-1.
OPN: IFVG (ext/uni_ifvg.pkl) split by price vs 00:00 and 08:30 opens: DISC2 (long below both / short above both), DISC1 (either), vs rest.
MOC: 14:00-14:55 range; CONSOL = range <= 33rd pct of 2018-21 days; BRK = first 15:00+ bar close beyond range -> with it, stop other side; FADE = bar pokes beyond and closes back inside -> fade, stop extreme+-1; 2R.
DRIFT: move 09:30 -> 10:00 (close of 09:55 bar) with |move| >= median |move| of 2018-21; WITH = trade the direction to 11:30; FADE = against, to 11:30.
GAPF: RTH gap 10-30 pts; trade toward fill at the 09:30 open; stop = |gap| beyond open; target prev close; flat 15:55 (also split 10-15 / 15-30)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; COST=.5; BL='2022-01-01'
nq=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); es=pd.read_parquet('/tmp/claude-0/ml/es5_real.parquet')[['dt','high','low','close']]
b=nq.merge(es,on='dt',suffixes=('','_es')).sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute; by=dict(tuple(b.groupby('d'))); days=sorted(by)
RT=b[(b.m>=570)&(b.m<960)].groupby('d').agg(h=('high','max'),l=('low','min'),c=('close','last'))
def sim(h,l,c,f,ent,stop,dr,xend,tgt):
    for q in range(f,xend+1):
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -abs(ent-stop)-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): return abs(tgt-ent)-COST
    return (c[xend]-ent)*dr-COST
R=[]; OP={}; D14={}
def add(cfg,D,pts): R.append((cfg,D,pts))
pend=[]  # MOC needs threshold from A
for i,D in enumerate(days):
    if i<2 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30') or days[i-1] not in RT.index: continue
    x=by[D]; x=x[x.m<=955]; p_=by[days[i-1]]; p_=p_[p_.m>=1080]; x=pd.concat([p_,x]).reset_index(drop=True); m=x.m.values; ms=np.where(m>=1080,m-1440,m)
    need=[0,495,510,565,570,595,685,840,900,955]
    if any((ms==t).sum()!=1 for t in need) or (m>=1080).sum()<20: continue
    h,l,c,o=[x[k].values.astype(float) for k in ('high','low','close','open')]; he,le=x.high_es.values.astype(float),x.low_es.values.astype(float)
    ix=lambda t:int(np.where(ms==t)[0][0]); i0,iC2,i830,i925,i930,i955,i1130,i1400,i1500,xend=[ix(t) for t in need]; iON=int(np.where(m>=1080)[0][0])
    OP[D]=(o[i0],o[i830]); win=range(i925,int(np.searchsorted(ms,641)))
    ONh,ONl=h[iON:i925].max(),l[iON:i925].min(); PDh,PDl=RT.h[days[i-1]],RT.l[days[i-1]]
    # Goldbach fades
    for nm,(lo,hi) in (('GB_ON',(ONl,ONh)),('GB_PD',(PDl,PDh))):
        Rg=hi-lo; eh,el=hi+.111*Rg,lo-.111*Rg
        for j in win:
            done=False
            for dr in (-1,1):
                if (h[j]>eh and c[j]<eh) if dr<0 else (l[j]<el and c[j]>el):
                    stop=(h[j]+1) if dr<0 else (l[j]-1); ent=c[j]; rk=abs(ent-stop)
                    if 6<=rk<=40: add(nm,D,sim(h,l,c,j+1,ent,stop,dr,xend,ent+2*rk*dr)); done=True; break
            if done: break
    # continuation with sweep
    refs={'CYC':(h[iC2:i925].max(),l[iC2:i925].min()),'ON':(ONh,ONl),'PDHL':(PDh,PDl)}
    for rn,(rh,rl) in refs.items():
        for wn,wend in (('S',570),('B',640)):
            done=False
            for j in win:
                if ms[j]>wend or done: break
                hh=h[i925:j].max() if j>i925 else -1e9; ll=l[i925:j].min() if j>i925 else 1e9
                for dr in (1,-1):
                    if (c[j]>rh and hh<=rh) if dr>0 else (c[j]<rl and ll>=rl):
                        ent=c[j]; stop=(rh-1) if dr>0 else (rl+1); rk=abs(ent-stop)
                        if 6<=rk<=40: add(f'CONT_{rn}_{wn}',D,sim(h,l,c,j+1,ent,stop,dr,xend,ent+2*rk*dr)); done=True; break
    # cycle-3 SMT
    rh,rl=h[iC2:i925].max(),l[iC2:i925].min(); reh,rel=he[iC2:i925].max(),le[iC2:i925].min(); done=False
    for j in win:
        if done: break
        hh=h[i925:j].max() if j>i925 else -1e9; ll=l[i925:j].min() if j>i925 else 1e9
        for dr in (-1,1):
            if (h[j]>rh and hh<=rh and he[i925:j+1].max()<=reh) if dr<0 else (l[j]<rl and ll>=rl and le[i925:j+1].min()>=rel):
                ent=c[j]; stop=(h[j]+1) if dr<0 else (l[j]-1); rk=abs(ent-stop)
                if 6<=rk<=40: add('SMTC',D,sim(h,l,c,j+1,ent,stop,dr,xend,ent+2*rk*dr)); done=True; break
    # MOC
    rg=(h[i1400:i1500].max(),l[i1400:i1500].min()); D14[D]=rg[0]-rg[1]
    pend.append((D,rg,i1500,xend,h,l,c))
    # drift
    mv=c[i955]-o[i930]; pend.append(('DR',D,mv,i955,i1130,c,xend))
    # gap fill
    pc=RT.c[days[i-1]]; gap=o[i930]-pc
    if 10<=abs(gap)<=30:
        dr=int(np.sign(pc-o[i930])); ent=o[i930]; rk=abs(gap); stop=ent-dr*rk
        cfg='GAPF_10_15' if abs(gap)<15 else 'GAPF_15_30'; add(cfg,D,sim(h,l,c,i930,ent,stop,dr,xend,pc)); add('GAPF_ALL',D,R[-1][2])
A_=[v for k,v in D14.items() if k<pd.Timestamp(BL)]; thr=np.percentile(A_,33)
mvA=np.median([abs(q[2]) for q in pend if q[0]=='DR' and q[1]<pd.Timestamp(BL)])
for q in pend:
    if q[0]=='DR':
        _,D,mv,i955,i1130,c,xend=q
        if abs(mv)>=mvA:
            ent=c[i955]; dr=int(np.sign(mv)); add('DRIFT_WITH',D,(c[i1130]-ent)*dr-COST); add('DRIFT_FADE',D,-(c[i1130]-ent)*dr-COST)
    else:
        D,(rh,rl),i1500,xend,h,l,c=q
        if rh-rl>thr: continue
        for kind in ('BRK','FADE'):
            for j in range(i1500,xend):
                if kind=='BRK' and (c[j]>rh or c[j]<rl):
                    dr=1 if c[j]>rh else -1; ent=c[j]; stop=rl if dr>0 else rh; rk=abs(ent-stop)
                    if 6<=rk<=40: add('MOC_BRK',D,sim(h,l,c,j+1,ent,stop,dr,xend,ent+2*rk*dr))
                    break
                if kind=='FADE' and ((h[j]>rh and c[j]<rh) or (l[j]<rl and c[j]>rl)):
                    dr=-1 if h[j]>rh and c[j]<rh else 1; ent=c[j]; stop=(h[j]+1) if dr<0 else (l[j]-1); rk=abs(ent-stop)
                    if 6<=rk<=40: add('MOC_FADE',D,sim(h,l,c,j+1,ent,stop,dr,xend,ent+2*rk*dr))
                    break
# OPN on IFVG
t=pd.DataFrame(pd.read_pickle(S+'ext/uni_ifvg.pkl')); t=t[t.model=='IFVG']; t['d']=pd.to_datetime(t.d)
def below(r,k): 
    o0,o8=OP.get(r.d,(np.nan,np.nan)); return (r.ent<o0,r.ent<o8)
for _,r in t.iterrows():
    if r.d not in OP: continue
    o0,o8=OP[r.d]; lo=(r.ent<o0,r.ent<o8); hi=(r.ent>o0,r.ent>o8); good2=all(lo) if r.dr>0 else all(hi); good1=any(lo) if r.dr>0 else any(hi)
    add('OPN_all',r.d,r.pts); add('OPN_DISC2' if good2 else 'OPN_notDISC2',r.d,r.pts); add('OPN_DISC1' if good1 else 'OPN_notDISC1',r.d,r.pts)
T=pd.DataFrame(R,columns=['cfg','d','pts']); T['d']=pd.to_datetime(T.d); T['era']=np.where(T.d<BL,'A','B'); T.to_pickle(S+'bat155.pkl')
def stt(s):
    n=len(s); t_=(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0); return n,s.mean(),t_,(s>0).mean()*100
print(f'MOC consolidation thr (33rd pct 2018-21 14:00-15:00 range) {thr:.1f} pts; drift median |move| {mvA:.1f}\nA=2018-21 | B=2022+ pts/trade')
for cfg,g in T.groupby('cfg',sort=False):
    a=stt(g[g.era=='A'].pts); bb=stt(g[g.era=='B'].pts)
    ps=(np.sign(a[1])==np.sign(bb[1]) and abs(a[2])>=2 and abs(bb[2])>=2 and a[0]>=30 and bb[0]>=30)
    print(f'{cfg:18s} A n{a[0]:4d} {a[1]:+6.2f} t{a[2]:+5.1f} w{a[3]:3.0f}% | B n{bb[0]:4d} {bb[1]:+6.2f} t{bb[2]:+5.1f} w{bb[3]:3.0f}%  {"PASS" if ps else ""}')
