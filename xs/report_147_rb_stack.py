"""report 147: REJECTION BLOCK AS A CONFLUENCE STACK on 1m / 5m / 15m NQ (Rey, Nov 15: 'if 1m RB has no edge at 50%, higher time frame should have more; it worked because Fibonacci aligned and it was mid range; ICT traders stack context').
DECLARED BEFORE RESULTS. RB candidate (long; short = mirror) on the chosen timeframe (bars TF): wick >= 60% of range, range >= 0.7 ATR14, low = lowest low of the last 15 bars (swept), close in top 40%; limit at the wick midpoint (valid 15 bars, 0.25 pt through); stop = wick low - 2; risk 4..40 (1m) / 6..60 (5m, 15m).
CONTEXT FLAGS (all known at the RB bar): DEALING RANGE = high/low of the previous 40 bars of the same timeframe.
  MID  : entry within 0.08 of the range midpoint (0.42-0.58) - Rey's '0.5 of current range'.
  FIB  : retracement depth (hi-ent)/(hi-lo) (longs; mirrored for shorts) within 0.04 of 0.5/0.618/0.705/0.786 (OTE + equilibrium).
  LOWRES target: TARGET = nearest UNTAKEN swing high (3-bar fractal on this timeframe, formed within the last 150 bars and not traded through since) above entry at >= 1.5R and <= 6R (shorts: swing low);
                 LOWRES = no other untaken pivot / PDH / ONH / Asia / London extreme between entry and that target ('low resistance liquidity': the pivot is likely to be taken).
Trade = the FIRST candidate of the day (09:35-14:00 ET) that satisfies the stack. Outcomes: 2R target and pivot-target (candidates without a pivot target are skipped for the pivot column). Flat 15:55, cost 0.5 pt, NQ pts, 1 contract.
A 2018-02..2021-12 | B 2022-01..2026-01. PASS = same sign, |t|>=2 both eras, n>=30 each. Stacks tested (8): base, MID, FIB, LOWRES, MID+FIB, MID+LOWRES, FIB+LOWRES, MID+FIB+LOWRES. Chance level of a PASS with 8x2 looks ~ 1."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
TF=sys.argv[1]; SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; COST=.5; RMAX=40 if TF=='1m' else 60
if TF=='1m':
    d=pd.read_csv('/home/claude/nqdrive/NQ_1m_ETH.csv',parse_dates=['timestamp']); d['dt']=d.timestamp-pd.Timedelta(hours=6); d=d[['dt','open','high','low','close']]
else:
    d=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet')[['dt','open','high','low','close']]
    if TF=='15m': d=d.set_index('dt').resample('15min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
d=d.sort_values('dt').drop_duplicates('dt').set_index('dt'); mm=d.index.hour*60+d.index.minute; days=sorted(set(d.index[(mm>=570)&(mm<960)].normalize()))
rt=d[(mm>=570)&(mm<960)]; RT=rt.groupby(rt.index.normalize()).agg(h=('high','max'),l=('low','min'))
def sim(h,l,c,f,ent,stop,dr,xend,tgt):
    for q in range(f,xend+1):
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -abs(ent-stop)-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): return abs(tgt-ent)-COST
    return (c[xend]-ent)*dr-COST
R=[]; FIBS=np.array([.5,.618,.705,.786])
for i,D in enumerate(days):
    if i<5 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30') or days[i-1] not in RT.index: continue
    x=d.loc[days[i-1]+pd.Timedelta(hours=18):D+pd.Timedelta(hours=15,minutes=56)]
    if len(x)<(80 if TF=='15m' else 250 if TF=='5m' else 800): continue
    t=x.index; m=(t.hour*60+t.minute).values; msx=np.where(m>=1080,m-1080,m+360); h,l,c,o=[x[k].values.astype(float) for k in ('high','low','close','open')]; n=len(x)
    tr=np.maximum.reduce([h-l,np.abs(h-np.r_[c[0],c[:-1]]),np.abs(l-np.r_[c[0],c[:-1]])]); atr=pd.Series(tr).rolling(14).mean().values
    i930=int(np.searchsorted(msx,930)); i03=int(np.searchsorted(msx,540)); i08=int(np.searchsorted(msx,840)); ie=int(np.searchsorted(msx,1320)); xend=n-1
    if i930>=n or i930<(30 if TF=='15m' else 45): continue
    ext=[RT.loc[days[i-1]].h,RT.loc[days[i-1]].l,h[:i930].max(),l[:i930].min(),h[:i03].max(),l[:i03].min(),h[i03:i08].max(),l[i03:i08].min()]
    # swing pivots (3-bar fractal), confirmed 3 bars later
    ph=np.array([k for k in range(3,n-3) if h[k]==h[k-3:k+4].max()]); pl=np.array([k for k in range(3,n-3) if l[k]==l[k-3:k+4].min()])
    for k in range(max(i930+3,41),min(ie,n-20)):
        rg=h[k]-l[k]
        if not(rg>=.7*atr[k] and rg>0): continue
        for dr in (1,-1):
            if dr>0: ok=(min(o[k],c[k])-l[k])>=.6*rg and c[k]>=l[k]+.6*rg and l[k]<=l[k-15:k].min()
            else: ok=(h[k]-max(o[k],c[k]))>=.6*rg and c[k]<=h[k]-.6*rg and h[k]>=h[k-15:k].max()
            if not ok: continue
            ent=(l[k]+min(o[k],c[k]))/2 if dr>0 else (h[k]+max(o[k],c[k]))/2; stop=l[k]-2 if dr>0 else h[k]+2; risk=abs(ent-stop)
            if risk<(4 if TF=='1m' else 6) or risk>RMAX: continue
            f=None
            for q in range(k+1,min(k+16,n)):
                if (l[q]<=ent-.25) if dr>0 else (h[q]>=ent+.25): f=q; break
            if f is None: continue
            hi,lo=h[k-40:k].max(),l[k-40:k].min()
            if hi<=lo: continue
            dep=(hi-ent)/(hi-lo) if dr>0 else (ent-lo)/(hi-lo); mid=.42<=dep<=.58; fib=bool(np.abs(dep-FIBS).min()<=.04)
            # untaken pivots known at k (confirmed: index <= k-3)
            if dr>0: pv=[h[p] for p in ph if k-150<=p<=k-3 and h[p]>ent and h[p+1:k+1].max()<=h[p]+1e-9]
            else: pv=[l[p] for p in pl if k-150<=p<=k-3 and l[p]<ent and l[p+1:k+1].min()>=l[p]-1e-9]
            lvl=[v for v in ext if (v>ent if dr>0 else v<ent)]
            allp=sorted(set(pv+lvl)) if dr>0 else sorted(set(pv+lvl),reverse=True)
            cand=[v for v in sorted(pv) if (v-ent)>=1.5*risk and (v-ent)<=6*risk] if dr>0 else [v for v in sorted(pv,reverse=True) if (ent-v)>=1.5*risk and (ent-v)<=6*risk]
            tg=cand[0] if cand else np.nan; low=False
            if tg==tg: low=not any((ent<v<tg-.25*risk) if dr>0 else (tg+.25*risk<v<ent) for v in allp)
            R.append(dict(d=D,dr=dr,pts2=sim(h,l,c,f,ent,stop,dr,xend,ent+2*risk*dr),ptsP=(sim(h,l,c,f,ent,stop,dr,xend,tg) if tg==tg else np.nan),mid=mid,fib=fib,low=low,hasP=tg==tg,risk=risk,k=k))
T=pd.DataFrame(R); T['A']=T.d<'2022-01-01'; T.to_pickle(SC+f'ext/rb_stack_{TF}.pkl'); print(TF,len(T),'candidates on',T.d.nunique(),'days')
def t_(a): return a.mean()/(a.std()/np.sqrt(len(a))) if len(a)>8 else np.nan
def run(lab,m,col):
    S_=T[m&T[col].notna()].sort_values(['d','k']).groupby('d').head(1)
    a,b=S_[S_.A][col],S_[~S_.A][col]; ps=len(a)>=30 and len(b)>=30 and np.sign(a.mean())==np.sign(b.mean()) and abs(t_(a))>=2 and abs(t_(b))>=2
    return f'  {lab:22s} A n{len(a):4d} {a.mean():+6.2f} win{100*(a>0).mean():3.0f}% t{t_(a):+5.1f} | B n{len(b):4d} {b.mean():+6.2f} win{100*(b>0).mean():3.0f}% t{t_(b):+5.1f}{"  <== PASS" if ps else ""}'
Z=T.pts2.notna()
for col,lab in (('pts2','2R target'),('ptsP','pivot target')):
    print('\n==',TF,lab)
    for nm,m in (('base',Z),('MID',T.mid),('FIB',T.fib),('LOWRES',T.low),('MID+FIB',T.mid&T.fib),('MID+LOWRES',T.mid&T.low),('FIB+LOWRES',T.fib&T.low),('MID+FIB+LOWRES',T.mid&T.fib&T.low)): print(run(nm,m,col))
