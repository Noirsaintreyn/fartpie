"""report 163: RSI / EMA trend / TREND LINES on NQ. DECLARED BEFORE RESULTS; nothing tuned; strict split A 2018-21 | B 2022+, PASS = same sign, |t|>=2, n>=30 both.
A) CONTEXT on IFVG trades (ext/uni_ifvg.pkl, entry bar = fill bar): MOM = trade dir agrees with 5m RSI14 side of 50 (bar before fill); EXT = long with RSI<30 / short with RSI>70 (contrarian extreme); EMAAL = dir agrees with price vs 5m EMA200 AND EMA50 vs EMA200; DEMA = dir agrees with prior-day daily EMA20 vs EMA50; DRSI = dir agrees with prior-day daily RSI14 side of 50.
B) STANDALONE daily (next RTH open->close, cost .5): RSI2 (<10 long / >90 short, Connors), RSI14 70/30 as MOMENTUM (>70 long) and as CONTRARIAN (>70 short) both reported, EMA20/50 daily trend with-the-trend.
C) TREND LINES on 15m (pivot order 4, confirmed 4 bars later; ascending support from the last two pivot lows if rising and 2..130 bars apart, descending resistance from last two pivot highs if falling), traded on 5m bars 09:45-14:00, first per day per type, risk 6..40, 2R, flat 15:55, cost .5:
 BOUNCE: low<=line and close>line (short mirror on resistance), stop = extreme -+1.
 BREAK: close beyond the line by >=0.1*ATR5(14) after the prior close was on the right side -> trade the break; stop = max/min of last 3 bars +-1.
 BRKRT (advanced): after such a break, within 24 bars price retests the line from the other side (high>=line-1 for a break down) and closes back beyond it -> trade the break direction; stop = retest extreme +-1."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; COST=.5; BL=pd.Timestamp('2022-01-01')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
def wild(x,n): return x.ewm(alpha=1/n,adjust=False).mean()
def rsi(c,n): dlt=c.diff(); up=wild(dlt.clip(lower=0),n); dn=wild((-dlt).clip(lower=0),n); return 100-100/(1+up/dn.replace(0,np.nan))
b['rsi']=rsi(b.close,14); b['e50']=b.close.ewm(span=50,adjust=False).mean(); b['e200']=b.close.ewm(span=200,adjust=False).mean()
rt=b[(b.m>=570)&(b.m<=955)]; D=rt.groupby('d').agg(o=('open','first'),c=('close','last'),h=('high','max'),l=('low','min')); D['rsi2']=rsi(D.c,2); D['rsi14']=rsi(D.c,14); D['e20']=D.c.ewm(span=20,adjust=False).mean(); D['e50']=D.c.ewm(span=50,adjust=False).mean()
Dp=D.shift(1)
def st(s):
    n=len(s); 
    if n==0: return 'n   0'
    return f'n{n:4d} {s.mean():+6.2f} t{(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0):+5.1f} w{(s>0).mean()*100:3.0f}%'
def line(nm,df):
    a,bb=df[df.d<BL].pts,df[df.d>=BL].pts; ps=lambda s:len(s)>=30 and abs(s.mean()/(s.std()/np.sqrt(len(s))))>=2; ok=ps(a) and ps(bb) and np.sign(a.mean())==np.sign(bb.mean())
    print(f'{nm:22s} A {st(a)} | B {st(bb)} {"PASS" if ok else ""}')
# ---- A) IFVG context
t=pd.DataFrame(pd.read_pickle(S+'ext/uni_ifvg.pkl')); t=t[t.model=='IFVG'].copy(); t['d']=pd.to_datetime(t.d)
ix=pd.Series(np.arange(len(b)),index=b.dt.values); rs=[];e5=[];e2=[];dm=[];dr_=[]
for _,r in t.iterrows():
    ts=r.d+pd.Timedelta(minutes=int(r.m)); j=ix.get(ts,None)
    if j is None: rs.append(np.nan); e5.append(np.nan); continue
    rs.append(b.rsi.iloc[j-1]); e5.append(float(r.dr*(r.ent-b.e200.iloc[j-1])>0 and r.dr*(b.e50.iloc[j-1]-b.e200.iloc[j-1])>0))
t['rsi']=rs; t['emaal']=e5; t['mom']=(t.dr>0)==(t.rsi>50); t['ext']=((t.dr>0)&(t.rsi<30))|((t.dr<0)&(t.rsi>70))
t['dema']=[float(r.dr*(Dp.e20.get(r.d,np.nan)-Dp.e50.get(r.d,np.nan))>0) for _,r in t.iterrows()]; t['drsi']=[float((r.dr>0)==(Dp.rsi14.get(r.d,np.nan)>50)) for _,r in t.iterrows()]
print('A) IFVG context (pts/trade)'); line('IFVG all',t)
for nm,m in (('MOM',t.mom),('notMOM',~t.mom),('EXT',t.ext),('EMAAL',t.emaal==1),('notEMAAL',t.emaal==0),('DEMA',t.dema==1),('notDEMA',t.dema==0),('DRSI',t.drsi==1),('notDRSI',t.drsi==0)): line(nm,t[m&t.rsi.notna()])
# ---- B) daily standalone
print('\nB) daily standalone (next day open->close)')
oc=(D.c-D.o); X=pd.DataFrame(dict(d=D.index,oc=oc.values,r2=Dp.rsi2.values,r14=Dp.rsi14.values,tr=np.sign(Dp.e20-Dp.e50).values)); X=X[X.d>='2018-02-01']
def addp(nm,mask,sgn): s=X[mask].copy(); s['pts']=s.oc*sgn[mask]-COST; line(nm,s)
for nm,m,sg in (('RSI2<10 long',X.r2<10,1),('RSI2>90 short',X.r2>90,-1),('RSI14>70 long(mom)',X.r14>70,1),('RSI14>70 short(contra)',X.r14>70,-1),('RSI14<30 long(contra)',X.r14<30,1),('RSI14<30 short(mom)',X.r14<30,-1)):
    s=X[m].copy(); s['pts']=s.oc*sg-COST; line(nm,s)
s=X.copy(); s['pts']=s.oc*s.tr-COST; line('EMA20/50 with trend',s.dropna())
# ---- C) trend lines
b15=b.set_index('dt').resample('15min').agg(dict(open='first',high='max',low='min',close='last')).dropna(); h15,l15=b15.high.values,b15.low.values; idx15=b15.index
K=4
def pivots(a,hi):
    out=[]
    for i in range(K,len(a)-K):
        w=a[i-K:i+K+1]
        if (hi and a[i]==w.max() and a[i]>w[:K].max()) or ((not hi) and a[i]==w.min() and a[i]<w[:K].min()): out.append(i)
    return out
pH,pL=pivots(h15,True),pivots(l15,False)
# per 15m index: last two confirmed pivots at time i (confirmed at pivot+K)
def last2(pv,i): 
    k=np.searchsorted(pv,i-K,'right'); return (pv[k-2],pv[k-1]) if k>=2 else None
b['t15']=b.dt.dt.floor('15min'); pos15=pd.Series(np.arange(len(b15)),index=idx15); b['i15']=b.t15.map(pos15).values
tr5=np.maximum.reduce([b.high-b.low,(b.high-b.close.shift()).abs(),(b.low-b.close.shift()).abs()]); b['atr']=pd.Series(tr5).rolling(14).mean().values
by=dict(tuple(b.groupby('d'))); days=sorted(by); R=[]
def sim(h,l,c,f,ent,stop,dr,xend,tgt):
    for q in range(f,xend+1):
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -abs(ent-stop)-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): return abs(tgt-ent)-COST
    return (c[xend]-ent)*dr-COST
for D_ in days:
    if D_<pd.Timestamp('2018-02-01') or D_>pd.Timestamp('2026-01-30'): continue
    x=by[D_]; x=x[(x.m>=570)&(x.m<=955)].reset_index(drop=True)
    if len(x)<70 or x.m.iloc[-1]!=955: continue
    h,l,c,o,atr=[x[k].values.astype(float) for k in ('high','low','close','open','atr')]; i15=x.i15.values; xend=len(x)-1; mm=x.m.values
    done=set(); brk={}
    for j in range(1,xend):
        if mm[j]<585 or mm[j]>840 or np.isnan(i15[j]): continue
        ii=int(i15[j]); frac=((x.dt.iloc[j]-x.t15.iloc[j]).total_seconds()/900)
        for kind,pv,hi in (('SUP',pL,False),('RES',pH,True)):
            p2=last2(pv,ii)
            if p2 is None: continue
            a,bq=p2; ya,yb=(l15[a],l15[bq]) if not hi else (h15[a],h15[bq])
            if not (2<=bq-a<=130): continue
            if (kind=='SUP' and yb<=ya) or (kind=='RES' and yb>=ya): continue
            slope=(yb-ya)/(bq-a); ln=yb+slope*(ii+frac-bq); lp=yb+slope*(ii+frac-1-bq)  # line now / one bar before (approx)
            side_prev=c[j-1]-lp; side_now=c[j]-ln
            dr_b=1 if kind=='SUP' else -1   # bounce direction
            # BOUNCE
            key=('BOUNCE',kind)
            if key not in done and ((l[j]<=ln and c[j]>ln) if kind=='SUP' else (h[j]>=ln and c[j]<ln)) and abs(ln-(l[j] if kind=='SUP' else h[j]))<=5*max(atr[j],1):
                ent=c[j]; stop=(l[j]-1) if kind=='SUP' else (h[j]+1); rk=abs(ent-stop)
                if 6<=rk<=40: R.append(dict(d=D_,cfg='BOUNCE_'+kind,pts=sim(h,l,c,j+1,ent,stop,dr_b,xend,ent+2*rk*dr_b))); done.add(key)
            # BREAK
            thr=.1*atr[j]
            brk_now=(c[j]<ln-thr and side_prev>0) if kind=='SUP' else (c[j]>ln+thr and side_prev<0)
            key=('BREAK',kind)
            if brk_now and key not in done:
                dd=-dr_b; ent=c[j]; stop=(h[max(0,j-2):j+1].max()+1) if dd<0 else (l[max(0,j-2):j+1].min()-1); rk=abs(ent-stop)
                if 6<=rk<=40: R.append(dict(d=D_,cfg='BREAK_'+kind,pts=sim(h,l,c,j+1,ent,stop,dd,xend,ent+2*rk*dd))); done.add(key)
                brk[kind]=(j,dd,ln)
            # BREAK-RETEST
            if kind in brk and ('BRKRT',kind) not in done:
                j0,dd,l0=brk[kind]
                if 1<=j-j0<=24:
                    touch=(h[j]>=ln-1 and c[j]<ln) if dd<0 else (l[j]<=ln+1 and c[j]>ln)
                    if touch:
                        ent=c[j]; stop=(h[j]+1) if dd<0 else (l[j]-1); rk=abs(ent-stop)
                        if 6<=rk<=40: R.append(dict(d=D_,cfg='BRKRT_'+kind,pts=sim(h,l,c,j+1,ent,stop,dd,xend,ent+2*rk*dd))); done.add(('BRKRT',kind))
T=pd.DataFrame(R); T.to_pickle(S+'tl163.pkl'); print('\nC) trend lines on 15m pivots (pts/trade, 2R)')
for cfg in sorted(T.cfg.unique()): line(cfg,T[T.cfg==cfg])
for pre in ('BOUNCE','BREAK','BRKRT'): line(pre+'_ALL',T[T.cfg.str.startswith(pre+'_')].sort_values('d').drop_duplicates('d'))
