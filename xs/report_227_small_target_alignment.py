"""Report 227: small-target trades when several things align (Rey: 'squeeze 10 points ... when more than one thing aligns ... targeting an ATR distance instead of a fixed price'). DECLARED before results.
Data: NQ 5m bars (ET), 2018-01..2026-07. Context from higher timeframes, known at the 5m bar's close: V15 = vector state on 15m bars; V60 = vector state on 1H bars. 'Aligned' = V15 and V60 agree (both bull -> longs only, both bear -> shorts only); 'V15 only' = trade with the 15m state.
Triggers at a 5m close, RTH 09:35-15:25, entry at the next bar's open, one position at a time:
 T1 EMA pullback: the bar's low (high for shorts) touches the 5m EMA20 and the bar closes back on the trend side of it.
 T2 breakout: the 5m close is the highest (lowest) close of the last 6 bars.
 RND placebo: every 5m close in the same context (no trigger) = does the trigger add anything beyond the context.
 COIN placebo: T1 bars but direction by coin flip = does the context add anything.
Exits: F10/10 target 10 pts, stop 10; F10/20 target 10, stop 20; A1/1 target 1.0 x ATR14(5m), stop 1.0 x ATR; A1/2 target 1 x ATR, stop 2 x ATR; all with a 12-bar (1 hour) time stop at the close and flat by 15:55. Stop wins a bar that touches both. Cost 1.0 pt round trip. CVD is NOT tested here: signed-volume data covers only Dec 2024-Nov 2025 (no split possible).
Split: train 2018-21 | test 2022-26. PASS = mean net pts same sign in both eras, |t| >= 2 (day-clustered), n >= 30; 16 strategy cells -> Bonferroni |t| >= 3.0 noted. Also shown: win rate, trades per day, share of days with net >= +10 pts."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b=b[b.dt>='2017-06-01'].reset_index(drop=True)
def htf(rule):
    g=b.set_index('dt').resample(rule,label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(); reg=np.asarray(run_state_machine(g.assign(datetime=g.index))[0]); return pd.Series(reg,index=g.index)
v15=htf('15min'); v60=htf('60min'); ce=b.dt+pd.Timedelta(minutes=5)
b['V15']=v15.reindex(ce,method='ffill').values; b['V60']=v60.reindex(ce,method='ffill').values
c=b.close.values; h=b.high.values; l=b.low.values; o=b.open.values; pc=np.r_[c[0],c[:-1]]; tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); atr=pd.Series(tr).rolling(14).mean().values; ema=pd.Series(c).ewm(span=20,adjust=False).mean().values
m=(b.dt.dt.hour*60+b.dt.dt.minute).values; day=b.dt.dt.normalize().values; yr=b.dt.dt.year.values; n=len(b); rth=(m>=575)&(m<=925)&(yr>=2018)
hi6=pd.Series(c).rolling(6).max().values; lo6=pd.Series(c).rolling(6).min().values
t1L=(l<=ema)&(c>ema); t1S=(h>=ema)&(c<ema); t2L=c>=hi6; t2S=c<=lo6; rng=np.random.default_rng(4); coin=rng.integers(0,2,n)*2-1
def sig(trig,ctx):
    if ctx=='ALIGNED': L=(b.V15.values==1)&(b.V60.values==1); S=(b.V15.values==-1)&(b.V60.values==-1)
    else: L=b.V15.values==1; S=b.V15.values==-1
    if trig=='T1': d=np.where(L&t1L,1,np.where(S&t1S,-1,0))
    elif trig=='T2': d=np.where(L&t2L,1,np.where(S&t2S,-1,0))
    elif trig=='RND': d=np.where(L,1,np.where(S,-1,0))
    else: d=np.where((L&t1L)|(S&t1S),coin,0)
    return np.where(rth,d,0)
EX={'F10/10':(10,10,False),'F10/20':(10,20,False),'A1/1':(1,1,True),'A1/2':(1,2,True)}
def run(d,ex):
    T,S,isatr=EX[ex]; out=[]; i=0
    while i<n-14:
        if d[i]==0: i+=1; continue
        e=i+1; s=d[i]; pe=o[e]; tg=T*atr[i] if isatr else T; st=S*atr[i] if isatr else S
        if not np.isfinite(tg) or day[e]!=day[i]: i+=1; continue
        res=None
        for j in range(e,min(e+12,n)):
            adv=(pe-l[j]) if s==1 else (h[j]-pe); fav=(h[j]-pe) if s==1 else (pe-l[j])
            if adv>=st: res=-st; break
            if fav>=tg: res=tg; break
            if m[j]>=955 or day[j]!=day[e]: res=s*(c[j]-pe); break
        if res is None: j=min(e+11,n-1); res=s*(c[j]-pe)
        out.append((day[e],res-1.0)); i=j+1
    return pd.DataFrame(out,columns=['day','pts'])
def stats(R):
    out=[]
    for nm,mk in (('train',R.day<np.datetime64('2022-01-01')),('test',R.day>=np.datetime64('2022-01-01'))):
        x=R[mk]; dd=x.groupby('day').pts.sum(); t=dd.mean()/(dd.std(ddof=1)/np.sqrt(len(dd))) if len(dd)>5 else np.nan; nd=len(np.unique(day[rth&((yr<2022) if nm=='train' else (yr>=2022))]))
        out.append(f'{nm}: n{len(x):6d} ({len(x)/nd:4.1f}/day) win {np.mean(x.pts>0)*100:3.0f}% mean {x.pts.mean():+5.2f} pts t{t:+5.1f} | day sum mean {dd.sum()/nd:+5.1f} pts, days >= +10 pts {np.sum(dd>=10)/nd*100:3.0f}%')
    return ' || '.join(out)
for ctx in ('ALIGNED','V15 only'):
    for trig in ('T1','T2','RND','COIN'):
        d=sig(trig,ctx)
        for ex in EX: print(f'{ctx:9s}{trig:5s}{ex:7s}',stats(run(d,ex)),flush=True)
