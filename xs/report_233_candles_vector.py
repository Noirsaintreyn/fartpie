"""Report 233: candle patterns paired with vector flips (Rey Nov 17: 'candle patterns before, pair some with vector'). DECLARED before results; 7 patterns, each with a stated mechanism, on 15m and 1H flips, no tuning.
Features on the flip bar i (long; mirror for short) and the bars before it, from the SAME timeframe as the flip:
 CLOSE  close location in the top quarter of the bar (displacement: buyers closed it)            | BODY  body >= 1.5 x mean body of the prior 10 bars (impulse)
 ENGULF flip-bar body engulfs the prior bar's body (absorption then reversal)                    | COMP  mean range of the 3 bars before the flip < 0.7 x ATR14 (compression -> expansion)
 REJ    prior bar has a >= 50% wick on the flip side (a rejection just before the flip)         | PB3   the 3 bars before the flip all closed against it (a clean pullback, then the vector turns)
 GAPLVL flip bar opens on the wrong side of the OLD vector level and closes through it > 0.5 ATR (a decisive cross, not a poke)
Test: same entries/exits as 229-232 (entry = next 5m open, stop = vector level, FIX 1R and BE@.5R tgt 1R, 1 pt cost). For each pattern: net pts per trade when TRUE vs FALSE, train 2019-21 | test 2018+2022-26, Welch t of the difference; a pattern helps only if TRUE > FALSE in both eras with t(diff) >= 2 in test and the same sign in train. 7 patterns x 2 timeframes x 2 exits -> read t ~3 as the bar."""
import sys; src=open('/home/claude/fartpie/xs/report_232_vx_exits.py').read(); exec(src[:src.index("MODES=")])
def feats(tf):
    gg=b.set_index('dt').resample(tf,label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    r_=run_state_machine(gg.assign(datetime=gg.index)); rg=np.asarray(r_[0]); lv=np.asarray(r_[1],float); en=gg.index.values
    O,H,L,C=[gg[x].values for x in ('open','high','low','close')]; p=pd.Series(C).shift(1); trr=np.maximum(H-L,np.maximum(np.abs(H-p),np.abs(L-p))); at=pd.Series(trr).rolling(14).mean().values; body=np.abs(C-O); bm=pd.Series(body).shift(1).rolling(10).mean().values
    rows=[]
    for i in range(14,len(gg)):
        s=1 if (rg[i]==1 and rg[i-1]!=1) else -1 if (rg[i]==-1 and rg[i-1]!=-1) else 0
        if s==0 or not np.isfinite(lv[i]): continue
        e=np.searchsorted(t5,en[i])
        if e>=n or t5[e]!=en[i]: continue
        R=abs(o[e]-lv[i])
        if R<3: continue
        rng=max(H[i]-L[i],1e-9); clv=(C[i]-L[i])/rng; clv=clv if s==1 else 1-clv
        pr=max(H[i-1]-L[i-1],1e-9); wick=((min(O[i-1],C[i-1])-L[i-1]) if s==1 else (H[i-1]-max(O[i-1],C[i-1])))/pr
        eng=(s==1 and C[i]>=max(O[i-1],C[i-1]) and O[i]<=min(O[i-1],C[i-1])) or (s==-1 and C[i]<=min(O[i-1],C[i-1]) and O[i]>=max(O[i-1],C[i-1]))
        old=lv[i-1]; thr=.5*at[i]
        gap=np.isfinite(old) and ((s==1 and C[i]-old>thr) or (s==-1 and old-C[i]>thr))
        rows.append(dict(e=e,s=s,R=R,CLOSE=clv>=.75,BODY=body[i]>=1.5*bm[i],ENGULF=bool(eng),COMP=np.mean(trr[i-3:i])<.7*at[i],REJ=wick>=.5,PB3=bool(all((C[i-k]<O[i-k]) if s==1 else (C[i-k]>O[i-k]) for k in (1,2,3))),GAPLVL=bool(gap)))
    return pd.DataFrame(rows)
PATS=['CLOSE','BODY','ENGULF','COMP','REJ','PB3','GAPLVL']
def welch(a,b_):
    if len(a)<8 or len(b_)<8: return np.nan
    return (a.mean()-b_.mean())/np.sqrt(a.var(ddof=1)/len(a)+b_.var(ddof=1)/len(b_))
for tf in ('60min','15min'):
    F=feats(tf)
    for xn,mode,par in (('FIX 1R','fix',1.0),('BE@.5R tgt 1R','be',(.5,1.0))):
        out=[]; last=-1
        for k_,r_ in F.iterrows():
            e=int(r_.e)
            if e<=last or e>=n-2 or not (575<=m[e]<=930) or day[e]<np.datetime64('2018-01-01'): continue
            gp,mfe,j=trade(e,int(r_.s),r_.R,mode,par); out.append(dict(day=pd.Timestamp(day[e]),pts=gp-1.0,**{p:bool(r_[p]) for p in PATS})); last=j
        D=pd.DataFrame(out); y=D.day.dt.year; eras={'train':(y>=2019)&(y<=2021),'test':(y==2018)|(y>=2022)}
        print(f'\n=== {tf} flips, exit {xn}: all train n{eras["train"].sum()} net {D[eras["train"]].pts.mean():+.1f} | test n{eras["test"].sum()} net {D[eras["test"]].pts.mean():+.1f} ===')
        for p in PATS:
            cells=[]
            for era,mk in eras.items():
                x=D[mk]; T_=x[x[p]].pts; F_=x[~x[p]].pts
                cells.append(f'{era}: TRUE n{len(T_):4d} {T_.mean():+6.1f} | FALSE n{len(F_):4d} {F_.mean():+6.1f} | diff t{welch(T_,F_):+5.1f}')
            print(f'{p:7s}',' || '.join(cells),flush=True)
