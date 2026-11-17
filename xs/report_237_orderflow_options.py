"""Report 237: order-flow proxies and options Greeks as filters on vector flips (Rey Nov 17: 'try a few things with orderflow and options data'). EXPLORATORY, features declared before results; flips on 1H and 15m NQ; exit FIX 1R (stop = vector level, target = same distance), entry next 5m open, cost 1 pt, one trade at a time.
ORDER-FLOW PROXIES (no tick data before Dec 2024, so signed volume is estimated from 5m bars: delta = volume x (2 close - high - low)/(high - low)): DLT agree = delta over the flip bar has the flip's sign; DLTS strong = |delta|/volume in the top third of 2019-21 flips; RVOL = flip-bar volume >= 1.5 x its 20-bar average; ABSORB = cumulative delta of the 6 bars before the flip is AGAINST the flip (the move reverses against the crowd's flow).
OPTIONS (QQQ daily aggregates at the prior close, read in the trade direction as in report dte_greeks): DEX agree (sign of net dealer delta = flip side), VANNA flow agree (sign(net vanna) x -sign(5d IV change) = flip side), CHARM agree, ROOM >= 0.5 ADR to the wall the trade runs into (call wall for longs, put wall for shorts; walls end 2025-12), IV high (IV >= its trailing-252 median), SKEW high, PUTCALL volume high.
Read: TRUE vs FALSE net pts per era, Welch t of the difference; a feature helps only if TRUE > FALSE in both eras and t(diff) >= 2 in test; 11 features x 2 timeframes -> ~3 is the real bar. Train 2019-21 | test 2018 + 2022-26."""
import sys; src=open('/home/claude/fartpie/xs/report_232_vx_exits.py').read(); exec(src[:src.index("MODES=")])
dsrc=open('/home/claude/fartpie/xs/dte_greeks.py').read(); seg=dsrc[dsrc.index("U = '/mnt"):dsrc.index("DAYF =")].replace("f'{U}/xs/ext/gex/QQQ.csv'","'/home/claude/fartpie/xs/ext/gex/QQQ.csv'"); exec(seg)
Gm=pd.read_csv('/home/claude/fartpie/xs/ext/gex/QQQ.csv',parse_dates=['date']); 
a14=pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/fv5_neg_trades.pkl') if False else None
rth=(m>=570)&(m<=955); rg5=pd.Series(np.where(rth,h-l,np.nan)).groupby(day).transform('max')
DR=pd.DataFrame(dict(day=day[rth],hi=h[rth],lo=l[rth])).groupby('day').agg(hi=('hi','max'),lo=('lo','min')); ATR14=(DR.hi-DR.lo).rolling(14).mean().shift(1)
rng=np.maximum(h-l,1e-9); delta=v*(2*c-h-l)/rng; cumd=np.cumsum(delta); cumv=np.cumsum(v)
def fe(tf,L5):
    gg=b.set_index('dt').resample(tf,label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    r_=run_state_machine(gg.assign(datetime=gg.index)); rg=np.asarray(r_[0]); lv=np.asarray(r_[1],float); en=gg.index.values; rows=[]
    vs=pd.Series(v).rolling(L5).sum().values; vavg=pd.Series(vs).rolling(L5*20).mean().values
    for i in range(14,len(gg)):
        s=1 if (rg[i]==1 and rg[i-1]!=1) else -1 if (rg[i]==-1 and rg[i-1]!=-1) else 0
        if s==0 or not np.isfinite(lv[i]): continue
        e=np.searchsorted(t5,en[i])
        if e>=n or t5[e]!=en[i] or e<L5*8: continue
        R=abs(o[e]-lv[i])
        if R<3: continue
        dflip=cumd[e-1]-cumd[e-1-L5]; vf=cumv[e-1]-cumv[e-1-L5]; dpre=cumd[e-1-L5]-cumd[e-1-7*L5]
        d=pd.Timestamp(day[e]); f=prior(d); at=ATR14.get(d,np.nan)
        row=dict(e=e,s=s,R=R,DLT=np.sign(dflip)==s,strength=abs(dflip)/max(vf,1),RVOL=(vs[e-1]/vavg[e-1]>=1.5) if np.isfinite(vavg[e-1]) else False,ABSORB=np.sign(dpre)==-s)
        if f is not None:
            row.update(DEX=np.sign(f.dex)==s if np.isfinite(f.dex) else False,VANNA=(f.vflow==s) if np.isfinite(f.vflow) else False,CHARM=np.sign(f.net_charm)==s if np.isfinite(f.net_charm) else False,
              ROOM=(((f.call_wall_nq-o[e]) if s==1 else (o[e]-f.put_wall_nq))/at>=.5) if np.isfinite(f.call_wall_nq) and np.isfinite(at) else False,
              IVHI=bool(f.ivp>=.5) if np.isfinite(f.ivp) else False,SKEWHI=bool(f.skewp>=.5) if np.isfinite(f.skewp) else False,PCVHI=bool(f.pcvp>=.5) if np.isfinite(f.pcvp) else False)
        rows.append(row)
    return pd.DataFrame(rows).fillna(False)
OF=['DLT','DLTS','RVOL','ABSORB']; OP=['DEX','VANNA','CHARM','ROOM','IVHI','SKEWHI','PCVHI']
def welch(a,b_): return np.nan if len(a)<8 or len(b_)<8 else (a.mean()-b_.mean())/np.sqrt(a.var(ddof=1)/len(a)+b_.var(ddof=1)/len(b_))
for tf,L5 in (('60min',12),('15min',3)):
    F_=fe(tf,L5); cut=F_[(pd.to_datetime(day[F_.e.values]).year>=2019)&(pd.to_datetime(day[F_.e.values]).year<=2021)].strength.quantile(2/3); F_['DLTS']=F_.strength>=cut
    out=[]; last=-1
    for k_,r_ in F_.iterrows():
        e=int(r_.e)
        if e<=last or e>=n-2 or not (575<=m[e]<=930) or day[e]<np.datetime64('2018-01-01'): continue
        gp,mfe,j=trade(e,int(r_.s),r_.R,'fix',1.0); out.append(dict(day=pd.Timestamp(day[e]),pts=gp-1.0,**{p:bool(r_[p]) for p in OF+OP})); last=j
    D=pd.DataFrame(out); y=D.day.dt.year; eras={'train':(y>=2019)&(y<=2021),'test':(y==2018)|(y>=2022)}
    print(f'\n=== {tf} flips FIX 1R: train n{eras["train"].sum()} net {D[eras["train"]].pts.mean():+.1f} | test n{eras["test"].sum()} net {D[eras["test"]].pts.mean():+.1f}  (options features need data: ROOM walls end 2025-12) ===')
    for p in OF+OP:
        cells=[]
        for era,mk in eras.items():
            x=D[mk]; T_=x[x[p]].pts; Fl=x[~x[p]].pts; cells.append(f'{era}: T n{len(T_):4d} {T_.mean():+6.1f} | F n{len(Fl):4d} {Fl.mean():+6.1f} | t{welch(T_,Fl):+5.1f}')
        print(f'{p:7s}',' || '.join(cells),flush=True)
