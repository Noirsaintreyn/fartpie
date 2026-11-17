"""Report 238: Rey Nov 17: if order flow works with the vector, order flow into negative gamma with the vector should work; if it filters too much, go to lower timeframes. DECLARED before results.
Features (report 237 definitions): ABSORB = cumulative estimated delta of the 6 bars before the flip is against the flip; IVHI = prior-close QQQ ATM IV >= its trailing-252 median; NEG = prior-close QQQ net GEX < 0.
Cells on 1H flips: ABSORB, IVHI, NEG singles; ABSORB&NEG, ABSORB&IVHI, NEG&IVHI pairs; ABSORB&NEG&IVHI. Then, because the combos shrink n, ABSORB and ABSORB&NEG on 30m, 15m and 5m flips (the 6-bar look-back scales with the timeframe).
Exit FIX 1R (stop = vector level, target = same distance), entry next 5m open, 1 pt cost, one trade at a time. Net pts per trade, day-clustered t vs 0, per era, trades per year; PASS = same sign both eras, n >= 30 each, t >= 2 test; 7 + 6 cells so ~3 is the bar. Train 2019-21 | test 2018+2022-26."""
import sys; src=open('/home/claude/fartpie/xs/report_237_orderflow_options.py').read(); exec(src[:src.index("OF=['DLT'")])
def tstat(x):
    d=x.groupby('day').pts.sum(); return d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>5 else np.nan
def neg_at(d):
    k=G.index.searchsorted(d)-1; return bool(G.iloc[k]<0) if k>=0 else False
def build(tf,L5):
    F_=fe(tf,L5); out=[]; last=-1
    for k_,r_ in F_.iterrows():
        e=int(r_.e)
        if e<=last or e>=n-2 or not (575<=m[e]<=930) or day[e]<np.datetime64('2018-01-01'): continue
        gp,mfe,j=trade(e,int(r_.s),r_.R,'fix',1.0); d=pd.Timestamp(day[e]); out.append(dict(day=d,pts=gp-1.0,ABS=bool(r_.ABSORB),IV=bool(r_.get('IVHI',False)),NEG=neg_at(d))); last=j
    return pd.DataFrame(out)
def cell(D,nm,mask):
    y=D.day.dt.year; res=[]
    for era,mk in (('train',(y>=2019)&(y<=2021)),('test',(y==2018)|(y>=2022))):
        x=D[mk&mask]; yrs=3 if era=='train' else 6.5
        res.append(f'{era}: n{len(x):4d} ({len(x)/yrs:4.1f}/yr) win{np.mean(x.pts>0)*100:3.0f}% net {x.pts.mean():+6.1f} t{tstat(x):+5.1f}' if len(x)>5 else f'{era}: n{len(x)}')
    print(f'{nm:24s}',' || '.join(res),flush=True)
D=build('60min',12); print('=== 1H flips ===')
cell(D,'ALL',D.pts==D.pts)
for nm,mk in (('ABSORB',D.ABS),('IVHI',D.IV),('NEG',D.NEG),('ABSORB & NEG',D.ABS&D.NEG),('ABSORB & IVHI',D.ABS&D.IV),('NEG & IVHI',D.NEG&D.IV),('ABSORB & NEG & IVHI',D.ABS&D.NEG&D.IV)): cell(D,nm,mk)
for tf,L5 in (('30min',6),('15min',3),('5min',1)):
    D=build(tf,L5); print(f'\n=== {tf} flips ===')
    cell(D,'ALL',D.pts==D.pts); cell(D,'ABSORB',D.ABS); cell(D,'NEG',D.NEG); cell(D,'ABSORB & NEG',D.ABS&D.NEG); cell(D,'NOT ABSORB & NEG',(~D.ABS)&D.NEG)
