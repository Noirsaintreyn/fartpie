"""Nov 14 2026: would 'stock above its 200 EMA' help the screener's vector flips? (a) WIDE: ~12k daily vector flips on S&P names (report_27 events): stock close above its own 200-day EMA (prior close, PIT) vs below; fwd 21/63d excess vs universe; strict eras + the five regimes. (b) 1h vector flip A on 9 LSE names (bp @8/16 bars, hold-to-bear): stock above its 200-bar 1h EMA, and above its 200-day EMA. PASS = above > below in BOTH eras, same sign."""
import sys, glob, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'/home/claude/fartpie/xs')
from vector_pine import run_state_machine
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
ERA={'18-21':('2018-01-01','2022-01-01'),'22+':('2022-01-01','2027-01-01')}; RGS=[('R1','2018-01-01','2020-03-01'),('R2','2020-03-01','2022-01-01'),('R3','2022-01-01','2023-04-01'),('R4','2023-04-01','2025-04-01'),('R5','2025-04-01','2027-01-01')]
rg=lambda t: next((n for n,a,b in RGS if pd.Timestamp(a)<=t<pd.Timestamp(b)),None)
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>5 and len(b)>5 else np.nan
O=pickle.load(open(SC+'flipval_ohlc.pkl','rb')); E=pd.read_pickle(SC+'flipval_events.pkl')
ema={t:(x.close>x.close.ewm(span=200,adjust=False).mean()).shift(1) for t,x in O.items()}
E['up']=[ema[t].get(d,np.nan) if t in ema else np.nan for t,d in zip(E.tk,E.d)]; E=E.dropna(subset=['up']); E['up']=E.up.astype(bool); E['rg']=[rg(t) for t in E.d]
print('(a) WIDE daily vector flips, stock above its 200-day EMA vs below; n=',len(E),'| above share',round(E.up.mean(),2))
for h in (21,63):
    print(f' fwd {h}d excess vs universe')
    for e,(a,b) in ERA.items():
        s=E[(E.d>=a)&(E.d<b)]; u=s[s.up][f'r{h}']; d=s[~s.up][f'r{h}']; print(f'   {e:6s} above {u.mean()*100:+.2f}% (n{len(u)}) below {d.mean()*100:+.2f}% (n{len(d)}) diff {(u.mean()-d.mean())*100:+.2f}% t {tt(u,d):+.1f}')
    print('   by regime above/below: '+'  '.join(f"{n} {E[(E.rg==n)&E.up][f'r{h}'].mean()*100:+.2f}/{E[(E.rg==n)&~E.up][f'r{h}'].mean()*100:+.2f}" for n,_,_ in RGS))
print('\n(b) 1h vector flip A, 9 LSE names (bp/trade)')
def load(f):
    d=pd.read_parquet(f); d['datetime']=pd.to_datetime(d.datetime); d=d.drop_duplicates('datetime').set_index('datetime').sort_index(); d.index=d.index.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None); return d[['open','high','low','close','volume']].astype(float).between_time('09:30','15:59')
rows=[]
for f in sorted(glob.glob('/home/claude/fartpie/lse_clean/*.parquet')):
    x=load(f); z=x.resample('60min',offset='30min').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna(subset=['close'])
    dc=x.close.groupby(x.index.normalize()).last(); dup=(dc>dc.ewm(span=200,adjust=False).mean()).shift(1)
    reg,lev,_,_=run_state_machine(z.assign(datetime=z.index)); reg=np.asarray(reg); lb=z.low.rolling(20).min().rolling(20).max().values; c=z.close.values; o=z.open.values; e1=z.close.ewm(span=200,adjust=False).mean().values; n=len(z)
    for i in range(300,n-17):
        if reg[i]==1 and reg[i-1]!=1 and c[i]>lb[i]:
            en=o[i+1]; xx=i+1
            while xx<n-1 and reg[xx]==1 and c[xx]>=lev[xx]: xx+=1
            rows.append(dict(t=z.index[i],up1h=c[i]>e1[i],upd=dup.get(z.index[i].normalize(),np.nan),r8=(c[i+8]/en-1)*1e4,r16=(c[i+16]/en-1)*1e4,nat=(c[xx]/en-1)*1e4))
P=pd.DataFrame(rows).dropna(subset=['upd']); P['upd']=P.upd.astype(bool)
for col,lab in (('up1h','above 200-bar 1h EMA'),('upd','above 200-day EMA')):
    print(f' {lab}')
    for ex in ('r8','r16','nat'):
        line=f'   exit {ex:3s}'
        for e,(a,b) in ERA.items():
            s=P[(P.t>=a)&(P.t<b)]; u=s[s[col]][ex]; d=s[~s[col]][ex]; line+=f' | {e}: above {u.mean():+7.1f} (n{len(u)}) below {d.mean():+7.1f} (n{len(d)}) t {tt(u,d):+.1f}'
        print(line)
