"""Maximin pivot on NQ legs. Longs: LB=highest(lowest(low,L),L) (rising support). Shorts: mirror LBs=lowest(highest(high,L),L) (falling resistance). Feature = dir*(close_prev-LB)/ATR(14) on 5m, 15m, 1h, 1d (completed bars only). Positive = price on the trade's side of the pivot."""
import pickle,numpy as np,pandas as pd
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
r=pickle.load(open(SC+'mix_tiers.pkl','rb'))['rows'].reset_index(drop=True); print(r.leg.value_counts().to_dict())
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').reset_index(drop=True)
b=b[(b.dt.dt.hour*60+b.dt.dt.minute>=570)&(b.dt.dt.hour*60+b.dt.dt.minute<960)]
def tf(bars,rule):
    x=bars.set_index('dt').resample(rule,origin='start_day',offset='30min' if rule!='1D' else '0min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    return x
def feats(x,L,minutes):
    lbL=x.low.rolling(L).min().rolling(L).max(); lbS=x.high.rolling(L).max().rolling(L).min()
    tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean()
    return pd.DataFrame({'known':x.index+pd.Timedelta(minutes=minutes),'lbL':lbL.values,'lbS':lbS.values,'cl':x.close.values,'atr':atr.values}).dropna()
TFS={'5m':('5min',5),'15m':('15min',15),'1h':('60min',60),'1d':('1D',16*60)}
r['t']=pd.to_datetime(r.day)+pd.to_timedelta(r.m,unit='m'); r=r.sort_values('t'); out={}
for L in (10,20):
    for nm,(rule,mn) in TFS.items():
        S=feats(tf(b,rule),L,mn).sort_values('known')
        x=pd.merge_asof(r[['t']],S,left_on='t',right_on='known',direction='backward'); x.index=r.index
        lb=np.where(r.dir>0,x.lbL,x.lbS); out[f'{nm}_{L}']=r.dir*(x.cl-lb)/x.atr
r=r.join(pd.DataFrame(out)).dropna(); print(len(r),'trades with features')
def row(x): return f'n {len(x):4d} {x.pts.mean():+6.1f}pt win {(x.pts>0).mean():.0%}' if len(x)>=25 else f'n {len(x)}'
for L in (10,20):
  for nm in TFS:
    c=f'{nm}_{L}'; print(f'== {nm} L={L}: aligned-distance (below pivot / 0-0.5 ATR / 0.5-1.5 / >1.5 ATR on trade side)')
    for e in ('A','B'):
        z=r[r.era==e]; q=[z[z[c]<0],z[(z[c]>=0)&(z[c]<.5)],z[(z[c]>=.5)&(z[c]<1.5)],z[z[c]>=1.5]]; print(f'   era {e}: '+' | '.join(row(y) for y in q))
print('=== skip rule effect on total points per era (all legs pooled)')
for L in (10,20):
  for nm in TFS:
    c=f'{nm}_{L}'; s=[]
    for e in ('A','B'):
        z=r[r.era==e]; k=z[z[c]>=0]; s.append(f'{e}: keep {len(k)}/{len(z)} avg {k.pts.mean():+.1f} vs {z.pts.mean():+.1f} total {k.pts.sum():+.0f} vs {z.pts.sum():+.0f}')
    print(f' skip wrong-side {nm} L{L}: '+' | '.join(s))
for leg in r.leg.unique():
    z=r[r.leg==leg]; print(leg,len(z),'| 15m L20 wrong-side avg',[round(z[(z.era==e)&(z['15m_20']<0)].pts.mean(),1) for e in 'AB'],'right-side',[round(z[(z.era==e)&(z['15m_20']>=0)].pts.mean(),1) for e in 'AB'])
