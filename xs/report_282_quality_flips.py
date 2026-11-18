"""Report 282 (declared before results). Rey: instead of raw vector flips on other assets, only take HIGH-QUALITY flips (volatility, orderflow, trend alignment) and, across assets, the best trade each day. No GEX for DIA/GLD/SPY/IWM (only QQQ options exist), so quality = price/volume features only. Assets (15m RTH, vector parameters UNCHANGED): NQ (resampled from 5m), QQQ, SPY, DIA, IWM, GLD, SLV, USO, TLT. Trade = vector regime flip bar i (regime changes to +1 / -1), enter next bar open in the new direction, exit at the next opposite flip's next open or 15:45, same session only; cost 2 bps round trip. Pre-declared quality flags (no fitting), at the flip bar: F1 flow (est delta over bars i-2..i has the flip's sign), F2 ATRP (ATR14/price in the top half of the asset's trailing 250 sessions), F3 participation (flip-bar volume >= 1.3x prior-12-bar mean), F4 daily trend alignment (flip direction equals sign(prior daily close - its 20-day mean)). Quality score = F1+F2+F3+F4. Cells: all flips, score>=3, score==4, score<=1, by asset and pooled, train 2019-21 | test 2018+2022-26, mean % per trade, day-clustered t. Then DAILY BEST-OF: each day take only the highest-score flip across assets (ties by lower ATRP-independent order: first by time), one trade per day. Pass: score>=3 positive in both splits pooled with test |t|>=2 AND positive in >=5 of 9 assets; else no multi-asset edge from this recipe."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'/home/claude/fartpie/xs')
from vector_pine import run_state_machine
P='/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/'
def load(t):
    if t=='NQ':
        b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index().between_time('09:30','15:55')
        return b.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    f=glob.glob(P+t+'_15m_*.parquet')[0]; d=pd.read_parquet(f); d['datetime']=pd.to_datetime(d.datetime).dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None); return d.set_index('datetime').sort_index().between_time('09:30','15:45')[['open','high','low','close','volume']]
rows=[]
for t in ('NQ','QQQ','SPY','DIA','IWM','GLD','SLV','USO','TLT'):
    d=load(t); d=d[~d.index.duplicated()]; reg=np.asarray(run_state_machine(d.assign(datetime=d.index))[0]); d['reg']=reg
    c=d.close.values;h=d.high.values;l=d.low.values;o=d.open.values;v=d.volume.values.astype(float); idx=d.index; day=idx.normalize(); mm=idx.hour*60+idx.minute
    dl=v*(2*c-h-l)/np.maximum(h-l,1e-12); tr=np.maximum(h-l,np.maximum(abs(h-np.r_[c[0],c[:-1]]),abs(l-np.r_[c[0],c[:-1]]))); atr=pd.Series(tr).rolling(14).mean().values/c
    dcl=pd.Series(c,index=idx).groupby(day).last(); m20=dcl.rolling(20).mean().shift(1); dprev=dcl.shift(1)
    atrd=pd.Series(atr,index=idx).groupby(day).mean(); rk=atrd.rolling(250,min_periods=100).rank(pct=True).shift(1)
    flips=np.where((np.diff(reg)!=0)&(reg[1:]!=0))[0]+1; flipi=list(flips)
    for n_,i in enumerate(flipi):
        if i+2>=len(c) or day[i]!=day[min(i+1,len(c)-1)] or mm[i]<585 or mm[i]>=900: continue
        dr=int(reg[i]); e=o[i+1]
        j=len(c)-1
        for q in range(i+1,len(c)):
            if day[q]!=day[i]: j=q-1; break
            if q>i+1 and reg[q]!=reg[q-1] and reg[q]!=dr: j=min(q,len(c)-1); break
            if mm[q]>=945: j=q; break
        x=o[j+1] if j+1<len(c) and day[j+1]==day[i] and reg[j]!=dr and False else c[j]
        r=dr*(x/e-1)-2e-4
        d0=day[i]; f1=dr*dl[max(0,i-2):i+1].sum()>0; f2=bool(rk.get(d0,np.nan)>=.5); pm=v[max(0,i-12):i].mean() if i>0 else np.nan; f3=bool(pm>0 and v[i]>=1.3*pm); f4=bool(np.isfinite(m20.get(d0,np.nan)) and dr*(dprev[d0]-m20[d0])>0)
        rows.append((t,d0,mm[i],dr,int(f1)+int(f2)+int(f3)+int(f4),r))
E=pd.DataFrame(rows,columns=['t','d','m','dr','s','r']); E['sp']=np.where((E.d.dt.year>=2019)&(E.d.dt.year<=2021),'train',np.where((E.d.dt.year==2018)|(E.d.dt.year>=2022),'test','x'))
print('flips',len(E),E.groupby('t').size().to_dict())
def cell(m):
    s=[]
    for sp in ('train','test'):
        z=E[m&(E.sp==sp)]
        if len(z)<15: s.append(f'n{len(z):5d} -'); continue
        ds=z.groupby('d').r.agg(['sum','size']); mean=z.r.mean(); se=np.sqrt(((ds['sum']-mean*ds['size'])**2).sum())/len(z); s.append(f'n{len(z):5d} {mean*100:+.3f}% (t{mean/se:+.1f})')
    return ' | '.join(s)
print('\nPOOLED (all 9 assets)')
for nm,m in (('all flips',E.s>=0),('score>=3',E.s>=3),('score==4',E.s==4),('score<=1',E.s<=1)): print(f' {nm:10s}',cell(m))
print('\nBY ASSET: score>=3 vs all flips')
pos=0
for t in E.t.unique():
    a=E.t==t; print(f' {t:4s} all {cell(a)}\n      q3  {cell(a&(E.s>=3))}')
print('\nDAILY BEST-OF across assets (highest score, first by time), one trade per day')
B=E[E.s>=3].sort_values(['d','s','m'],ascending=[True,False,True]).groupby('d').head(1)
E2=E; E=B
print(' best-of (score>=3 only):',cell(E.s>=0)); E=E2
