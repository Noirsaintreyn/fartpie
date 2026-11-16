"""report 176: pair IFVG and the S1 premium fade with VWAP and CVD (Rey). DECLARED BEFORE RESULTS.
VWAP (full 2018-2026 from NQ 5m volume): IFVG DISC_VWAP = long below / short above the RTH VWAP (09:30 cumulative hlc3*vol) at the fill bar, vs the opposite (trend side). S1: PREM_ONVWAP = open above the overnight (18:00-09:29) VWAP (premium to overnight value), vs not.
CVD (only 2024-12-18 .. end of mnq_orderflow_1min.parquet, MNQ 1m signed volume, UTC -> ET): IFVG ALIGN30 = dir * (sum of signed volume over the 30 min before the fill) > 0 (buyers/sellers pushing with the trade), DIV = opposite. S1: ONCVD_UP = overnight (18:00-09:29) signed volume > 0 before a short (buyers already exhausted?) vs <=0. n is small: leads only.
Reports A 2018-21 | B 2022+ for VWAP, and the CVD window alone."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute; b['tp']=(b.high+b.low+b.close)/3*b.volume
rt=b[(b.m>=570)&(b.m<=955)].copy(); rt['cv']=rt.groupby('d').volume.cumsum(); rt['ct']=rt.groupby('d').tp.cumsum(); rt['vw']=rt.ct/rt.cv; V=rt.set_index(['d','m']).vw
I=pd.DataFrame(pd.read_pickle(S+'ext/uni_ifvg.pkl')); I=I[I.model=='IFVG'].copy(); I['d']=pd.to_datetime(I.d)
I['vw']=[V.get((r.d,int(r.m)//5*5),np.nan) for _,r in I.iterrows()]; I['disc']=(I.dr*(I.ent-I.vw)<0)
S1=pd.read_pickle(S+'s1_trades162.pkl').rename(columns={'day':'d'}); S1['d']=pd.to_datetime(S1.d)
on=b.assign(d2=(b.dt+pd.to_timedelta(np.where(b.m>=1080,1,0),unit='D')).dt.normalize())[lambda x:(x.m>=1080)|(x.m<570)]; ov=(on.groupby('d2').tp.sum()/on.groupby('d2').volume.sum()); op=rt.groupby('d').open.first()
S1['prem']=S1.d.map(op)>S1.d.map(ov)
def st(s):
    n=len(s); return f'n{n:4d} {s.mean():+7.2f} t{(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0):+5.1f} w{(s>0).mean()*100:3.0f}%' if n else 'n   0'
def row(lab,T,m,col='pts'):
    a,bb=T[(T.d<'2022-01-01')&m],T[(T.d>='2022-01-01')&m]; ok=len(a)>=30 and len(bb)>=30 and abs(a[col].mean()/(a[col].std()/np.sqrt(len(a))))>=2 and abs(bb[col].mean()/(bb[col].std()/np.sqrt(len(bb))))>=2 and np.sign(a[col].mean())==np.sign(bb[col].mean()); print(f'  {lab:22s} A {st(a[col])} | B {st(bb[col])} {"PASS" if ok else ""}')
print('VWAP'); print('IFVG:'); row('all',I,I.pts==I.pts); row('DISC (long<VWAP/short>VWAP)',I,I.disc); row('TREND side',I,~I.disc & I.vw.notna())
print('S1 premium fade:'); row('all',S1,S1.pts==S1.pts); row('open > overnight VWAP',S1,S1.prem); row('open <= overnight VWAP',S1,~S1.prem)
# CVD
f=pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet'); f['dt']=pd.to_datetime(f.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); f=f.set_index('dt').sort_index(); print('\nCVD window',f.index.min(),'->',f.index.max())
sv=f.signed_vol
rows=[]
for _,r in I[I.d>=f.index.min().normalize()].iterrows():
    t1=r.d+pd.Timedelta(minutes=int(r.m)); x=sv[t1-pd.Timedelta(minutes=30):t1]
    if len(x)>5: rows.append(dict(d=r.d,pts=r.pts,al=(r.dr*x.sum()>0),cv=r.dr*x.sum()))
C=pd.DataFrame(rows); print('IFVG CVD n',len(C)); 
for lab,m in (('all',C.pts==C.pts),('ALIGN30',C.al),('DIV',~C.al)): print(f'  {lab:9s} {st(C[m].pts)}')
rows=[]
for _,r in S1[S1.d>=f.index.min().normalize()].iterrows():
    x=sv[r.d-pd.Timedelta(hours=6):r.d+pd.Timedelta(hours=9,minutes=29)]
    if len(x)>50: rows.append(dict(d=r.d,pts=r.pts,up=x.sum()>0))
C=pd.DataFrame(rows); print('S1 CVD n',len(C))
for lab,m in (('all',C.pts==C.pts),('overnight CVD up',C.up),('overnight CVD down',~C.up)): print(f'  {lab:19s} {st(C[m].pts)}')
