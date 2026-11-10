"""Nov 10 2026: CROSS-ASSET LEAD-LAG NQ vs ES + A VECTOR OVER THE NQ/ES RATIO. DECLARED BEFORE RESULTS.
(1) 5m/30m lead-lag IC: ES return minus NQ return (last 5m, 30m) vs NQ next-bar return, Spearman, 2014-21 vs 2022-26.
(2) pivot-ratchet vector (the screener's) on the 15m NQ/ES ratio (known at each 15m close): bull = NQ strengthening vs ES.
    Noise-engine trades and RA / V15a trades (they carry dir): points WITH the ratio vector (side*reg>0) vs AGAINST vs flat, both periods.
PASS = WITH minus AGAINST same sign and |t|>=2 in both periods."""
import sys, numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/fartpie/discord_bot')
from pivot_ratchet_core import run_state_machine
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
def ld(f): X=pd.read_parquet(f'/tmp/claude-0/ml/{f}.parquet'); return X.set_index('dt')[['open','high','low','close','volume']]
nq=ld('nq5_real'); es=ld('es5_real'); idx=nq.index.intersection(es.index); nq=nq.loc[idx]; es=es.loc[idx]
# (1) lead-lag IC
r_nq=np.log(nq.close).diff(); r_es=np.log(es.close).diff(); rth=(idx.hour*60+idx.minute>=570)&(idx.hour*60+idx.minute<960)
def ic(x,y,m):
    d=pd.DataFrame({'x':x,'y':y})[m].dropna(); return d.x.rank().corr(d.y.rank()), len(d)
print('LEAD-LAG IC (spearman), RTH only; x = ES ret - NQ ret over the window, y = NQ next-window ret')
for win,lab in ((1,'5m'),(6,'30m')):
    rn=np.log(nq.close).diff(win); re=np.log(es.close).diff(win); x=(re-rn); y=rn.shift(-win)
    ye=re.shift(-win)
    for per,m in (('14-21',idx.year<2022),('22-26',idx.year>=2022)):
        mm=rth&m; a,n1=ic(x,y,mm); b,_=ic(x,ye,mm); print(f' window {lab} {per}: IC(ES-NQ -> NQ next) {a:+.4f}  IC(ES-NQ -> ES next) {b:+.4f}  n={n1}')
# (2) vector over the ratio
o=nq.open/es.open; c=nq.close/es.close; rr=pd.DataFrame({'o':o,'c':c}); rr['open']=rr.o; rr['close']=rr.c; rr['high']=rr[['o','c']].max(axis=1); rr['low']=rr[['o','c']].min(axis=1)
b15=rr.resample('15min',origin='start_day').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
reg,lv,atr,_=run_state_machine(b15.assign(datetime=b15.index)); reg=pd.Series(np.asarray(reg).astype(int),index=b15.index)
known=reg.copy(); known.index=known.index+pd.Timedelta(minutes=15)           # state known at the close of its bar
print('\nratio vector state share:',known.value_counts(normalize=True).round(2).to_dict())
def state_at(day,minute): 
    t=pd.Timestamp(day)+pd.Timedelta(minutes=int(minute)); i=known.index.searchsorted(t,side='right')-1; return known.iloc[i] if i>=0 else 0
G=pd.read_pickle(SP+'/G_noise.pkl'); G['day']=pd.to_datetime(G.day).dt.normalize(); G=G[G.day>='2018-02-01']
G['st']=[state_at(d,t+5) for d,t in zip(G.day,G.tod)]; G['rel']=np.sign(G.st)*G.side
O,N=pickle.load(open(SP+'/s6.pkl','rb')); legs=dict(zip(N,O['V2']))
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=10 and len(b)>=10 else np.nan
def rep(name,df,wcol):
    o=[]
    for lab,m in (('18-21',df.day.dt.year<2022),('22-26',df.day.dt.year>=2022)):
        d=df[m]; w=d[wcol][d.rel>0]; a=d[wcol][d.rel<0]; f=d[wcol][d.rel==0]
        o.append(f'{lab}: WITH {w.mean():+6.1f} (n{len(w)}) AGAINST {a.mean():+6.1f} (n{len(a)}) flat {f.mean():+5.1f} | w-a t {tt(w,a):+.2f}')
    print(f' {name:6}',' || '.join(o))
print('\nWITH vs AGAINST the NQ/ES ratio vector (pts per trade)'); rep('NOISE',G,'pts')
for k in ('RA','V15a'):
    df=legs[k][0].copy()
    if 'dir' not in df.columns: continue
    df['day']=pd.to_datetime(df.day).dt.normalize(); df=df[(df.sz>0)&(df.day>='2018-02-01')].copy(); df['w']=df.pts*df.sz
    df['st']=[state_at(d,m) for d,m in zip(df.day,df.m)]; df['rel']=np.sign(df.st)*np.sign(df.dir); rep(k,df,'w')
