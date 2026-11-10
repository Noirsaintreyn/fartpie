"""Nov 10 2026: RSI. DECLARED BEFORE RESULTS. RSI(14) Wilder on 15m NQ closes (RTH+premarket continuous) known at entry, and daily RSI(14) of prior close.
Noise trades (G_noise) + RA / V15a: signed RSI = RSI*side (>50 = momentum agrees). Buckets: <40 / 40-60 / >60 of signed-agree RSI (long: RSI, short: 100-RSI); pts/trade per bucket, 2018-21 vs 2022-26, PASS = (high - low) same sign and |t|>=2 both."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
X=pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet').set_index('dt')
def rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean(); return 100-100/(1+up/dn)
c15=X.close.resample('15min',origin='start_day').last().dropna(); r15=rsi(c15); r15.index=r15.index+pd.Timedelta(minutes=15)   # known at bar close
rth=X[X.rth]; cd=rth.groupby(pd.to_datetime(rth.day).dt.normalize()).close.last(); rd=rsi(cd).shift(1)
def at(day,m):
    t=pd.Timestamp(day)+pd.Timedelta(minutes=int(m)); i=r15.index.searchsorted(t,side='right')-1; return r15.iloc[i] if i>=0 else np.nan
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=10 and len(b)>=10 else np.nan
def rep(name,df,w,mcol):
    for src,vals in (('15m',[at(d,m) for d,m in zip(df.day,df[mcol])]),('daily',rd.reindex(df.day).values)):
        v=np.asarray(vals,float); agree=np.where(df.side>0,v,100-v); o=[]
        for lab,m in (('18-21',df.day.dt.year<2022),('22-26',df.day.dt.year>=2022)):
            x=df[w][m]; a=agree[m.values]; lo=a<40; mid=(a>=40)&(a<=60); hi=a>60
            o.append(f'{lab}: lo {x[lo].mean():+5.1f}({lo.sum()}) mid {x[mid].mean():+5.1f}({mid.sum()}) hi {x[hi].mean():+5.1f}({hi.sum()}) hi-lo t {tt(x[hi],x[lo]):+.2f} hi-mid t {tt(x[hi],x[mid]):+.2f}')
        print(f' {name:6} RSI {src:5}',' | '.join(o))
G=pd.read_pickle(SP+'/G_noise.pkl'); G['day']=pd.to_datetime(G.day).dt.normalize(); G=G[G.day>='2018-02-01'].copy(); rep('NOISE',G,'pts','tod')
O,N=pickle.load(open(SP+'/s6.pkl','rb')); legs=dict(zip(N,O['V2']))
for k in ('RA','V15a'):
    df=legs[k][0].copy(); df['day']=pd.to_datetime(df.day).dt.normalize(); df=df[(df.sz>0)&(df.day>='2018-02-01')].copy(); df['w']=df.pts*df.sz; df['side']=np.sign(df.dir); rep(k,df,'w','m')
