"""Nov 14 2026: REGIME ATLAS v1. Rey's point: no single narrative; the market's participants and plumbing changed (stimulus/retail, tightening, 0DTE, AI bull, tariffs/policy shocks), so models work in the regimes they were 'made for'. DECLARED regime calendar from external events (NOT fit to results), 5 regimes:
 R1 2018-01..2020-02 late-cycle low-vol, pre-COVID; R2 2020-03..2021-12 COVID crash -> stimulus/retail/meme; R3 2022-01..2023-03 Fed tightening/bear (daily-expiry 0DTE ramps from May 2022; SVB Mar 2023);
 R4 2023-04..2025-03 AI-led bull, 0DTE dominant, low realized vol; R5 2025-04..now tariff/policy-shock regime.
Descriptive map: market microstructure features per regime (range, overnight share of variance, VR, overnight continuation) and every model's result per regime. Standard for a regime EXPLANATION: a model that works in two separate regimes must share a measurable feature that the other regimes lack, and that mapping must be fit on earlier regimes and confirmed on later ones."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
RG=[('R1 pre-COVID','2018-01-01','2020-03-01'),('R2 COVID/stimulus/retail','2020-03-01','2022-01-01'),('R3 tightening/bear','2022-01-01','2023-04-01'),('R4 AI bull/0DTE','2023-04-01','2025-04-01'),('R5 tariff/policy','2025-04-01','2027-01-01')]
def rg(t):
    t=pd.Timestamp(t)
    for n,a,b in RG:
        if pd.Timestamp(a)<=t<pd.Timestamp(b): return n
D=pd.read_pickle(SC+'atlas_D.pkl'); D['rg']=[rg(t) for t in D.index]; D['rng']=D.hi-D.lo
print('\n#### NQ MARKET FEATURES BY REGIME (days, avg RTH range pts, overnight share of variance, VR(7) trailing, overnight->RTH continuation, strong-inventory share)')
for n,_,_ in RG:
    s=D[D.rg==n]; on=s.on.dropna(); idv=s.id.dropna()
    print(f'  {n:26s} days {len(s):4d} | range {s.rng.mean():6.1f} | ON var share {on.var()/(on.var()+idv.var()):.2f} | VR {s.vr.mean():.3f} | ON->RTH cont {s.cont.mean():+.3f} | corr(ON,RTH) {s.on.corr(s.id):+.2f} | strong-inv {s.strong.mean():.0%} | RTH eff {s.eff.mean():.3f}')
def per(df,col,pcol,name,fmt='{:+.2f}',per_trade=True):
    out=[]
    for n,_,_ in RG:
        s=df[df[col]==n][pcol].dropna()
        out.append(f'{n.split()[0]} {fmt.format(s.mean()) if len(s) else "   - "}(n{len(s)})')
    print(f'  {name:34s} '+' | '.join(out))
S=pickle.load(open(SC+'s6.pkl','rb')); legs=S[0]['V2']; nm=S[1]; tot=[]
print('\n#### MAIN NQ ALGO (V2 legs x unit), pts per TRADE by regime')
for (T,u,_),n in zip(legs,nm):
    x=T.copy(); x['day']=pd.to_datetime(x.day).dt.normalize(); x['p']=x.pts*x.sz*u; x['rg']=[rg(t) for t in x.day]; per(x,'rg','p',f'leg {n} (u={u})'); tot.append(x[['day','p']])
T_=pd.concat(tot).groupby('day').p.sum().reset_index(); T_['rg']=[rg(t) for t in T_.day]
print('  --- all legs combined, pts per DAY, win-day share, worst day:')
for n,_,_ in RG:
    s=T_[T_.rg==n].p; print(f'  {n:26s} mean/day {s.mean():+6.2f} (n{len(s)}) win-days {np.mean(s>0):.0%} sd {s.std():5.1f} worst {s.min():6.1f}')
A=pd.read_pickle(SC+'amt_models_T.pkl'); A['day']=pd.to_datetime(A.d).dt.normalize(); A['rg']=[rg(t) for t in A.day]
C=pd.read_pickle(SC+'icc_T.pkl'); C['day']=pd.to_datetime(C.d).dt.normalize(); C['rg']=[rg(t) for t in C.day]; C=C.join(D[['on','strong']],on='day'); C['with']=np.sign(C.on)==C.dr
print('\n#### NQ INTRADAY MODELS, pts per trade by regime')
for m in ('VA-B','VA-F','VA-80','IB-B','INV-F','SHAPE'): per(A[A.model==m],'rg','pts',m)
per(C,'rg','pts','ICT chain (all)'); per(C[C.strong==True],'rg','pts','ICT chain, strong overnight'); per(C[C['with']],'rg','pts','ICT chain WITH overnight move'); per(C[~C['with']],'rg','pts','ICT chain AGAINST overnight move')
P=pd.read_pickle(SC+'persist_R.pkl'); P=P[P.kind=='A flip|over pivot'].copy(); P['rg']=[rg(t) for t in P.t]
print('\n#### 9 LSE stocks, 1h vector flip A (bp/trade) by regime')
for col in ('8','16','nat'): per(P,'rg',col,f'flip A exit {col}','{:+.0f}')
C2=pd.read_pickle(SC+'persist_R.pkl'); C2=C2[C2.kind=='ctl bull bars'].copy(); C2['rg']=[rg(t) for t in C2.t]
for col in ('8','16'): per(C2,'rg',col,f'control: any bull bar, exit {col}','{:+.0f}')

print('\n#### H-b (declared after the atlas, mechanism: vector flips are upside-trend continuation, so they need an up-trending index): flip A by QQQ trend state at the flip (QQQ close > its 200-day SMA, PIT) -- by regime')
import glob
q=pd.read_parquet(glob.glob('/home/claude/fartpie/lse_clean/QQQ_15m*.parquet')[0]); q['datetime']=pd.to_datetime(q.datetime); q=q.drop_duplicates('datetime').set_index('datetime').sort_index()
q.index=q.index.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None); dq=q.between_time('09:30','15:59').close.groupby(lambda t:t.normalize()).last()
up=(dq>dq.rolling(200).mean()).shift(1)  # state as of the prior close
P['up']=up.reindex(P.t.dt.normalize()).values; C2['up']=up.reindex(C2.t.dt.normalize()).values
for col in ('8','16','nat'):
    for st,lab in ((True,'QQQ above 200d'),(False,'QQQ below 200d')):
        per(P[P.up==st],'rg',col,f'flip A exit {col} | {lab}','{:+.0f}')
for col in ('8','16'):
    for st,lab in ((True,'above'),(False,'below')): per(C2[C2.up==st],'rg',col,f'control bull bar exit {col} | {lab}','{:+.0f}')
