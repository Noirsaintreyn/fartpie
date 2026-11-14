"""Nov 13-14 2026: TEST THE SEVEN ALIGNMENT PREDICTIONS of xs/IDEOLOGY.md. Persistence state at day D is known at the open (PIT, bars through D-1): vr_nq = VR(7) of NQ 1h RTH log returns over the trailing 60 sessions (>1 persistent, <1 reverting); cont60 = trailing 60-day mean of sign(overnight move) x sign(RTH open->close move) (inventory continuation). Strict eras 2018-21 / 2022+. Each prediction states its expected sign; a model 'aligns' only if the sign is as predicted in BOTH eras."""
import pickle, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; ERA={'18-21':('2018-01-01','2022-01-01'),'22+':('2022-01-01','2027-01-01')}
nq=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); nq['dt']=pd.to_datetime(nq.dt); nq=nq.set_index('dt').sort_index()
rth=nq.between_time('09:30','15:59'); h1=rth.resample('60min',offset='30min').agg(dict(open='first',high='max',low='min',close='last')).dropna()
r1=np.log(h1.close).diff(); q=7
# rolling VR over 420 hourly bars, then take the value as of the last bar of each session (known at the next open)
vr=(np.log(h1.close).diff(q).rolling(420).var()/(q*r1.rolling(420).var()))
vrD=vr.groupby(vr.index.normalize()).last()
g=rth.groupby(rth.index.normalize()); D=pd.DataFrame(dict(o=g.open.first(),c=g.close.last(),hi=g.high.max(),lo=g.low.min()))
D['pc']=D.c.shift(1); D['on']=D.o-D.pc; D['id']=D.c-D.o; D['cont']=np.sign(D.on)*np.sign(D.id)
D['cont60']=D.cont.rolling(60).mean().shift(1); D['vr']=vrD.shift(1).reindex(D.index)
# overnight inventory (share of ETH bars from 18:00 to 09:25 closing above prior RTH settle) and RTH efficiency
eth=nq[(nq.index.hour>=18)|((nq.index.hour<9)|((nq.index.hour==9)&(nq.index.minute<30)))].copy(); eth['td']=(eth.index+pd.Timedelta(hours=6)).normalize()
settle=D.pc.reindex(eth.td.values).values; eth['above']=(eth.close.values>settle)
inv=eth.groupby('td').above.mean(); D['inv']=inv.reindex(D.index)
rb=rth.copy(); rb['td']=rb.index.normalize(); rb['mv']=rb.close.diff().abs(); eff=(rb.groupby('td').close.last()-rb.groupby('td').open.first()).abs()/rb.groupby('td').mv.sum(); D['eff']=eff
D['strong']=(D.inv>=.9)|(D.inv<=.1); D.index.name='day'
print('days',len(D),'vr valid',D.vr.notna().sum()); D.to_pickle(SC+'atlas_D.pkl')
def era(df,col,e): a,b=ERA[e]; return df[(df[col]>=a)&(df[col]<b)]
def tt(x): return x.mean()/(x.std()/len(x)**.5) if len(x)>5 else np.nan
def split(df,dcol,pcol,sel,name,pred):
    print(f'\n{name}   [prediction: {pred}]')
    for e in ERA:
        s=era(df,dcol,e).copy(); s=s.join(D[['vr','cont60']],on=dcol).dropna(subset=['vr','cont60'])
        for sc,lab,cut in (('vr','VR(7)',1.0),('cont60','inv-continuation',0.0)):
            p=s[s[sc]>cut][pcol]; r=s[s[sc]<=cut][pcol]
            if len(p)<10 or len(r)<10: continue
            dd=p.mean()-r.mean(); se=np.sqrt(p.var()/len(p)+r.var()/len(r))
            print(f'   {e:6s} by {lab:17s}: persistent {p.mean():+7.2f} (n{len(p)}) | reverting {r.mean():+7.2f} (n{len(r)}) | diff {dd:+6.2f} t {dd/se:+.1f}')
A=pd.read_pickle(SC+'amt_models_T.pkl'); A['day']=pd.to_datetime(A.d).dt.normalize()
print('\n#### 1. VA-B (continuation) vs VA-F (fade): VA-B should be better when persistent, VA-F when reverting')
for m,pr in (('VA-B','persistent > reverting'),('VA-F','reverting > persistent'),('VA-80','?'),('IB-B','persistent > reverting')): split(A[A.model==m],'day','pts',None,f'NQ {m} (pts/trade)',pr)
C=pd.read_pickle(SC+'icc_T.pkl'); C['day']=pd.to_datetime(C.d).dt.normalize(); C=C.join(D[['on','inv','strong']],on='day')
C['with']=np.sign(C.on)==C.dr
print('\n#### 2. ICT chain WITH vs AGAINST the overnight move, by persistence: with-ON should win when persistent / cont60>0, against when reverting')
for e in ERA:
    s=era(C,'day',e).join(D[['vr','cont60']],on='day').dropna(subset=['vr','cont60'])
    for sc,lab in (('vr','VR(7)'),('cont60','inv-continuation')):
        up=s[s[sc]>(1.0 if sc=='vr' else 0)]; dn=s[s[sc]<=(1.0 if sc=='vr' else 0)]
        print(f'   {e:6s} {lab:17s}>cut: with {up[up["with"]].pts.mean():+6.1f} (n{up["with"].sum()}) vs against {up[~up["with"]].pts.mean():+6.1f} (n{(~up["with"]).sum()}) | <=cut: with {dn[dn["with"]].pts.mean():+6.1f} (n{dn["with"].sum()}) vs against {dn[~dn["with"]].pts.mean():+6.1f} (n{(~dn["with"]).sum()})')
print('\n#### 4. strong-inventory days should be MORE persistent: RTH efficiency ratio (|open->close| / sum of 5m moves) strong vs balanced, and day-level inventory continuation')
for e in ERA:
    s=era(D.reset_index(),'day',e).dropna(subset=['eff','inv'])
    print(f'   {e:6s} efficiency strong {s[s.strong].eff.mean():.3f} (n{s.strong.sum()}) vs balanced {s[~s.strong].eff.mean():.3f} (n{(~s.strong).sum()}) | on-continuation strong {s[s.strong].cont.mean():+.3f} vs balanced {s[~s.strong].cont.mean():+.3f}')
print('   chain pts by VR bucket (all chain trades):'); split(C,'day','pts',None,'ICT chain (pts/trade)','persistent > reverting if chain = persistence')
print('\n#### 6. main NQ algo legs (V2 legs x unit, pts/day) by regime')
S=pickle.load(open(SC+'s6.pkl','rb')); legs=S[0]['V2']; nm=S[1]
for (T,u,_),n in zip(legs,nm):
    x=T.copy(); x['day']=pd.to_datetime(x.day).dt.normalize(); x['p']=x.pts*x.sz*u; split(x,'day','p',None,f'leg {n} (u={u})','?')
print('\n#### 3. persistence spectrum (VR at several horizons): vector 1h should live where VR>1, 15m where VR<1')
import warnings
def ld(f):
    d=pd.read_parquet(f); d['datetime']=pd.to_datetime(d.datetime); d=d.drop_duplicates('datetime').set_index('datetime').sort_index(); d.index=d.index.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None); return d.between_time('09:30','15:59').close
def spec(c,qs,lab):
    out=[]
    lc=np.log(c)
    for e,(a,b) in ERA.items():
        x=lc[(lc.index>=a)&(lc.index<b)]; r=x.diff()
        # within-day-safe approximation: use all consecutive bars (overnight gaps included once per day)
        out.append(f'{e}: '+' '.join(f'q{q}:{x.diff(q).var()/(q*r.var()):.3f}' for q in qs))
    print(f'   {lab:22s} '+' | '.join(out))
nq15=rth.close.resample('15min').last().dropna(); nq5=rth.close; nq1=h1.close
spec(nq5,[3,6,12,24],'NQ 5m bars (15m..2h)'); spec(nq15,[2,4,8,16],'NQ 15m (30m..4h)'); spec(nq1,[2,4,7,14,35],'NQ 1h (2h..1wk)'); spec(D.c.dropna(),[2,5,10,20],'NQ daily (2d..4wk)')
sv=[ld(f).resample('60min',offset='30min').last().dropna() for f in sorted(glob.glob('/home/claude/fartpie/lse_clean/*.parquet'))]
s15=[ld(f) for f in sorted(glob.glob('/home/claude/fartpie/lse_clean/*.parquet'))]
def pool(L,qs,lab):
    for e,(a,b) in ERA.items():
        v=[]
        for c in L:
            lc=np.log(c[(c.index>=a)&(c.index<b)]); r=lc.diff(); v.append([lc.diff(q).var()/(q*r.var()) for q in qs])
        print(f'   {lab:22s} {e}: '+' '.join(f'q{q}:{x:.3f}' for q,x in zip(qs,np.mean(v,axis=0))))
pool(s15,[2,4,8,16],'9 stocks 15m (30m..4h)'); pool(sv,[2,4,7,14,35],'9 stocks 1h (2h..1wk)')
