"""Nov 10 2026: BAD-REGIME SCAN, EVERYTHING. DECLARED BEFORE RESULTS.
Units: S6 V2 legs (RA M1 R1 ON+ ON- RULES V15a; sz-weighted pts per trade), S6all (sz-weighted pts per day), NZ (noise engine, pts per day).
Features known at prior close, causal 252d rank (top third vs bottom third) or flag vs rest:
 TREND: slope20 (reg slope / ATR), r2_20, dist20, dist50, ma50-200, adx14, pos20 (close in 20d range), ret5, ret20, hh/ll count
 PRICE SQUEEZE: BB bandwidth rank, TTM squeeze (BB inside KC), NR7, inside day, rc (5d/20d range)
 VOLUME: vol1 (prior RTH vol / 20d), vol5 (5d/20d), obv10 slope, sq_vol_up (TTM squeeze & vol1>1) , sq_vol_dn (squeeze & vol1<1)
 VOL/GREEKS: ivr, iv5, ivrv, VIX, VIXd5, YZr, HARf, gexp, dexp, vanna, charm, gapabs, sq_iv (ivr<.2 & rc<med)
PASS = same sign and |t|>=2 in 2018-21 and 2022-26. Null: same scan with each feature circularly shifted by a random lag (20 draws) -> expected false passes.
Then: SQUEEZE BREAKS vs REGULAR BREAKS: noise trades on TTM-squeeze days vs not, with and without volume confirmation."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
exec(open(f'{SP}/ctxhead.py').read())
Xb=pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); Xb=Xb[Xb.rth].copy(); Xb['day']=pd.to_datetime(Xb.day).dt.normalize()
V=Xb.groupby('day').volume.sum().reindex(ix)
O_=Dn.O.astype(float); H=Dn.DH.astype(float); L=Dn.DL.astype(float); C=Dn.DC.astype(float)
tr=pd.concat([H-L,(H-C.shift()).abs(),(L-C.shift()).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean()
F=pd.DataFrame(index=ix)
def slope(s,n):
    x=np.arange(n); xm=x-x.mean(); return s.rolling(n).apply(lambda y: (xm*(y-y.mean())).sum()/(xm**2).sum(),raw=True)
def r2(s,n):
    return s.rolling(n).apply(lambda y: np.corrcoef(np.arange(n),y)[0,1]**2,raw=True)
F['slope20']=slope(C,20)/atr; F['r2_20']=r2(C,20); F['dist20']=(C/C.rolling(20).mean()-1); F['dist50']=(C/C.rolling(50).mean()-1)
F['ma50_200']=C.rolling(50).mean()/C.rolling(200).mean()-1
up=H.diff(); dn=-L.diff(); pdm=np.where((up>dn)&(up>0),up,0); mdm=np.where((dn>up)&(dn>0),dn,0)
pdi=100*pd.Series(pdm,ix).rolling(14).mean()/atr; mdi=100*pd.Series(mdm,ix).rolling(14).mean()/atr
F['adx14']=(100*(pdi-mdi).abs()/(pdi+mdi)).rolling(14).mean()
F['pos20']=(C-L.rolling(20).min())/(H.rolling(20).max()-L.rolling(20).min())
F['ret5']=C.pct_change(5); F['ret20']=C.pct_change(20)
F['hh']=(H>H.shift()).rolling(10).sum()-(L<L.shift()).rolling(10).sum()
sd=C.rolling(20).std(); ma=C.rolling(20).mean(); bw=4*sd/ma; F['bbw']=bw.rolling(126,min_periods=60).rank(pct=True)
kc=1.5*atr.rolling(20).mean(); F['ttm']=((2*sd<kc)).astype(float).where(sd.notna())
rg=H-L; F['nr7']=(rg<=rg.rolling(7).min()).astype(float); F['inside']=((H<H.shift())&(L>L.shift())).astype(float)
F['rc']=rg.rolling(5).mean()/rg.rolling(20).mean()
F['vol1']=V/V.rolling(20).mean(); F['vol5']=V.rolling(5).mean()/V.rolling(20).mean()
obv=(np.sign(C.diff())*V).cumsum(); F['obv10']=slope(obv,10)/V.rolling(20).mean()
F['sq_vol_up']=((F.ttm==1)&(F.vol1>1)).astype(float).where(F.ttm.notna()); F['sq_vol_dn']=((F.ttm==1)&(F.vol1<1)).astype(float).where(F.ttm.notna())
iv=G1.atm_iv; ivr=iv.rolling(252,min_periods=120).rank(pct=True); rv20=np.log(C).diff().rolling(20).std()*np.sqrt(252)
sh=lambda s: s.shift(1).reindex(ix,method='ffill')
F=F.shift(1)   # known at prior close (all above are computed on day t-1 bars -> shift so row t uses data through t-1)
F['ivr']=sh(ivr).values; F['iv5']=sh(iv.diff(5)).values; F['ivrv']=(sh(iv)-rv20.shift(1)).values
F['VIX']=sh(vx).values; F['VIXd5']=sh(vx.diff(5)).values; F['YZr']=sh(YZr).values
M_=pd.DataFrame({'d':lr.shift(1),'w':np.log(R.rolling(5).mean().shift(1)),'m':np.log(R.rolling(22).mean().shift(1))}); ok=M_.notna().all(axis=1)&(ix<'2020-01-01')
from sklearn.linear_model import LinearRegression
hf=LinearRegression().fit(M_[ok],lr[ok]); F['HARf']=pd.Series(np.where(M_.notna().all(axis=1),hf.predict(M_.fillna(0)),np.nan),ix)
F['gexp']=sh(OWN.rolling(252,min_periods=120).rank(pct=True)).values; F['dexp']=sh((G1.call_dex+G1.put_dex).rolling(252,min_periods=120).rank(pct=True)).values
F['vanna']=sh(G2.net_vanna).values; F['charm']=sh(G2.net_charm).values; F['gapabs']=(Dn.gap.abs()/Dn.rng20.shift(1)).values
F['sq_iv']=((F.ivr<.2)&(F.rc<F.rc.median())).astype(float).where(F.ivr.notna())
FLAGS=['ttm','nr7','inside','sq_vol_up','sq_vol_dn','sq_iv']; CONT=[c for c in F.columns if c not in FLAGS]
Rk=F[CONT].apply(lambda s: s.rolling(252,min_periods=120).rank(pct=True))
# ---- units
O,N=pickle.load(open(f'{SP}/s6.pkl','rb')); legs=O['V2']; U={}
alld=[]
for nm,(df,_,_) in zip(N,legs):
    df=df.copy(); df['day']=pd.to_datetime(df.day).dt.normalize(); df['w']=df.pts*df.sz; U[nm]=df[['day','w']]; alld.append(df[['day','w']])
A=pd.concat(alld).groupby('day').w.sum(); U['S6all']=A.reset_index()
G=pd.read_pickle(f'{SP}/G_noise.pkl'); G['day']=pd.to_datetime(G.day).dt.normalize(); G=G[G.day>='2018-02-01']
NZd=G.groupby('day').pts.sum(); U['NZ']=NZd.reset_index().rename(columns={'pts':'w'})
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=15 and len(b)>=15 else np.nan
def scan(Fc,Rkc,verbose=False):
    res=[]
    for un,d in U.items():
        d=d[(d.day>='2018-02-01')]
        for f in list(Rkc.columns)+FLAGS:
            row=[]
            for lab,m in (('a',d.day.dt.year<2022),('b',d.day.dt.year>=2022)):
                dd=d[m]
                if f in FLAGS: x=Fc[f].reindex(dd.day).values; hi=x==1; lo=x==0
                else: x=Rkc[f].reindex(dd.day).values; hi=x>=.67; lo=x<=.33
                row+= [tt(dd.w[hi],dd.w[lo]), dd.w[hi].mean()-dd.w[lo].mean(), hi.sum()]
            res.append((un,f)+tuple(row))
    R_=pd.DataFrame(res,columns=['unit','feat','t_a','d_a','n_a','t_b','d_b','n_b']); R_['pass']=(np.sign(R_.t_a)==np.sign(R_.t_b))&(R_.t_a.abs()>=2)&(R_.t_b.abs()>=2); return R_
R0=scan(F,Rk); P=R0[R0['pass']].copy(); P['mind']=P[['d_a','d_b']].abs().min(axis=1)
print('REAL scan: tests',len(R0),'passes',len(P)); print(P.sort_values('unit').round(2).to_string(index=False))
rng=np.random.default_rng(1); nulls=[]
for k in range(20):
    lag=int(rng.integers(120,len(ix)-120)); Fs=F.apply(lambda s: pd.Series(np.roll(s.values,lag),index=s.index)); Rs=Rk.apply(lambda s: pd.Series(np.roll(s.values,lag),index=s.index))
    nulls.append(int(scan(Fs,Rs)['pass'].sum()))
print('NULL passes per scan (20 shifts): mean %.1f  max %d  min %d'%(np.mean(nulls),max(nulls),min(nulls)))
# ---- squeeze breaks vs regular breaks (noise trades)
print('\nSQUEEZE vs REGULAR BREAKS (noise engine, pts/day; 18-21 | 22-26)')
days=pd.DatetimeIndex(sorted(G.day.unique())); Pd=NZd.reindex(days).fillna(0)
for nm in ['ttm','nr7','inside','sq_vol_up','sq_vol_dn','sq_iv']:
    s=F[nm].reindex(days); o=[]
    for lab,m in (('18-21',days.year<2022),('22-26',days.year>=2022)):
        a=s[m]==1; b=s[m]==0; o.append(f'{lab}: brk {Pd[m][a].mean():+5.1f} (n{a.sum()}) reg {Pd[m][b].mean():+5.1f} t {tt(Pd[m][a],Pd[m][b]):+.2f}')
    print(f'  {nm:10}',' | '.join(o))

print('\nGAPABS terciles (causal 252d rank): mean pts, std, mean/std; 18-21 | 22-26')
for un in ['S6all','NZ','V15a','RA']:
    d=U[un]; d=d[d.day>='2018-02-01']; o=[]
    for lab,m in (('a',d.day.dt.year<2022),('b',d.day.dt.year>=2022)):
        dd=d[m]; x=Rk['gapabs'].reindex(dd.day).values; s=[]
        for nm,k in (('lo',x<=.33),('mid',(x>.33)&(x<.67)),('hi',x>=.67)):
            w=dd.w[k]; s.append(f'{nm} {w.mean():+6.1f}/{w.std():5.1f}/{w.mean()/w.std():+.2f} n{k.sum()}')
        o.append(lab+': '+' | '.join(s))
    print(f' {un:6}',' || '.join(o))
print('\nSKIP lowest-third gap days: S6all & NZ total pts and mean/std (18-21 | 22-26)')
for un in ['S6all','NZ']:
    d=U[un]; d=d[d.day>='2018-02-01']; o=[]
    for lab,m in (('a',d.day.dt.year<2022),('b',d.day.dt.year>=2022)):
        dd=d[m]; x=Rk['gapabs'].reindex(dd.day).values; k=~(x<=.33)
        o.append(f'{lab}: all {dd.w.sum():8.0f} ({dd.w.mean()/dd.w.std():.3f}) -> skip-lo {dd.w[k].sum():8.0f} ({dd.w[k].mean()/dd.w[k].std():.3f}), days kept {k.mean():.0%}')
    print(f' {un:6}',' || '.join(o))

print('\n=== LOOK-AHEAD CHECK: S6 WITHOUT overnight legs (ON+, ON-) ===')
alld2=[]
for nm,(df,_,_) in zip(N,legs):
    if nm.startswith('ON'): continue
    df=df.copy(); df['day']=pd.to_datetime(df.day).dt.normalize(); df['w']=df.pts*df.sz; alld2.append(df[['day','w']])
U['S6day']=pd.concat(alld2).groupby('day').w.sum().reset_index()
for un in ['S6day','NZ','RA','M1','R1','RULES','V15a','ON+','ON-']:
    d=U[un]; d=d[d.day>='2018-02-01']; o=[]
    for lab,m in (('18-21',d.day.dt.year<2022),('22-26',d.day.dt.year>=2022)):
        dd=d[m]; x=Rk['gapabs'].reindex(dd.day).values; hi=x>=.67; lo=x<=.33
        o.append(f'{lab}: lo {dd.w[lo].mean():+6.1f} hi {dd.w[hi].mean():+6.1f} t {tt(dd.w[hi],dd.w[lo]):+.2f}')
    print(f' {un:6}',' | '.join(o))
d=U['S6day']; d=d[d.day>='2018-02-01']
for lab,m in (('18-21',d.day.dt.year<2022),('22-26',d.day.dt.year>=2022)):
    dd=d[m]; x=Rk['gapabs'].reindex(dd.day).values; k=~(x<=.33)
    print(f' S6day skip-lo {lab}: all {dd.w.sum():.0f} ({dd.w.mean()/dd.w.std():.3f}) -> {dd.w[k].sum():.0f} ({dd.w[k].mean()/dd.w[k].std():.3f})')
