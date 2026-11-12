"""Literature-guided mean reversion (declared before results). Hints: Baltussen et al./Gao et al. intraday momentum (early return predicts late return); 'Where does gamma hedge drive the intraday move' (long-gamma dealers -> intraday REVERSAL when the move stays inside the gamma-theta breakeven range, short gamma -> momentum once it is hit); Lou-Polk-Skouras overnight vs intraday.
1) early return (open->T1) vs late return (T1->15:55) by prior-day QQQ net-GEX tercile (252d rank, lagged) and by quiet/loud (|early| < implied 1-day sigma = ATM IV/sqrt(252) x open).  Trade = fade the early move, 0.5pt cost, only if |early| >= 0.15 ADR.
2) overnight GAP fade (entry 09:30 open toward the prior close, exit 15:55) by gamma tercile and gap size.
3) VIX state (yfinance ^VIX, ^VIX3M; term ratio, 1-day spike) vs NQ open->close and vs early/late reversal.  RTH only, flat by 15:55."""
import numpy as np, pandas as pd, warnings, math; warnings.filterwarnings('ignore')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
mn=b.dt.dt.hour*60+b.dt.dt.minute; b['d']=b.dt.dt.normalize(); rth=b[(mn>=570)&(mn<955)].copy(); rth['mn']=rth.dt.dt.hour*60+rth.dt.dt.minute
G={d:g for d,g in rth.groupby('d')}
D=rth.groupby('d').agg(o=('open','first'),c=('close','last'),h=('high','max'),l=('low','min')); D['rng']=D.h-D.l; D['adr']=D.rng.rolling(20).mean().shift(); D['pc']=D.c.shift()
Q=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); Q.index=Q.index.normalize()
Q['net']=Q.call_gex-Q.put_gex; Q['gr']=Q.net.rolling(252,min_periods=120).rank(pct=True); Q['iv']=Q.atm_iv
Q=Q[['gr','iv','net']].shift(1).reindex(D.index,method='ffill')
rows=[]
for d,g in G.items():
    if d not in D.index or not D.adr[d]==D.adr[d] or d<pd.Timestamp('2018-01-01'): continue
    o=g.open.iloc[0]; adr=D.adr[d]; gr=Q.gr.get(d); iv=Q.iv.get(d)
    f=dict(day=d,gr=gr,iv=iv,gap=o-D.pc[d],adr=adr,o=o,oc=D.c[d]-o)
    last=g.close.iloc[-1]
    for T in (30,60,120):
        x=g[g.mn<570+T]
        if len(x)<2: continue
        pT=g[g.mn>=570+T].open.iloc[0]; f[f'e{T}']=pT-o; f[f'l{T}']=last-pT
    # first 30 vs last 30
    f['first30']=g[g.mn<600].close.iloc[-1]-o if (g.mn<600).any() else np.nan; l30=g[g.mn>=925]; f['last30']=last-l30.open.iloc[0] if len(l30) else np.nan
    rows.append(f)
F=pd.DataFrame(rows); F['yr']=F.day.dt.year; F['era']=np.where(F.yr<=2021,'18-21','22-26'); F['sig']=F.iv/math.sqrt(252)*F.o
F['gam']=pd.cut(F.gr,[0,1/3,2/3,1.0],labels=['neg/low','mid','pos/high'],include_lowest=True)
def ln(x): return f'n {len(x):4d}' if len(x)<20 else f'n {len(x):4d} avg {x.pts.mean():+6.2f} win {(x.pts>0).mean():.0%} t {x.pts.mean()/(x.pts.std()/len(x)**.5):+.1f}'
print('=== 1. FADE the early move (pts = -sign(early)*late move - .5), |early|>=0.15 ADR')
for T in (30,60,120):
    z=F.dropna(subset=[f'e{T}',f'l{T}','gam','sig']).copy(); z=z[z[f'e{T}'].abs()>=.15*z.adr]; z['pts']=-np.sign(z[f'e{T}'])*z[f'l{T}']-.5; z['quiet']=z[f'e{T}'].abs()<z.sig
    print(f'-- early window = open->+{T}m')
    for g_ in ('neg/low','mid','pos/high'):
        for qn,qm in (('quiet',True),('loud',False)):
            s=z[(z.gam==g_)&(z.quiet==qm)]; print(f'   gamma {g_:8s} {qn:5s}: '+' || '.join(f'{e} {ln(s[s.era==e])}' for e in ('18-21','22-26')))
    print('   ALL days: '+' || '.join(f'{e} {ln(z[z.era==e])}' for e in ('18-21','22-26')))
print('=== 1b. first-30m vs last-30m (Gao et al.): corr and fade/follow payoff by gamma tercile')
z=F.dropna(subset=['first30','last30','gam']); z=z[z.first30.abs()>=.1*z.adr].copy(); z['pts']=np.sign(z.first30)*z.last30-.5       # FOLLOW payoff
for g_ in ('neg/low','mid','pos/high'): s=z[z.gam==g_]; print(f'   gamma {g_:8s} FOLLOW first30: '+' || '.join(f'{e} {ln(s[s.era==e])}' for e in ('18-21','22-26')))
print('=== 2. GAP FADE (entry open, exit 15:55; toward prior close), |gap|>=0.15 ADR, by gamma and by gap size vs implied sigma')
z=F.dropna(subset=['gap','gam','sig']).copy(); z=z[z.gap.abs()>=.15*z.adr]; z['pts']=-np.sign(z.gap)*z.oc-.5; z['big']=z.gap.abs()>=z.sig
for g_ in ('neg/low','mid','pos/high'):
    for bn,bm in (('small',False),('big',True)):
        s=z[(z.gam==g_)&(z.big==bm)]; print(f'   gamma {g_:8s} gap {bn:5s}: '+' || '.join(f'{e} {ln(s[s.era==e])}' for e in ('18-21','22-26')))
print('   ALL: '+' || '.join(f'{e} {ln(z[z.era==e])}' for e in ('18-21','22-26')))
# 3. VIX
import yfinance as yf
try:
    v=yf.download(['^VIX','^VIX3M'],start='2017-06-01',progress=False,auto_adjust=False)['Close']; v.index=pd.to_datetime(v.index).normalize()
    V=pd.DataFrame({'vix':v['^VIX'],'v3m':v['^VIX3M']}); V['term']=V.vix/V.v3m; V['chg']=V.vix.pct_change(); V['pct']=V.vix.rolling(252,min_periods=120).rank(pct=True)
    V=V[['vix','term','chg','pct']].shift(1).reindex(F.day,method='ffill'); V.index=F.index; F2=F.join(V)
    print('=== 3. VIX state (prior close) -> NQ intraday: open->close long payoff and early/late fade, by state')
    for nm,col,bins,labs in (('term VIX/VIX3M',"term",[0,.9,1.0,9],['contango<.9','flat .9-1','backwardation>1']),('1-day VIX change','chg',[-9,-.05,.08,9],['VIX down>5%','calm','VIX spike>8%']),('VIX 1y percentile','pct',[0,.33,.67,1],['low','mid','high'])):
        F2['st']=pd.cut(F2[col],bins,labels=labs,include_lowest=True)
        for l in labs:
            s=F2[F2.st==l].dropna(subset=['oc']); s=s.assign(pts=s.oc-.5)
            t=F2[F2.st==l].dropna(subset=['e60','l60']); t=t[t.e60.abs()>=.15*t.adr]; t=t.assign(pts=-np.sign(t.e60)*t.l60-.5)
            print(f'   {nm:18s} {l:17s} LONG open->close: '+' || '.join(f'{e} {ln(s[s.era==e])}' for e in ('18-21','22-26'))+'  | FADE early(60m): '+' || '.join(f'{e} {ln(t[t.era==e])}' for e in ('18-21','22-26')))
except Exception as ex: print('VIX download failed',str(ex)[:120])
