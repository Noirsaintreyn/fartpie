"""Nov 10 2026: GOOD vs BETTER, BIG vs BIGGER. DECLARED BEFORE RESULTS. Everything we know, measured on the noise-engine trades (2018-2026, 2,087 trades), asking what separates
 (A) winners from losers, (B) BIG winners (>=100 pts) from ordinary winners (size among winners), (C) any win >= 40.
Features known at entry: engine features (band distance, VWAP distance, return, range, rv, wick, accel, efficiency, since-high/low),
 VP/TPO vs prior-day profile (distance to POC, beyond prior value area in trade direction, room to prior H/L, all /r20), HAR forecast ratio (HAR-RV on Yang-Zhang daily vol vs its 20d mean), GJR-GARCH(1,1,1) one-step vol forecast ratio (walk-forward, params refit each Jan),
 Yang-Zhang 20d vol percentile, QQQ net GEX (sign, 1y percentile), call/put volume ratio, VXN percentile, balance ov5. [CWAO: not defined in repo - skipped]
Tercile tables (low vs high) on P(win), P(big|win), P(>=40), mean pts; PASS: high-low same sign and |t|>=2 in both 2018-21 and 2022-26. Then walk-forward L2 logistic: OOS AUC for A and B."""
import sys; sys.path.insert(0,'/home/claude/fartpie/xs')
import numpy as np, pandas as pd, warnings, yfinance as yf; warnings.filterwarnings('ignore')
import nz_core as Z; from nz_core import A, SP
from arch import arch_model
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import roc_auc_score
ns={}; src=open('idea3_levels.py').read(); exec(src[:src.index("days=Z.DAYS; LV={}")],ns); prof=ns['prof']
D=pd.read_pickle(SP+'/nq_days.pkl'); D.index=pd.to_datetime(D.index).normalize(); D=D[D.index>='2014-01-01']
# daily YZ vol from RTH OHLC
o,h,l,c=np.log(D.O),np.log(D.H),np.log(D.L),np.log(D.C); pc=c.shift(1); k=0.34/(1.34+(21)/(19))
yz=(np.log(D.O/D.C.shift(1))**2).rolling(20).var()*0+ ( (o-pc)**2 )  # overnight^2 placeholder per day
rs=(h-c)*(h-o)+(l-c)*(l-o); oc=(c-o)
yzd=np.sqrt(((o-pc)**2).rolling(20).mean()+k*(oc**2).rolling(20).mean()+(1-k)*rs.rolling(20).mean())   # 20d YZ vol (daily)
yzp=yzd.rolling(252,min_periods=120).rank(pct=True).shift(1)
rv=np.sqrt(((o-pc)**2)+k*(oc**2)+(1-k)*rs)                              # daily YZ-style realized vol proxy for HAR
rv5=rv.rolling(5).mean(); rv22=rv.rolling(22).mean()
har=pd.Series(np.nan,index=D.index)
for yr in range(2016,2027):
    tr=D.index<pd.Timestamp(f'{yr}-01-01'); Xh=pd.concat([rv,rv5,rv22],axis=1).shift(1); yh=rv
    m=pd.concat([Xh,yh],axis=1).dropna(); mt=m[m.index<pd.Timestamp(f'{yr}-01-01')]
    if len(mt)<250: continue
    b=np.linalg.lstsq(np.c_[np.ones(len(mt)),mt.iloc[:,:3].values],mt.iloc[:,3].values,rcond=None)[0]
    te=D.index[(D.index>=pd.Timestamp(f'{yr}-01-01'))&(D.index<pd.Timestamp(f'{yr+1}-01-01'))]; xt=Xh.reindex(te); har[te]=b[0]+xt.values@b[1:]
har_ratio=har/rv22.shift(1)
# GJR walk-forward one-step vol forecast
r=(np.log(D.C).diff()*100).dropna(); gjr=pd.Series(np.nan,index=D.index)
for yr in range(2016,2027):
    trr=r[r.index<pd.Timestamp(f'{yr}-01-01')]
    if len(trr)<500: continue
    res=arch_model(trr,mean='Constant',vol='GARCH',p=1,o=1,q=1).fit(disp='off'); P=res.params; om,al,ga,be=P['omega'],P['alpha[1]'],P['gamma[1]'],P['beta[1]']
    s2=res.conditional_volatility.iloc[-1]**2; rl=trr.iloc[-1]-P['mu']; s2n=om+(al+ga*(rl<0))*rl**2+be*s2
    for d in r.index[(r.index>=pd.Timestamp(f'{yr}-01-01'))&(r.index<pd.Timestamp(f'{yr+1}-01-01'))]:
        gjr[d]=np.sqrt(s2n); rr=r[d]-P['mu']; s2n=om+(al+ga*(rr<0))*rr**2+be*s2n
gjr_ratio=gjr/(r.rolling(20).std().shift(1))
# GEX / vxn / balance
g=pd.read_csv('/home/claude/fartpie/xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); g['pcr']=g.put_vol/g.call_vol.replace(0,np.nan)
gx=g.net_gex.rolling(252,min_periods=120).rank(pct=True); pcrp=g.pcr.rolling(252,min_periods=120).rank(pct=True)
Y2=yf.download(['^VXN'],start='2015-01-01',interval='1d',progress=False,auto_adjust=True)['Close'].ffill(); Y2.index=pd.to_datetime(Y2.index).normalize(); vxp=Y2.iloc[:,0].rolling(252,min_periods=120).rank(pct=True)
def pr(s,d): return s.reindex([pd.Timestamp(d)-pd.Timedelta(days=1)],method='ffill').iloc[0]
# VP/TPO prior-day levels
LV={d:prof(d) for d in Z.ALLD}; prev={d:Z.ALLD[i-1] for i,d in enumerate(Z.ALLD) if i>0}
G=pd.read_pickle(SP+'/G_noise.pkl'); G['day']=pd.to_datetime(G.day).dt.normalize(); G=G[G.day>='2018-02-01'].reset_index(drop=True)
r20=D.rng.rolling(20).mean().shift(1); rows=[]
for t in G.itertuples():
    a=A[t.day][0]; j=int((t.tod-570)//5)+1; ep=a[0][j] if j<len(a[0]) else np.nan; lv=LV.get(prev.get(t.day)); R=r20.get(t.day,np.nan)
    if lv is None or not np.isfinite(ep) or not np.isfinite(R): rows.append([np.nan]*7); continue
    H,L,POC,VAH,VAL=lv; s=t.side
    rows.append([(ep-POC)*s/R,(ep-(VAH if s>0 else VAL))*s/R,float(ep>VAH if s>0 else ep<VAL),((H-ep) if s>0 else (ep-L))/R,float(VAL<=ep<=VAH),0,0])
V=pd.DataFrame(rows,columns=['d_poc','d_va_edge','out_of_va','room_pHL','in_va','_a','_b']).drop(columns=['_a','_b'])
G=pd.concat([G,V],axis=1)
G['yz_pct']=[yzp.reindex([d]).iloc[0] if d in yzp.index else np.nan for d in G.day]; G['har_ratio']=har_ratio.reindex(G.day).values; G['gjr_ratio']=gjr_ratio.reindex(G.day).values
G['gex_pct']=[pr(gx,d) for d in G.day]; G['gex_pos']=[float(pr(g.net_gex,d)>0) for d in G.day]; G['pcr_pct']=[pr(pcrp,d) for d in G.day]; G['vxn_pct']=[pr(vxp,d) for d in G.day]
Gf=G.copy(); win=Gf.pts>0; Gf['y_win']=win.astype(int); Gf['big']=(Gf.pts>=100).astype(int); Gf['ge40']=(Gf.pts>=40).astype(int)
FEAT=['dist','vwd','ret','rngs','gap','rv','rv3','wick','acc','er','sinceH','sinceL','d_poc','d_va_edge','out_of_va','room_pHL','yz_pct','har_ratio','gjr_ratio','gex_pct','gex_pos','pcr_pct','vxn_pct']
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=15 and len(b)>=15 else np.nan
print(f'trades {len(Gf)} | win {Gf.y_win.mean():.0%} | >=40 {Gf.ge40.mean():.0%} | >=100 {Gf.big.mean():.1%} | big share among winners {(Gf.big[win]).mean():.0%}')
print('\nFEATURE | P(win) lo->hi | P(big|win) lo->hi | mean pts lo->hi | per-period high-low: win%, big|win%, pts (t)  [* = same sign & |t|>=2 both periods on that metric]')
rep=[]
for f in FEAT:
    x=Gf.dropna(subset=[f]); 
    if x[f].nunique()<3: continue
    out={}
    for lab,m in (('a',x.day.dt.year<2022),('b',x.day.dt.year>=2022)):
        z=x[m]; q=np.nanquantile(z[f],[1/3,2/3]); lo=z[z[f]<=q[0]]; hi=z[z[f]>=q[1]]
        if z[f].nunique()==2: lo=z[z[f]==0]; hi=z[z[f]==1]
        out[lab]=(hi.y_win.mean()-lo.y_win.mean(), hi.big[hi.pts>0].mean()-lo.big[lo.pts>0].mean(), hi.pts.mean()-lo.pts.mean(), tt(hi.y_win,lo.y_win), tt(hi.big[hi.pts>0],lo.big[lo.pts>0]), tt(hi.pts,lo.pts))
    flag=lambda i: '*' if (np.sign(out['a'][i])==np.sign(out['b'][i]) and abs(out['a'][i+3])>=2 and abs(out['b'][i+3])>=2) else ''
    print(f' {f:10} win {out["a"][0]*100:+4.0f}/{out["b"][0]*100:+4.0f}pp (t {out["a"][3]:+.1f}/{out["b"][3]:+.1f}){flag(0)} | big|win {out["a"][1]*100:+4.0f}/{out["b"][1]*100:+4.0f}pp (t {out["a"][4]:+.1f}/{out["b"][4]:+.1f}){flag(1)} | pts {out["a"][2]:+5.1f}/{out["b"][2]:+5.1f} (t {out["a"][5]:+.1f}/{out["b"][5]:+.1f}){flag(2)}')
# walk-forward logistic
use=[f for f in FEAT if Gf[f].notna().mean()>.8]; Gm=Gf.dropna(subset=use).copy(); print('\nwalk-forward L2 logistic (refit each Jan on prior years), features used',len(use),'trades',len(Gm))
for tgt,sub in (('y_win',None),('big','win')):
    pr_=pd.Series(np.nan,index=Gm.index)
    for yr in range(2020,2027):
        tr=Gm[Gm.day<pd.Timestamp(f'{yr}-01-01')]; te=Gm[(Gm.day>=pd.Timestamp(f'{yr}-01-01'))&(Gm.day<pd.Timestamp(f'{yr+1}-01-01'))]
        if sub: tr=tr[tr.pts>0]; te=te[te.pts>0]
        if len(te)<20 or tr[tgt].nunique()<2: continue
        ss=StandardScaler().fit(tr[use]); m=LogisticRegression(C=0.05,max_iter=500).fit(ss.transform(tr[use]),tr[tgt]); pr_[te.index]=m.predict_proba(ss.transform(te[use]))[:,1]
    o=Gm.join(pr_.rename('p')).dropna(subset=['p']); 
    if sub: o=o[o.pts>0]
    a=[roc_auc_score(o[tgt][m],o.p[m]) if o[tgt][m].nunique()>1 else np.nan for m in (o.day.dt.year<2023,o.day.dt.year>=2023)]
    q=pd.qcut(o.p,5,labels=False); print(f' target {tgt:5}{"|win" if sub else "    "}: OOS AUC 2020-22 {a[0]:.3f} | 2023-26 {a[1]:.3f} | n {len(o)} | by predicted quintile Q1..Q5 mean pts: '+' '.join(f'{o.pts[q==k].mean():+6.1f}' for k in range(5))+' | big rate '+' '.join(f'{o.big[q==k].mean():.0%}' for k in range(5)))

# ---- use the SIZE model on ALL trades (what you would know ex ante): score = P(big | win) from the big|win model, trained walk-forward on winners only
sc=pd.Series(np.nan,index=Gm.index)
for yr in range(2020,2027):
    tr=Gm[(Gm.day<pd.Timestamp(f'{yr}-01-01'))&(Gm.pts>0)]; te=Gm[(Gm.day>=pd.Timestamp(f'{yr}-01-01'))&(Gm.day<pd.Timestamp(f'{yr+1}-01-01'))]
    if len(te)<20: continue
    ss=StandardScaler().fit(tr[use]); m=LogisticRegression(C=0.05,max_iter=500).fit(ss.transform(tr[use]),tr['big']); sc[te.index]=m.predict_proba(ss.transform(te[use]))[:,1]
O=Gm.join(sc.rename('sc')).dropna(subset=['sc']); print(f'\nSIZE-MODEL SCORE applied to ALL trades, OOS 2020-2026 (n={len(O)}):')
for lab,mk in (('2020-22',O.day.dt.year<2023),('2023-26',O.day.dt.year>=2023)):
    x=O[mk]; q=pd.qcut(x.sc,5,labels=False)
    print(f'  {lab}: quintile Q1..Q5 | win% '+' '.join(f'{x.y_win[q==k].mean():4.0%}' for k in range(5))+' | mean pts '+' '.join(f'{x.pts[q==k].mean():+6.1f}' for k in range(5))+' | big% '+' '.join(f'{x.big[q==k].mean():4.1%}' for k in range(5))+f' | Q5-Q1 pts t {tt(x.pts[q==4],x.pts[q==0]):+.2f}')
# per-day sizing on the noise engine: multiplier by trailing score quantile; day pts = sum of mult*pts
O=O.sort_values(['day','tod']); lo=O.sc.expanding(100).quantile(.2).shift(1); hi=O.sc.expanding(100).quantile(.8).shift(1)
S=lambda x: x.mean()/x.std()*np.sqrt(252)
for up,dn in ((1.5,0.75),(2.0,0.5),(1.0,0.5),(2.0,1.0)):
    m=pd.Series(1.0,index=O.index); m[O.sc>=hi]=up; m[O.sc<=lo]=dn; O['w']=O.pts*m; o=[]
    for lab,mk in (('2020-22',O.day.dt.year<2023),('2023-26',O.day.dt.year>=2023)):
        x=O[mk]; b=x.groupby('day').pts.sum(); s2=x.groupby('day').w.sum(); o.append(f'{lab}: Sharpe {S(b):.2f}->{S(s2):.2f} pts {b.sum():+.0f}->{s2.sum():+.0f} worst day {b.min():+.0f}->{s2.min():+.0f}')
    print(f'  size top20% x{up} / bottom20% x{dn}: '+' | '.join(o))
