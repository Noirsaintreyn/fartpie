"""Report 253: CAUSAL predictors of mean-reverting days + VIX. DECLARED before results. Generator = report 252 reversion fade (k in .5,.75 ATR from VWAP, stop 1.0 ATR beyond, target VWAP; k=.5 added after 252 for sample size). Predictors known before entry, hypothesis directions fixed: MR if (a) VIX prior close below its prior-5d mean (vol falling), (b) |open gap|/ATR below median, (c) QQQ prior-close net GEX > 0 (positive gamma pins), (d) QQQ ATM IV prior close below trailing-252 median? (control), (e) prior-day ER low (control). Check 1: day-level AUC of the predictor for MR vs TREND (ex-post label). Check 2: fade pts per trade with the predictor ON vs OFF, train 2019-21 vs test 2018+2022-26. Also: main algo daily pts correlation with VIX change/level."""
"""Report 252: mean-reversion CEILING test. DECLARED before results.
Day type (EX-POST, hindsight, ceiling only): RTH 09:30-16:00 efficiency ER = |close-open|/(high-low); MR day = bottom third of ER within the split (tertiles over all days), TREND = top third.
Generic reversion model (no tuning beyond the declared 2x2 grid): when price (5m close) is k x ATR14(daily RTH range) away from session VWAP (k in 0.75, 1.0), after 10:00 and before 15:00, enter AGAINST at next 5m open; stop = entry-side extreme + s x ATR beyond entry (s in 0.5, 1.0) ; target = VWAP; flat 15:55; one trade per day per direction; cost 1 pt.
Report net pts/trade and t by day type and by split (train 2019-21, test 2018 + 2022-26). If it loses on MR days even with perfect day-type knowledge = payoff problem; if it wins = identification problem.
Also: main algo daily pts by ER tercile (shareable250.pkl daily, deploy size, NQ pts)."""
import pandas as pd, numpy as np, pickle
df=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); df['dt']=pd.to_datetime(df.dt); df=df.set_index('dt').sort_index()
df=df.between_time('09:30','15:55'); df['d']=df.index.normalize()
days=[]; trades=[]
D=df.groupby('d')
dd=pd.DataFrame({'o':D.open.first(),'h':D.high.max(),'l':D.low.min(),'c':D.close.last(),'n':D.size()}); dd=dd[dd.n>=70]
dd['rng']=dd.h-dd.l; dd['er']=(dd.c-dd.o).abs()/dd.rng; dd['atr']=dd.rng.rolling(14).mean().shift(1)
dd['type']=pd.qcut(dd.er,3,labels=['MR','MID','TREND'])
for k in (0.5,0.75):
  for sm in (1.0,):
    for d,g in D:
        if d not in dd.index or np.isnan(dd.atr[d]): continue
        a=dd.atr[d]; h=g.high.values;l=g.low.values;c=g.close.values;o=g.open.values;v=g.volume.values; t=g.index
        tp=(h+l+c)/3; vw=np.cumsum(tp*v)/np.maximum(np.cumsum(v),1)
        done={1:False,-1:False}
        for i in range(6,len(g)-2):
            if t[i].hour>=15: break
            dev=(c[i]-vw[i])/a
            for dr in (1,-1):   # dr=1 long fade when below VWAP
                if done[dr] or dev*(-dr)<k: continue
                e=o[i+1]; stop=(l[i] if dr==1 else h[i]) - dr*sm*a; done[dr]=True
                res=None
                for j in range(i+1,len(g)):
                    if dr==1:
                        if l[j]<=stop: res=stop-e;break
                        if h[j]>=vw[i]+0*1 and h[j]>=vw[j]: res=vw[j]-e;break
                    else:
                        if h[j]>=stop: res=e-stop;break
                        if l[j]<=vw[j]: res=e-vw[j];break
                if res is None: res=dr*(c[-1]-e)
                trades.append((k,sm,d,dr,res-1.0))
T=pd.DataFrame(trades,columns=['k','sm','d','dr','pts']); T['type']=T.d.map(dd.type)

vix=pd.read_parquet('data/yahoo/daily_v/VIX.parquet').close; vix.index=pd.to_datetime(vix.index).normalize()
gx=pd.read_csv('xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date')
dd['pc']=dd.c.shift(1); dd['gap']=(dd.o-dd.pc).abs()/dd.atr; dd['per']=dd.er.shift(1)
prev=lambda s: s.shift(1).reindex(dd.index,method='ffill')
v1=vix.shift(1); v5=vix.shift(1).rolling(5).mean()
dd['vixfall']=((v1<v5).reindex(dd.index,method='ffill')).astype(float); dd['vix_prev']=v1.reindex(dd.index,method='ffill')
dd['gexpos']=(gx.net_gex.shift(1)>0).reindex(dd.index,method='ffill').astype(float)
dd['ivlow']=(gx.atm_iv.shift(1)<gx.atm_iv.shift(1).rolling(252).median()).reindex(dd.index,method='ffill').astype(float)
dd['gaplow']=(dd.gap<dd.gap.median()).astype(float); dd['perlow']=(dd.per<dd.per.median()).astype(float)
dd['vixhi']=(dd.vix_prev>dd.vix_prev.rolling(252).median()).astype(float)
dd['yr']=dd.index.year; dd['split']=np.where((dd.yr>=2019)&(dd.yr<=2021),'train','test')
from itertools import product
def auc(x,y):
    from scipy.stats import rankdata
    r=rankdata(x); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
print('--- day-level AUC (feature ON predicts MR vs TREND; 0.5=none)')
mt=dd[dd.type.isin(['MR','TREND'])].dropna(subset=['vixfall','gaplow','gexpos','ivlow','perlow','vixhi'])
for f in ['vixfall','vixhi','gaplow','gexpos','ivlow','perlow']:
    print(f'  {f:8s}', ' '.join(f'{sp} {auc(g[f].values,(g.type=="MR").astype(int).values):.3f} (n {len(g)})' for sp,g in mt.groupby('split')))
T['split']=np.where((T.d.dt.year>=2019)&(T.d.dt.year<=2021),'train','test')
T=T.merge(dd[['vixfall','vixhi','gaplow','gexpos','ivlow','perlow']],left_on='d',right_index=True)
print('--- fade pts/trade, feature ON vs OFF (train | test), k=.75 and .5')
for (k,sm),g in T.groupby(['k','sm']):
    print(f' k={k}  all: '+' | '.join(f'{sp} n {len(h)} {h.pts.mean():+.1f}' for sp,h in g.groupby('split')))
    for f in ['vixfall','vixhi','gaplow','gexpos','ivlow','perlow']:
        out=[]
        for on in (1,0):
            out.append(' / '.join(f'{sp[:2]} n{len(h)} {h.pts.mean():+.1f}' for sp,h in g[g[f]==on].groupby('split')))
        print(f'   {f:8s} ON: {out[0]}   OFF: {out[1]}')
    g2=g[(g.gexpos==1)&(g.gaplow==1)]; print('   gexpos&gaplow: '+' / '.join(f'{sp[:2]} n{len(h)} {h.pts.mean():+.1f}' for sp,h in g2.groupby('split')))
    g3=g[(g.vixfall==1)&(g.gexpos==1)]; print('   vixfall&gexpos: '+' / '.join(f'{sp[:2]} n{len(h)} {h.pts.mean():+.1f}' for sp,h in g3.groupby('split')))
Z=pickle.load(open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/shareable250.pkl','rb')); dly=Z['daily']/20
x=pd.DataFrame({'p':dly}); x['dv']=vix.diff().reindex(x.index); x['v']=vix.shift(1).reindex(x.index,method='ffill'); x['er']=dd.er.reindex(x.index)
print('--- main algo daily pts: corr with VIX change same day',round(x.p.corr(x.dv),3),'| with prior VIX level',round(x.p.corr(x.v),3),'| with ER',round(x.p.corr(x.er),3))
print('    pts/day by prior VIX tercile', x.groupby(pd.qcut(x.v,3,labels=['low','mid','high'])).p.mean().round(2).to_dict())
