"""True intraday alert sim, last ~60 sessions (Yahoo 5m incl. pre/post, 2,018 $0.30-20 stocks). Declared before results.
Alert = first bar of a day where cumulative day volume (incl premarket) >= VM x median daily volume of prior 20 sessions AND price >= prior close*(1+GM), price $0.30-20 at alert.
Entry = NEXT 5m bar open +1% slippage (illiquid), exit = last regular-session bar close (15:55 bar) [alerts after 15:30 skipped], also with -10% hard stop (checked on bar lows, stop fill = stop price -1%).
Report by (VM,GM) grid and by alert time bucket (premarket / 9:30-10:30 / 10:30-15:30), and by gain-at-alert bucket.  ~60 days only: tiny sample, no train/test; treat as sanity check."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; D5=pickle.load(open(SC+'lowpx_5m.pkl','rb')); DD=pickle.load(open(SC+'lowpx_daily.pkl','rb'))
R=[]
for tk,X in D5.items():
    d=DD[tk]; d=d[~d.index.duplicated()]; dv=d.Volume.astype(float); med=dv.shift(1).rolling(20).median(); dc=d.Close
    X=X.copy(); X.index=X.index.tz_convert('America/New_York') if X.index.tz is not None else X.index; days=X.index.normalize().tz_localize(None); 
    for day,g in X.groupby(days):
        dk=pd.Timestamp(day); 
        if dk not in med.index or not np.isfinite(med.get(dk,np.nan)) or med[dk]<=0: continue
        j=dc.index.get_loc(dk)
        if j<1: continue
        pcl=dc.iloc[j-1]; 
        if not(0.3<=pcl<=20): continue
        cv=g.Volume.cumsum().values; c=g.Close.values; o=g.Open.values; lo=g.Low.values; hi=g.High.values; tm=g.index.hour*60+g.index.minute
        reg=np.where((tm>=570)&(tm<=955))[0]
        if len(reg)<20: continue
        last=reg[-1]; exitp=c[last]
        for VM in (10,20,50):
            for GM in (.15,.30):
                ok=np.where((cv>=VM*med[dk])&(c>=pcl*(1+GM)))[0]
                if len(ok)==0: continue
                a=ok[0]
                if tm[a]>=930 or a+1>=len(g) or a+1>last: continue
                en=o[a+1]*1.01; mx=hi[a+1:last+1].max(); mn=lo[a+1:last+1].min()
                r_close=exitp/en-1-.005; stop=en*.90; hit=np.where(lo[a+1:last+1]<=stop)[0]
                r_stop=(stop*.99/en-1-.005) if len(hit) else r_close
                bucket='pre' if tm[a]<570 else ('open' if tm[a]<630 else 'mid')
                R.append(dict(tk=tk,day=dk,VM=VM,GM=GM,bucket=bucket,gain_at=c[a]/pcl-1,en=en/ (pcl) -1,rc=r_close,rs=r_stop,mfe=mx/en-1,mae=mn/en-1,day_hi=hi[reg].max()/pcl-1,runner=float(hi[reg].max()/pcl>=1.5)))
E=pd.DataFrame(R); E.to_pickle(SC+'alert5m_E.pkl'); print('alerts',len(E),'days',E.day.nunique())
def st(g):
    if len(g)<8: return 'n%d'%len(g)
    a=g.rc.sort_values(); return 'n%-4d P(runner>=+50%% day) %2.0f%% | to-close mean %+6.2f%% med %+6.2f%% win %2.0f%% ex-top1 %+6.2f | w/-10%%stop mean %+6.2f%% | MFE med %+5.1f%% MAE med %+5.1f%%'%(len(g),g.runner.mean()*100,g.rc.mean()*100,g.rc.median()*100,(g.rc>0).mean()*100,a.iloc[:max(int(len(a)*.99),1)].mean()*100,g.rs.mean()*100,g.mfe.median()*100,g.mae.median()*100)
for VM in (10,20,50):
    for GM in (.15,.30): print('VM>=%d gain>=%d%% :'%(VM,GM*100),st(E[(E.VM==VM)&(E.GM==GM)]))
B=E[(E.VM==20)&(E.GM==.15)]; print('\n-- VM20/G15 by alert time'); [print('  ',b,st(B[B.bucket==b])) for b in ('pre','open','mid')]
print('-- VM20/G15 by gain already at alert'); [print('   %3d-%3s%%'%(lo*100,'inf' if hi>9 else int(hi*100)),st(B[(B.gain_at>=lo)&(B.gain_at<hi)])) for lo,hi in ((.15,.30),(.30,.60),(.60,1.0),(1.0,99))]
