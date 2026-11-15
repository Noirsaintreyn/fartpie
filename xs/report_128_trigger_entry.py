"""report_128 (Nov 15 2026): ENTER AT THE VECTOR TRIGGER INTRADAY (idea 2: "we enter so late"). Daily-bar approximation over the full 2019-26 universe (report_88 had only 60 days of 5m). Trigger T[i] = level[i-1] + 0.35*ATR[i-1] while the state is not bull (reproduces 435/437 flips in report_88). Cross on day i if high[i] >= T. Entry = max(open[i], T). If the day closes as a TRUE flip (state bull) hold to the usual bear-flip exit (exit price = open[i+1]*(1+r) from the flip table); if it closes back below (FALSE cross) exit at the close of day i. Cost 30bp. Compared per signal with the baseline (true flips only, next-open entry), and with the frozen OS/score filter known BEFORE the cross (ATRP, EXT, turnover are prior-close features)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D=pd.read_pickle(X+'wide_daily_ohlcv.pkl'); O=pd.read_pickle(X+'combined_scores.pkl'); COST=.003
F=O.set_index(['tk','d']); rows=[]
for tk,q in D.items():
    q=q[['open','high','low','close','volume']].copy(); q=q[~q.index.duplicated()].dropna(); q.index=pd.to_datetime(q.index).tz_localize(None).normalize()
    if len(q)<300: continue
    reg,lev,atr,_=run_state_machine(q.assign(datetime=q.index)); reg=np.asarray(reg); lev=np.asarray(lev,float); atr=np.asarray(atr,float)
    o,h,l,c=[q[k].values.astype(float) for k in ('open','high','low','close')]; dates=q.index
    pcl=np.r_[c[0],c[:-1]]; trr=np.maximum(h-l,np.maximum(abs(h-pcl),abs(l-pcl))); a14=pd.Series(trr).rolling(14).mean().shift(1).values; lo60=pd.Series(l).rolling(60).min().shift(1).values
    for i in range(130,len(q)-2):
        if reg[i-1]==1 or dates[i]<pd.Timestamp('2019-01-01'): continue
        T=lev[i-1]+0.35*atr[i-1]
        if not (T==T) or h[i]<T or c[i-1]<1 or c[i-1]>=20: continue
        E=max(o[i],T); true=(reg[i]==1)
        if true:
            k=(tk,dates[i])
            if k not in F.index: continue
            f=F.loc[k]; f=f.iloc[0] if isinstance(f,pd.DataFrame) else f
            ex=o[i+1]*(1+f.r); ret=ex/E-1-COST; base=f.n; key=dict(os=f.OS,lr=f.lr,mb=f['m_BIG_0.5'],test=f.test)
        else:
            ret=c[i]/E-1-COST; base=np.nan; key=dict(os=np.nan,lr=np.nan,mb=np.nan,test=dates[i]>=pd.Timestamp('2022-01-01'))
        rows.append(dict(ATRP=a14[i]/c[i-1],EXT=c[i-1]/lo60[i]-1,tk=tk,d=dates[i],true=true,ret=ret,base=base,px=c[i-1],gapup=o[i]/T-1,**key))
R=pd.DataFrame(rows); R.to_pickle(X+'trigger_entry.pkl'); print(len(R),'crosses; true',R.true.sum())
for era,e in (('TRAIN 2019-21',~R.test),('TEST 2022+',R.test)):
    for lab,m in (('px 1-5',(R.px<5)),('px 5-20',(R.px>=5))):
        z=R[e&m]; t=z[z.true]
        print(f'{era} {lab}: crosses {len(z):5d} true {len(t):4d} ({len(t)/max(1,len(z))*100:3.0f}%) | baseline true flips (next open) mean {t.base.mean()*100:+6.2f}% | trigger entry TRUE {t.ret.mean()*100:+6.2f}% FALSE {z[~z.true].ret.mean()*100:+7.2f}% -> per cross {z.ret.mean()*100:+6.2f}% (if only true flips counted at trigger: gain {(t.ret-t.base).mean()*100:+.2f}pp)')
tr=O[~O.test]; ec=lambda col:(lambda x,s=np.sort(tr[col].dropna().values):np.searchsorted(s,x,side='right')/len(s))
R=R.dropna(subset=['ATRP','EXT']); R['OS2']=ec('ATRP')(R.ATRP.values)+ec('EXT')(R.EXT.values); cut=R[~R.test].OS2.quantile(.8)
print('\nFILTERED to OS2 top 20% (ATRP+EXT ranks, train cut, known at the prior close), px<5')
for era,e in (('TRAIN',~R.test),('TEST ',R.test)):
    z=R[e&(R.px<5)&(R.OS2>=cut)]; t=z[z.true]; f=z[~z.true]
    print(f'{era}: crosses {len(z):4d} true {len(t):4d} ({len(t)/max(1,len(z))*100:3.0f}%) | baseline true-flip next-open {t.base.mean()*100:+6.2f}% (total {t.base.sum()*100:+7.0f}pp) | trigger: true {t.ret.mean()*100:+6.2f}% false {f.ret.mean()*100:+6.2f}% total {z.ret.sum()*100:+7.0f}pp, per cross {z.ret.mean()*100:+6.2f}%')
print('\nFILTER by score known before the cross: false-cross cost cannot be filtered (OS only exists for true flips here); filter by prior-close features not tested beyond this')
