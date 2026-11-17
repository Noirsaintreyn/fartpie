"""Report 257: (A) does implied vol tell the day type? (B) AMD / PO3 / Wyckoff-style sweep-and-reclaim as the day-type model. DECLARED before results (Rey Nov 17).
A: QQQ prior-close ATM IV -> implied daily move = IV/sqrt(252). Features (causal, prior close): IV rank (trailing 252), IV change prior day, skew_25d, ratio_prev = prior-day NQ RTH range% / implied move (realized vs implied), 5-day mean of ratio_prev. Target: today's RTH efficiency ER and today's range / implied move. Spearman by split (train 2019-21 | test 2018+2022-26). Hypothesis: low ratio_prev (realized << implied = vol being sold) and falling IV -> chop (low ER, small range/implied). Also the contemporaneous claim: IV falls (day-over-day change) on choppy days.
B: ACCUMULATION = Asia range 20:00-02:00 ET prior evening->morning (qualifier: range <= 1.0 ATR14 'tight'). MANIPULATION = price takes out the Asia high (low) before or during 09:30-11:30 and a 5m close comes back inside the range (sweep + reclaim). DISTRIBUTION = trade toward the opposite side: short after a high sweep / long after a low sweep; entry next 5m open after the reclaim close; stop = sweep extreme +/-1 pt; R>=10 (cap 150); targets 1R, 2R, and the opposite side of the Asia range (TGT_A); flat 15:55; cost 1 pt. Wyckoff variant: same with the prior-day range (PDH/PDL sweep + reclaim, spring/upthrust). Day-type check: P(MR/MID/TREND day) for days with a reclaim vs sweep that holds (5m close >= .25 ATR beyond, never reclaims by 11:30) vs no sweep.
Pass = same sign and meaningful size in train and test."""
import numpy as np, pandas as pd, pickle
from scipy.stats import spearmanr
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index()
rth=b.between_time('09:30','15:55'); D=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'o':D.open.first(),'h':D.high.max(),'l':D.low.min(),'c':D.close.last(),'n':D.size()}); dd=dd[dd.n>=70]
dd['rng']=dd.h-dd.l; dd['er']=(dd.c-dd.o).abs()/dd.rng; dd['atr']=dd.rng.rolling(14).mean().shift(1); dd['type']=pd.qcut(dd.er,3,labels=['MR','MID','TREND'])
dd['yr']=dd.index.year; dd['split']=np.where((dd.yr>=2019)&(dd.yr<=2021),'train','test'); dd=dd[dd.yr>=2018]
ov=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); ov.index=ov.index.normalize()
iv=ov.atm_iv; ivr=iv.rolling(252,min_periods=120).rank(pct=True); ivd=iv.diff(); sk=ov.skew_25d
def prior(s): 
    k=s.index.searchsorted(dd.index.values.astype('datetime64[ns]'))-1; k=np.clip(k,0,len(s)-1); return pd.Series(s.values[k],index=dd.index)
dd['ivr']=prior(ivr); dd['ivd']=prior(ivd); dd['sk']=prior(sk); dd['imp']=prior(iv)/np.sqrt(252)
dd['rpct']=dd.rng/dd.o; dd['ratio']=dd.rpct/dd.imp; dd['ratio_prev']=dd.ratio.shift(1); dd['ratio5']=dd.ratio.shift(1).rolling(5).mean()
dd['ivchg_today']=pd.Series(ivd.reindex(dd.index,method=None).values,index=dd.index)   # contemporaneous (IV at close d minus d-1)
print('=== A: IV vs day type (Spearman rho with today ER | today range/implied), train | test')
for f in ['ivr','ivd','sk','ratio_prev','ratio5']:
    out=[]
    for sp in ('train','test'):
        g=dd[dd.split==sp].dropna(subset=[f,'er','ratio']); out.append((spearmanr(g[f],g.er)[0],spearmanr(g[f],g.ratio)[0]))
    print(f'  {f:11s} ER {out[0][0]:+.3f}|{out[1][0]:+.3f}   range/implied {out[0][1]:+.3f}|{out[1][1]:+.3f}')
out=[]
for sp in ('train','test'):
    g=dd[dd.split==sp].dropna(subset=['ivchg_today','er']); out.append(spearmanr(g.ivchg_today,g.er)[0])
print(f'  contemporaneous: IV change on the day vs that day ER rho {out[0]:+.3f}|{out[1]:+.3f} (negative = IV falls on choppy days... sign: positive rho means IV rises on trend days)')
print('  mean range/implied by ER tercile:',dd.groupby('type').ratio.mean().round(2).to_dict())
# ---- B
rows=[]; tag=[]
bi=b.index
def tt(x): return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>2 and x.std()>0 else np.nan
pdh=dd.h.shift(1); pdl=dd.l.shift(1)
import os; LAG=int(os.environ.get('LAG',0)); ASIA={}
for d_ in dd.index:
    a_=b.loc[d_-pd.Timedelta(hours=4):d_+pd.Timedelta(hours=2)-pd.Timedelta(minutes=5)]
    if len(a_)>=30: ASIA[d_]=(a_.high.max(),a_.low.min())
DL=list(dd.index)
for d,g in D:
    if d not in dd.index or not np.isfinite(dd.atr[d]): continue
    a=dd.atr[d]
    ii=DL.index(d)-LAG
    if ii<0 or DL[ii] not in ASIA or d not in ASIA: continue
    AH,AL=ASIA[DL[ii]]; tight=(AH-AL)<=a
    pre=b.loc[d+pd.Timedelta(hours=2):d+pd.Timedelta(hours=9,minutes=29)]
    o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;t=g.index; mm=t.hour*60+t.minute
    for lab,H_,L_ in (('asia',AH,AL),('pd',pdh[d],pdl[d])):
        if not np.isfinite(H_): continue
        sw_h=bool(len(pre) and pre.high.max()>H_); sw_l=bool(len(pre) and pre.low.min()<L_)
        mh=pre.high.max() if len(pre) else -np.inf; ml=pre.low.min() if len(pre) else np.inf
        done={'S':False,'L':False}; first=None; held=False
        for i in range(len(g)-2):
            if mm[i]>690: break
            if h[i]>H_: sw_h=True; mh=max(mh,h[i])
            if l[i]<L_: sw_l=True; ml=min(ml,l[i])
            if sw_h and not done['S'] and c[i]<H_ and mm[i]<=690:
                done['S']=True; e=o[i+1]; stop=mh+1; R=stop-e; first=first or 'rc'
                if 10<=R<=150:
                    res={}
                    for nm,tgt in (('r1',e-R),('r2',e-2*R),('ta',L_ if L_<e-5 else e-R)):
                        r=None
                        for j in range(i+1,len(g)):
                            if h[j]>=stop: r=-R;break
                            if l[j]<=tgt: r=e-tgt;break
                            if mm[j]>=955: r=e-c[j];break
                        if r is None: r=e-c[-1]
                        res[nm]=r-1
                    rows.append((d,lab,-1,R,res['r1'],res['r2'],res['ta'],tight))
            if sw_l and not done['L'] and c[i]>L_ and mm[i]<=690:
                done['L']=True; e=o[i+1]; stop=ml-1; R=e-stop; first=first or 'rc'
                if 10<=R<=150:
                    res={}
                    for nm,tgt in (('r1',e+R),('r2',e+2*R),('ta',H_ if H_>e+5 else e+R)):
                        r=None
                        for j in range(i+1,len(g)):
                            if l[j]<=stop: r=-R;break
                            if h[j]>=tgt: r=tgt-e;break
                            if mm[j]>=955: r=c[j]-e;break
                        if r is None: r=c[-1]-e
                        res[nm]=r-1
                    rows.append((d,lab,1,R,res['r1'],res['r2'],res['ta'],tight))
        # day-type tag
        sw=sw_h or sw_l; reclaimed=done['S'] or done['L']
        tag.append((d,lab,'reclaim' if reclaimed else ('sweep_hold' if sw else 'none')))
T=pd.DataFrame(rows,columns=['d','lab','dr','R','r1','r2','ta','tight']); T['yr']=T.d.dt.year; T['split']=np.where((T.yr>=2019)&(T.yr<=2021),'train','test'); T=T[T.yr>=2018]
G=pd.DataFrame(tag,columns=['d','lab','state']).merge(dd[['type','er','split']],left_on='d',right_index=True)
print('\n=== B: sweep + reclaim then trade toward the opposite side (net pts, train | test)')
for lab in ('asia','pd'):
    for cond,nm in ((np.ones(len(T),bool),'all'),(T.tight.values,'tight Asia range')):
        if lab=='pd' and nm!='all': continue
        g=T[(T.lab==lab)&cond]
        print(f' {lab:5s} {nm:16s} n {len(g)} ({len(g)/8:.0f}/yr)',end='')
        for sp in ('train','test'):
            x=g[g.split==sp]; print(f' | {sp[:2]} n{len(x)} 1R {x.r1.mean():+5.1f}(t{tt(x.r1):+.1f}) 2R {x.r2.mean():+5.1f}(t{tt(x.r2):+.1f}) opp-side {x.ta.mean():+5.1f}(t{tt(x.ta):+.1f})',end='')
        print()
print(' by year (asia, 1R / opp-side):',{y:(int(len(x)),round(x.r1.mean(),1),round(x.ta.mean(),1)) for y,x in T[T.lab=='asia'].groupby('yr')})
print('\n=== day type by state (share MR / MID / TREND, mean ER) train | test')
for lab in ('asia','pd'):
    for st in ('none','sweep_hold','reclaim'):
        s=[]
        for sp in ('train','test'):
            x=G[(G.lab==lab)&(G.state==st)&(G.split==sp)]; s.append(f'n{len(x)} MR {(x.type=="MR").mean():.0%} TR {(x.type=="TREND").mean():.0%} ER {x.er.mean():.2f}')
        print(f' {lab:5s} {st:11s} {s[0]} | {s[1]}')
pickle.dump((T,G),open(SP+f'amd257_lag{LAG}.pkl','wb'))
x=T[T.lab=='asia']
for sp in ('train','test'):
    y=x[x.split==sp]; print(f'  {sp}: median R {y.R.median():.0f} pts, win@1R {(y.r1>0).mean():.0%}, win@2R {(y.r2>0).mean():.0%}, avg win2R {y[y.r2>0].r2.mean():.0f} avg loss2R {y[y.r2<=0].r2.mean():.0f}')
