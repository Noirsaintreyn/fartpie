"""report 151b (CORRECTION: M1/R1/RULES had no direction column in report 151 so all were treated as longs; direction now inferred from the price path, determined trades only, infer_leg_dir.py). Original: report 151: ICT 2022-mentorship ideas never tested here, as context for IFVG / unicorn / plain FVG and the core NQ legs. DECLARED BEFORE RESULTS (Rey: 'test all those').
H1 SILVER BULLET windows: fill minute in 10:00-11:00 or 14:00-15:00 ET (IFVG, unicorn; core legs by entry minute). Prediction: window > outside.
H2 IPDA 20/40/60-day: DRAW_N = trade direction points toward the NEARER of the prior-N-RTH-day high / low measured from the 09:30 open (draw on liquidity); DISC_N = long below / short above the midpoint of the prior-N-day range;
   TREND_N = direction equals sign(prior close - close N days earlier). Prediction: aligned > against (ICT: bias to the draw; buy discount).
H3 CBDR / Asia: NARROW = prior-day CBDR (14:00-20:00 ET) range, resp. Asia (20:00-00:00 ET) range, in the lowest third (cut on 2018-21) -> coiled range -> expansion (explains / improves trend-type entries). WIDE = top third.
H4 NDOG / NWOG: new-day opening gap = [16:55 close, 18:00 open] of the prior session; new-week opening gap = [Fri 16:55 close, Sun 18:00 open] of the current week. TOWARD = entry is outside the gap and the trade points toward it (gap as magnet), AWAY = points away.
Signals: IFVG, UNICORN (report 144 trades; 2R, cost 0.5), plain FVG (report 148; no minute so no H1), core legs RA M1 R1 RULES V15a (s6 V2 trades, report 138 loader; pts per trade; no ent price so no H4).
A 2018-21 | B 2022-26; thresholds cut on A only. PASS = same sign, |t|>=2 both eras, n>=30 each. ~25 slices per signal: expect ~0.1-0.3 chance PASS per signal."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').set_index('dt'); m_=b.index.hour*60+b.index.minute
rt=b[(m_>=570)&(m_<960)]; g=rt.groupby(rt.index.normalize()); RTD=pd.DataFrame({'o':g.open.first(),'h':g.high.max(),'l':g.low.min(),'c':g.close.last()}); days=list(RTD.index)
rows={}
for i,D in enumerate(days):
    if i<61: continue
    r={'open':RTD.o.iloc[i],'pc':RTD.c.iloc[i-1]}
    for N in (20,40,60):
        H=RTD.h.iloc[i-N:i].max(); L=RTD.l.iloc[i-N:i].min(); r[f'H{N}']=H; r[f'L{N}']=L; r[f'tr{N}']=np.sign(RTD.c.iloc[i-1]-RTD.c.iloc[i-1-N])
    P=days[i-1]; cb=b.loc[P+pd.Timedelta(hours=14):P+pd.Timedelta(hours=19,minutes=55)]; r['cbdr']=cb.high.max()-cb.low.min() if len(cb)>40 else np.nan
    asia=b.loc[P+pd.Timedelta(hours=20):P+pd.Timedelta(hours=23,minutes=55)]; r['asia']=asia.high.max()-asia.low.min() if len(asia)>30 else np.nan
    c17=b.loc[P+pd.Timedelta(hours=16,minutes=55):P+pd.Timedelta(hours=16,minutes=55),'close']; o18=b.loc[P+pd.Timedelta(hours=18):P+pd.Timedelta(hours=18,minutes=10),'open']
    # NDOG: prior session close (16:55 bar) vs the 18:00 open that follows it (same calendar evening of day P)
    if len(c17) and len(o18): r['ndog0'],r['ndog1']=sorted((c17.iloc[0],o18.iloc[0]))
    rows[D]=r
C=pd.DataFrame(rows).T; C.index=pd.DatetimeIndex(C.index)
# NWOG: Friday 16:55 close vs the Sunday 18:00 open of the following week -> assign to all days of that week (Mon-Fri) once known
wk={}
for D in days:
    if D.dayofweek==4:
        c=b.loc[D+pd.Timedelta(hours=16,minutes=55):D+pd.Timedelta(hours=16,minutes=55),'close']; o=b.loc[D+pd.Timedelta(days=2,hours=18):D+pd.Timedelta(days=2,hours=18,minutes=10),'open']
        if len(c) and len(o): wk[D+pd.Timedelta(days=3)]=sorted((c.iloc[0],o.iloc[0]))
for D in C.index:
    mon=D-pd.Timedelta(days=D.dayofweek)
    if mon in wk: C.loc[D,'nwog0'],C.loc[D,'nwog1']=wk[mon]
def t_(a): return a.mean()/(a.std()/np.sqrt(len(a))) if len(a)>8 else np.nan
def flags(T,hasm,hasent):
    T=T.join(C,on='d'); T=T[T.open.notna()].copy(); L=T.dr>0
    if hasm: T['SB']=((T.m>=600)&(T.m<660))|((T.m>=840)&(T.m<900)); T['NOT_SB']=~T.SB
    for N in (20,40,60):
        dh=T[f'H{N}']-T.open; dl=T.open-T[f'L{N}']; up=dh<dl; T[f'DRAW{N}']=(L&up)|(~L&~up); T[f'DRAWag{N}']=~T[f'DRAW{N}']
        mid=(T[f'H{N}']+T[f'L{N}'])/2; T[f'DISC{N}']=(L&(T.open<mid))|(~L&(T.open>mid)); T[f'TREND{N}']=(T[f'tr{N}']==np.where(L,1,-1))
    for nm in ('cbdr','asia'):
        q=T[T.A][nm].quantile([1/3,2/3]).values; T[f'{nm}_NARROW']=T[nm]<=q[0]; T[f'{nm}_WIDE']=T[nm]>q[1]
    if hasent:
        for nm,(a,bb) in (('NDOG',('ndog0','ndog1')),('NWOG',('nwog0','nwog1'))):
            out_up=T.ent>T[bb]; out_dn=T.ent<T[a]; ok=T[a].notna()
            T[f'{nm}_TOWARD']=ok&((out_up&~L)|(out_dn&L)); T[f'{nm}_AWAY']=ok&((out_up&L)|(out_dn&~L))
    return T
def line(T,lab,m):
    a,bb=T[m&T.A].pts,T[m&~T.A].pts; ps=len(a)>=30 and len(bb)>=30 and np.sign(a.mean())==np.sign(bb.mean()) and abs(t_(a))>=2 and abs(t_(bb))>=2
    return f'  {lab:16s} A n{len(a):4d} {a.mean():+6.2f} win{100*(a>0).mean():3.0f}% t{t_(a):+5.1f} | B n{len(bb):4d} {bb.mean():+6.2f} win{100*(bb>0).mean():3.0f}% t{t_(bb):+5.1f}{"  <== PASS" if ps else ""}'
def show(name,T,hasm,hasent):
    T=flags(T,hasm,hasent); cols=[c for c in T.columns if c.split('_')[0] in ('SB','NOT','DRAW','DRAWag','DISC','TREND','cbdr','asia','NDOG','NWOG') or c.startswith(('DRAW','DISC','TREND'))]
    cols=[c for c in dict.fromkeys(cols) if T[c].dtype==bool]
    print(f'\n===== {name}: {len(T)} trades'); print(line(T,'ALL',T.pts.notna()))
    for c in cols: print(line(T,c,T[c]))
U=pd.read_pickle(SC+'ext/uni_ifvg.pkl'); F=pd.read_pickle(SC+'ext/fvg_all_vp.pkl')
import os
if not os.getenv('LEGS_ONLY'): show('IFVG',U[U.model=='IFVG'][['d','dr','pts','A','m','ent']],True,True); show('UNICORN',U[U.model=='UNI'][['d','dr','pts','A','m','ent']],True,True); show('FVG plain',F[['d','dr','pts','A','ent']],False,True)
LD=pickle.load(open(SC+'ext/leg_dir.pkl','rb'))
OUT,NAMES=pickle.load(open(SC+'s6.pkl','rb')); BM=dict(pickle.load(open(SC+'base_mv30.pkl','rb')))
for (t,u,y),n in zip(OUT['V2'],NAMES):
    if n in ('ON+','ON-'): continue
    t=t.copy(); t['d']=pd.to_datetime(t.day).dt.normalize(); B_=BM[n].copy(); B_['d']=pd.to_datetime(B_.day).dt.normalize(); B_=B_[B_.sz>0][['d','m','sz']].rename(columns={'sz':'szb'}); t=t.merge(B_,on=['d','m']); t=t[(t.d>='2018-02-01')&(t.d<='2026-01-30')].copy()
    t['A']=t.d<'2022-01-01'
    if 'dir' in t.columns: t['dr']=t['dir']
    else:
        s_=LD[n]; t['dr']=[s_.get((a,m_),0) for a,m_ in zip(t.d,t.m)]; t=t[t.dr!=0]   # direction inferred from the price path (determined trades only)
    show('LEG '+n+(' (dir inferred, determined subset)' if 'dir' not in t.columns or n in ('M1','R1','RULES') else ''),t[['d','dr','pts','A','m']],True,False)
