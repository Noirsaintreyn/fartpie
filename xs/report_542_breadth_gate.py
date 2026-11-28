"""Report 542 (Nov 28 2026): Rey: does market breadth (as in the weekly prep sheet) gate small-cap flips? DECLARED BEFORE RESULTS.
Breadth from our own universe (wide_daily_ohlcv, all names with a close that day, price>=1): B50 = share of names above own 50d mean, B50CH = B50 minus its value 5 sessions ago (thrust), HL = (share at 20d high) - (share at 20d low). Also SUB5 version of B50 (names closing <5). Read at the flip-day close. Terciles cut on train 2019-21. Hypothesis (mechanism, from 515): sub-$5 flips run when speculative participation is broad/expanding; narrow or collapsing breadth = flips fail. Prediction: reach and mean r rise with B50 / B50CH / HL. Pass: top tercile beats bottom tercile in BOTH eras, perm p<.05 (3 features x 2 metrics = 6 trials, Bonferroni .008), shown within HI-ATRP and on all flips.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D=pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/wide_daily_ohlcv.pkl'); C=pd.DataFrame({k:v.close for k,v in D.items()}); C.index=pd.DatetimeIndex(C.index).astype('datetime64[ns]'); C=C[C.index>='2018-01-01']; C=C.where(C>=1)
ab=(C>C.rolling(50).mean()).where(C.notna()); B50=ab.sum(axis=1)/C.notna().sum(axis=1)
hi=(C>=C.rolling(20).max()).where(C.notna()); lo=(C<=C.rolling(20).min()).where(C.notna()); HL=(hi.sum(axis=1)-lo.sum(axis=1))/C.notna().sum(axis=1)
s5=C.where(C<5); ab5=(s5>s5.rolling(50,min_periods=40).mean()).where(s5.notna()); B5=ab5.sum(axis=1)/s5.notna().sum(axis=1)
F=pd.DataFrame(dict(B50=B50,B50CH=B50-B50.shift(5),HL=HL,B5=B5,B5CH=B5-B5.shift(5)))
print('universe names',C.shape[1],'days',len(F))
H=pd.read_csv('flip_history.csv',parse_dates=['d']); H=H[H.px<5].merge(F,left_on='d',right_index=True).dropna(); H['yr']=H.d.dt.year; H['hit']=H.hit.astype(float)
tr=H.yr.between(2019,2021); te=H.yr>=2022; cut=H.loc[tr,'ATRP'].quantile(2/3); H['HA']=H.ATRP>cut
print('flips',len(H),'train',tr.sum(),'test',te.sum())
rng=np.random.default_rng(3)
for f in ('B50','B50CH','HL','B5','B5CH'):
    q=H.loc[tr,f].quantile([1/3,2/3]); H['T']=np.where(H[f]>q.iloc[1],'TOP',np.where(H[f]<=q.iloc[0],'BOT','MID')); print(f'\n{f} cuts {q.iloc[0]:+.3f} {q.iloc[1]:+.3f}')
    for nm,mk in (('TRAIN',tr),('TEST',te)):
        for lab,sub in (('HI-ATRP',H.HA),('ALL',H.HA|~H.HA)):
            b=H[mk&sub]; o=f'  {nm} {lab:7}'
            for t in('BOT','MID','TOP'):
                d=b[b['T']==t]
                if len(d)<8: o+=f' | {t} n{len(d)} few'; continue
                p=''
                if t=='TOP' or t=='BOT':
                    R=np.array([(lambda s:(s.hit.mean(),s.r.mean()))(b.sample(len(d),random_state=int(rng.integers(1e9)))) for _ in range(800)])
                    p=f' p({"<=" if t=="BOT" else ">="}:{((R[:,0]<=d.hit.mean()) if t=="BOT" else (R[:,0]>=d.hit.mean())).mean():.2f},{((R[:,1]<=d.r.mean()) if t=="BOT" else (R[:,1]>=d.r.mean())).mean():.2f})'
                o+=f' | {t} n{len(d):3d} reach {d.hit.mean()*100:3.0f}% mean {d.r.mean()*100:+6.1f} med {d.r.median()*100:+5.1f}{p}'
            print(o)
print('\nmean feature by era:',H.groupby(np.where(tr,'train','test'))[['B50','B50CH','HL','B5']].mean().round(3).to_dict())
