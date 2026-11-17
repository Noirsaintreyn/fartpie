"""Report 278 (declared before results). Rey: keep going on the volatility + orderflow reversal (report 264). 264 events (rejected new session extreme 10:00-14:00, n=2,862, with R^ forecast, range used U, V1 volume building, V2 absorption, p618 = pts net to a .618 R^ target with stop beyond the extreme, held = the extreme is the day's extreme). Pre-declared integer SCORE = V1 + V2 + (U>=.5) + (prior-close QQQ net GEX>0), no fitting. Report per score 0..4: n, hold rate, p618 and p100 mean pts per trade (net), day-clustered t, train 2019-21 | test 2018+2022-26 (observation-weighted, NOT mean of day-means). Also single-flag and pair cells. Pass: positive p618 in both splits with n>=40 per split at some score >= 2."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
E=pickle.load(open(SP+'vz264b.pkl','rb')); E['d']=pd.to_datetime(E.d)
gx=pd.read_csv('xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').sort_index(); kg=np.clip(gx.index.searchsorted(E.d.values.astype('datetime64[ns]'))-1,0,len(gx)-1); E['gpos']=gx.net_gex.values[kg]>0
E['u5']=E.U>=.5; E['score']=E.vb.astype(int)+E.ab.astype(int)+E.u5.astype(int)+E.gpos.astype(int)
print('events',len(E),' split n',E.split.value_counts().to_dict(),'\ncolumns used: vb,ab,U,gpos; p618/p100 are net pts')
def cell(m,col):
    s=[]
    for sp in ('train','test'):
        z=E[m&(E.split==sp)]
        if len(z)<5: s.append(f'n{len(z):4d} -'); continue
        ds=z.groupby('d')[col].agg(['sum','size']); mean=z[col].mean(); se=np.sqrt(((ds['sum']-mean*ds['size'])**2).sum())/len(z)
        s.append(f'n{len(z):4d} {mean:+6.1f} (t{mean/se:+.1f})')
    return ' | '.join(s)
def hold(m):
    return ' | '.join(f'{E[m&(E.split==sp)].held.mean():.0%}' for sp in ('train','test'))
print('\nby score (hold rate train|test ; p618 ; p100)')
for sc in range(5):
    m=E.score==sc; print(f' score {sc}: hold {hold(m):9s} p618 {cell(m,"p618")}   p100 {cell(m,"p100")}')
print(' score>=2  : hold',hold(E.score>=2),'p618',cell(E.score>=2,'p618'),'  p100',cell(E.score>=2,'p100'))
print(' score>=3  : hold',hold(E.score>=3),'p618',cell(E.score>=3,'p618'),'  p100',cell(E.score>=3,'p100'))
print('\nsingle flags / pairs (p618)')
for nm,m in (('V1',E.vb),('V2',E.ab),('U>=.5',E.u5),('gamma+',E.gpos),('V1&V2',E.vb&E.ab),('V1&gamma+',E.vb&E.gpos),('V2&gamma+',E.ab&E.gpos),('U>=.5&gamma+',E.u5&E.gpos),('V1&V2&gamma+',E.vb&E.ab&E.gpos),('none',E.score==0)):
    print(f' {nm:14s} hold {hold(m):9s} p618 {cell(m,"p618")}')
print('\nscore>=2 by year p618:',{y:(round(E[(E.score>=2)&(E.d.dt.year==y)].p618.mean(),1),int(((E.score>=2)&(E.d.dt.year==y)).sum())) for y in range(2018,2027)})
