"""Report 180: session-structure gates for IFVG and S1 premium fade (Rey, Nov 15). DECLARED BEFORE RESULTS.
Mechanism: IFVG is a mean-reversion entry that needs chop/rotation; premium fade needs an opening drive that is NOT accepted. Features known before entry:
ASIA_R = range 19:00-24:00 ET prior eve / ATR14; LON_R = range 02:00-05:00 ET / ATR14; OR30 = RTH 09:30-09:55 range / ATR14 (IFVG fills at >=10:00 only);
ACC = fraction of overnight (18:00-09:25) 5m closes outside the prior-day RTH range (acceptance outside yesterday's range); ACCD = same but only on the side of the open (above PDH if open above prior close else below PDL);
OPENPOS = (open-PDL)/(PDH-PDL) (S1 only; >1 = open above yesterday's high).
Lower / upper tercile cuts fit on 2018-21 trades, judged 2022+; PASS = same sign, |t|>=2, n>=30 both eras. Legs: IFVG (all), S1 regime-ON, S1 all."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>8 and len(b)>8 else np.nan
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
rt=b[(b.m>=570)&(b.m<=955)]; R=rt.groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last')); atr=(R.h-R.l).rolling(14).mean().shift(1); pdh=R.h.shift(1); pdl=R.l.shift(1); pc=R.c.shift(1)
# overnight session of day d: 18:00 of previous calendar day to 09:25 of d
b['sd']=(b.dt+pd.to_timedelta(np.where(b.m>=1080,1,0),unit='D')).dt.normalize()
on=b[(b.m>=1080)|(b.m<565)]; asia=b[(b.m>=1140)].assign(sd=lambda x:(x.dt+pd.Timedelta(days=1)).dt.normalize()); lon=b[(b.m>=120)&(b.m<300)]
AS=asia.groupby('sd').apply(lambda g:g.high.max()-g.low.min()); LO=lon.groupby('d').apply(lambda g:g.high.max()-g.low.min()); OR=rt[(rt.m>=570)&(rt.m<595)].groupby('d').apply(lambda g:g.high.max()-g.low.min())
def accf(d):
    g=on[on.sd==d]
    if len(g)<20 or d not in pdh.index or not np.isfinite(pdh[d]): return (np.nan,np.nan)
    up=(g.close>pdh[d]).mean(); dn=(g.close<pdl[d]).mean(); o=R.o.get(d,np.nan); return (up+dn, up if o>pc[d] else dn)
days=R.index; F=pd.DataFrame(index=days); F['ASIA_R']=AS.reindex(days)/atr; F['LON_R']=LO.reindex(days)/atr; F['OR30']=OR.reindex(days)/atr
ac=[accf(d) for d in days]; F['ACC']=[a[0] for a in ac]; F['ACCD']=[a[1] for a in ac]; F['OPENPOS']=(R.o-pdl)/(pdh-pdl)
I=pd.read_pickle(SP+'ext/uni_ifvg.pkl'); I=I[I.model=='IFVG'].copy(); I['day']=pd.to_datetime(I.d).dt.normalize()
S1=pd.read_pickle(SP+'s1_trades162.pkl'); S1['day']=pd.to_datetime(S1.day).dt.normalize()
legs={'IFVG':(I,['ASIA_R','LON_R','OR30','ACC','ACCD']),'S1on':(S1[S1.reg>0],['ASIA_R','LON_R','ACC','ACCD','OPENPOS']),'S1all':(S1,['ASIA_R','LON_R','ACC','ACCD','OPENPOS'])}
def passed(r): return np.sign(r[0][0])==np.sign(r[1][0]) and all(np.isfinite(x[1]) and abs(x[1])>=2 and x[2]>=30 for x in r)
for n,(t,fe) in legs.items():
    t=t.reset_index(drop=True).copy()
    for c in fe: t[c]=F[c].reindex(t.day).values
    if n=='IFVG': t.loc[t.m<600,'OR30']=np.nan
    e=(t.day<'2022-01-01').values; print(f'\n{n}: {len(t)} trades, train {e.sum()} test {(~e).sum()}; avg pts {t.pts[e].mean():+.2f} | {t.pts[~e].mean():+.2f}')
    for c in fe:
        v=t[c]; q=v[e].quantile([1/3,2/3]).values
        for nm,sel in (('low',v<=q[0]),('high',v>q[1])):
            res=[]
            for mm in (e,~e):
                ok=mm&v.notna().values; a,bb=t.pts[ok&sel.values],t.pts[ok&~sel.values]; res.append((a.mean()-bb.mean(),tt(a,bb),len(a),a.mean()))
            tag='  <== PASS' if passed(res) else ''
            print(f'  {c:8s} {nm:4s} spread {res[0][0]:+6.1f} (t{res[0][1]:+.1f},n{res[0][2]},avg {res[0][3]:+.1f}) // {res[1][0]:+6.1f} (t{res[1][1]:+.1f},n{res[1][2]},avg {res[1][3]:+.1f}){tag}')
