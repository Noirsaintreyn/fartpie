"""report_126 (Nov 15 2026): RULE A with VOL-SCALED stop/target. Finding from report_121: RA pts/trade by vol state (ATR5/ATR60 hysteresis machine fit on train): EXP -1.6 (train) / -0.8 (test), QUIET +11.3 / +14.7, MID +0.4 / +16.2. Mechanism (declared): in expansion the fixed -50 stop sits inside normal noise (whipsaw) so the leg is a coin flip minus cost; scaling stop and target to prior-day volatility (S = clip(ATR14d / TRAIN median ATR14d, 0.7, 2.0)) keeps the stop outside the noise. Risk-equal comparison: results in R (pts / stop used) because live sizing is by stop. Replay as report_111 (BASE validated corr .998). Train <=2021, test 2022+."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
T=pd.read_csv('/mnt/attach/outputs/acct/nq_rule_a_trades.csv',parse_dates=['day','entry']); d=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').set_index('dt').sort_index()
st=pickle.load(open(SC+'regime_states.pkl','rb')); SMv=st['SMv']; D=pd.read_pickle(SC+'atlas_D.pkl'); D.index=pd.to_datetime(D.index)
atr14=(D.hi-D.lo).rolling(14).mean().shift(1); med=atr14[D.index<'2022-01-01'].median()
h=d.high.values; l=d.low.values; c=d.close.values; o_=d.open.values; ix={t:k for k,t in enumerate(d.index)}; COST=1.0
def replay(r,S,stop0=50.,tgt0=75.):
    i=ix.get(r.entry)
    if i is None: return np.nan,np.nan
    day=r.entry.normalize(); noon=day+pd.Timedelta(hours=12); s=r.dir; e=o_[i]; STOP=stop0*S; TGT=tgt0*S; k=i
    while True:
        t=d.index[k]
        if t>=noon: return (o_[k]-e)*s-COST,STOP
        if (l[k]<=e-STOP) if s==1 else (h[k]>=e+STOP): return -STOP-COST,STOP
        if (h[k]>=e+TGT) if s==1 else (l[k]<=e-TGT): return TGT-COST,STOP
        k+=1
        if k>=len(d) or d.index[k].normalize()!=day: return (c[k-1]-e)*s-COST,STOP
rows=[]
for _,r in T.iterrows():
    dd=r.day.normalize(); a=atr14.get(dd,np.nan); S=float(np.clip(a/med,.7,2.0)) if a==a else np.nan
    if S!=S: continue
    b,sb=replay(r,1.0); sc,ss=replay(r,S); rows.append(dict(day=dd,state=SMv.get(dd,np.nan),S=S,base=b,Rb=b/50,scaled=sc,Rs=sc/ss,te=dd>=pd.Timestamp('2022-01-01')))
R=pd.DataFrame(rows).dropna(subset=['state']); print('trades',len(R))
for lab,z in (('ALL',R),)+tuple((s,R[R.state==s]) for s in ('QUIET','MID','EXP')):
    for te in (False,True):
        x=z[z.te==te]; print(f'{lab:6s} {"TE" if te else "TR"} n {len(x):3d} | base: pts {x.base.mean():+6.2f} R {x.Rb.mean():+.3f} win {np.mean(x.base>0)*100:4.1f}% | scaled: pts {x.scaled.mean():+6.2f} R {x.Rs.mean():+.3f} win {np.mean(x.scaled>0)*100:4.1f}% | mean S {x.S.mean():.2f}')
R.to_pickle(SC+'rulea_volscale.pkl')
