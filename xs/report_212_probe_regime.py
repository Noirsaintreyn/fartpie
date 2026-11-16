"""Report 212: the probe confirmation rate as a real-time regime sensor. DECLARED before results.
From report 211: a 1H flip is a probe; it either fails (1H bear flip first) or is confirmed (daily vector bull at a daily close). ~5 probes resolve per day across the universe, so the trailing confirmation rate is a fast, well-powered read on whether early moves are following through (the earlier look-back test, report 127, used 5-20 trades with a 6-9% base rate and had no power).
REG(t) = share of probes that RESOLVED in the 10 trading days before the entry day (failed -> its exit time; confirmed -> its confirming daily close) that were confirmed; needs >= 30 resolved. Known at the entry time, no lookahead.
H: higher REG -> higher net return of trades entered that day, same sign in H1 (2024-02..2025-06) and H2 (2025-07..2026-11), for (a) hybrid trades, (b) daily-only flips. Cuts = H1 terciles. Test: day-clustered (mean return per entry day, weighted by trades) Welch t, top vs bottom tercile. Gate sim: trade only when REG >= H1 median; sum of net returns and mean vs ungated."""
import numpy as np, pandas as pd
from scipy import stats
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
H=pd.read_pickle(SP+'hybrid211.pkl'); T=pd.read_pickle(SP+'flips_1h_vs_d210.pkl'); Dl=T[T.tf=='D'].copy()
H['rt']=pd.to_datetime(np.where(H.conf,H.ct,H.exit)); R=H.dropna(subset=['rt']).sort_values('rt'); rt=R.rt.values; cf=R.conf.values.astype(float); cs=np.r_[0,np.cumsum(cf)]
days=pd.DatetimeIndex(sorted(set(H.entry.dt.normalize())|set(Dl.entry.dt.normalize()))); reg={}
for k,d in enumerate(days):
    if k<10: continue
    a=np.searchsorted(rt,np.datetime64(days[k-10])); b=np.searchsorted(rt,np.datetime64(d)); 
    if b-a>=30: reg[d]=(cs[b]-cs[a])/(b-a)
reg=pd.Series(reg); print('days with REG',len(reg),'REG mean %.3f sd %.3f; lag-1 autocorr of daily REG %.2f, lag-10 (non-overlapping) %.2f'%(reg.mean(),reg.std(),reg.autocorr(1),reg.autocorr(10)))
MID=pd.Timestamp('2025-07-01'); q=reg[reg.index<MID].quantile([1/3,2/3]).values; med=reg[reg.index<MID].median(); print('H1 tercile cuts',q.round(3),'median',round(med,3))
def clus(s):
    g=s.groupby(s.entry.dt.normalize()).r.mean(); return g
for nm,S in (('hybrid trades',H),('daily-only flips',Dl)):
    S=S.copy(); S['r']=S.ret-.01; S['reg']=S.entry.dt.normalize().map(reg); S=S.dropna(subset=['reg']); S['half']=np.where(S.entry<MID,'H1','H2'); S['b']=np.where(S.reg<=q[0],'low',np.where(S.reg>=q[1],'high','mid'))
    print(f'\n{nm}')
    for hf in ('H1','H2'):
        s=S[S.half==hf]; out=[]
        for b in ('low','mid','high'):
            x=s[s.b==b]; rs=x.r.sort_values(); out.append(f'{b}: n{len(x):4d} mean {x.r.mean()*100:6.2f}% x3 {rs.iloc[:-3].mean()*100 if len(x)>3 else np.nan:6.2f}% med {x.r.median()*100:5.2f}% >=+50% {np.mean(x.r>=.5)*100:4.1f}%')
        a=clus(s[s.b=='high']); c=clus(s[s.b=='low']); t=stats.ttest_ind(a,c,equal_var=False).statistic if len(a)>5 and len(c)>5 else np.nan; ic=stats.spearmanr(s.reg,s.r)[0]
        print(f'  {hf} '+' | '.join(out)+f' | day-clustered t(high-low) {t:+.1f}, IC {ic:+.3f}')
        g=s[s.reg>=med]; print(f'     gate REG>=H1 median: n{len(g)} of {len(s)}, mean {g.r.mean()*100:6.2f}% sum {g.r.sum()*100:7.0f}% vs ungated mean {s.r.mean()*100:6.2f}% sum {s.r.sum()*100:7.0f}%; blocked trades mean {s[s.reg<med].r.mean()*100:6.2f}%')
m=reg.groupby(reg.index.to_period('M')).mean(); print('\nmonthly mean REG:',' '.join(f'{k}:{v:.2f}' for k,v in m.items()))
