"""Report 211: HYBRID = 1H entry, daily hold. DECLARED before results (after report 210: 1H enters with ~7% of a doubling move gone vs ~24% for daily and has a flip in 86-91% of doubling episodes vs 45-61%, but its bear flip exits in ~5 days and it rarely holds a double).
Rule (one position per name): enter at the open of the bar after a 1H bull flip (the probe). At each daily close while in the trade: if the DAILY vector is bull, the trade is CONFIRMED and from then on ignores 1H bear flips; it exits at the next day's first open after a daily close with the daily vector bear. If a 1H bear flip comes before any confirmation, exit at the next 1H open (failed probe). Same data, filters, cost (100bp), halves (H1 2024-02..2025-06, H2 2025-07..2026-11) and episode definition as report 210.
H_HYBRID: in both halves, (1) mean net return per trade above 1H-only and sum of net returns above daily-only, (2) share of trades >= +100% at least the daily share, (3) doubling-episode capture (>= +50% realized, as a share of ALL episodes) above both. Also shown: failed-probe share and its mean, confirmed-trade stats. Variant V2 (declared): probe only when the daily vector is NOT already bull at the prior daily close (pure early entries), to see where the gain comes from."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
E='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SP=E[:-4]
P=pickle.load(open(E+'lowpx_1h_3y.pkl','rb')); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); S0=pd.Timestamp('2024-02-01'); MID=pd.Timestamp('2025-07-01'); T0=pd.read_pickle(SP+'flips_1h_vs_d210.pkl'); EPd=pd.read_pickle(SP+'episodes210.pkl'); rows=[]
for tk,h in P.items():
    x=h.copy(); x.columns=['open','high','low','close','volume']; x.index=x.index.tz_convert('America/New_York').tz_localize(None); x=x[~x.index.duplicated()].dropna(); m=x.index.hour*60+x.index.minute; x=x[(m>=570)&(m<960)]
    if len(x)<1500 or tk not in D: continue
    d=D[tk]; d=d[~d.index.duplicated()]; d=d[d.index>='2022-06-01'][['open','high','low','close','volume']].dropna()
    r1=np.asarray(run_state_machine(x.assign(datetime=x.index))[0]); rD=pd.Series(np.asarray(run_state_machine(d.assign(datetime=d.index))[0]),index=d.index)
    day=x.index.normalize(); o,c=x.open.values,x.close.values; n=len(x); last=np.r_[day[1:]!=day[:-1],True]; rd=rD.reindex(day,method='ffill').values; rd_prev=rD.shift(1).reindex(day,method='ffill').values
    i=1
    while i<n-2:
        if r1[i]==1 and r1[i-1]!=1:
            e=i+1; conf=False; j=e; xi=None; pre=rd_prev[e]==1; ct=None
            while j<n-1:
                if last[j]:
                    if rd[j]==1:
                        if not conf: ct=x.index[j]
                        conf=True
                    elif conf and rd[j]==-1: xi=j+1; break
                if not conf and r1[j]==-1 and j>e-1: xi=j+1; break
                j+=1
            op=xi is None; x_=n-1 if op else xi; xp=c[n-1] if op else o[x_]; g=o[e+1:x_+1]/c[e:x_] if x_>e else np.array([1.])
            rows.append(dict(tk=tk,entry=x.index[e],exit=x.index[x_],px=o[e],ret=xp/o[e]-1,conf=conf,pre=pre,ct=ct,disc=bool(len(g) and (g.max()>2 or g.min()<.5)),open=op)); i=x_
        else: i+=1
H=pd.DataFrame(rows); H=H[(H.entry>=S0)&(H.px<5)&(~H.disc)].copy(); H['half']=np.where(H.entry<MID,'H1','H2'); H['tf']='HY'
def line(s,nm):
    r=s.ret-.01; rs=r.sort_values(); return f'  {nm:26s} n{len(s):5d} mean {r.mean()*100:6.2f}% x3 {rs.iloc[:-3].mean()*100:6.2f}% med {r.median()*100:6.2f}% win {np.mean(r>0)*100:3.0f}% >=+30% {np.mean(r>=.3)*100:4.1f}% >=+50% {np.mean(r>=.5)*100:4.1f}% >=+100% {np.mean(r>=1)*100:4.1f}% | sum {r.sum()*100:7.0f}% | hold {(s.exit-s.entry).dt.days.median():.0f}d'
for hf in ('H1','H2'):
    print(f'\n{hf}'); print(line(T0[(T0.tf=="D")&(T0.half==hf)],'daily only')); print(line(T0[(T0.tf=="1H")&(T0.half==hf)],'1H only')); s=H[H.half==hf]; print(line(s,'HYBRID')); print(line(s[~s.conf],'  failed probes')); print(line(s[s.conf],'  confirmed')); print(line(s[~s.pre],'V2: daily not yet bull')); print(line(s[s.pre],'    daily already bull'))
print('\nDoubling episodes: share of ALL episodes where a trade entered in the window realized >= +50% / >= +100% net (first trade in window, not the best)')
A=pd.concat([T0[['tk','tf','entry','ret','half']],H[['tk','tf','entry','ret','half']]]); G={k:g.sort_values('entry') for k,g in A.groupby(['tk','tf'])}
for hf in ('H1','H2'):
    e=EPd[EPd.half==hf]
    for tf in ('D','1H','HY'):
        first=[]; anyb=[]
        for _,ep in e.iterrows():
            g=G.get((ep.tk,tf)); w=g[(g.entry>=ep.t0m5)&(g.entry<=ep.t1+pd.Timedelta(hours=23))] if g is not None else []
            first.append(w.ret.iloc[0]-.01 if len(w) else np.nan); anyb.append((w.ret-.01).sum() if len(w) else np.nan)
        f=np.array(first); a=np.array(anyb); print(f'  {hf} {tf:2s}: episodes {len(e):3d} | first trade >=+50%: {np.nansum(f>=.5)/len(e)*100:4.1f}%  >=+100%: {np.nansum(f>=1)/len(e)*100:4.1f}% | sum of ALL trades in the episode window, mean per episode {np.nan_to_num(a).mean()*100:6.1f}%, median {np.nanmedian(a)*100:6.1f}%')
H.to_pickle(SP+'hybrid211.pkl')
