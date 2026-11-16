"""Report 189: FIG-V (IFVG) FILL-BAR AUDIT (found while building an example trade, Nov 15). In report 144 the 5m sim starts at the FILL bar and checks the target on the whole bar range, but the limit fills at an unknown time inside that bar: a low (short: high) that printed BEFORE the fill can be credited as a target hit (example: 2024-03-12 13:15 short, bar opened below the target, low printed first, recorded +27.25).
FIX: re-simulate every IFVG trade on 1-minute bars. Fill = first 1m bar inside the recorded fill 5m bar where price trades 0.25 through the limit (same rule); from the fill minute on, stop checked before target within a minute (conservative); same stop/target/2R/flat 15:55/cost 0.5. Also reported: + stop slippage 0.5 pt on stopped trades, and the effect by era / direction.
DECLARED BEFORE RESULTS; strict split 2018-21 | 2022+."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; COST=.5
I=pd.read_pickle(SP+'ext/uni_ifvg.pkl'); I=I[I.model=='IFVG'].copy().reset_index(drop=True); I['d']=pd.to_datetime(I.d)
b=pd.read_csv('/home/claude/nqdrive/NQ_1m_ETH.csv',parse_dates=['timestamp']); b['dt']=b.timestamp-pd.Timedelta(hours=6); b=b.sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
by={d:g[(g.m>=570)&(g.m<=955)] for d,g in b.groupby('d')}
out=[]
for r in I.itertuples():
    g=by.get(r.d)
    if g is None or len(g)<300: out.append((np.nan,np.nan,np.nan)); continue
    m=g.m.values; h=g.high.values; l=g.low.values; c=g.close.values; dr=r.dr; ent=r.ent; R=r.risk
    stop=ent-dr*R; tgt=ent+dr*2*R; i0=np.where(m>=int(r.m)//5*5)[0]
    if not len(i0): out.append((np.nan,np.nan,np.nan)); continue
    i0=i0[0]; f=None
    for q in range(i0,min(i0+5,len(m))):
        if (l[q]<=ent-.25) if dr>0 else (h[q]>=ent+.25): f=q; break
    if f is None: out.append((np.nan,np.nan,np.nan)); continue
    worst=0.; res=None; stopped=False
    for q in range(f,len(m)):
        worst=min(worst,((l[q] if dr>0 else h[q])-ent)*dr)
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): res=-R-COST; stopped=True; break
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): res=2*R-COST; break
    if res is None: res=(c[-1]-ent)*dr-COST
    out.append((res,worst-COST,stopped))
O=pd.DataFrame(out,columns=['pts1m','mae1m','stopped']); I=pd.concat([I,O],axis=1)
print('trades',len(I),'unfilled on 1m (dropped)',int(I.pts1m.isna().sum()))
J=I.dropna(subset=['pts1m']).copy(); J['pts_slip']=J.pts1m-0.5*(J.stopped==True)
def st(s): n=len(s); return n,s.mean(),(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0)
for lab,col in (('original 5m (report 144)','pts'),('1m fill-resolved','pts1m'),('1m + 0.5pt stop slippage','pts_slip')):
    a=st(J[col][J.d<'2022-01-01']); b_=st(J[col][J.d>='2022-01-01']); ok=np.sign(a[1])==np.sign(b_[1]) and abs(a[2])>=2 and abs(b_[2])>=2
    print(f'  {lab:28s} A {a[1]:+6.2f} (t{a[2]:+.1f}, n{a[0]}) | B {b_[1]:+6.2f} (t{b_[2]:+.1f}, n{b_[0]}){"  PASS" if ok else "  FAIL"}')
J['chg']=(J.pts-J.pts1m).abs()>.6; print('trades whose result changed:',int(J.chg.sum()),'of',len(J),'; of those original win -> now loss',int(((J.pts>0)&(J.pts1m<=0)).sum()),'| original loss -> win',int(((J.pts<=0)&(J.pts1m>0)).sum()))
for nm,m_ in (('long',J.dr>0),('short',J.dr<0)):
    a=st(J.pts1m[(J.d<'2022-01-01')&m_]); b_=st(J.pts1m[(J.d>='2022-01-01')&m_]); print(f'  1m {nm:5s}: A {a[1]:+6.2f} (t{a[2]:+.1f},n{a[0]}) | B {b_[1]:+6.2f} (t{b_[2]:+.1f},n{b_[0]})')
print('win rate 1m: A %.0f%% | B %.0f%%'%(100*(J.pts1m[J.d<'2022-01-01']>0).mean(),100*(J.pts1m[J.d>='2022-01-01']>0).mean()))
I.to_pickle(SP+'ifvg_1m189.pkl')
