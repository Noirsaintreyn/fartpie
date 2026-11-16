"""Report 221: ATRP + insider buying + flip-bar volume as cuts on the 7,260 1H sub-$5 flips (Rey's mid/small-cap filter lead, now on the 1H pool). DECLARED before results.
F_ATRP: daily ATR14/close at the prior daily close, top H1 tercile (sign + on mean; it is the size effect, so the median may fall). F_INS: an open-market insider purchase (Form 4 code P, acquired) with a transaction date in the 30 days before the flip (sign +). Insider file ends 2026-05-06, so insider rows use flips before 2026-05-06 only. F_VOL: flip-bar volume / mean of the previous 100 bars, top H1 tercile (sign +; report 218: passed H1, failed H2). Combos: INS&VOL, ATRP&VOL, INS&ATRP, all three.
Outcomes RHY (hybrid exit) and R1H, net 100bp, halves H1 2024-02..2025-06 | H2 2025-07..2026-11. PASS = same sign both halves, day-clustered |t| >= 2.8 (Bonferroni 14) for selected-minus-rest on returns winsorized at the 99th pct, n >= 30 per half. Any pass then needs the random-bar placebo before it counts."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; E=SP+'ext/'
F=pd.read_pickle(SP+'flips1h_feats218.pkl'); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); ins=pd.read_pickle(E+'insider.pkl'); ins=ins[(ins.TRANS_CODE=='P')&(ins.TRANS_ACQUIRED_DISP_CD=='A')].copy(); ins['d']=pd.to_datetime(ins.TRANS_DATE,errors='coerce',format='mixed'); ins=ins.dropna(subset=['d']); G={t:np.sort(g.d.values) for t,g in ins.groupby('ISSUERTRADINGSYMBOL')}; INS_END=pd.Timestamp('2026-05-06')
at={}
for tk in F.tk.unique():
    B=D[tk]; B=B[~B.index.duplicated()]; c,h,l=B.close,B.high,B.low; pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); at[tk]=(tr.rolling(14).mean()/c).shift(1)
dd=F.entry.dt.normalize(); F['ATRP']=[at[t].get(d,np.nan) for t,d in zip(F.tk,dd)]
def insf(t,d):
    a=G.get(t)
    if a is None: return False
    k=np.searchsorted(a,np.datetime64(d)); return bool(k>0 and (np.datetime64(d)-a[k-1])<=np.timedelta64(30,'D'))
F['INS']=[insf(t,d) for t,d in zip(F.tk,dd)]; h1=F[F.half=='H1']; F['A']=F.ATRP>=h1.ATRP.quantile(2/3); F['V']=F.VOLX>=h1.VOLX.quantile(2/3)
print('flips',len(F),'| insider-flagged',int(F.INS.sum()),'(names with any insider buy:',F[F.INS].tk.nunique(),') | ATRP top third',int(F.A.sum()),'| VOLX top third',int(F.V.sum()))
def st(s,col): r=s[col]; return f'n{len(s):5d} mean {r.mean()*100:6.2f}% x3 {r.sort_values().iloc[:-3].mean()*100 if len(s)>3 else np.nan:6.2f}% med {r.median()*100:5.2f}% win {np.mean(r>0)*100:3.0f}% >=+30% {np.mean(r>=.3)*100:4.1f}%'
def tt(a,b,col):
    if len(a)<30: return np.nan
    cap=np.quantile(np.r_[a[col].values,b[col].values],.99); A=a.assign(w=np.minimum(a[col],cap)).groupby(a.entry.dt.normalize()).w.mean(); B=b.assign(w=np.minimum(b[col],cap)).groupby(b.entry.dt.normalize()).w.mean(); return (A.mean()-B.mean())/np.sqrt(A.var(ddof=1)/len(A)+B.var(ddof=1)/len(B))
SEL={'ATRP top third':lambda s:s.A,'VOL top third':lambda s:s.V,'INSIDER buy 30d':lambda s:s.INS,'INS & VOL':lambda s:s.INS&s.V,'ATRP & VOL':lambda s:s.A&s.V,'INS & ATRP':lambda s:s.INS&s.A,'INS & ATRP & VOL':lambda s:s.INS&s.A&s.V}
for col in ('RHY','R1H'):
    print(f'\n=== {col}')
    for nm,f in SEL.items():
        for hf in ('H1','H2'):
            s=F[F.half==hf]; s=s[s.entry<INS_END] if 'INS' in nm else s; a=s[f(s)]; b=s[~f(s)]
            print(f'  {nm:18s}{hf}: selected {st(a,col) if len(a)>3 else "n"+str(len(a))} | rest mean {b[col].mean()*100:5.2f}% med {b[col].median()*100:5.2f}% | t {tt(a,b,col):+.1f}')
F.to_pickle(SP+'flips1h_feats221.pkl')
