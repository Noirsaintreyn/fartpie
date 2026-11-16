"""Report 205: (A) SUPPLY: do big moves exist in late 2026 or are we missing them? For every name in the wide daily universe (1,656) and every month 2019-2026: among names with close < $5 on the first trading day of the month, the share whose max close over the next 60 trading days is >= 2x that close ('doubler supply'); compare with the vector flips that doubled (report 199 'peak' >= +100%, 2026 only). (B) CATALYST theory, declared before looking, one test each, sign declared, on the report-203 flips (era train 2019-21 | test 2022-Feb26 | late 2026): H1 an 8-K filed in the 5 trading days before the flip (fresh information) -> BIG up; H2 open-market insider purchase (code P, acquired) in the prior 30 days -> BIG up; H3 SC 13D / 13D-A in the prior 30 days (activist / large holder) -> BIG up; H4 a prospectus / shelf offering (424B*, S-3, S-1, F-3) in the prior 10 days (dilution) -> BIG down. Report coverage (date range of each source), rate of BIG with vs without, and the BIG-rate ratio."""
import pandas as pd, numpy as np
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; E=X+'ext/'
D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); rows=[]
for tk,B in D.items():
    B=B[~B.index.duplicated()]; B=B[B.index>='2019-01-01']; c=B.close.values; idx=B.index
    if len(c)<100: continue
    fm=pd.Series(np.arange(len(c)),index=idx).groupby(idx.to_period('M')).first()
    cm=pd.Series(c).rolling(60).max().shift(-60).values
    for m,i in fm.items():
        if i+60<len(c) and c[i]<5: rows.append((m,tk,cm[i]/c[i]>=2))
S=pd.DataFrame(rows,columns=['m','tk','dbl']); g=S.groupby('m').agg(sub5=('tk','size'),dbl=('dbl','sum')); g['per100']=g.dbl/g.sub5*100
print('(A) doubler supply, names <$5 at month start whose max close over the next 60 trading days >= 2x (per 100 sub-$5 names)')
print(g.loc['2025-01':].round(1).to_string()); y=g.groupby(g.index.year).agg(dbl=('dbl','sum'),sub5=('sub5','sum')); y['per100']=(y.dbl/y.sub5*100).round(1); print('by year:',y.per100.to_dict())
R=pd.read_pickle(X+'penny2026_199.pkl'); R=R[~R.disc]; R['m']=R.entry.dt.to_period('M'); v=R.groupby('m').agg(flips=('ret','size'),peak100=('peak',lambda s:(s>=1).sum()),ret100=('ret',lambda s:(s>=1).sum())); print('\nvector flips by entry month 2026: flips, flips whose peak reached +100%, flips that ended >=+100%'); print(v.to_string())
Z=pd.read_pickle(X+'selectors203.pkl'); Z['sd']=Z.d
e8=pd.read_csv(E+'8k_wide.csv',parse_dates=['filed']); f13=pd.read_csv(E+'13d_wide.csv',parse_dates=['filed']); fm_=pd.read_csv(E+'forms.csv',parse_dates=['filed']); ins=pd.read_pickle(E+'insider.pkl')
ins=ins[(ins.TRANS_CODE=='P')&(ins.TRANS_ACQUIRED_DISP_CD=='A')].copy(); ins['d']=pd.to_datetime(ins.TRANS_DATE,errors='coerce'); ins['tk']=ins.ISSUERTRADINGSYMBOL
print('\n(B) coverage: 8-K',e8.filed.min().date(),e8.filed.max().date(),len(e8),'| 13D/G',f13.filed.min().date(),f13.filed.max().date(),'| forms',fm_.filed.min().date(),fm_.filed.max().date(),len(fm_),'| insider P',ins.d.min().date(),ins.d.max().date(),len(ins))
f13=f13[f13.form.str.contains('13D',na=False)]; fm_=fm_[fm_.form.str.contains('424B|S-3|S-1|F-3',na=False)]
def has(df,col,tk,d,lo,hi):
    s=df[df.tk==tk]; return bool(((s[col]>=d-pd.Timedelta(days=lo))&(s[col]<=d).values).any()) if len(s) else False
e8=e8.rename(columns={'ticker':'tk'}); f13=f13.rename(columns={'ticker':'tk'}); fm_=fm_.rename(columns={'ticker':'tk'})
K={'H1_8K':(e8,'filed',7),'H3_13D':(f13,'filed',30),'H4_offering':(fm_,'filed',14),'H2_insiderP':(ins,'d',30)}
for k,(df,col,lo) in K.items():
    grp={t:x for t,x in df.groupby('tk')}; Z[k]=[bool(len(grp.get(t,[]))) and bool(((grp[t][col]>=d-pd.Timedelta(days=lo))&(grp[t][col]<=d)).any()) for t,d in zip(Z.tk,Z.d)]
print('\nBIG (>=+50%) rate with vs without the catalyst; ratio; n with; (only flips inside each source coverage window are meaningful)')
for k,(df,col,lo) in K.items():
    lo_d=df[col].min()
    for e in ('train','test','late26'):
        s=Z[(Z.era==e)&(Z.d>=lo_d+pd.Timedelta(days=lo))]
        if len(s)<20: print(f'  {k:12s}{e:7s} n={len(s)} (no coverage)'); continue
        a=s[s[k]]; b=s[~s[k]]; print(f'  {k:12s}{e:7s} with n{len(a):4d} BIG {a.big.mean():5.1%} mean {a.r.mean()*100:6.1f}% | without n{len(b):4d} BIG {b.big.mean():5.1%} mean {b.r.mean()*100:6.1f}% | ratio {a.big.mean()/max(b.big.mean(),1e-9):4.2f}')
print('\n(C) persistence of the doubler supply: month-to-month autocorrelation of per100 (2019-2026, months with full 60d forward window)')
s=g.per100; print('lag1 %.2f lag2 %.2f lag3 %.2f'%(s.autocorr(1),s.autocorr(2),s.autocorr(3)))
ex=s.drop(s.index[(s.index>='2020-01')&(s.index<='2020-12')]); print('ex-2020 lag1 %.2f lag2 %.2f'%(ex.autocorr(1),ex.autocorr(2)))
# trailing: does last quarter's doubler supply (observed by month-start M, using only windows completed by M) predict this month's supply?
T=pd.Series({m:g.per100.loc[:m-3].tail(3).mean() for m in g.index if len(g.per100.loc[:m-3])>=3}); J=pd.concat([T,g.per100],axis=1,keys=['trail','now']).dropna(); print('corr(trailing 3m supply completed before month start, supply now) %.2f (n%d); Spearman %.2f'%(J.trail.corr(J.now),len(J),J.trail.corr(J.now,method='spearman')))
