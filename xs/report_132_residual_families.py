"""report_132 (Nov 15 2026): RESIDUAL PROGRAMME. What the OS/score (ATRP + EXT + turnover [+ mcap, price]) does NOT answer, and which model FAMILY might. Residual questions: (Q1) false positives: ~55-65% of top-20% sub-$5 flips still lose (median -10%); (Q2) misses: ~40-50% of +100% flips rank outside the top 20%; (Q3) 2019-21 vs 2022+ level gap. Families, mechanical, declared signs [+ better returns] before results, all at flip-bar close (PIT), universes B px<5 / C px<5 & top-20% frozen score, train 2019-21 terciles, test 2022+ (pass: same sign, test |t|>=2):
AMT/supply (auction & overhead supply): OH = volume-weighted share of the last 252 bars' volume traded ABOVE the flip close [-] (no trapped holders overhead = air pocket); OH30 = share traded within +30% above close [-].
ICT: FVG = (flip-bar low - high two bars earlier)/ATR (bullish displacement gap / fair value gap) [+].
Participation/institutional-proxy: FINRA daily short-sale volume ratio: SV_FLIP = flip-day short volume / total [-] (supply sold INTO strength = take-profit/short pressure, Rey's mechanism), SV_CH = SV_FLIP - mean of prior 20d [-], SV5 = mean prior 5d [+] (squeeze fuel).
Wyckoff/physics already tested (reports 122, 129): excluded here.
Incremental test: ridge with the 5 base features (ATRP, EXT, LTURN, LMC, LPX) vs base + family feature, train-only fit, test AUC for BIG/HUGE."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D=pd.read_pickle(X+'wide_daily_ohlcv.pkl'); O=pd.read_pickle(X+'combined_scores.pkl'); O=O[O.px<5].copy()
FS=pd.read_pickle(X+'finra_sv_wide.pkl'); FS['Date']=pd.to_datetime(FS.Date); FS['svr']=FS.ShortVolume/FS.TotalVolume.replace(0,np.nan); fg={k:g.set_index('Date').svr for k,g in FS.groupby('Symbol')}
feat={}
for tk,g in O.groupby('tk'):
    q=D.get(tk)
    if q is None: continue
    q=q[['open','high','low','close','volume']].copy(); q=q[~q.index.duplicated()].dropna(); q.index=pd.to_datetime(q.index).tz_localize(None).normalize()
    h,l,c,v=[q[k].values.astype(float) for k in ('high','low','close','volume')]; pos={d:i for i,d in enumerate(q.index)}
    pc=np.r_[c[0],c[:-1]]; trr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); a14=pd.Series(trr).rolling(14).mean().shift(1).values; s=fg.get(tk)
    for _,r in g.iterrows():
        i=pos.get(r.d)
        if i is None or i<260: continue
        hh,ll,vv=h[i-251:i+1],l[i-251:i+1],v[i-251:i+1]; cc=c[i]; rg=np.maximum(hh-ll,1e-12)
        fa=np.clip((hh-np.maximum(cc,ll))/rg,0,1); fb=np.clip((np.minimum(hh,1.3*cc)-np.maximum(ll,cc))/rg,0,1)
        OH=(vv*fa).sum()/max(vv.sum(),1); OH30=(vv*fb).sum()/max(vv.sum(),1); FVG=(l[i]-h[i-2])/a14[i-1] if a14[i-1]>0 else np.nan
        sv=np.nan;svc=np.nan;sv5=np.nan
        if s is not None:
            w=s.loc[:r.d]
            if len(w)>25 and w.index[-1]==r.d: sv=w.iloc[-1]; svc=sv-w.iloc[-21:-1].mean(); sv5=w.iloc[-6:-1].mean()
        feat[(tk,r.d)]=dict(OH=OH,OH30=OH30,FVG=FVG,SV_FLIP=sv,SV_CH=svc,SV5=sv5)
I=pd.DataFrame.from_dict(feat,orient='index'); I.index=pd.MultiIndex.from_tuples(I.index,names=['tk','d']); A=O.merge(I.reset_index(),on=['tk','d']); A=A[A.d>='2019-01-01']; A['top']=A.lr>=O[~O.test].lr.quantile(.8)
A['LTURN']=np.log(A.TURN.clip(lower=1e-9)); A.to_pickle(X+'residual_families.pkl'); print('px<5 flips with features',len(A),'train',(~A.test).sum(),'test',A.test.sum())
def t_(d): return d.mean()/(d.std()/np.sqrt(len(d))) if len(d)>5 and d.std()>0 else np.nan
sign=dict(OH='-',OH30='-',FVG='+',SV_FLIP='-',SV_CH='-',SV5='+')
for un,U in (('B px<5',A),('C px<5 & top20',A[A.top])):
    print(f'\n#### {un}: train {(~U.test).sum()} test {U.test.sum()}')
    for f,sg in sign.items():
        tr=U[~U.test].dropna(subset=[f]); q1,q2=tr[f].quantile([1/3,2/3]).values; out=[];ok=True
        for era,z in (('TR',U[~U.test]),('TE',U[U.test])):
            z=z.dropna(subset=[f]); hi=z[z[f]>=q2]; lo=z[z[f]<=q1]; dd=(hi.groupby('mo').n.mean()-lo.groupby('mo').n.mean()).dropna(); tt=t_(dd); sp=hi.n.mean()-lo.n.mean()
            out.append(f'{era}: T3 {hi.n.mean()*100:+6.1f}%/T1 {lo.n.mean()*100:+6.1f}% med {hi.n.median()*100:+5.1f}/{lo.n.median()*100:+5.1f} big30 {np.mean(hi.n>=.3)*100:4.1f}/{np.mean(lo.n>=.3)*100:4.1f} t {tt:+.2f}')
            ok&=((sp>0)==(sg=='+')) and (era=='TR' or (tt==tt and abs(tt)>=2 and (tt>0)==(sg=='+')))
        print(f'  {f:8s}[{sg}] '+' | '.join(out)+('  <== PASS' if ok else ''))
base=['ATRP','EXT','LTURN','LMC','LPX']; tr=A[~A.test]
def rk(cols):
    d={}
    for f in cols: s=np.sort(tr[f].dropna().values); d[f]=np.searchsorted(s,A[f].fillna(tr[f].median()).values,side='right')/len(s)
    return pd.DataFrame(d,index=A.index)
print('\nINCREMENTAL AUC on test (px<5): base5 vs base5+feature')
for tgt in ('BIG','HUGE'):
    R0=rk(base); m0=LogisticRegression(C=.5,max_iter=2000).fit(R0[~A.test],A[~A.test][tgt]); a0=roc_auc_score(A[A.test][tgt],m0.decision_function(R0)[A.test.values]); line=f'  {tgt}: base {a0:.3f}'
    for f in sign: 
        R=rk(base+[f]); m=LogisticRegression(C=.5,max_iter=2000).fit(R[~A.test],A[~A.test][tgt]); line+=f' | +{f} {roc_auc_score(A[A.test][tgt],m.decision_function(R)[A.test.values]):.3f}'
    print(line)
# Q1/Q2 diagnostics
te=A[A.test].copy(); te['rk']=te['m_BIG_0.5'].rank(pct=True); top=te[te.rk>=.8]
print('\nQ1 false positives (test, px<5, top-20% m_BIG): n',len(top),'| win',round(float(np.mean(top.n>0))*100,1),'% | big30',round(float(np.mean(top.n>=.3))*100,1),'% | loss>10%',round(float(np.mean(top.n<-.1))*100,1),'%')
H=te[te.HUGE]; print('Q2 misses: HUGE flips in test px<5:',len(H),'| ranked in top20%:',int((H.rk>=.8).sum()),'| outside:',int((H.rk<.8).sum()))
cols=['ATRP','EXT','LPX','VOLX','F2','F3','OH','SV_FLIP','FVG']
print(pd.DataFrame({'caught HUGE':H[H.rk>=.8][cols].median(),'missed HUGE':H[H.rk<.8][cols].median(),'top20 non-huge':top[~top.HUGE][cols].median()}).round(3).to_string())
