"""report_134 (Nov 15 2026): EXIT RULE BY PREDICTED TAIL CLASS (theory: for a diffusion with drawdown stop b and mixture drift, noise-state names (mu~0) have peak ~ Exp(mean b): the stop should be TIGHT, the expected gain from waiting is ~0; real-state names (mu>0) have peak scale >> b: a WIDE stop keeps the tail. So the optimal stop width should RISE with the model's P(real/+100%).) Declared policy BEFORE results: top-20% predicted P(+100%) (train 80th pct cut, M1 mixture) -> vector exit only (v); rest -> chandelier M=3 with vector backstop (c3). Compared with uniform v, uniform c2/c3/c4/c5 and the reverse policy (tight on top, wide on rest). Daily flips (report_108 exits), net 30bp, train 2019-21 / test 2022+; universes ALL and px<5."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
C=pd.read_pickle(X+'chand_daily.pkl'); S=pd.read_pickle(X+'physics_scores.pkl'); O=pd.read_pickle(X+'combined_scores.pkl')[['tk','d','px']]
A=C.merge(S,on=['tk','d']).merge(O,on=['tk','d']); A=A[A.d>='2019-01-01'].copy(); COST=.003
for c in ('v','c2','c3','c4','c5'): A[c]=A[c]-COST
def ex3(s): s=s.sort_values(ascending=False); return s.iloc[3:].mean()
for sname in ('M1P','M0P'):
  print('\n######## ranking score',sname)
  for un,mk in (('ALL',A.px>0),('px<5',A.px<5)):
    U=A[mk]; cut=U[~U.test][sname].quantile(.8); U=U.assign(top=U[sname]>=cut)
    pol={'uniform v (vector exit)':U.v,'uniform c2':U.c2,'uniform c3':U.c3,'uniform c4':U.c4,'uniform c5':U.c5,
         'DECLARED: top->v, rest->c3':np.where(U.top,U.v,U.c3),'top->v, rest->c2':np.where(U.top,U.v,U.c2),'top->c5, rest->c3':np.where(U.top,U.c5,U.c3),'REVERSE: top->c3, rest->v':np.where(U.top,U.c3,U.v)}
    print(f' {un}: train n {(~U.test).sum()} test n {U.test.sum()}')
    for nm,r in pol.items():
        r=pd.Series(np.asarray(r),index=U.index); out=[]
        for era,e in (('TR',~U.test),('TE',U.test)):
            z=r[e]; out.append(f'{era}: mean {z.mean()*100:+6.2f}% med {z.median()*100:+6.2f}% ex3 {ex3(z)*100:+6.2f}% win {np.mean(z>0)*100:4.1f}%')
        print(f'   {nm:30s} '+' | '.join(out))
    # subgroup view: top vs rest by exit
    for lab,m in (('TOP20',U.top),('REST',~U.top)):
        print(f'   -- {lab}: '+' | '.join(f'{c}: TR {U[m&~U.test][c].mean()*100:+6.1f}/TE {U[m&U.test][c].mean()*100:+6.1f} (med TE {U[m&U.test][c].median()*100:+5.1f})' for c in ('v','c2','c3','c4','c5')))
