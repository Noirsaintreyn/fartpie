import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; A=pd.read_pickle(X+'vh_features.pkl'); A=A[A.d>='2019-01-01']
A['top']=A.lr>=A[~A.test].lr.quantile(.8); sign=dict(BEARLEN='-',DEPTH='+',WHIP='-',BRD='+',BRDCH='+',HEALTH='+')
UN={'A all':A,'B px<5':A[A.px<5],'C px<5 & top20':A[(A.px<5)&A.top]}
def t_(d): return d.mean()/(d.std()/np.sqrt(len(d))) if len(d)>5 and d.std()>0 else np.nan
for un,U in UN.items():
    print(f'\n#### {un}: train n {(~U.test).sum()} test n {U.test.sum()}')
    for f,sg in sign.items():
        tr=U[~U.test].dropna(subset=[f]); q1,q2=tr[f].quantile([1/3,2/3]).values; out=[]; ok=True
        for era,z in (('TR',U[~U.test]),('TE',U[U.test])):
            z=z.dropna(subset=[f]); hi=z[z[f]>=q2]; lo=z[z[f]<=q1]; d=(hi.groupby('mo').n.mean()-lo.groupby('mo').n.mean()).dropna(); tt=t_(d); sp=hi.n.mean()-lo.n.mean()
            out.append(f'{era}: hiT {hi.n.mean()*100:+6.1f}% loT {lo.n.mean()*100:+6.1f}% med {hi.n.median()*100:+5.1f}/{lo.n.median()*100:+5.1f} big30 {np.mean(hi.n>=.3)*100:4.1f}/{np.mean(lo.n>=.3)*100:4.1f}% t {tt:+.2f}')
            ok&=((sp>0)==(sg=='+')) and (era=='TR' or (abs(tt)>=2 and (tt>0)==(sg=='+')))
        print(f'  {f:7s}[{sg}] cuts {q1:+.3f}/{q2:+.3f} '+' | '.join(out)+('  <== PASS' if ok else ''))
