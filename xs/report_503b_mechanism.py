"""Report 503b: mechanism read for 503's surprise (BUY ELEV, unusual vs the stock's own baseline, is WORSE: opposite of the declared sign). Descriptive, no new hypotheses tested for significance: what do ELEV buys look like vs ROUTINE buys and the rest?"""
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
S = pd.read_pickle(SP + 'ext/insider_own_baseline_flips.pkl')
A = pd.read_pickle(SP + 'ext/insider_PS_all.pkl'); A['d'] = pd.to_datetime(A.FILING_DATE, format='%d-%b-%Y', errors='coerce'); A = A[(A.TRANS_CODE == 'P') & (A.TRANS_ACQUIRED_DISP_CD == 'A')].dropna(subset=['d'])
A['usd'] = pd.to_numeric(A.TRANS_SHARES, errors='coerce') * pd.to_numeric(A.TRANS_PRICEPERSHARE, errors='coerce'); G = {k: v for k, v in A.groupby('ISSUERTRADINGSYMBOL')}
def roles(r):
    x = G.get(r.tk)
    if x is None: return pd.Series(dict(nbuy=0, own10=np.nan, offc=np.nan, price_rel=np.nan))
    w = x[(x.d > r.d - pd.Timedelta(days=30)) & (x.d <= r.d)]
    rel = w.RPTOWNER_RELATIONSHIP.fillna('')
    return pd.Series(dict(nbuy=w.RPTOWNER_CIK.nunique() if 'RPTOWNER_CIK' in w else w.RPTOWNERCIK.nunique(), own10=rel.str.contains('TenPercent').mean() if len(w) else np.nan, offc=rel.str.contains('Officer').mean() if len(w) else np.nan, price_rel=np.nan))
Z = S[S.B_ANY].copy(); Z = pd.concat([Z, Z.apply(roles, axis=1)], axis=1)
print(f"{'class':9s} {'n':>3s} | prior60d ret med | LMC med | buyers med | share with 10% owner | share with officer | reach | mean r | median r")
for c in ('NEW', 'ELEV', 'ROUTINE'):
    X = Z[Z.buy_c == c]; print(f"{c:9s} {len(X):3d} | {X.R60.median():+.1%} | {X.LMC.median():.1f} | {X.nbuy.median():.0f} | {X.own10.mean():.0%} | {X.offc.mean():.0%} | {X.hit.mean():.0%} | {X.rw.mean():+.1%} | {X.rw.median():+.1%}")
X = S[~S.B_ANY]; print(f"{'no buys':9s} {len(X):3d} | {X.R60.median():+.1%} | {X.LMC.median():.1f} | - | - | - | {X.hit.mean():.0%} | {X.rw.mean():+.1%} | {X.rw.median():+.1%}")
