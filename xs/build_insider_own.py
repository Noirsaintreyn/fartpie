"""insider_own.json for the live card: each small-price issuer's open-market buy/sell history through 2026Q1 (own-baseline), plus the measured class stats from report 503 (sub-$5 flips)."""
import pandas as pd, numpy as np, json
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
A = pd.read_pickle(SP + 'ext/insider_PS_all.pkl'); A['d'] = pd.to_datetime(A.FILING_DATE, format='%d-%b-%Y', errors='coerce'); A['px'] = pd.to_numeric(A.TRANS_PRICEPERSHARE, errors='coerce')
A['usd'] = pd.to_numeric(A.TRANS_SHARES, errors='coerce') * A.px; A = A.dropna(subset=['d', 'usd', 'ISSUERTRADINGSYMBOL']); A = A[A.usd > 0]
A['side'] = np.where((A.TRANS_CODE == 'P') & (A.TRANS_ACQUIRED_DISP_CD == 'A'), 'b', np.where((A.TRANS_CODE == 'S') & (A.TRANS_ACQUIRED_DISP_CD == 'D'), 's', 'x')); A = A[A.side != 'x']
keep = A.groupby('ISSUERTRADINGSYMBOL').px.median(); keep = set(keep[keep < 12].index); A = A[A.ISSUERTRADINGSYMBOL.isin(keep)]
H = {}
for (tk, sd), x in A.groupby(['ISSUERTRADINGSYMBOL', 'side']): H.setdefault(tk, {})[sd] = [[int(d.strftime('%Y%m%d')), int(u)] for d, u in zip(x.d, x.usd)]
S = pd.read_pickle(SP + 'ext/insider_own_baseline_flips.pkl'); st = {}
base = dict(n=len(S), reach=float(S.hit.mean()), mean=float(S.rw.mean()), med=float(S.rw.median()))
for side in ('buy', 'sell'):
    for c in ('NEW', 'ELEV', 'ROUTINE'): X = S[S[side + '_c'] == c]; st[f'{side}_{c}'] = dict(n=int(len(X)), reach=float(X.hit.mean()) if len(X) else None, mean=float(X.rw.mean()) if len(X) else None, med=float(X.rw.median()) if len(X) else None)
json.dump(dict(hist=H, stats=st, base=base, through='2026-03-31'), open('insider_own.json', 'w'), separators=(',', ':')); print(len(H), 'issuers', json.dumps(st)[:400])
