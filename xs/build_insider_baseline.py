"""Insider-size baseline for sub-$5 flips (Rey Nov 26 2026: 'explain how much insider buying or selling is noticeable vs normal vs outlier, on small caps').
For every sub-$5 flip in structure_small.pkl: open-market insider $ (code P buys 2018+, code S sales Apr-Oct 2025 only, the depth the SEC dataset on hand covers) in the 30d before the flip,
as $ and as % of market cap (shares_pit). Writes insider_baseline.json: share of flips with any trade, and percentile cuts among flips that had one."""
import pandas as pd, numpy as np, json, os
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
s = pd.read_pickle('structure_small.pkl' if os.path.exists('structure_small.pkl') else SP + 'ext/structure_small.pkl'); s = s[s.px < 5][['tk', 'd', 'px']].copy(); s['d'] = pd.to_datetime(s.d)
sh = pd.read_pickle(SP + 'ext/shares_pit.pkl'); sh['filed'] = pd.to_datetime(sh.filed); sh = sh.sort_values('filed')
P = pd.read_pickle(SP + 'ext/insider_P_hist.pkl'); P = P[(P.TRANS_CODE == 'P') & (P.TRANS_ACQUIRED_DISP_CD == 'A')].copy(); P['d'] = pd.to_datetime(P.fd); P['tk'] = P.ISSUERTRADINGSYMBOL.astype(str)
A = pd.read_pickle(SP + 'ext/insider.pkl'); S_ = A[(A.TRANS_CODE == 'S') & (A.TRANS_ACQUIRED_DISP_CD == 'D')].copy() if 'TRANS_CODE' in A else None
S_['d'] = pd.to_datetime(S_.FILING_DATE, format='%d-%b-%Y', errors='coerce'); S_['tk'] = S_.ISSUERTRADINGSYMBOL.astype(str)
for D in (P, S_): D['usd'] = pd.to_numeric(D.TRANS_SHARES, errors='coerce') * pd.to_numeric(D.TRANS_PRICEPERSHARE, errors='coerce')
def agg(D, lo=None):
    D = D.dropna(subset=['d', 'usd']); g = {k: v for k, v in D.groupby('tk')}; out = []
    for r in s.itertuples():
        if lo is not None and not (lo[0] <= r.d <= lo[1]): out.append((np.nan, np.nan, np.nan)); continue
        x = g.get(r.tk); u = 0.; n = 0
        if x is not None: m = x[(x.d > r.d - pd.Timedelta(days=30)) & (x.d <= r.d)]; u = m.usd.sum(); n = len(m)
        q = sh[(sh.tk == r.tk) & (sh.filed <= r.d)].val; cap = (q.iloc[-1] * r.px) if len(q) else np.nan
        out.append((u, n, u / cap * 100 if cap == cap and cap > 0 else np.nan))
    return pd.DataFrame(out, columns=['usd', 'n', 'pcap'])
res = {}
for nm, D, lo in (('buy', P, None), ('sell', S_, (pd.Timestamp('2025-04-15'), pd.Timestamp('2025-10-31')))):
    r = agg(D, lo); r = r.dropna(subset=['usd']); h = r[r.usd > 0]
    res[nm] = dict(flips=int(len(r)), any=float(len(h) / len(r)), usd={q: float(h.usd.quantile(q)) for q in (.5, .75, .9, .975)}, pcap={q: float(h.pcap.dropna().quantile(q)) for q in (.5, .75, .9, .975)}, n_ge3=float((h.n >= 3).mean()) if len(h) else 0.)
    print(nm, json.dumps(res[nm], indent=1))
json.dump(res, open('insider_baseline.json', 'w'), indent=1)
