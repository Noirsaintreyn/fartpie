"""[SMALL/MID version: 374 names, 1h, Apr 2025-Nov 2026, halves instead of eras; 8-K coverage checked] Nov 14 2026 CATALYST LAYER (declared before results). Idea (Rey): the vector times the entry; fundamentals/news explain WHY participants enter. Test: do vector flips that follow a material SEC 8-K (a reason for participants to act) become the tail more often than flips with no catalyst? 8-K accept time must be <= flip-bar time (causal). Groups by 8-K item: EARN 2.02, AGREE 1.01, DILUTE 3.02/2.03, MGMT 5.02, OTHER 7.01/8.01, ANY (any item except 9.01-only). Windows: filed within the prior 1, 3, 5 trading days. Outcome: vol-normalized return (ret / ATR%) and raw ret, tail = top decile of vol-normalized return per direction. Large caps: strict eras 2019-21 vs 2022+ (xs/ext/8k_items.csv has 575 tickers 2006-2026). PASS = same sign both eras, pooled |t| >= 2."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import sys, os; sys.path.insert(0, '/home/claude/fartpie/xs')
from participant_feats import flips_for
CA = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/part_small.pkl'
if os.path.exists(CA): P = pd.read_pickle(CA)
else:
    D = pd.read_pickle('live/midsmall_bars.pkl')['d']; rows = []
    for tk, B in D.items():
        if B.close.median() < 0.3: continue
        rows += flips_for(B, 7, 500, tk)
    P = pd.DataFrame(rows); P.to_pickle(CA)
P = P.dropna(subset=['RISK']).sort_values('t').reset_index(drop=True); P['retn'] = P.ret / np.exp(P.RISK)
K = pd.read_csv('xs/ext/8k_items.csv'); K['acc'] = pd.to_datetime(K.accept, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); K['items'] = K['items'].astype(str)
G = {'ANY': lambda s: bool(set(s.split(',')) - {'9.01', 'nan'}), 'EARN': lambda s: '2.02' in s.split(','), 'AGREE': lambda s: '1.01' in s.split(','), 'DILUTE': lambda s: bool({'3.02', '2.03'} & set(s.split(','))), 'MGMT': lambda s: '5.02' in s.split(','), 'OTHER': lambda s: bool({'7.01', '8.01'} & set(s.split(',')))}
print('flip tickers', P.tk.nunique(), 'with 8-K coverage', len(set(P.tk) & set(K.ticker)))
bd = pd.offsets.BDay
for g, fn in G.items():
    m = K[K['items'].map(fn)]; per = {tk: x.acc.sort_values().values for tk, x in m.groupby('ticker')}
    for W in (1, 3, 5):
        flag = np.zeros(len(P), bool)
        for tk, ix in P.groupby('tk').groups.items():
            a = per.get(tk)
            if a is None: continue
            t = P.loc[ix, 't'].values; pos = np.searchsorted(a, t, side='right') - 1; ok = pos >= 0; last = np.where(ok, a[np.clip(pos, 0, None)], np.datetime64('NaT'))
            lim = (pd.to_datetime(P.loc[ix, 't']) - bd(W)).values; flag[P.index.get_indexer(ix)] = ok & (last >= lim)
        P[f'{g}{W}'] = flag
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
for d, nm in ((1, 'LONG'), (-1, 'SHORT')):
    S = P[P.d == d].copy(); S['tl'] = S.retn >= S.retn.quantile(0.9); print(f'\n{nm} n {len(S)} | cols: era | share with catalyst | tail% with vs without | mean retn(ATR-scaled)% | raw mean ret% with / without | t(retn)')
    for g in G:
        for W in (1, 3, 5):
            line = f' {g:6s}{W}d'
            for e, (a, b) in (('h1', ('2025', '2026-03-01')), ('h2', ('2026-03-01', '2027')), ('all', ('2025', '2027'))):
                s = S[(S.t >= a) & (S.t < b)]; y = s[s[f'{g}{W}']]; n_ = s[~s[f'{g}{W}']]
                line += f' | {e}: {len(y)/len(s)*100:4.1f}% tail {y.tl.mean()*100:4.1f}/{n_.tl.mean()*100:4.1f} raw {y.ret.mean()*100:+.2f}/{n_.ret.mean()*100:+.2f} t {tt(y.retn, n_.retn):+.1f}'
            print(line)
