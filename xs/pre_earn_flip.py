"""Nov 1 2026 (from the CVNA / ORCL case study: both winners flipped 1-2 days before a catalyst): flips in the days before an earnings report.
k = trading days from entry to the report's gap day. Avg trade by k, and 'pre-earnings flips first' as the tie-break."""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = pd.read_parquet('/tmp/claude-0/ml/mtf_states.parquet').sort_values('entry').reset_index(drop=True)
k = np.full(len(L), np.nan)
for tk, ix in L.groupby('tk').groups.items():
    f = f'earn/{tk}.csv'; b = bars(tk)
    if not os.path.exists(f) or b is None: continue
    e = pd.read_csv(f)
    if not len(e): continue
    D = b[1].index; t = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    G = sorted({d.normalize() if d.hour < 12 else (D[D > d.normalize()][0] if (D > d.normalize()).any() else d.normalize()) for d in t})
    G = pd.DatetimeIndex(G)
    for i in ix:
        en = L.at[i, 'entry'].normalize(); nx = G[G >= en]
        if len(nx): k[i] = np.searchsorted(D, nx[0]) - np.searchsorted(D, en)
L['k'] = k; TK = taken(L).merge(L[['tk', 'entry', 'k']], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
b = pd.cut(TK.k, [-1, 0, 1, 3, 5, 10, 20, 999], labels=['gap day', '1 day before', '2-3', '4-5', '6-10', '11-20', '21+'])
print('avg trade % (n) by trading days until the next earnings gap day'); print((TK.groupby([b, 'h']).ret.agg(['size', 'mean']).unstack('h') * [1, 1, 100, 100]).round(2).to_string())
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def sh(nm, Z): print(f'  {nm:<44}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b_) for a, b_ in P)), flush=True)
print('\nbook'); sh('live', L); pr = L.ratio.groupby(L.entry.dt.year).rank(pct=True)
for lo, hi in ((1, 3), (1, 5), (2, 10)):
    Z = L.copy(); Z['ratio'] = (pr + ((L.k >= lo) & (L.k <= hi)) * 10).values; sh(f'flips {lo}-{hi} days before earnings first', Z)
