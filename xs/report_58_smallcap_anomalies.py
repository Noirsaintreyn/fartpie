"""Nov 14 2026 KNOWN SMALL-CAP ANOMALIES AS FLIP FILTERS (declared before results). Long flips, 374 small/mid names, 1h, Apr 2025-Nov 2026, net of 15bp, halves. Hypotheses (direction fixed from the literature before looking): H1 DILUTION: a recent 8-K item 3.02 (unregistered equity sale) or 1.01/2.03 financing within 5 trading days -> WORSE flips (issuers underperform after offerings). H2 LOTTERY/MAX: top-third highest single-day return over the prior 20 days -> WORSE (lottery demand overprices, strongest in small caps). H3 LOW PRICE: entry price < $5 -> WORSE. H4 ILLIQUIDITY: lowest-third prior-20d dollar volume -> flips pay MORE gross but cost more (tested gross, flagged). H5 MGMT CHANGE: 5.02 within 5 days -> descriptive. Terciles are PIT-expanding."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
P = pd.read_pickle(SC + 'part_small.pkl'); D = pd.read_pickle('live/midsmall_bars.pkl')['d']; K = pd.read_csv('/home/claude/fartpie/xs/ext/8k_items_midsmall.csv'); K['items'] = K['items'].astype(str); K['acc'] = pd.to_datetime(K.accept, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
P = P[P.d == 1].copy().reset_index(drop=True); P['net'] = P.ret - 0.0015; P['MAX'] = np.nan; P['PX'] = np.nan; P['DV'] = np.nan
for tk, ix in P.groupby('tk').groups.items():
    B = D[tk]; day = B.index.normalize(); dcl = B.close.groupby(day).last(); dv = (B.close * B.volume).groupby(day).sum(); r = dcl.pct_change(); mx = r.rolling(20).max().shift(1); dvm = dv.rolling(20).mean().shift(1); pxs = dcl.shift(1)
    td = P.loc[ix, 't'].dt.normalize().values; P.loc[ix, 'MAX'] = mx.reindex(td).values; P.loc[ix, 'DV'] = dvm.reindex(td).values; P.loc[ix, 'PX'] = pxs.reindex(td).values
bd = pd.offsets.BDay(5)
for nm, items in (('DILUTE', {'3.02'}), ('FINANCE', {'1.01', '2.03', '3.02'}), ('MGMT', {'5.02'}), ('NEWS', {'7.01', '8.01'})):
    m = K[K['items'].map(lambda s: bool(items & set(s.split(','))))]; per = {tk: x.acc.sort_values().values for tk, x in m.groupby('ticker')}; flag = np.zeros(len(P), bool)
    for tk, ix in P.groupby('tk').groups.items():
        a = per.get(tk)
        if a is None: continue
        t = P.loc[ix, 't'].values; pos = np.searchsorted(a, t, side='right') - 1; ok = pos >= 0; last = a[np.clip(pos, 0, None)]; lim = (pd.to_datetime(P.loc[ix, 't']) - bd).values; flag[P.index.get_indexer(ix)] = ok & (last >= lim)
    P[nm] = flag
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
P = P.sort_values('t').reset_index(drop=True); mid = pd.Timestamp('2026-03-01'); P['LOW'] = P.PX < 5
for c in ('MAX', 'DV'):
    v = P[c].values; o = np.full(len(v), np.nan)
    for i in range(300, len(v)):
        h = v[:i]; h = h[~np.isnan(h)]
        if not np.isnan(v[i]): q1, q2 = np.quantile(h, [1 / 3, 2 / 3]); o[i] = 0 if v[i] <= q1 else 2 if v[i] >= q2 else 1
    P['T_' + c] = o
print('LONG flips net mean % (feature on | off) and t, halves + all; BV>=1.5 subset second line')
def line(name, on, off):
    out = f' {name:22s}'
    for lab, m in (('h1', P.t < mid), ('h2', P.t >= mid), ('all', P.t >= P.t.min())):
        a = P[m & on]; b = P[m & off]; out += f' | {lab}: {a.net.mean()*100:+.2f} (n {len(a)}) vs {b.net.mean()*100:+.2f} t {tt(a.net, b.net):+.1f}'
    print(out)
for sub, msk in (('ALL FLIPS', np.ones(len(P), bool)), ('BV>=1.5 ONLY', (P.BV >= 1.5).values)):
    print(f'\n[{sub}]'); mk = pd.Series(msk)
    for nm in ('DILUTE', 'FINANCE', 'MGMT', 'NEWS'): line(f'8-K {nm} <=5d', mk & P[nm], mk & ~P[nm])
    line('MAX top3rd (lottery)', mk & (P.T_MAX == 2), mk & (P.T_MAX == 0)); line('price < $5', mk & P.LOW, mk & ~P.LOW & P.PX.notna()); line('illiquid 3rd (DV low)', mk & (P.T_DV == 0), mk & (P.T_DV == 2)); line('(gross) illiquid 3rd', mk & (P.T_DV == 0), mk & (P.T_DV == 2)) if False else None
