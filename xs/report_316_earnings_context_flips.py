"""Nov 18 2026 report 316. FUNDAMENTAL/EVENT LAYER on the screener (Rey: 'macro and fundamentals are the thesis, TA is the entry'; desk docs D1a/D1b: composite revision/quality + PEAD). Data on disk: earnings (when, est, rep, surp%) for the 54 screener names, ~50 quarters each.
THEORY (declared before running). (H1) PEAD (Bernard-Thomas 1989): the market under-reacts to earnings surprises, so a flip that occurs 1-60 trading days after a surprise in the top tercile (SUE proxy = surp %) should out-earn a flip after a bottom-tercile surprise (same sign both halves; cuts frozen on train). Why it would fail: megacap PEAD has decayed; flips already condition on price having turned. (H2) Event-risk: flips within 5 trading days BEFORE the next report have higher dispersion (std of ret), not higher mean; it is a sizing lever (the report-303 finding: edge flat, sigma is what moves). (H3) the reaction window (flip within 3 trading days after a report) mixes the event into the flip.
Universe: b302 trades for the 54 names with earnings files (12,360 trades). Train/test = b302.tr. Reported: mean ret %, t, std, n."""
import numpy as np, pandas as pd, glob, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; B = pd.read_parquet(SP + 'b302.parquet'); B['ed'] = pd.to_datetime(B.entry).dt.tz_localize(None).dt.normalize() if getattr(pd.to_datetime(B.entry).dt, 'tz', None) is not None else pd.to_datetime(B.entry).dt.normalize()
rows = []
for tk, g in B.groupby('tk'):
    try: e = pd.read_parquet(f'/home/claude/fartpie/data/fund/earn/{tk}.parquet')
    except Exception: continue
    e = e.dropna(subset=['when']).copy(); e['d'] = pd.to_datetime(e['when']).dt.normalize(); e = e.sort_values('d')
    if getattr(e.d.dt, 'tz', None) is not None: e['d'] = e.d.dt.tz_localize(None)
    ds = e.d.values.astype('datetime64[D]'); sur = e.surp.values
    for i, r in g.iterrows():
        k = np.searchsorted(ds, np.datetime64(r.ed, 'D'), side='right') - 1
        prev = ds[k] if k >= 0 else None; nxt = ds[k + 1] if k + 1 < len(ds) else None
        rows.append((i, (np.busday_count(prev, np.datetime64(r.ed, 'D')) if prev is not None else np.nan), (np.busday_count(np.datetime64(r.ed, 'D'), nxt) if nxt is not None else np.nan), sur[k] if k >= 0 else np.nan))
X = pd.DataFrame(rows, columns=['i', 'since', 'until', 'surp']).set_index('i'); B = B.join(X); print('trades', len(B), 'with earnings context', int(B.since.notna().sum()), '| since<=60:', int((B.since <= 60).sum()))
def cell(m, lab):
    out = []
    for nm, t in (('train', B.tr), ('test', ~B.tr)):
        r = B[t & m].ret.dropna()
        out.append(f'{nm} {100 * r.mean():+.2f}% (t{r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 2 else np.nan:+.1f}, sd {100 * r.std():.1f}, n{len(r)})')
    print(f'  {lab:34s}', ' | '.join(out))
print('\n(H1) PEAD: surprise tercile (cuts on train) among flips 1-60 busdays after a report')
w = (B.since >= 1) & (B.since <= 60); q = B[B.tr & w].surp.quantile([1 / 3, 2 / 3]).values; print('  surprise cuts', np.round(q, 1))
cell(w & (B.surp <= q[0]), 'bottom tercile surprise'); cell(w & (B.surp > q[0]) & (B.surp < q[1]), 'mid'); cell(w & (B.surp >= q[1]), 'top tercile surprise'); cell(w & (B.surp < 0), 'any miss (surp<0)'); cell(w & (B.surp > 5), 'beat >5%'); cell(~w, 'flips >60d after a report')
print('\n(H2) event risk: flips by days until NEXT report'); cell(B.until <= 5, 'next report <=5 bdays'); cell((B.until > 5) & (B.until <= 20), '6-20'); cell(B.until > 20, '>20')
print('\n(H3) reaction window'); cell(B.since <= 3, 'flip <=3 bdays after report'); cell((B.since > 3) & (B.since <= 10), '4-10')
print('\nPEAD by year (top minus bottom tercile surprise, ret %):', {y: round(100 * (g[(g.since >= 1) & (g.since <= 60) & (g.surp >= q[1])].ret.mean() - g[(g.since >= 1) & (g.since <= 60) & (g.surp <= q[0])].ret.mean()), 2) for y, g in B.groupby(pd.to_datetime(B.entry).dt.year) if len(g) > 100})
B.to_parquet(SP + 'b316.parquet')
