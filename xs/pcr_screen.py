"""Oct 26 2026: put/call option volume as a second screener input (like the options ratio), on the 23 live names that have daily
option summaries (xs/ext/opt_summary, 2018-2026). Each live trade gets the put/call volume ratio known before its entry (prior day):
 pcr      prior-day put volume / call volume
 pcr5     5-day average
 pcr_rel  prior day vs that stock's own 60-day median (unusual put buying for that name)
 pcr_xs   cross-sectional percentile among the 23 names that day
Then: average trade by tercile, and the live book with (a) skip covered trades in the worst tercile, (b) tie-break by low put/call."""
import numpy as np, pandas as pd, glob, os
exec(open(os.path.expanduser('xs/book_mtm.py')).read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
O = {os.path.basename(f)[:-4]: pd.read_csv(f, parse_dates=['date']).set_index('date') for f in glob.glob('/mnt/user-data/uploads/fartpie/xs/ext/opt_summary/*.csv')}
pc = pd.DataFrame({t: d.put_vol / d.call_vol for t, d in O.items()}).sort_index()
F = {'pcr': pc, 'pcr5': pc.rolling(5).mean(), 'pcr_rel': pc / pc.rolling(60, min_periods=20).median(), 'pcr_xs': pc.rank(axis=1, pct=True)}
L = prep(glob.glob('xs/ext/tm/*.parquet')); L = L[L['var'] == 'vec'].copy()
for k, Fm in F.items():
    Fs = Fm.shift(1)                                                     # known before the entry day
    L[k] = [Fs[t].get(d, np.nan) if t in Fs.columns else np.nan for t, d in zip(L.tk, L.d)]
C = L[L.pcr.notna()]; print(f'live trades {len(L)}, with put/call data {len(C)} ({C.tk.nunique()} names)\n')
for k in F:
    q = pd.qcut(C[k], 3, labels=['low puts', 'mid', 'high puts'])
    g = C.groupby([q, np.where(C.d.dt.year >= 2022, '22-26', '19-21')]).ret.agg(['mean', 'size']).unstack(); g[[('mean', '19-21'), ('mean', '22-26')]] *= 100
    print(k, '(avg trade %, n)'); print(g.round(3).to_string(), '\n')
P_ = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<56}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P_)), flush=True)
print(f'{"":<58}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', L)
for k in F:
    cut = C[C.d.dt.year < 2022][k].quantile(2 / 3)                         # cut fixed on 2019-21
    show(f'skip covered trades with {k} in the top third (> {cut:.2f})', L[~(L[k] > cut)])
    cl = C[C.d.dt.year < 2022][k].quantile(1 / 3); show(f'skip covered trades with {k} in the bottom third (< {cl:.2f})', L[~(L[k] < cl)])
    z = L.copy(); z['ratio'] = z.ratio - np.nan_to_num(z[k].rank(pct=True), nan=.5) * 0.5; show(f'tie-break nudge: lower {k} ranks first', z)
