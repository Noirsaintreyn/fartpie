"""Nov 3 2026: how fast should the post-report cut be? Per-trade accounting (cut at the checkpoint vs the live exit) for delays of 15 / 30 / 45 / 60 min after the report
and thresholds -1/-2/-3/-5/-8%. Book trades held through a report (live engine, taken by the live book). Fill = bar close at the checkpoint minus 15bp."""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('tm/*.parquet')); L = L[L['var'] == 'vec'].reset_index(drop=True)
ex = open('/home/claude/fartpie/xs/earn_both.py').read(); exec(ex[ex.index('FULL = {}'):ex.index('MODES = ')])
DEL = (15, 30, 45, 60); ev = []
for x in L.itertuples():
    S_ = full(x.tk); b = bars(x.tk)
    if S_ is None or b is None or x.tk not in REP: continue
    op, dc = b; e, xt = pd.Timestamp(x.entry), pd.Timestamp(x.exit); px0 = op.get(e); D = dc.index
    for tt in REP[x.tk]:
        d = tt.normalize(); ac = tt.hour >= 12; g = d if not ac else (D[D > d][0] if (D > d).any() else None)
        if g is None or g not in D: continue
        gopen = g + pd.Timedelta('9h30min'); pdy = D[D < g]
        if not (len(pdy) and e < gopen <= xt and px0 == px0 and px0): continue
        c0 = dc[pdy[-1]]; row = dict(tk=x.tk, entry=x.entry, hold=x.ret, rep=tt, gopen=gopen, o1=op.get(gopen, np.nan) / c0 - 1, ok=True)
        for m in DEL:
            p, tc = px_at(S_, tt + pd.Timedelta(minutes=m))
            if p is None: row['ok'] = False; break
            row[f'rea{m}'] = p / c0 - 1; row[f'pr{m}'] = p * (1 - .0015) / px0 - 1
        if row['ok']: ev.append(row)
        break
E = pd.DataFrame(ev); TK = taken(L); key = set(zip(TK.tk, TK.entry.astype('datetime64[ns]'))); E['taken'] = [(a, np.datetime64(b, 'ns')) in key for a, b in zip(E.tk, E.entry)]; T = E[E.taken]
print(f'{len(T)} book trades held through a report with every checkpoint available; mean total gap to the open {T.o1.mean():+.2%}')
print('\nrule "cut if reaction <= -X% at +N min after the report": fires / net effect on the book over 2019-26 (sum of (cut - hold)/6) / mean per fired trade / 2019-21 | 2022-26 net')
for m in DEL:
    print(f' +{m} min')
    for th in (.01, .02, .03, .05, .08):
        g = T[T[f'rea{m}'] <= -th]; dd = g[f'pr{m}'] - g.hold; y = pd.to_datetime(g.entry).dt.year
        print(f'   <= -{th:.0%}: {len(g):3d} fires  net {dd.sum() / 6:+.2%}  mean {dd.mean():+.2%}   {dd[y <= 2021].sum() / 6:+.2%} | {dd[y > 2021].sum() / 6:+.2%}')
print('\nhow far did the stock fall AFTER the checkpoint to the next open (bad reactions only)?')
for m in DEL:
    g = T[T[f'rea{m}'] <= -.03]; print(f'   +{m} min, reaction <= -3%: n {len(g)}, mean total gap to the open {g.o1.mean():+.2%}, mean reaction at checkpoint {g[f"rea{m}"].mean():+.2%}, so mean move after the checkpoint {g.o1.mean() - g[f"rea{m}"].mean():+.2%}')
