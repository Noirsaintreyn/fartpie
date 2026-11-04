"""Nov 4 2026: Dow 30 with the live model. DECLARED BEFORE RESULTS. Dow names are all S&P members, so 'live + Dow' = (a) Dow-only book by track (15m only / 1h only / both tracks), (b) blends of two separate 6-slot books (no rebalancing): 70% S&P live + 30% Dow, 50/50, and Nasdaq-100 + Dow 50/50,
(c) Dow names get queue priority inside the live S&P book. 40 matched runs (same signals dropped in every book), both periods; DD is the point."""
_s = open('xs/universe_test.py').read(); exec(_s[:_s.index("B0 = ev(W[W.sp]")].replace("print('unclassified", "pass #").replace("print('parity", "pass #"))
LIVE = W[W.sp].reset_index(drop=True); DOWF = W[W.dow].reset_index(drop=True); NDXF = W[W.ndx].reset_index(drop=True)
def daily(Y, i):
    HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl
    s = sim(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], Y.ratio.values[m].astype(float), Y.tk.values[m])
    return np.exp(s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0))
def blend(A, Bk, w):
    res = []
    for i in range(1, 41):
        e = w * daily(A, i) + (1 - w) * daily(Bk, i); res.append(stat2(np.log(e)))
    return np.array(res)
B0 = ev(LIVE); print('40 matched runs, 2019-21 / 2022-26'); line('L live (S&P)', B0)
for nm, Y in (('Dow 30, both tracks', DOWF), ('Dow 30, 15m track only', DOWF[~DOWF.h1]), ('Dow 30, 1h track only', DOWF[DOWF.h1])):
    Y = Y.reset_index(drop=True); a, b = Y[Y.entry < '2022'], Y[Y.entry >= '2022']; print(f'  {nm}: {Y.tk.nunique()} tickers, {len(Y)} trades, avg {a.ret.mean()*100:+.2f}% / {b.ret.mean()*100:+.2f}%, runners {(Y.ret >= .15).sum()}'); line(nm, ev(Y), B0)
for nm, A, Bk, w in (('70% S&P live + 30% Dow', LIVE, DOWF, .7), ('50% S&P live + 50% Dow', LIVE, DOWF, .5), ('50% Nasdaq-100 + 50% Dow', NDXF, DOWF, .5), ('50% S&P live + 50% Nasdaq-100', LIVE, NDXF, .5)): line(nm, blend(A, Bk, w), B0)
P = LIVE.copy(); P['ratio'] = P.ratio + 1000 * P.dow.astype(float); line('Dow names first in the S&P queue', ev(P), B0)
