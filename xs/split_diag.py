"""Nov 4 2026: WHERE does the split books' backtest edge come from? DECLARED BEFORE RESULTS. Same engine/yardstick (40 matched runs, 2019-21 / 2022-26).
PART A attribution (full trade set, no drops): per-year return of LIVE vs the two split books; P&L contribution by ticker group
         (S&P-only names, names in BOTH S&P and Nasdaq-100, Nasdaq-100-only names, Dow names) and the top tickers that add/subtract vs LIVE.
PART B mechanisms, each vs LIVE (S&P, 6 slots):
   B1 more slots only: S&P, 12 slots (each slot 1/12, same size as a slot in a split book)
   B2 one merged queue: S&P + Nasdaq-100 in ONE 12-slot book
   B3 split, second book = Nasdaq-100 names NOT in the S&P only
   B4 split, second book = names in BOTH S&P and Nasdaq-100 (doubles exposure to the overlap)
   B5 split, second book = the S&P book itself (identical twin: should equal LIVE; sanity check)
PART C extract into the current book (one 6-slot S&P book):
   C1 Nasdaq-100 members first in the tie-break   C2 Nasdaq-100 members get ratio x 1.5   C3 8 slots + Nasdaq-100 first"""
_s = open('xs/universe_test.py').read(); exec(_s[:_s.index("B0 = ev(W[W.sp]")].replace("print('unclassified", "pass #").replace("print('parity", "pass #"))
LIVE = W[W.sp].reset_index(drop=True); DOWF = W[W.dow].reset_index(drop=True); NDXF = W[W.ndx].reset_index(drop=True)
body = src[src.index('def sim('):src.index('def ev(')]
def mksim(N): ns = {}; exec(body.replace('>= 6', f'>= {N}').replace('/ 6', f'/ {N}').replace('def sim(', 'def simN('), globals(), ns); return ns['simN']
def taken(Y, N=6):
    """same order/slots/breaker as sim; returns mask of trades taken (full set)"""
    E, Xx, R, K, TK = Y.entry.values, Y.exit.values, Y.ret.values, Y.ratio.values.astype(float), Y.tk.values
    o = np.lexsort((TK, -K, E)); op = []; pend = []; real = 0.0; win = deque(); lvl0 = 0.0; hist = []; tk_ = np.zeros(len(Y), bool)
    for idx in o:
        e, x, r = E[idx], Xx[idx], R[idx]
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q > e]
        if len(op) >= N: continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): continue
        op.append(x); pend.append((x, r)); tk_[idx] = True
    return tk_
def daily_runs(Y, N=6):
    sm = mksim(N); HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); out = []
    for i in range(1, 41):
        m = ~(((HSH * 2654435761 + i * 40503) % 1000) < 50); s = sm(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], Y.ratio.values[m].astype(float), Y.tk.values[m])
        out.append(np.exp(s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0)).values)
    return np.array(out)
def stats_from(E):  return np.array([stat2(pd.Series(np.log(e), index=IDX)) for e in E])
def show(nm, E, B):
    a = stats_from(E); m = np.median(a, axis=0); w1, w2 = (a[:, 2] > B[:, 2]).mean(), (a[:, 5] > B[:, 5]).mean()
    print(f'  {nm:<52} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}   beats live {w1:.0%} / {w2:.0%}' + ('  PASS' if w1 >= .7 and w2 >= .7 else ''), flush=True)
# ---------------- PART A
SPN = set(LIVE.tk); NDN = set(NDXF.tk); DWN = set(DOWF.tk)
grp = lambda t: 'both S&P+NDX' if (t in SPN and t in NDN) else 'NDX only' if t in NDN else 'S&P only'
print('names: S&P', len(SPN), ' NDX', len(NDN), ' both', len(SPN & NDN), ' NDX-only', sorted(NDN - SPN), ' Dow', len(DWN), flush=True)
A = {}
for nm, Y in (('LIVE', LIVE), ('NDX', NDXF), ('DOW', DOWF)):
    m = taken(Y); Z = Y[m].copy(); Z['contrib'] = Z.ret / 6; A[nm] = Z
    eq = Z.groupby(Z.exit.dt.year).contrib.sum(); print(f'  {nm} book: {m.sum()} trades taken of {len(Y)}; simple P&L by exit year (sum of ret/6):', ' '.join(f'{y}:{v*100:+.0f}%' for y, v in eq.items()), flush=True)
def contrib(parts):
    out = pd.concat([Z.assign(contrib=Z.contrib * w) for Z, w in parts]); return out
SPLIT1 = contrib([(A['LIVE'], .5), (A['NDX'], .5)]); SPLIT2 = contrib([(A['NDX'], .5), (A['DOW'], .5)]); L1 = A['LIVE']
print('\n  P&L by ticker group (sum of slot-weighted returns, 2019-26):')
for nm, Z in (('LIVE', L1), ('S&P+NDX split', SPLIT1), ('NDX+Dow split', SPLIT2)):
    g = Z.groupby(Z.tk.map(grp)).contrib.sum(); per = Z.groupby(np.where(Z.entry.dt.year <= 2021, '19-21', '22-26')).contrib.sum()
    print(f'   {nm:<16}', ' | '.join(f'{k} {v*100:+.0f}%' for k, v in g.items()), f'| total {Z.contrib.sum()*100:+.0f}% (19-21 {per.get("19-21", 0)*100:+.0f}%, 22-26 {per.get("22-26", 0)*100:+.0f}%)', flush=True)
for nm, Z in (('S&P+NDX split', SPLIT1), ('NDX+Dow split', SPLIT2)):
    d = Z.groupby('tk').contrib.sum().sub(L1.groupby('tk').contrib.sum(), fill_value=0).sort_values()
    print(f'   {nm} minus LIVE, by ticker: top +', ' '.join(f'{t} {v*100:+.0f}' for t, v in d.tail(8)[::-1].items()), ' | top -', ' '.join(f'{t} {v*100:+.0f}' for t, v in d.head(6).items()), flush=True)
    yy = Z.groupby(Z.exit.dt.year).contrib.sum().sub(L1.groupby(L1.exit.dt.year).contrib.sum(), fill_value=0); print(f'   {nm} minus LIVE, by exit year:', ' '.join(f'{y}:{v*100:+.0f}' for y, v in yy.items()), flush=True)
# ---------------- PART B
print('\nPART B mechanisms (40 matched runs)', flush=True)
EL = daily_runs(LIVE); B0 = stats_from(EL); show('L live S&P 6 slots', EL, B0)
EN, ED = daily_runs(NDXF), daily_runs(DOWF)
show('ref: 50% S&P + 50% NDX (split)', .5 * EL + .5 * EN, B0); show('ref: 50% NDX + 50% Dow (split)', .5 * EN + .5 * ED, B0)
show('B1 S&P only, 12 slots', daily_runs(LIVE, 12), B0)
MERGED = pd.concat([LIVE, NDXF]).drop_duplicates(['tk', 'entry', 'h1']).sort_values('entry').reset_index(drop=True); show('B2 S&P + NDX merged, ONE 12-slot book', daily_runs(MERGED, 12), B0)
NO = NDXF[~NDXF.tk.isin(SPN)].reset_index(drop=True); BO = NDXF[NDXF.tk.isin(SPN)].reset_index(drop=True)
print(f'   (NDX-only trades {len(NO)}, overlap trades {len(BO)})', flush=True)
show('B3 split: S&P + NDX-only-names book', .5 * EL + .5 * daily_runs(NO), B0); show('B4 split: S&P + overlap-names book (doubles them)', .5 * EL + .5 * daily_runs(BO), B0)
show('B5 split: S&P + S&P twin (sanity, = live)', .5 * EL + .5 * EL, B0)
# ---------------- PART C
print('\nPART C extract into the current 6-slot S&P book', flush=True)
def K(Y, k): Y = Y.copy(); Y['ratio'] = k; return Y
show('C1 NDX members first in the tie-break', daily_runs(K(LIVE, LIVE.ratio + 1000 * LIVE.tk.isin(NDN)), 6), B0)
show('C2 NDX members ratio x 1.5', daily_runs(K(LIVE, LIVE.ratio * (1 + .5 * LIVE.tk.isin(NDN))), 6), B0)
show('C3 8 slots + NDX members first', daily_runs(K(LIVE, LIVE.ratio + 1000 * LIVE.tk.isin(NDN)), 8), B0)
