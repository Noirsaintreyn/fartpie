"""Nov 8 2026: ACCEPTANCE EXIT - enter at the flip as now, but if the next 15m bar closes back on the wrong side of the flip bar's close (a fake-out),
get out at the next open; otherwise hold to the 15m vector exit (stop 50, flat 15:55). DECLARED BEFORE RESULTS (prompted by nq_more_features: accepted flips +32.6,
fake-outs -21.0 per flip, but waiting for the confirm bar to enter leaves +5.0). Every RTH flip and the Rule A subset; NQ points, 1 contract; per day by period."""
_tx = open('xs/nq_th_exits.py').read(); exec(_tx[:_tx.index("ALLD = pd.DatetimeIndex")])
bi = pd.Series(np.arange(len(B)), index=B.end.values)
def sim_acc(j, d, accept_exit):
    k = bi.get(t5[j]); fc = B.close.values[k] if k is not None else None
    e = o5[j]; kk = k5[j]
    for q in range(j, len(H)):
        if d5[q] != d5[j] or m5[q] >= 955: return (c5[q - 1] - e) * d * kk, q - 1
        if (((l5[q] if d > 0 else h5[q]) - e) * d * kk) <= -50: return -50.0, q
        nxt = q + 1 < len(H) and d5[q + 1] == d5[j]
        if not nxt: continue
        if m5[q] % 15 == 10:
            if accept_exit and q - j == 2 and fc is not None and (c5[q] - fc) * d <= 0: return (o5[q + 1] - e) * d * kk, q     # first 15m bar after entry: fake-out
            st = reg15.get(t5[q] + np.timedelta64(5, 'm'))
            if st is not None and st == -d: return (o5[q + 1] - e) * d * kk, q
    return None
ALLD = pd.DatetimeIndex(sorted(H.day.unique()))
def book(X, acc):
    rows = []
    for day, g in X.groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            if n >= 4 or r.j <= busy: continue
            res = sim_acc(int(r.j), int(r.dir), acc)
            if res is None: continue
            rows.append((day, res[0] - 1.0)); busy = res[1]; n += 1
    T = pd.DataFrame(rows, columns=['day', 'pts']); P = T.groupby('day').pts.sum().reindex(ALLD).fillna(0.0); M = P.groupby(P.index.to_period('M')).sum(); cum = P.cumsum()
    per = np.select([P.index < '2022-01-01', P.index < '2024-01-01', P.index < '2026-01-01'], ['18-21', '22-23', '24-25'], '2026')
    return P.mean(), P.mean() / P.std() * np.sqrt(252), M.min(), (cum - cum.cummax()).min(), (M >= 50).mean(), [P[per == p].mean() for p in ('18-21', '22-23', '24-25', '2026')], T.pts.gt(0).mean()
for nm, X in (('every RTH flip', FL), ('Rule A entries', FL[FL.ruleA])):
    print(f'\n{nm}: pts/day  Sharpe  worst month  max DD  months>=50  win | per day 18-21 / 22-23 / 24-25 / 2026', flush=True)
    for lab, acc in (('vector exit only', False), ('+ acceptance exit', True)):
        d, sh, w, dd, m50, per, win = book(X, acc)
        print(f'  {lab:<20} {d:+6.2f} {sh:6.2f} {w:+8.0f} {dd:+7.0f} {m50:8.0%} {win:5.0%} | ' + ' / '.join(f'{v:+.2f}' for v in per), flush=True)
