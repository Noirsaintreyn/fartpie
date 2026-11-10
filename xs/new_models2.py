"""Nov 10 2026: SECOND BATCH OF ENGINES, CORRECTED ET CLOCK, same data / cost / pass rule as xs/new_models.py (0.5 pt, 1 NQ, PASS = pts/day > 0 with t >= 2 in 2014-21 AND 2022-Jan26). DECLARED BEFORE RESULTS.
FAMILY S  PRIOR-DAY EXTREME SWEEP REVERSAL: a 5m bar trades through the prior RTH high (low) and CLOSES back inside -> fade at the next open (short at PDH, long at PDL); stop 40, target 80, flat 15:55, one per side a day, 10:00-14:30.
FAMILY R  NARROW-RANGE BREAKOUT: prior-day RTH range in the bottom fifth of the last 20 -> a 5m close beyond the prior high (low) after 10:00 -> go with it at the next open; stop 40, flat 15:55, one a day.
FAMILY O  OPENING-RANGE BREAKOUT corrected clock: OR of 15 / 30 / 60 minutes, first 5m close beyond it before 11:30 -> go with it, stop = the other side of the range (cap 60), flat 12:00 or 15:55.
FAMILY K  TREND-DAY FOLLOW (noise-area width as the clock): after 10:30, when the day's range so far exceeds k x the typical range-so-far at that time (mean over the prior 14 sessions of the range to that bar) and the close is in the outer 20% of the day's range, go with it; stop 40, flat 15:55; k in {1.25, 1.5}.
Reported as before: n, pts/day by period with t, long / short, corr with S6."""
_n = open('xs/new_models.py').read(); exec(_n[:_n.index("R = {}")])
def pdhl(d):
    i = SI[d]
    if i == 0: return np.nan, np.nan
    p = ALLD[i - 1]; a, sy = A[p]
    if sy != A[d][1]: return np.nan, np.nan
    return np.nanmax(a[1]), np.nanmin(a[2])
PDR = pd.Series({d: (lambda r: r[0] - r[1])(pdhl(d)) for d in ALLD}); NR = (PDR <= PDR.shift(1).rolling(20).quantile(.2))
def exit_path(a, j0, dr, e, stop, target, endj=NB - 1):
    o, h, l, c = a[0], a[1], a[2], a[3]; worst = 0.0
    for k in range(j0, endj + 1):
        if np.isnan(c[k]): continue
        lo, hi = ((l[k] - e), (h[k] - e)) if dr > 0 else ((e - h[k]), (e - l[k])); worst = min(worst, lo)
        if lo <= -stop: return -stop - COST, worst - COST
        if target is not None and hi >= target: return target - COST, worst - COST
    k = endj
    while k > j0 and np.isnan(c[k]): k -= 1
    return (c[k] - e) * dr - COST, worst - COST
def sweep(d):
    a, _ = A[d]; o, h, l, c = a[0], a[1], a[2], a[3]; pdh, pdl = pdhl(d); rows = []; done = set()
    if not np.isfinite(pdh): return rows
    for j in range(pos_of[600], pos_of[870] + 1):
        if np.isnan(c[j]) or np.isnan(o[j + 1]): continue
        if h[j] > pdh and c[j] < pdh and -1 not in done: e = o[j + 1]; p, w = exit_path(a, j + 1, -1, e, 40.0, 80.0); rows.append((d, j + 1, -1, p, w)); done.add(-1)
        if l[j] < pdl and c[j] > pdl and 1 not in done: e = o[j + 1]; p, w = exit_path(a, j + 1, 1, e, 40.0, 80.0); rows.append((d, j + 1, 1, p, w)); done.add(1)
    return rows
def nrb(d):
    if not NR.get(d, False): return []
    a, _ = A[d]; o, c = a[0], a[3]; pdh, pdl = pdhl(d)
    if not np.isfinite(pdh): return []
    for j in range(pos_of[600], NB - 2):
        if np.isnan(c[j]) or np.isnan(o[j + 1]): continue
        dr = 1 if c[j] > pdh else (-1 if c[j] < pdl else 0)
        if dr: e = o[j + 1]; p, w = exit_path(a, j + 1, dr, e, 40.0, None); return [(d, j + 1, dr, p, w)]
    return []
def orb(d, mins, endm):
    a, _ = A[d]; o, h, l, c = a[0], a[1], a[2], a[3]; n = mins // 5; w = slice(0, n)
    if np.isnan(h[w]).any(): return []
    orh, orl = np.nanmax(h[w]), np.nanmin(l[w]); rng = orh - orl
    for j in range(n, pos_of[685] + 1):
        if np.isnan(c[j]) or np.isnan(o[j + 1]): continue
        dr = 1 if c[j] > orh else (-1 if c[j] < orl else 0)
        if dr:
            e = o[j + 1]; stop = min(max(rng, 10.0), 60.0); p, w2 = exit_path(a, j + 1, dr, e, stop, None, pos_of[endm] - 1 if endm < 955 else NB - 1); return [(d, j + 1, dr, p, w2)]
    return []
def trend(d, k):
    a, _ = A[d]; o, h, l, c = a[0], a[1], a[2], a[3]; i = SI[d]
    if i < 15: return []
    rs = np.array([np.nanmax(np.where(np.arange(NB) <= 0, -np.inf, A[ALLD[i - m]][0][1])) for m in range(1, 1)]) if False else None
    hs = np.fmax.accumulate(np.nan_to_num(h, nan=-np.inf)); ls = np.fmin.accumulate(np.nan_to_num(l, nan=np.inf)); rng = hs - ls
    for j in range(pos_of[630], pos_of[900]):
        if np.isnan(c[j]) or np.isnan(o[j + 1]): continue
        typ = []
        for m in range(1, 15):
            q = A[ALLD[i - m]][0]; hh = np.fmax.accumulate(np.nan_to_num(q[1], nan=-np.inf)); ll = np.fmin.accumulate(np.nan_to_num(q[2], nan=np.inf)); typ.append(hh[j] - ll[j])
        if rng[j] > k * np.mean(typ):
            pos = (c[j] - ls[j]) / max(rng[j], 1e-9)
            dr = 1 if pos >= .8 else (-1 if pos <= .2 else 0)
            if dr: e = o[j + 1]; p, w = exit_path(a, j + 1, dr, e, 40.0, None); return [(d, j + 1, dr, p, w)]
    return []
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); UW = {'RA': 1.0, 'M1': .3, 'R1': .5, 'ON+': .4, 'ON-': .1, 'RULES': 1.0, 'V15a': .3}; base = pd.Series(0.0, index=DAYS)
for (t, u, y), n in zip(OUT['V2'], NAMES):
    t = t[t.sz > 0]; dd = pd.to_datetime(t.day).dt.normalize(); base = base.add((t.pts * t.sz * UW[n]).groupby(dd.values).sum().reindex(DAYS).fillna(0.0), fill_value=0)
R = {}
print('=== FAMILY S: prior-day extreme sweep reversal ==='); R['S'] = stats([r for d in DAYS for r in sweep(d)], 'S sweep PDH/PDL fade', base)
print('\n=== FAMILY R: narrow-range breakout ==='); R['R'] = stats([r for d in DAYS for r in nrb(d)], 'R NR (bottom fifth) breakout', base)
print('\n=== FAMILY O: opening-range breakout ===')
for mins in (15, 30, 60):
    for endm, nm in ((720, '12:00'), (955, '15:55')): R[f'O{mins}{endm}'] = stats([r for d in DAYS for r in orb(d, mins, endm)], f'O OR{mins} flat {nm}', base)
print('\n=== FAMILY K: trend-day follow ===')
for k in (1.25, 1.5): R[f'K{k}'] = stats([r for d in DAYS for r in trend(d, k)], f'K range-so-far > {k} x typical', base)
pickle.dump(R, open(f'{SP}/new_models2.pkl', 'wb'))
