"""Nov 3 2026: can the new exit (1h-confirmed 15m stop) plus the earlier filter findings beat the current model?
DECLARED BEFORE RESULTS. Exit rule E in {L live, A5 (1h-confirm + 5% hard stop), Ainf (1h-confirm, no hard stop)}  x  overlay O in
 none | M1 low-vol flips last in same-bar collisions | M3 size tilt 1.25/1/0.75 (high/mid/low 20d realized vol, within-year terciles: low = bottom fifth, high = top third)
 | D drop low-vol flips (6 slots)  | M3+M1.   15 cells, every one scored against the CURRENT model (L, none), 40 matched runs, both periods.
Pass: >= 70% of runs beat current in BOTH periods. 15 cells -> expect ~1 chance pass at looser thresholds; all periods are used for selection => paper-track only."""
import glob
src = open('/home/claude/fartpie/xs/stop_1h_ev.py').read().split("B0 = ev(XL)")[0]; exec(src)
RV = {}
for tk in sorted(V.tk.unique()):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'close']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    RV[tk] = H[(m >= 570) & (m < 960)].groupby(H.dt.dt.normalize()).close.last().pct_change().rolling(20).std().shift(1)
def prep(nm):
    Xv = build(nm); Xv['rv'] = [RV[t].asof(d) if t in RV else np.nan for t, d in zip(Xv.tk, Xv.d)]; r = Xv.groupby(Xv.entry.dt.year).rv.rank(pct=True)
    Xv['low'] = (r <= .2).fillna(False); Xv['high'] = (r > 2 / 3).fillna(False); return Xv
def simx(E, Xx, R, RA, TK, low, high, mode):
    key = -RA + (1e6 * low if 'M1' in mode else 0); o = np.lexsort((TK, key, E)); pend = {}; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for k in o:
        e = E[k]
        for kk in [kk for kk, (xt, _, _) in pend.items() if xt <= e]:
            xt, rr, w = pend.pop(kk); real += np.log1p(rr * w); hist.append((xt, real)); win.append((xt, real))
        gross = sum(v[2] for v in pend.values())
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): continue
        if len(pend) >= 6: continue
        w = (1.25 if high[k] else .75 if low[k] else 1) / 6 if 'M3' in mode else 1 / 6
        w = min(w, 1 - gross)
        if w < .02: continue
        pend[k] = (Xx[k], R[k], w)
    for xt, rr, w in sorted(pend.values(), key=lambda z: z[0]): real += np.log1p(rr * w); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
def evc(Xv, mode):
    HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Xv.tk, Xv.entry.astype('int64'))], dtype=np.uint64); res = []
    for i in range(1, 41):
        dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl
        if mode == 'D': m = m & ~Xv.low.values
        d = Xv[m]; res.append(stat2(simx(d.entry.values, d.exit.values, d.ret.values, d.ratio.values.astype(float), d.tk.values, d.low.values, d.high.values, mode)))
    return np.array(res)
PX = {nm: prep(nm) for nm in ('L', 'A5', 'Ainf')}; B0 = evc(PX['L'], 'none'); print('40 matched runs, 2019-21 / 2022-26'); line('CURRENT (L, none)', B0)
for nm in ('L', 'A5', 'Ainf'):
    for ov in ('none', 'M1', 'M3', 'D', 'M3+M1'):
        if nm == 'L' and ov == 'none': continue
        line(f'{nm} + {ov}', evc(PX[nm], ov), B0)
