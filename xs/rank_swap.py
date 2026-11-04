"""Nov 4 2026: SWAP rule - a better-ranked flip takes a full book's slot. DECLARED BEFORE RESULTS. Same budget (6 x 1/6), 8% breaker as live.
When the 6 slots are full and a signal arrives: eject one open position (closed at the bar open of the swap time, 2 bp cost) and admit the new one.
 A125  eject the open position with the LOWEST options ratio if new ratio >= 1.25 x its ratio       A2  same with 2.0 x
 B     eject the open position with the WORST unrealized return if that is <= 0 (new signal needs ratio >= its own median of open names)   B0  same with no ratio condition
 AB    eject worst unrealized (<= 0) first, else lowest ratio at 1.25 x
Reports 40 matched runs vs live both periods + runner accounting (>=15% trades: taken by swap rule that live missed / ejected early)."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
PX = {}
for tk in XL.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_values('dt').drop_duplicates('dt')
    PX[tk] = (H.dt.values, H.open.values)
def px_at(tk, e):
    t, o = PX[tk]; i = np.searchsorted(t, e, side='right') - 1; return o[i] if i >= 0 else np.nan
Z = XL.copy(); Z['px0'] = [px_at(a, b) for a, b in zip(Z.tk, Z.entry.values)]
def sims(E, Xx, R, K, TK, PX0, mode, n=6):
    o = np.lexsort((TK, -K, E)); op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; taken = []; ejected = []
    for ii in o:
        e, x, r = E[ii], Xx[ii], R[ii]
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q['x'] > e]
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): continue
        if len(op) >= n:
            if mode is None: continue
            for q in op: q['u'] = px_at(q['tk'], e) / q['p0'] - 1
            v = None; kk = K[ii]
            wu = min(op, key=lambda q: q['u']); wr = min(op, key=lambda q: q['k'])
            if mode in ('B', 'B0', 'AB') and wu['u'] <= 0 and (mode == 'B0' or mode == 'AB' or kk >= np.median([q['k'] for q in op])): v = wu
            if v is None and mode in ('A125', 'AB') and kk >= 1.25 * wr['k']: v = wr
            if v is None and mode == 'A2' and kk >= 2.0 * wr['k']: v = wr
            if v is None: continue
            op.remove(v); pend.append((e, v['u'] - 2e-4 if False else (px_at(v['tk'], e) / v['p0'] - 1 - 2e-4))); pend = [p for p in pend if p is not None]
            ejected.append(v['r'])
            pend = [p for p in pend if not (p[0] == v['x'] and p[1] == v['r'])]
        op.append(dict(x=x, k=K[ii], tk=TK[ii], p0=PX0[ii], r=r)); pend.append((x, r)); taken.append(r)
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(), taken, ejected
def evs(Y, mode):
    HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); K = Y.ratio.values.astype(float); res = []; tk_r = []; ej_r = []
    for i in range(1, 41):
        dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl
        s, tkn, ej = sims(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], K[m], Y.tk.values[m], Y.px0.values[m], mode)
        res.append(stat2(s)); tk_r.append(sum(1 for r in tkn if r >= .15)); ej_r.append((len(ej), sum(1 for r in ej if r >= .15), np.mean(ej) if ej else np.nan))
    return np.array(res), np.mean(tk_r), np.mean([a for a, _, _ in ej_r]), np.mean([b for _, b, _ in ej_r]), np.nanmean([c for _, _, c in ej_r])
B0r, tr0, _, _, _ = evs(Z, None); print('40 matched runs, 2019-21 / 2022-26'); line('L live', B0r); print(f'     runners (>=15%) taken per run: {tr0:.0f}')
for md in ('A125', 'A2', 'B', 'B0', 'AB'):
    r, tr, ne, nr, ma = evs(Z, md); line(f'swap {md}', r, B0r); print(f'     runners taken {tr:.0f} (live {tr0:.0f}); ejections/run {ne:.0f}, of which would-be runners {nr:.1f}, ejected trades avg original ret {ma*100:+.2f}%', flush=True)
