"""Nov 4 2026: quantify BAD open trades, then swap only those. DECLARED BEFORE RESULTS.
PART 1  at the close of session d after entry (d = 1, 2, 3, 5, 10) for trades still open: unrealized u = close/entry - 1. Table by u bucket: n, the REMAINING return (exit vs that close), share of remaining < 0,
        both periods. 'Bad' = bucket whose remaining avg is < 0 in both periods.
PART 2  swap (as xs/rank_swap.py) but ONLY when all 6 slots are full and a signal is waiting: eject the open position with the worst unrealized if u <= -a and age >= g days.
        grid a in {1%, 2%, 3%} x g in {1, 3} sessions-days (6 cells). 40 matched runs vs live, both periods; pass = >= 70% beat live in both."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/rank_swap.py').read(); exec(src[:src.index("def sims(")])
# PART 1
rows = []
for tk, G in Z.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_values('dt').drop_duplicates('dt')
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; R = H[(m >= 570) & (m < 960)]; dc = R.groupby(R.dt.dt.normalize()).close.last(); days = dc.index.values
    for i, r in G.iterrows():
        k0 = np.searchsorted(days, r.entry.normalize().to_datetime64(), side='left')
        for d in (1, 2, 3, 5, 10):
            k = k0 + d
            if k >= len(days): break
            if np.datetime64(days[k]) + np.timedelta64(16, 'h') >= np.datetime64(r.exit): break   # exited before that close
            c = dc.iloc[k]; u = c / r.px0 - 1; rows.append(dict(d=d, u=u, rem=(1 + r.ret) / (1 + u) - 1, per='19-21' if r.entry.year <= 2021 else '22-26', ratio=r.ratio))
P = pd.DataFrame(rows); P['ub'] = pd.cut(P.u, [-9, -.05, -.03, -.02, -.01, 0, .02, .05, 9], labels=['<-5%', '-5..-3', '-3..-2', '-2..-1', '-1..0', '0..2', '2..5', '>5%'])
print('PART 1: remaining return (close of day d -> exit) by unrealized bucket   n | avg remaining | % below 0   19-21 || 22-26   (* avg remaining < 0 in both)')
for d, g in P.groupby('d'):
    print(f' day {d}')
    for b, z in g.groupby('ub', observed=True):
        a, c2 = z[z.per == '19-21'], z[z.per == '22-26']
        star = '*' if len(a) >= 50 and len(c2) >= 50 and a.rem.mean() < 0 and c2.rem.mean() < 0 else ''
        print(f'   {str(b):<8} n{len(a):>5} {a.rem.mean()*100:+.2f}% {a.rem.lt(0).mean():.0%} || n{len(c2):>5} {c2.rem.mean()*100:+.2f}% {c2.rem.lt(0).mean():.0%} {star}')
def sims2(E, Xx, R, K, TK, PX0, a, g, n=6):
    o = np.lexsort((TK, -K, E)); op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; ej = []
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
            cand = []
            for q in op:
                q['u'] = px_at(q['tk'], e) / q['p0'] - 1
                if q['u'] <= -a and (e - q['e']) / np.timedelta64(1, 'D') >= g: cand.append(q)
            if not cand: continue
            v = min(cand, key=lambda q: q['u']); op.remove(v)
            pend = [p for p in pend if not (p[0] == v['x'] and p[1] == v['r'])]; pend.append((e, v['u'] - 2e-4)); ej.append(v['r'])
        op.append(dict(x=x, e=e, k=K[ii], tk=TK[ii], p0=PX0[ii], r=r)); pend.append((x, r))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(), ej
def evs2(Y, a, g):
    HSH = np.array([zlib.crc32(f'{p}{q}'.encode()) for p, q in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); K = Y.ratio.values.astype(float); res = []; ne = []; nr = []; mr = []
    for i in range(1, 41):
        dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl
        s, ej = sims2(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], K[m], Y.tk.values[m], Y.px0.values[m], a, g)
        res.append(stat2(s)); ne.append(len(ej)); nr.append(sum(1 for r in ej if r >= .15)); mr.append(np.mean(ej) if ej else np.nan)
    return np.array(res), np.mean(ne), np.mean(nr), np.nanmean(mr)
B0r = evs2(Z, 9, 1e9)[0]; print('\nPART 2: swap only bad positions, 40 matched runs, 2019-21 / 2022-26'); line('L live', B0r)
for a in (.01, .02, .03):
    for g in (1, 3):
        r, ne, nr, mr = evs2(Z, a, g); line(f'swap worst if u <= -{a:.0%} & age >= {g}d', r, B0r); print(f'     ejections/run {ne:.0f}, would-be runners {nr:.1f}, ejected trades avg original ret {mr*100:+.2f}%', flush=True)
