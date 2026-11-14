"""Nov 14 2026 PEAD + FLIPS (declared before results). Rey: post-earnings-announcement drift as the 'why participants enter' layer, vector flip as the entry. Mechanism: earnings news is under-reacted to; informed/slow money keeps buying after a positive surprise (and selling after a negative one) for weeks, so a flip in the SAME direction as the earnings reaction should continue; a flip AGAINST it should fail. Surprise proxy (no consensus data in repo): earnings-announcement return EAR = close of the reaction day / prior close - 1 (reaction day = filing day if 8-K item 2.02 accepted before 16:00 ET, else next trading day), z = EAR / std of the 20 prior daily returns. Flip is eligible once the reaction day has closed. ALIGN = sign(EAR) * flip direction (+1 aligned, -1 against). Lag buckets (trading days since reaction): 1-5, 6-20, 21-60; BIG = |z| >= 2. Compare aligned vs against; large caps strict eras 2019-21 vs 2022+, small/mid (374 names, 8-K pulled from SEC EDGAR, 1h, Apr 2025-) halves. Outcome raw trade return % and ATR-scaled retn; PASS = aligned > against in BOTH eras and pooled t >= 2."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
def events(K, dc):
    out = {}
    for tk, x in K[K['items'].astype(str).str.contains('2.02')].groupby('ticker'):
        c = dc.get(tk)
        if c is None or len(c) < 100: continue
        ix = c.index; r = c.pct_change(); sd = r.rolling(20).std().shift(1); ev = []
        for a in pd.to_datetime(x.accept, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None):
            d = a.normalize(); k = ix.searchsorted(d)
            if k >= len(ix): continue
            if a.hour >= 16 and ix[k] == d: k += 1
            elif ix[k] != d: pass
            if k < 21 or k >= len(ix): continue
            ear = c.iloc[k] / c.iloc[k - 1] - 1; z = ear / sd.iloc[k] if sd.iloc[k] > 0 else np.nan; ev.append((ix[k], ear, z))
        if ev: out[tk] = sorted(set(ev))
    return out
def tag(P, ev, dc):
    P = P.copy(); P['lag'] = np.nan; P['ear'] = np.nan; P['z'] = np.nan
    for tk, ix in P.groupby('tk').groups.items():
        e = ev.get(tk); c = dc.get(tk)
        if not e: continue
        days = np.array([x[0] for x in e], dtype='datetime64[ns]'); cd = c.index.values; t = P.loc[ix, 't'].dt.normalize().values; pos = np.searchsorted(days, t, side='left') - 1; ok = pos >= 0
        pe = np.where(ok, days[np.clip(pos, 0, None)], np.datetime64('NaT')); lag = np.searchsorted(cd, t) - np.searchsorted(cd, pe); lag = np.where(ok, lag, np.nan)
        P.loc[ix, 'lag'] = lag; P.loc[ix, 'ear'] = np.where(ok, np.array([e[p][1] for p in np.clip(pos, 0, None)]), np.nan); P.loc[ix, 'z'] = np.where(ok, np.array([e[p][2] for p in np.clip(pos, 0, None)]), np.nan)
    P['ALIGN'] = np.sign(P.ear) * P.d; return P
def show(P, eras, name):
    print(f'\n=== {name}: n flips {len(P)}, with an earnings reaction on file in prior 60d: {int((P.lag.between(1, 60)).sum())}')
    for lb, (lo, hi) in (('lag 1-5', (1, 5)), ('lag 6-20', (6, 20)), ('lag 21-60', (21, 60))):
        for sz, fz in (('all', lambda z: np.ones(len(z), bool)), ('BIG |z|>=2', lambda z: np.abs(z) >= 2)):
            for d, dn in ((0, 'both dirs'), (1, 'LONG'), (-1, 'SHORT')):
                line = f' {lb:9s} {sz:10s} {dn:9s}'
                for en, (a, b) in eras:
                    s = P[(P.t >= a) & (P.t < b) & P.lag.between(lo, hi) & fz(P.z.fillna(0).values)]
                    if d: s = s[s.d == d]
                    al = s[s.ALIGN > 0]; ag = s[s.ALIGN < 0]
                    line += f' | {en}: aligned {al.ret.mean()*100:+.2f}% (n {len(al)}) vs against {ag.ret.mean()*100:+.2f}% (n {len(ag)}) t {tt(al.ret, ag.ret):+.1f}'
                print(line)
    nr = P[~P.lag.between(1, 60)]; print(f' no recent earnings (baseline): mean ret {nr.ret.mean()*100:+.2f}% n {len(nr)}')
# ---- large caps
K = pd.read_csv('xs/ext/8k_items.csv'); K['items'] = K['items'].astype(str)
P = pd.read_pickle(SC + 'part_large.pkl'); dc = {}
for tk in P.tk.unique():
    f = glob.glob(f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'close']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)]; dc[tk] = H.groupby(H.dt.dt.normalize()).close.last()
ev = events(K, dc); print('large: tickers with events', len(ev), 'events', sum(len(v) for v in ev.values()))
show(tag(P, ev, dc), (('19-21', ('2019', '2022')), ('22+', ('2022', '2027')), ('all', ('2019', '2027'))), 'LARGE CAPS 15m')
def tag_pre(P, ev, dc):
    P = P.copy(); P['dto'] = np.nan; P['near'] = np.nan
    for tk, ix in P.groupby('tk').groups.items():
        e = ev.get(tk); c = dc.get(tk)
        if not e: continue
        days = np.array([x[0] for x in e], dtype='datetime64[ns]'); cd = c.index.values; t = P.loc[ix, 't'].dt.normalize().values; pos = np.searchsorted(days, t, side='right'); ok = pos < len(days)
        nx = np.where(ok, days[np.clip(pos, 0, len(days) - 1)], np.datetime64('NaT')); P.loc[ix, 'dto'] = np.where(ok, np.searchsorted(cd, nx) - np.searchsorted(cd, t), np.nan); P.loc[ix, 'near'] = np.where(ok, np.array([e[p][1] for p in np.clip(pos, 0, len(days) - 1)]), np.nan)
    P['LEAN'] = np.sign(P.near) * P.d; return P
def show_pre(P, eras, name):
    print(f'\n=== PRE-EARNINGS flips, {name} (entry before the reaction day; LEAN=+1 means flip direction matches the sign of the EAR that followed: 50% = no information; ret includes the earnings gap if held through)')
    for lb, (lo, hi) in (('1-3d before', (1, 3)), ('4-10d before', (4, 10)), ('11-30d before', (11, 30))):
        for d, dn in ((1, 'LONG'), (-1, 'SHORT')):
            line = f' {lb:13s} {dn:5s}'
            for en, (a, b) in eras:
                s = P[(P.t >= a) & (P.t < b) & P.dto.between(lo, hi) & (P.d == d)]; r = P[(P.t >= a) & (P.t < b) & ~P.dto.between(1, 30) & (P.d == d)]
                line += f' | {en}: n {len(s)} lean+ {(s.LEAN > 0).mean()*100:4.1f}% ret {s.ret.mean()*100:+.2f}% (lean+ {s[s.LEAN>0].ret.mean()*100:+.2f} lean- {s[s.LEAN<0].ret.mean()*100:+.2f}) vs no-earnings-soon {r.ret.mean()*100:+.2f}% t {tt(s.ret, r.ret):+.1f}'
            print(line)
show_pre(tag_pre(P, ev, dc), (('19-21', ('2019', '2022')), ('22+', ('2022', '2027')), ('all', ('2019', '2027'))), 'LARGE CAPS 15m')
# ---- small/mid
K2 = pd.read_csv('/home/claude/fartpie/xs/ext/8k_items_midsmall.csv'); K2['items'] = K2['items'].astype(str); P2 = pd.read_pickle(SC + 'part_small.pkl'); D = pd.read_pickle('live/midsmall_bars.pkl')['d']; dc2 = {tk: B.close.groupby(B.index.normalize()).last() for tk, B in D.items()}
ev2 = events(K2, dc2); print('\nsmall: tickers with events', len(ev2), 'events', sum(len(v) for v in ev2.values()))
show(tag(P2, ev2, dc2), (('h1', ('2025', '2026-03-01')), ('h2', ('2026-03-01', '2027')), ('all', ('2025', '2027'))), 'SMALL/MID 1h')

show_pre(tag_pre(P2, ev2, dc2), (('h1', ('2025', '2026-03-01')), ('h2', ('2026-03-01', '2027')), ('all', ('2025', '2027'))), 'SMALL/MID 1h')
