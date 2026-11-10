"""Nov 10 2026: SAME ENGINES ON FX AND GOLD, HOURLY BARS (yfinance 1h, only ~2.8 years: 2024-01 .. 2026-11, so ONE period, split into halves). DECLARED BEFORE RESULTS.
Symbols EURUSD, GBPUSD, USDJPY, XAUUSD (GC=F futures, converted to UTC). Sessions: LDN 07:00-16:00 UTC, NY 12:00-21:00 UTC. Cost per round trip (fraction of price): FX 0.010% (~1 pip EURUSD), gold 0.030%; sensitivity x2.
Families: N noise-area momentum (k 1, hourly decisions from bar 2, stop .5 sigma_end), long/short split; O ORB first hour -> flat session end.
Only about 650 sessions per series, so |t| >= 2 in each half is the bar and this is a screen, not a verdict."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import crypto_engines as ce
ce.BAR = 60
for f, nm, cost in (('FX_EURUSDX', 'EURUSD', .0001), ('FX_GBPUSDX', 'GBPUSD', .0001), ('FX_USDJPYX', 'USDJPY', .0001), ('FX_GCF', 'XAUUSD', .0003)):
    d = pd.read_parquet(f'/tmp/claude-0/crypto/{f}_1h.parquet'); d.columns = [c.lower() for c in d.columns]; d.index = pd.DatetimeIndex(d.index).tz_convert('UTC').tz_localize(None); d = d[~d.index.duplicated()]
    for clk, (a0, a1) in (('LDN 07-16', (420, 960)), ('NY 12-21', (720, 1260))):
        S, n = ce.prep(d, a0, a1)
        if len(S) < 100: print(nm, clk, 'too few sessions', len(S)); continue
        SIG = ce.sig_table(S, n); days = pd.DatetimeIndex(sorted(S)); days = days[days >= '2024-03-01']; mid = days[len(days) // 2]
        print(f'\n===== {nm} {clk} UTC: {len(days)} sessions, {n} bars =====')
        for c_ in (cost, 2 * cost):
            for lab, kw in (('N base', dict(fam='N', step=1, start=2)), ('N long only', dict(fam='N', step=1, start=2, side=1)), ('N short only', dict(fam='N', step=1, start=2, side=-1)), ('O ORB first hour', dict(fam='O', orm=1))):
                T = ce.run(S, n, SIG, cost=c_, **kw); s = T.groupby('day').r.sum().reindex(days).fillna(0) * 100; a, b = s[s.index < mid], s[s.index >= mid]
                tt = lambda x: x.mean() / (x.std() / np.sqrt(len(x))) if x.std() > 0 else 0
                ok = a.mean() > 0 and b.mean() > 0 and tt(a) >= 2 and tt(b) >= 2
                print(f'  cost {c_ * 100:.3f}% {lab:<18} n {len(T):4d} | %/day first half {a.mean():+.4f} (t {tt(a):+5.2f}) | second half {b.mean():+.4f} (t {tt(b):+5.2f}){"   <== PASS" if ok else ""}')
