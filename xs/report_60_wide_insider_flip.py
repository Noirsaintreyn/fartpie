"""Nov 14 2026 WIDE UNIVERSE: DAILY VECTOR FLIP + >=2 INSIDER BUYERS + FLIP-BAR VOLUME (declared before results; Rey's rule). Universe: ~1,700 mid/small names (close_px_mid + midsmall_universe.txt), daily OHLCV 2018-> via yfinance. SURVIVORSHIP BIAS: only names still listed today are included, which flatters long results in every variant equally, so read the variant-vs-base differences, not the levels. Long flips (vector bull flip -> enter next open; exit open after the bear flip). INS = >=1 open-market purchase (SEC Form 4 code P) by an officer/director filed in the 30 calendar days up to the flip bar; CLUSTER = >=2 distinct insiders; BV = flip-bar volume / prior-20-day mean >= 1.5. Filters: price >= $1, median 20d dollar volume >= $500k. Insider data covers filings through 2026-03-31, so all variants are compared on flips through then. STRICT SPLIT: 2019-21 vs 2022-26Q1. Costs: 30bp round trip (60bp stress). Variants: A all | B BV | C INS | D CLUSTER | E CLUSTER+BV (the rule). Stats: n, mean net %, median, win%, share of trades that are big (>=+15% net), avg hold; month-clustered t of variant minus complement (rest of flips)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl'); I = pd.read_pickle(SC + 'insider_P_hist.pkl'); I = I[I.fd.notna()].copy(); I['sym'] = I.ISSUERTRADINGSYMBOL.str.upper().str.strip(); Ig = {k: g.sort_values('fd') for k, g in I.groupby('sym')}
print('tickers', len(D), 'insider symbols', len(Ig), 'P rows', len(I), flush=True); rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; n = len(B)
    if n < 400: continue
    o = B.open.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; reg = run_state_machine(B.assign(datetime=B.index))[0]
    dv = pd.Series(c * v).rolling(20).median().shift(1).values; vm = pd.Series(v).rolling(20).mean().shift(1).values; ig = Ig.get(tk.upper()); fd = ig.fd.values if ig is not None else None; ow = ig.RPTOWNERCIK.values if ig is not None else None
    ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]
    for a in range(len(ch) - 1):
        i, j = ch[a], ch[a + 1]
        if reg[i] != 1 or j + 1 >= n or i + 1 >= n: continue
        if c[i] < 1 or np.isnan(dv[i]) or dv[i] < 5e5 or vm[i] <= 0: continue
        t0 = idx[i].to_datetime64(); ni = 0
        if fd is not None:
            m = (fd <= t0) & (fd > t0 - np.timedelta64(30, 'D'))
            if m.any(): ni = len(set(ow[m]))
        rows.append(dict(tk=tk, t=idx[i + 1], ret=o[j + 1] / o[i + 1] - 1, hold=j - i, BV=v[i] / vm[i], NI=ni))
P = pd.DataFrame(rows).sort_values('t').reset_index(drop=True); P = P[P.t < '2026-04-01']; print('long flips', len(P), 'names', P.tk.nunique(), 'with >=1 insider', (P.NI >= 1).sum(), 'cluster>=2', (P.NI >= 2).sum(), flush=True)
V = {'A all flips': np.ones(len(P), bool), 'B BV>=1.5': (P.BV >= 1.5).values, 'C INS>=1': (P.NI >= 1).values, 'D CLUSTER>=2': (P.NI >= 2).values, 'E CLUSTER+BV (rule)': ((P.NI >= 2) & (P.BV >= 1.5)).values, 'F INS>=1 + BV': ((P.NI >= 1) & (P.BV >= 1.5)).values}
ERAS = (('2019-21', '2019-01-01', '2022-01-01'), ('2022-26Q1', '2022-01-01', '2026-04-01'))
def mclust(S, mask):
    s = S[mask]; r = S[~mask]; a = s.groupby(s.t.dt.to_period('M')).ret_n.mean(); b = r.groupby(r.t.dt.to_period('M')).ret_n.mean(); dm = (a - b).dropna()
    return dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan
for cost, tag in ((0.003, '30bp'), (0.006, '60bp stress')):
    P['ret_n'] = P.ret - cost; print(f'\n##### cost {tag}')
    for en, a, b in ERAS + (('all', '2019-01-01', '2026-04-01'),):
        S = P[(P.t >= a) & (P.t < b)].reset_index(drop=True); Vs = {k: m[(P.t >= a).values & (P.t < b).values] for k, m in V.items()}; print(f'\n {en}: n={len(S)} | variant | n | mean net % | median % | win % | big(>=15%) % | avg hold d | clustered t vs rest')
        for k, m in Vs.items():
            s = S[m]
            if len(s) < 5: print(f'   {k:22s} n {len(s)}'); continue
            print(f'   {k:22s} n {len(s):6d} | {s.ret_n.mean()*100:+6.2f} | {s.ret_n.median()*100:+5.2f} | {(s.ret_n>0).mean()*100:4.1f} | {(s.ret_n>=0.15).mean()*100:4.1f} | {s.hold.mean():4.1f} | t {mclust(S, m):+.1f}')
big = P.ret >= P.ret.quantile(0.9); e = V['E CLUSTER+BV (rule)']; print(f'\nTAIL CAPTURE: top-decile flips (ret >= {P.ret.quantile(0.9)*100:.1f}%): rule E contains {(big & e).sum()} of {big.sum()} ({(big & e).sum()/big.sum()*100:.1f}%) while being {e.mean()*100:.2f}% of flips; share of rule-E flips that are top-decile {(big[e]).mean()*100:.1f}% (base 10%)')
print('signals per year (rule E):', P[e].groupby(P[e].t.dt.year).size().to_dict())
P.to_pickle(SC + 'wide_flips_ins.pkl')
