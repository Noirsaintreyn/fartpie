"""Nov 12 2026: vector on $1-2 names by timeframe (1h live | 4h | daily), same 753-name 1h cache, same window. DECLARED BEFORE RESULTS.
Rule: entry = next bar open after a bull flip, exit = next bar open after the bear flip; price (flip-bar close) in [1,2); sets: ALL flips, B = flip-bar volume >= 1.5x prior-20 mean.
Cost 40 bp round trip. Book sim: 8 slots, equal share of realised equity, higher volume first, one position per ticker. Train < 2025-07-01 | test after (no parameters fitted)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); COST = .004
def tfbars(Y, tf):
    if tf == '1h': return Y
    day = Y.index.normalize()
    if tf == '4h': g = [day, (Y.index.hour >= 13).astype(int)]
    else: g = [day]
    G = Y.groupby(g); o = G.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum'))
    idx = G.apply(lambda x: x.index[0]); o.index = pd.DatetimeIndex(idx.values); return o.sort_index()
def trades(tk, tf):
    Y = tfbars(data[tk].dropna(subset=['open', 'close']), tf)
    if len(Y) < 120: return []
    reg = run_state_machine(Y.assign(datetime=Y.index))[0]; v = Y.volume.values.astype(float); vrel = v / pd.Series(v).rolling(20).mean().shift().values; c = Y.close.values; o = Y.open.values; out = []; ent = None
    for i in range(1, len(Y) - 1):
        if reg[i] == reg[i - 1] or reg[i] == 0: continue
        if reg[i] == 1 and ent is None and 1 <= c[i] < 2: ent = (i + 1, vrel[i])
        elif reg[i] == -1 and ent is not None: out.append(dict(tk=tk, et=Y.index[ent[0]], xt=Y.index[i + 1], epx=o[ent[0]], xpx=o[i + 1], vr=ent[1], ret=o[i + 1] / o[ent[0]] - 1)); ent = None
        elif reg[i] == -1: ent = None
    return out
def book(T, K=8):
    eq = 1.0; op = []; curve = []; n = 0
    for r in sorted(T.to_dict('records'), key=lambda r: (r['et'], -r['vr'])):
        still = []
        for p in op:
            if p['xt'] <= r['et']: eq += p['sz'] * (p['ret'] - COST); curve.append((p['xt'], eq))
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r); n += 1
    for p in op: eq += p['sz'] * (p['ret'] - COST); curve.append((p['xt'], eq))
    s = pd.Series(dict(curve)).sort_index(); return s, n
def rep(nm, T):
    if not len(T): print(f'  {nm:<22} no trades'); return
    r = T.ret - COST; a, b = r[T.et < '2025-07-01'], r[T.et >= '2025-07-01']; s, n = book(T); yrs = (s.index[-1] - T.et.min()).days / 365.25; cagr = s.iloc[-1] ** (1 / max(yrs, .5)) - 1; dd = (s / s.cummax() - 1).min()
    print(f'  {nm:<22} {len(T):>5} tr {len(T)/yrs:>5.0f}/yr | mean {r.mean()*100:+.2f}% med {r.median()*100:+.1f}% win {(r>0).mean():.0%} | train {a.mean()*100:+.2f}% ({len(a)}) test {b.mean()*100:+.2f}% ({len(b)}) | book {cagr:+.0%}/yr DD {dd:.0%}', flush=True)
for tf in ('1h', '4h', '1d'):
    T = pd.DataFrame([x for tk in data for x in trades(tk, tf)]); T = T[T.et >= '2024-01-01'].reset_index(drop=True); print(f'== {tf}: entries since 2024-01, price $1-2')
    rep('all flips', T); rep('volume >= 1.5x (B)', T[T.vr >= 1.5]); rep('volume >= 2.5x', T[T.vr >= 2.5])
