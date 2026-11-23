"""Nov 23 2026 report 400. Small-cap (sub-$5, 108 LSE names) slot book with consolidation gate / priority.
Flips = q.trades on lse15s 15m bars (entry<$5 only), flags C_*/S_* from report 397 (amd397.parquet ds=B).
Sim: K=8 slots, equal weight 1/K, COST .004 round trip (report_sub5 convention), first-come order, skip if slots full.
Variants: BASE; GATE (any C_ flag); GATE_ALL (C_15m&C_1h); PRIORITY (when several flips at the same bar, consolidation-first, no gate); SEQ gate (any S_).
Split train 2019-21 | test 2022-26 (no 2018 small-cap data). Per-trade t-stats too. Pass: same sign both halves, |t|>=2."""
import os, sys, numpy as np, pandas as pd, multiprocessing as mp, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); import report_295_flip_quality as q
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; K = 8; COST = .004
def work(tk):
    try:
        d = pd.read_parquet(SP + f'lse15s/{tk}.parquet').rename(columns={'ts': 'datetime'}); d['datetime'] = pd.to_datetime(d.datetime).dt.tz_localize('UTC'); tmp = SP + f'tmp400_{tk}.parquet'; d.to_parquet(tmp); H = q.build(tmp); os.remove(tmp)
        T = pd.DataFrame(q.trades(H, tk)); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit); return T[['tk', 'entry', 'exit', 'ret']]
    except Exception as e: print('ERR', tk, str(e)[:80], flush=True); return None
def sim(d, t0, t1, prio=None):
    d = d[(d.entry >= t0) & (d.entry < t1)]; rows = d.assign(p=(prio(d) if prio else 0)).sort_values(['entry', 'p'], ascending=[True, False]).to_dict('records')
    eq = 1.0; op = []; cur = [1.0]; n = 0
    for r in rows:
        still = []
        for p in op:
            if p['exit'] <= r['entry']: eq *= 1 + (p["ret"] - COST) / K; cur.append(eq); n += 1
            else: still.append(p)
        op = still
        if len(op) < K and not any(p['tk'] == r['tk'] for p in op): op.append(r)
    for p in op: eq *= 1 + (p["ret"] - COST) / K; cur.append(eq); n += 1
    yrs = max((pd.Timestamp(t1) - pd.Timestamp(t0)).days, 1) / 365.25; c = np.array(cur); mdd = (c / np.maximum.accumulate(c) - 1).min()
    return dict(ret=eq - 1, cagr=eq ** (1 / yrs) - 1 if eq > 0 else -1, mdd=mdd, n=n)
if __name__ == '__main__':
    names = sorted(f[:-8] for f in os.listdir(SP + 'lse15s') if f.endswith('.parquet'))
    with mp.Pool(2) as P: out = [t for t in P.map(work, names) if t is not None]
    T = pd.concat(out, ignore_index=True); T.to_parquet(SP + 'tr400_B.parquet'); F = pd.read_parquet(SP + 'amd397.parquet'); F = F[F.ds == 'B']; F['entry'] = pd.to_datetime(F.entry)
    M = T.merge(F.drop(columns=['ret', 'ds']), on=['tk', 'entry'], how='inner'); M = M[(M.px < 5) & (M.entry >= '2019-01-01')]; print('trades', len(T), 'merged sub5', len(M), flush=True)
    M['Cany'] = M[['C_15m', 'C_1h', 'C_4h', 'C_D']].max(axis=1); M['Sany'] = M[['S_15m', 'S_1h', 'S_4h', 'S_D']].max(axis=1); M['Call'] = M.C_15m * M.C_1h
    M['csum'] = M[['C_15m', 'C_1h', 'C_4h', 'C_D']].sum(axis=1)
    halves = {'train 19-21': ('2019-01-01', '2022-01-01'), 'test 22-26': ('2022-01-01', '2026-12-31')}
    V = {'BASE': (M, None), 'GATE Cany': (M[M.Cany == 1], None), 'GATE C15m&C1h': (M[M.Call == 1], None), 'GATE Sany': (M[M.Sany == 1], None), 'ANTI-GATE (no cons)': (M[M.Cany == 0], None), 'PRIORITY cons-first': (M, lambda d: d.csum.values + 3 * d.Sany.values)}
    for nm, (d, pr) in V.items():
        s = ' | '.join(f"{h} CAGR {sim(d, a, b, pr)['cagr']*100:6.1f}% mdd {sim(d, a, b, pr)['mdd']*100:5.1f}% n={sim(d, a, b, pr)['n']}" for h, (a, b) in halves.items()); print(f'{nm:22s} {s}', flush=True)
    print('\nPer-trade mean net (bp) and t, by flag:')
    for h, (a, b) in halves.items():
        d = M[(M.entry >= a) & (M.entry < b)]; 
        for nm, m in [('all', d.ret.notna()), ('Cany', d.Cany == 1), ('no cons', d.Cany == 0), ('Sany', d.Sany == 1)]:
            x = (d[m].ret - COST) * 1e4; t = x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan; print(f'  {h} {nm:8s} n={len(x):6d} mean {x.mean():7.1f}bp t {t:5.2f}')
