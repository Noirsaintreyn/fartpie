"""Sub-$5 book: STACK the two filters that showed something (skip entries <=3 sessions before earnings; skip flip-bar close >1 ATR above session VWAP) and add up-volume/down-volume. UDV = share of the last 20 1h bars' volume (incl. flip bar) that traded on up bars (close>open); CL = flip-bar close location in its range. DECLARED: require UDV>=0.5 and CL>=0.5 (buyers dominate; Llorente et al. 2002 informed-volume continuation). Thresholds for DTE/VWAP were picked on this same data -> stacked numbers are in-sample; judge by H1/H2 halves and the random-skip p."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); E = pd.read_pickle(SC + 'earnmac_T2.pkl'); V = pd.read_pickle(SC + 'sub5_T.pkl')[['tk', 'et', 'vwx']]
T = E[E.sub5].merge(V, on=['tk', 'et'], how='left'); COST = .004; K = 8; udv = []; cl = []
for _, r in T.iterrows():
    Yc = M.closed_only(data[r.tk]); i = int(r.fi); a = Yc.iloc[max(0, i - 19):i + 1]; up = (a.close > a.open).values; v = a.volume.values.astype(float); udv.append(v[up].sum() / v.sum() if v.sum() > 0 else np.nan)
    b = Yc.iloc[i]; cl.append((b.close - b.low) / (b.high - b.low) if b.high > b.low else np.nan)
T['udv'] = udv; T['cl'] = cl; print('trades', len(T), 'vwx cov', T.vwx.notna().mean().round(2))
def sim(d, t0=pd.Timestamp('2023-12-01'), t1=pd.Timestamp('2027-01-01')):
    d = d[(d.et >= t0) & (d.et < t1)]; rows = sorted(d.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))); eq = 1.0; op = []; cv = [1.0]; n = 0; w = 0; hd = []
    def close(p):
        nonlocal eq, n, w
        eq += p['sz'] * (p['ret'] - COST); cv.append(eq); n += 1; w += p['ret'] - COST > 0; hd.append((p['xt'] - p['et']) / np.timedelta64(1, 'D'))
    for r in rows:
        still = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: close(p)
    cv = np.array(cv); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min(), n, w / max(n, 1), np.mean(hd) if hd else np.nan
H1, H2, H3 = (pd.Timestamp('2023-12-01'), pd.Timestamp('2025-04-01')), (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-01-01')), (pd.Timestamp('2026-01-01'), pd.Timestamp('2027-01-01')); ann = lambda r, y: ((1 + r) ** (1 / y) - 1) * 100 if r > -1 else -100
RULES = {'none (base)': lambda d: d.ret == d.ret, 'E: skip DTE<=3': lambda d: ~(d.dte <= 3), 'V: skip >1ATR above VWAP': lambda d: ~(d.vwx > 1), 'E+V': lambda d: ~(d.dte <= 3) & ~(d.vwx > 1), 'U: UDV>=0.5': lambda d: d.udv >= .5, 'CL>=0.5': lambda d: d.cl >= .5, 'U+CL': lambda d: (d.udv >= .5) & (d.cl >= .5), 'E+V+U': lambda d: ~(d.dte <= 3) & ~(d.vwx > 1) & (d.udv >= .5), 'E+V+U+CL': lambda d: ~(d.dte <= 3) & ~(d.vwx > 1) & (d.udv >= .5) & (d.cl >= .5)}
rng = np.random.default_rng(11)
for bn, bm in (('C vol>=1.5x + low-vol day', (T.vr >= 1.5) & T.lv), ('B vol>=1.5x', T.vr >= 1.5)):
    b = T[bm]; print(f'\n=== sub-$5 {bn}: {len(b)} flips'); print(f"{'rule':26s}{'kept':>5s} {'ann%':>6s} {'DD%':>5s} {'H1':>6s} {'H2':>6s} {'2026':>6s} {'win%':>5s} {'hold d':>6s} {'kept mean':>9s} {'skip mean':>9s} {'p(rand)':>8s}")
    for rn, f in RULES.items():
        kp = f(b).fillna(False).values; k_ = b[kp]; s_ = b[~kp]; r, dd, n, w, hd = sim(k_); rm = np.array([(b.ret.values[rng.permutation(len(b))[:len(k_)]] - COST).mean() for _ in range(2000)]) * 100; km = (k_.ret - COST).mean() * 100
        print(f"{rn:26s}{len(k_):5d} {ann(r, 2.94):+6.0f} {dd * 100:5.0f} {sim(k_, *H1)[0] * 100:+6.0f} {sim(k_, *H2)[0] * 100:+6.0f} {sim(k_, *H3)[0] * 100:+6.0f} {w * 100:5.0f} {hd:6.1f} {km:+8.2f}% {(s_.ret - COST).mean() * 100 if len(s_) else float('nan'):+8.2f}% {(rm >= km).mean():8.3f}")
b = T[(T.vr >= 1.5) & T.lv]; print('\nUDV buckets (C):', {str(k): (len(x), round((x.ret - COST).mean() * 100, 2)) for k, x in b.groupby(pd.cut(b.udv, [0, .3, .4, .5, .6, 1]))})
