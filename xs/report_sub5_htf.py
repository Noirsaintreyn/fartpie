"""SUB-$5 book: higher-timeframe 'extended' skip filters (weekly/monthly anchored VWAP, rolling 5d/20d VWAP, daily EMA10/20 distance, daily Bollinger %B) vs the intraday-VWAP skip. Base sets B (vol>=1.5x) and C (B + low-vol day). Same sim as report_sub5.py (8 slots, 40bp, higher flip volume first). All HTF levels use only bars/days known at the flip-bar close (daily stats lagged one day). Also a random-skip null: skipping the same number of trades at random from the base set -> p that kept-mean >= observed."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); T = pd.read_pickle(SC + 'sub5_T.pkl'); COST = .004; K = 8
def atr14(h, l, c): pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); return pd.Series(tr).rolling(14).mean().values
FE = {}
for tk, g in T.groupby('tk'):
    Yc = M.closed_only(data[tk]); o, h, l, c, v = (Yc[k].values.astype(float) for k in ('open', 'high', 'low', 'close', 'volume')); at = atr14(h, l, c); idx = Yc.index; day = idx.normalize(); tp = (h + l + c) / 3; pv = pd.Series(tp * v, index=idx); vv = pd.Series(v, index=idx)
    f = {}
    for nm, per in (('wk', idx.to_period('W')), ('mo', idx.to_period('M'))): f[nm + '_vwap'] = ((pv.groupby(per).cumsum() / (vv.groupby(per).cumsum() + 1e-9)).values)
    for nm, N in (('r5', 35), ('r20', 140)): f[nm + '_vwap'] = (pv.rolling(N, min_periods=N).sum() / (vv.rolling(N, min_periods=N).sum() + 1e-9)).values
    D = pd.DataFrame({'h': pd.Series(h, index=idx).groupby(day).max(), 'l': pd.Series(l, index=idx).groupby(day).min(), 'c': pd.Series(c, index=idx).groupby(day).last()}); pc = D.c.shift(1); dtr = np.maximum(D.h - D.l, np.maximum((D.h - pc).abs(), (D.l - pc).abs())); datr = dtr.rolling(14).mean()
    L = pd.DataFrame({'e10': D.c.ewm(span=10, adjust=False).mean(), 'e20': D.c.ewm(span=20, adjust=False).mean(), 'm20': D.c.rolling(20).mean(), 's20': D.c.rolling(20).std(), 'datr': datr}).shift(1).reindex(day)  # lagged one day: known before today's bars
    L.index = idx
    for _, r in g.iterrows():
        i = int(r.fi); a = at[i]; d = {}
        for k in ('wk', 'mo', 'r5', 'r20'): d[k + '_vwap'] = (c[i] - f[k + '_vwap'][i]) / a
        d['ema10_d'] = (c[i] - L.e10.iat[i]) / L.datr.iat[i]; d['ema20_d'] = (c[i] - L.e20.iat[i]) / L.datr.iat[i]
        lo, up = L.m20.iat[i] - 2 * L.s20.iat[i], L.m20.iat[i] + 2 * L.s20.iat[i]; d['bb_pctB'] = (c[i] - lo) / (up - lo) if up > lo else np.nan
        d['vwap_id'] = r.vwx; FE[r.name] = d
X = pd.DataFrame.from_dict(FE, orient='index').reindex(T.index); T = pd.concat([T, X.drop(columns='vwap_id')], axis=1)
def sim(d, t0=pd.Timestamp('2023-12-01'), t1=pd.Timestamp('2027-01-01')):
    d = d[(d.et >= t0) & (d.et < t1)]; rows = sorted(d.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))); eq = 1.0; op = []; cv = [1.0]; n = 0; w = 0
    def close(p):
        nonlocal eq, n, w
        eq += p['sz'] * (p['ret'] - COST); cv.append(eq); n += 1; w += p['ret'] - COST > 0
    for r in rows:
        still = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: close(p)
    cv = np.array(cv); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min(), n
H1, H2 = (pd.Timestamp('2023-12-01'), pd.Timestamp('2025-04-01')), (pd.Timestamp('2025-04-01'), pd.Timestamp('2027-01-01'))
FILT = [('wk VWAP', 'wk_vwap', [1, 2, 3], 'gt'), ('mo VWAP', 'mo_vwap', [1, 2, 3], 'gt'), ('5d roll VWAP', 'r5_vwap', [1, 2, 3], 'gt'), ('20d roll VWAP', 'r20_vwap', [2, 3, 5], 'gt'), ('daily EMA10', 'ema10_d', [1, 1.5, 2], 'gt'), ('daily EMA20', 'ema20_d', [1, 1.5, 2], 'gt'), ('daily BB %B', 'bb_pctB', [.85, 1.0, 1.1], 'gt'), ('intraday VWAP (old)', 'vwx', [1], 'gt')]
rng = np.random.default_rng(7); ann = lambda r, y: ((1 + r) ** (1 / y) - 1) * 100 if r > -1 else -100
for bn, base in (('B vol>=1.5x', (T.vr >= 1.5).values), ('C vol>=1.5x + low-vol day', ((T.vr >= 1.5) & T.lv).values)):
    b = T[base]; r0, d0, n0 = sim(b); print(f'\n=== BASE {bn}: {len(b)} flips | full {r0 * 100:+.0f}% (ann {ann(r0, 2.94):+.0f}%) DD {d0 * 100:.0f}% trades {n0}')
    print(f"{'filter':22s}{'thr':>5s} {'skip%':>6s} {'ann%':>6s} {'DD%':>5s} {'H1 ret':>7s} {'H2 ret':>7s} {'kept mean/tr':>12s} {'skipped mean':>12s} {'p(rand)':>8s}")
    for nm, col, thrs, _ in FILT:
        for t in thrs:
            sk = (b[col] > t).values & b[col].notna().values; keep = b[~sk]
            if len(keep) < 20: continue
            r, dd, n = sim(keep); ra, _, _ = sim(keep, *H1); rb, _, _ = sim(keep, *H2); km = (keep.ret - COST).mean() * 100; sm = (b[sk].ret - COST).mean() * 100 if sk.sum() else np.nan
            ns = int(sk.sum()); rm = np.array([(b.ret.values[rng.permutation(len(b))[:len(b) - ns]] - COST).mean() for _ in range(2000)]) * 100 if ns else np.array([km]); p = (rm >= km).mean()
            print(f'{nm:22s}{t:5.2f} {ns / len(b) * 100:5.0f}% {ann(r, 2.94):+6.0f} {dd * 100:5.0f} {ra * 100:+7.0f} {rb * 100:+7.0f} {km:+11.2f}% {sm:+11.2f}% {p:8.3f}')
print('\nZIP / NCMI feature values at their flip bar:')
for tk in ('ZIP', 'NCMI'):
    z = T[(T.tk == tk)].sort_values('et').iloc[-1]; print(tk, z.et, {k: round(float(z[k]), 2) for k in ('vwx', 'wk_vwap', 'mo_vwap', 'r5_vwap', 'r20_vwap', 'ema10_d', 'ema20_d', 'bb_pctB')}, 'ret', round(float(z.ret), 3))
