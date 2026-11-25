"""Report 476 (Nov 24 2026): Rey - 'maybe Bollinger makes more sense than assuming we break or not'. SAME universe/base/sim as 475 (book B, sub-$5/$10/$20, E+V base, vector exit, 8 slots, 40bp), with Bollinger(20, 2) on 1h / 4h / daily (higher timeframes use the last COMPLETED bar). DECLARED BEFORE RESULTS.
(1) At the flip bar: %B = (close-lower)/(upper-lower) in buckets <0, 0-.25, .25-.5, .5-.75, .75-1, >1 (>1 = already breaking out) and SQUEEZE = bandwidth (upper-lower)/basis in the bottom third of its own trailing 250 values (mechanism: low-volatility compression precedes expansion, a flip out of a squeeze has room; a flip that fires with price tagging the upper band is extended).
(2) After entry, rejection of the UPPER band: high>=upper and close<upper (failed break). Variants: BE = stop to breakeven after that bar, RX = exit at the next open. Also MIDX = exit at the next open when a bar closes below the basis (20-MA) after the trade has been above it. 3 timeframes x 3 variants x 3 buckets = 27 looks; any single pass is a lead.
(475 docstring follows) Report 475 (Nov 24 2026): Rey - (1) 'alignment' = price CLOSE ENOUGH to the EMA that it looks like we pass it rather than reject it (not 'already above'); (2) 'stop BE if we reject the ema'; (3) 'backtest the best time frames and EMA to use'.
DECLARED BEFORE RESULTS. Universe/base = report 474: bull 1h vector flips (earnmac_T2), entry next open, exit = the vector's own bear flip, base rules E+V (skip <=3 sessions before earnings, skip >1 ATR above session VWAP), vol>=1.5x book 'B', buckets sub-$5 / $10 / $20, Dec 2023-Nov 2026, same 8-slot sim, 40bp.
GRID (declared, nothing else is tried): timeframe in {1h, 4h, daily} x EMA length in {20, 50, 100, 200}  = 12 EMAs. EMA of a higher timeframe uses only its last COMPLETED bar (causal).
(1) PROXIMITY: dist = (flip-bar close - EMA) / ATR14(1h) at the flip bar. Buckets (ATR): <-3, -3..-1, -1..-0.25, -0.25..+0.25, +0.25..+1, +1..+3, >+3. Hypothesis: the dangerous bucket is price BELOW the EMA by a meaningful margin (it has to climb through it and meets supply), whereas price within ~1 ATR (below or above) or beyond it should not; read: mean net (after 40bp) per bucket, n, H1/H2 consistency. NEAR = -1 <= dist <= +1.
(2) REJECTION after entry (a signal, not a clock): bar j after entry with high >= EMA and close < EMA. Variant BE: from the bars after j, if low <= entry price, exit at min(open, entry) (stop moved to breakeven once rejected; gap-down fills at the open). Variant RX: exit at the open after the rejection bar. Ret recomputed from the bars; exit time moves, slots free up.
Judging: for each of the 12 EMAs, does the variant beat the base on ann%, DD AND both H1 and H2 on at least two of the three price buckets? 12 EMAs x 2 variants x 3 buckets = 72 looks: any pass is a lead, not a finding, unless it is monotone across the grid (neighbouring EMAs agree)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); E = pd.read_pickle(SC + 'earnmac_T2.pkl'); T = E[(E.epx < 20) & (E.vr >= 1.5)].copy().reset_index(drop=True)
COST = .004; K = 8; GRID = [(tf, n) for tf in ('1h', '4h', 'd') for n in (20, 50, 100, 200)]

def bands(Y):
    c = Y.close.astype(float); be = Y.index + pd.to_timedelta(np.where((Y.index.hour == 15) & (Y.index.minute == 30), 30, 60), unit='m'); out = {}; day = Y.index.normalize()
    for tf in ('1h', '4h', 'd'):
        if tf == '1h': ends, cl = be, c
        else:
            G = Y.groupby([day, (Y.index.hour >= 13).astype(int)] if tf == '4h' else [day]); cl = G.close.last(); ends = pd.DatetimeIndex([x.index[-1] + pd.Timedelta(minutes=30 if x.index[-1].hour == 15 else 60) for _, x in G])
        m = cl.rolling(20).mean(); sd = cl.rolling(20).std(ddof=0); up, lo = m + 2 * sd, m - 2 * sd; bw = (up - lo) / m; rk = bw.rolling(250, min_periods=60).apply(lambda z: (z[:-1] < z[-1]).mean(), raw=True)
        pos = np.searchsorted(np.asarray(ends), np.asarray(be), side='right') - 1; ok = pos >= 20; pc = np.clip(pos, 0, None)
        g = lambda v: np.where(ok, v.values[pc], np.nan); out[tf] = dict(m=g(m), up=g(up), lo=g(lo), rk=g(rk))
    return out
rows = []
for tk, g in T.groupby('tk'):
    Y = M.closed_only(data[tk]); BD = bands(Y); h, l, c, o = (Y[x].values.astype(float) for x in ('high', 'low', 'close', 'open')); ix = Y.index; F = M.features(Y)
    for r in g.itertuples():
        fi, ei = int(r.fi), int(r.ei); xi = ix.get_indexer([r.xt])[0]
        if fi < 200 or xi < 0 or xi <= ei: continue
        rec = dict(tk=tk, et=r.et, xt=r.xt, vr=r.vr, dte=r.dte, epx=r.epx, xpx=r.xpx, ret=r.ret, vwx=float(F.vwx.iat[fi]))
        for tf, B in BD.items():
            w = B['up'][fi] - B['lo'][fi]; rec[f'pb_{tf}'] = (c[fi] - B['lo'][fi]) / w if w > 0 else np.nan; rec[f'sq_{tf}'] = B['rk'][fi]
            # UP rejection variants
            rj = None
            for j in range(ei, xi):
                if np.isfinite(B['up'][j]) and h[j] >= B['up'][j] and c[j] < B['up'][j]: rj = j; break
            be_t, be_px, rx_t, rx_px = r.xt, r.xpx, r.xt, r.xpx
            if rj is not None:
                for kk in range(rj + 1, xi):
                    if l[kk] <= r.epx: be_t, be_px = ix[kk], min(o[kk], r.epx); break
                if rj + 1 < xi: rx_t, rx_px = ix[rj + 1], o[rj + 1]
            rec[f'be_{tf}'] = be_px / r.epx - 1; rec[f'bet_{tf}'] = be_t; rec[f'rx_{tf}'] = rx_px / r.epx - 1; rec[f'rxt_{tf}'] = rx_t
            # MIDX: after being above basis at some bar, first close below basis -> exit next open
            above = False; mx_t, mx_px = r.xt, r.xpx
            for j in range(ei, xi):
                if not np.isfinite(B['m'][j]): continue
                if c[j] > B['m'][j]: above = True
                elif above and j + 1 < xi: mx_t, mx_px = ix[j + 1], o[j + 1]; break
            rec[f'mx_{tf}'] = mx_px / r.epx - 1; rec[f'mxt_{tf}'] = mx_t
        rows.append(rec)
D = pd.DataFrame(rows); D['et'] = pd.to_datetime(D.et); print('trades', len(D))
def sim(d, retc, xtc, t0=pd.Timestamp('2023-12-01'), t1=pd.Timestamp('2027-01-01')):
    d = d[(d.et >= t0) & (d.et < t1)]; rs = sorted(d.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))); eq = 1.0; op = []; cv = [1.0]; n = w = 0
    def close(p):
        nonlocal eq, n, w
        rr = p[retc] - COST; eq += p['sz'] * rr; cv.append(eq); n += 1; w += rr > 0
    for r in rs:
        still = []
        for p in op:
            if p[xtc] <= r['et']: close(p)
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: close(p)
    cv = np.array(cv); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min(), n, w / max(n, 1)
H1, H2 = (pd.Timestamp('2023-12-01'), pd.Timestamp('2025-04-01')), (pd.Timestamp('2025-04-01'), pd.Timestamp('2027-01-01')); ann = lambda r: ((1 + r) ** (1 / 2.94) - 1) * 100 if r > -1 else -100
D['bet_base'] = D.xt; D['ret0'] = D.ret; base_ok = (~(D.dte <= 3)) & (~(D.vwx > 1)); PB = [-99, 0, .25, .5, .75, 1, 99]; PL = ['<0', '0-.25', '.25-.5', '.5-.75', '.75-1', '>1']; summary = {}
for cap in (5, 10, 20):
    b = D[base_ok & (D.epx < cap)].copy(); r0, dd0, n0, w0 = sim(b, 'ret0', 'bet_base'); h1, h2 = sim(b, 'ret0', 'bet_base', *H1)[0], sim(b, 'ret0', 'bet_base', *H2)[0]
    print(f'\n########## sub-${cap} book B + E+V: {len(b)} trades | base ann {ann(r0):+.0f}% DD {dd0 * 100:.0f}% H1 {h1 * 100:+.0f} H2 {h2 * 100:+.0f} win {w0:.0%} mean net {(b.ret0 - COST).mean() * 100:+.2f}%')
    hh = lambda x, H: f'{(x[(x.et >= H[0]) & (x.et < H[1])].ret0 - COST).mean() * 100:+.1f}'
    for tf in ('1h', '4h', 'd'):
        q = pd.cut(b[f'pb_{tf}'], PB, labels=PL); print(f'  %B {tf:>2s}: ' + ' | '.join(f'{l_} {(x.ret0 - COST).mean() * 100:+.1f}%({len(x)})' for l_, x in b.groupby(q, observed=False) if len(x)))
        sq = b[b[f'sq_{tf}'] <= 1 / 3]; nsq = b[b[f'sq_{tf}'] > 1 / 3]; print(f'  SQUEEZE {tf:>2s}: bottom-third bandwidth {(sq.ret0 - COST).mean() * 100:+.2f}%({len(sq)}) [H1 {hh(sq, H1)} H2 {hh(sq, H2)}] vs rest {(nsq.ret0 - COST).mean() * 100:+.2f}%({len(nsq)}) [H1 {hh(nsq, H1)} H2 {hh(nsq, H2)}]')
    for tf in ('1h', '4h', 'd'):
        for v, nm in (('be', 'BE-stop after upper-band reject'), ('rx', 'exit after upper-band reject'), ('mx', 'exit on close below basis')):
            r, dd, nn, w = sim(b, f'{v}_{tf}', f'{v}t_{tf}'); a1, a2 = sim(b, f'{v}_{tf}', f'{v}t_{tf}', *H1)[0], sim(b, f'{v}_{tf}', f'{v}t_{tf}', *H2)[0]; ok = ann(r) > ann(r0) and dd > dd0 and a1 > h1 and a2 > h2; summary.setdefault((tf, v), []).append(ok)
            print(f'  {tf:>2s} {nm:33s} ann {ann(r):+5.0f} DD {dd * 100:4.0f} H1 {a1 * 100:+5.0f} H2 {a2 * 100:+5.0f} win {w:.0%} mean {(b[f"{v}_{tf}"] - COST).mean() * 100:+.2f}% changed {(b[f"{v}_{tf}"] != b.ret0).mean():.0%} {"<-- beats base on all four" if ok else ""}')
print('\nPASS counts:', {k: sum(v) for k, v in summary.items() if sum(v)})
