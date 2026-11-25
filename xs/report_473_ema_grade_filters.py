"""Report 473 (Nov 24 2026): Rey - 'cut out the low-grade plays, and try EMA aligning: a lot of these flips look like they are rejecting the EMA'. Sub-$5 1h bull vector flips (the live mid/small book), Dec 2023 - Nov 2026, same sim as report_sub5_stack (8 slots, 40bp, H1 Dec23-Mar25 / H2 Apr25-Dec25 / 2026 split, p(rand) = share of 2000 random same-size subsets that match the kept mean).
DECLARED BEFORE RESULTS. Base = live production rules already adopted (skip <=3 sessions before earnings AND skip flip close >1 ATR above session VWAP), books C (vol>=1.5x + low-vol day) and B (vol>=1.5x).
Mechanism: a flip that fires while price sits BELOW a falling longer-term EMA is a bounce into overhead supply (rejection), so it should fail more than a flip with trend alignment; flips already above rising EMAs have buyers underneath.
Features at the flip bar (1h EMAs, causal): A50 close>EMA50; A200 close>EMA200; A4H close>4h EMA200; STACK close>EMA20>EMA50>EMA200; RISE50 EMA50 above its value 20 bars ago; REJ = close below EMA50 AND a high in the last 10 bars reached EMA50 (just rejected it); DAILYAL = daily close>daily EMA20>EMA50. KAL = causal Kalman slope>0 (same filter as live).
Live grade (as in midsmall_screener): TOP = KAL & A200 & A4H, STRONG = KAL & A200, WEAK = else. Filters tested (declared): grade>=STRONG (live default), grade=TOP, A200, STACK, RISE50, ~REJ, DAILYAL, and 'A200 & RISE50'.
Judging: not a significance hunt - 3 years, ~650-1750 flips per book, thresholds not tuned; PASS = kept mean above skipped mean AND ann% and drawdown better than base AND H1,H2 both positive-improvement vs base, p(rand)<.1."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); E = pd.read_pickle(SC + 'earnmac_T2.pkl'); V = pd.read_pickle(SC + 'sub5_T.pkl')[['tk', 'et', 'vwx']]
T = E[E.sub5].merge(V, on=['tk', 'et'], how='left'); COST = .004; K = 8
def kal_series(c):
    y = np.log(c); R = max(np.nanvar(np.diff(y)[:500]), 1e-8); x = np.array([y[0], 0.]); P = np.diag([R, R * .01]); F = np.array([[1, 1], [0, 1]]); Q = np.diag([.05 * R, .001 * R]); out = np.zeros(len(y), bool)
    for i in range(1, len(y)):
        x = F @ x; P = F @ P @ F.T + Q; k = P[:, 0] / (P[0, 0] + R); x = x + k * (y[i] - x[0]); P = P - np.outer(k, P[0]); out[i] = x[1] > 0
    return out
feat = {}
for tk, g in T.groupby('tk'):
    Y = M.closed_only(data[tk]); c = Y.close.astype(float); e20, e50, e200 = (c.ewm(span=s, adjust=False).mean() for s in (20, 50, 200)); kal = kal_series(c.values)
    day = Y.index.normalize(); G = Y.groupby([day, (Y.index.hour >= 13).astype(int)]); cl4 = G.close.last(); end4 = [x.index[-1] + pd.Timedelta(minutes=30 if x.index[-1].hour == 15 else 60) for _, x in G]; e4 = pd.Series(cl4.ewm(span=200, adjust=False).mean().values, index=pd.DatetimeIndex(end4))
    d = Y.groupby(day).agg(close=('close', 'last')); de20, de50 = d.close.ewm(span=20, adjust=False).mean(), d.close.ewm(span=50, adjust=False).mean(); dal = ((d.close > de20) & (de20 > de50)).shift(1)       # last COMPLETED daily bar
    hi = Y.high.astype(float); touch = (hi.rolling(10).max() >= e50)
    for r in g.itertuples():
        i = int(r.fi)
        if i >= len(Y) or i < 200: continue
        be = Y.index[i] + pd.Timedelta(minutes=30 if Y.index[i].hour == 15 else 60); e4v = e4[e4.index <= be]
        feat[(r.tk, r.et)] = dict(A50=c.iat[i] > e50.iat[i], A200=c.iat[i] > e200.iat[i], A4H=bool(len(e4v) and c.iat[i] > e4v.iat[-1]), STACK=bool(c.iat[i] > e20.iat[i] > e50.iat[i] > e200.iat[i]), RISE50=bool(e50.iat[i] > e50.iat[i - 20]), REJ=bool(c.iat[i] < e50.iat[i] and touch.iat[i]), DAILYAL=bool(dal.reindex([Y.index[i].normalize()]).fillna(False).iat[0]), KAL=bool(kal[i]))
F = pd.DataFrame.from_dict(feat, orient='index'); F.index = pd.MultiIndex.from_tuples(F.index, names=['tk', 'et']); T = T.merge(F.reset_index(), on=['tk', 'et'], how='inner')
T['GRADE'] = np.where(T.KAL & T.A200 & T.A4H, 'TOP', np.where(T.KAL & T.A200, 'STRONG', 'WEAK')); print('flips with features', len(T), T.GRADE.value_counts().to_dict())
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
RULES = {'base (E+V)': lambda d: d.ret == d.ret, 'grade>=STRONG (live default)': lambda d: d.GRADE != 'WEAK', 'grade=TOP only': lambda d: d.GRADE == 'TOP', 'A200 (above 1h EMA200)': lambda d: d.A200, 'A50': lambda d: d.A50, 'STACK 20>50>200': lambda d: d.STACK, 'RISE50': lambda d: d.RISE50, 'not REJ (not just rejected EMA50)': lambda d: ~d.REJ, 'DAILYAL': lambda d: d.DAILYAL, 'A200 & RISE50': lambda d: d.A200 & d.RISE50}
rng = np.random.default_rng(11); base_ok = (~(T.dte <= 3)) & (~(T.vwx > 1))
for bn, bm in (('C vol>=1.5x + low-vol day', (T.vr >= 1.5) & T.lv), ('B vol>=1.5x', T.vr >= 1.5)):
    b = T[bm & base_ok]; print(f'\n=== sub-$5 {bn} + E+V base: {len(b)} flips | grade mix {b.GRADE.value_counts().to_dict()}'); print(f"{'rule':34s}{'kept':>5s} {'ann%':>6s} {'DD%':>5s} {'H1':>6s} {'H2':>6s} {'2026':>6s} {'win%':>5s} {'hold d':>6s} {'kept mean':>9s} {'skip mean':>9s} {'p(rand)':>8s}")
    for rn, f in RULES.items():
        kp = f(b).fillna(False).values; k_ = b[kp]; s_ = b[~kp]
        if len(k_) < 30: print(rn, 'n<30'); continue
        r, dd, n, w, hd = sim(k_); rm = np.array([(b.ret.values[rng.permutation(len(b))[:len(k_)]] - COST).mean() for _ in range(2000)]) * 100; km = (k_.ret - COST).mean() * 100
        print(f"{rn:34s}{len(k_):5d} {ann(r, 2.94):+6.0f} {dd * 100:5.0f} {sim(k_, *H1)[0] * 100:+6.0f} {sim(k_, *H2)[0] * 100:+6.0f} {sim(k_, *H3)[0] * 100:+6.0f} {w * 100:5.0f} {hd:6.1f} {km:+8.2f}% {(s_.ret - COST).mean() * 100 if len(s_) else float('nan'):+8.2f}% {(rm >= km).mean():8.3f}")
    print('  mean net by grade:', {g: (len(x), round((x.ret - COST).mean() * 100, 2), f'{((x.ret - COST) > 0).mean():.0%} win') for g, x in b.groupby('GRADE')})
    print('  REJ vs not:', {str(k): (len(x), round((x.ret - COST).mean() * 100, 2)) for k, x in b.groupby('REJ')})
