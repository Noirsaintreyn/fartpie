"""Share-count change (net issuance / buyback) and post-earnings entry on the vector flips. Yahoo get_shares_full = dated share counts (approx point in time; last count dated <= entry). dil = shares(entry)/shares(>=365d earlier)-1; buyback yield = -dil. DECLARED BEFORE RESULTS: DIL skip dil>5% (dilution funds cash-burning small caps; Pontiff-Woodgate 2008, Daniel-Titman 2006 net issuance); BB require dil<0; PE post-earnings entry dse<=1 & last surprise>0 (drift). Same sim/halves as report_sub5_earn_macro.py."""
import pickle, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; T = pd.read_pickle(SC + 'earnmac_T.pkl'); COST = .004; K = 8
def g(tk):
    try:
        s = yf.Ticker(tk).get_shares_full(start='2022-06-01'); s = s[~s.index.duplicated()].sort_index(); s.index = pd.DatetimeIndex(s.index).tz_localize(None); return tk, s
    except Exception: return tk, None
with ThreadPoolExecutor(8) as ex: S = dict(ex.map(g, T.tk.unique()))
dil = np.full(len(T), np.nan)
for ix, r in T.iterrows():
    s = S.get(r.tk)
    if s is None or len(s) < 2: continue
    a = s[:r.et]; b = s[:r.et - pd.Timedelta(days=365)]
    if len(a) and len(b) and b.iloc[-1] > 0: dil[ix] = a.iloc[-1] / b.iloc[-1] - 1
T['dil'] = dil; print('share-count coverage', np.isfinite(dil).mean().round(2), 'sub5', np.isfinite(dil[T.sub5.values]).mean().round(2)); T.to_pickle(SC + 'earnmac_T2.pkl')
exec(open('xs/report_sub5_earn_macro.py').read().split('def sim(')[1].join(['def sim(', '']).split('RULES =')[0]) if False else None
def sim(d, t0=pd.Timestamp('2023-12-01'), t1=pd.Timestamp('2027-01-01')):
    d = d[(d.et >= t0) & (d.et < t1)]; rows = sorted(d.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))); eq = 1.0; op = []; cv = [1.0]
    def close(p):
        nonlocal eq
        eq += p['sz'] * (p['ret'] - COST); cv.append(eq)
    for r in rows:
        still = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: close(p)
    cv = np.array(cv); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min()
H1, H2 = (pd.Timestamp('2023-12-01'), pd.Timestamp('2025-04-01')), (pd.Timestamp('2025-04-01'), pd.Timestamp('2027-01-01')); ann = lambda r, y: ((1 + r) ** (1 / y) - 1) * 100 if r > -1 else -100
RULES = {'DIL skip dil>5%': lambda d: ~(d.dil > .05), 'DIL skip dil>15%': lambda d: ~(d.dil > .15), 'BB net buyback dil<0': lambda d: d.dil < 0, 'PE post-earn dse<=1 & beat': lambda d: (d.dse <= 1) & (d.surp > 0), 'PE any dse<=1': lambda d: d.dse <= 1}
rng = np.random.default_rng(5)
for pn, pm in (('SUB-$5', T.sub5), ('>= $5', ~T.sub5)):
    for bn, bm in (('B', T.vr >= 1.5), ('C', (T.vr >= 1.5) & T.lv)):
        b = T[pm & bm]; r0, d0 = sim(b); print(f'\n=== {pn} | {bn}: {len(b)} flips mean {(b.ret - COST).mean() * 100:+.2f}% | ann {ann(r0, 2.94):+.0f}% DD {d0 * 100:.0f}%')
        print('   dilution buckets:', {str(k): (int(len(x)), round((x.ret - COST).mean() * 100, 2)) for k, x in b.groupby(pd.cut(b.dil, [-1, -.01, .02, .05, .15, 99]))})
        print(f"   {'rule':26s} {'kept':>5s} {'ann%':>6s} {'DD%':>5s} {'H1':>6s} {'H2':>6s} {'kept mean':>9s} {'skip mean':>9s} {'p(rand)':>8s}")
        for rn, f in RULES.items():
            kp = f(b).fillna(False).values; k_ = b[kp]; s_ = b[~kp]
            if len(k_) < 15: print(f'   {rn:26s} {len(k_):5d} (too few)'); continue
            r, dd = sim(k_); ra = sim(k_, *H1)[0]; rb = sim(k_, *H2)[0]; km = (k_.ret - COST).mean() * 100; sm = (s_.ret - COST).mean() * 100
            rm = np.array([(b.ret.values[rng.permutation(len(b))[:len(k_)]] - COST).mean() for _ in range(2000)]) * 100
            print(f'   {rn:26s} {len(k_):5d} {ann(r, 2.94):+6.0f} {dd * 100:5.0f} {ra * 100:+6.0f} {rb * 100:+6.0f} {km:+8.2f}% {sm:+8.2f}% {(rm >= km).mean():8.3f}')
