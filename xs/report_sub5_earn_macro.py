"""Vector flips (1h, B = flip volume >=1.5x, C = B + low-vol day) paired with EARNINGS (surprise, sessions to/since the earnings reaction) and MACRO regime (IWM vs 50d, VIX vs trailing median). Universe = 753-name 1h cache; two price classes: sub-$5 and >=$5 (mid/small). DECLARED BEFORE RESULTS (mechanism first):
 R1 skip if the next earnings reaction is <=3 sessions away (binary gap risk, tight stops get run -> chop)
 R2 require last reported EPS surprise > 0 (post-earnings-announcement drift: Bernard-Thomas 1989; = the lead signal of the old mid book 'pead_combo_x')
 R3 = R1 and R2
 M1 require IWM prior close > its 50d mean   M2 require VIX prior close <= its trailing 252d median
Judged chronologically: H1 = Dec23-Mar25 (choose), H2 = Apr25-Nov26 (judge). 40bp, 8 slots, higher flip volume first. Prior-day macro values only. Fundamental ratios (P/E, margins) are NOT point-in-time on Yahoo and the SEC is not reachable from here, so not tested."""
import sys, pickle, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); EARN = pickle.load(open(SC + 'earn_uni.pkl', 'rb')); COST = .004; K = 8
def atr14(h, l, c): pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); return pd.Series(tr).rolling(14).mean().values
R = []
for tk, Y in data.items():
    try: Yc = M.closed_only(Y); F = M.features(Yc)
    except Exception: continue
    n = len(F); r = F.reg.values; o, h, l, c = (Yc[k].values.astype(float) for k in ('open', 'high', 'low', 'close')); at = atr14(h, l, c); op = None
    for i in range(1, n):
        if r[i] == r[i - 1] or r[i] == 0: continue
        if r[i] == 1 and i + 1 < n and op is None and at[i] == at[i]: op = dict(tk=tk, fi=i, ei=i + 1, epx=o[i + 1], et=Yc.index[i + 1], vr=float(F.vrel.iat[i]) if F.vrel.iat[i] == F.vrel.iat[i] else np.nan, lv=bool(F.lv.iat[i]))
        elif r[i] == -1 and op is not None and i + 1 < n: op['xpx'] = o[i + 1]; op['xt'] = Yc.index[i + 1]; op['open'] = False; R.append(op); op = None
        elif r[i] == -1: op = None
    if op is not None: op['xpx'] = c[-1]; op['xt'] = Yc.index[-1]; op['open'] = True; R.append(op)
T = pd.DataFrame(R); T['ret'] = T.xpx / T.epx - 1; T = T[T.et >= pd.Timestamp('2023-12-01')].reset_index(drop=True); T['sub5'] = T.epx < 5; print('trades', len(T), 'sub5', T.sub5.sum(), flush=True)
# ---- earnings features
dte = np.full(len(T), np.nan); dse = dte.copy(); surp = dte.copy(); span = np.zeros(len(T), bool)
for tk, g in T.groupby('tk'):
    E = EARN.get(tk)
    if E is None: continue
    E = E.copy(); E.index = pd.DatetimeIndex(E.index).tz_localize(None); E = E.sort_index()
    rt = [(d.normalize() + pd.Timedelta(days=1) if d.hour >= 16 else d.normalize()) + pd.Timedelta(hours=9, minutes=30) for d in E.index]; rt = pd.DatetimeIndex(rt); rep = E['Reported EPS'].values; su = E['Surprise(%)'].values
    for ix, r_ in g.iterrows():
        nxt = np.searchsorted(rt.values, np.datetime64(r_.et), side='right')
        if nxt < len(rt): dte[ix] = np.busday_count(r_.et.date(), rt[nxt].date()); span[ix] = rt[nxt] <= r_.xt
        j = nxt - 1
        while j >= 0 and not np.isfinite(rep[j]): j -= 1
        if j >= 0: dse[ix] = np.busday_count(rt[j].date(), r_.et.date()); surp[ix] = su[j] if np.isfinite(su[j]) else np.nan
T['dte'] = dte; T['dse'] = dse; T['surp'] = surp; T['span'] = span
# ---- macro
mac = yf.download(['IWM', '^VIX'], start='2023-01-01', progress=False, auto_adjust=True)['Close']; iw = mac['IWM'].dropna(); vx = mac['^VIX'].dropna()
m1 = (iw > iw.rolling(50).mean()).shift(1); m2 = (vx <= vx.rolling(252).median()).shift(1); d_ = T.et.dt.normalize()
T['m1'] = m1.reindex(d_).values.astype(float); T['m2'] = m2.reindex(d_).values.astype(float)
T.to_pickle(SC + 'earnmac_T.pkl')
def sim(d, t0=pd.Timestamp('2023-12-01'), t1=pd.Timestamp('2027-01-01')):
    d = d[(d.et >= t0) & (d.et < t1)]; rows = sorted(d.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))); eq = 1.0; op = []; cv = [1.0]; n = 0
    def close(p):
        nonlocal eq, n
        eq += p['sz'] * (p['ret'] - COST); cv.append(eq); n += 1
    for r in rows:
        still = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: close(p)
    cv = np.array(cv); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min(), n
H1, H2 = (pd.Timestamp('2023-12-01'), pd.Timestamp('2025-04-01')), (pd.Timestamp('2025-04-01'), pd.Timestamp('2027-01-01')); ann = lambda r, y: ((1 + r) ** (1 / y) - 1) * 100 if r > -1 else -100
RULES = {'R1 skip DTE<=3': lambda d: ~(d.dte <= 3), 'R2 last surprise>0': lambda d: d.surp > 0, 'R3 R1+R2': lambda d: ~(d.dte <= 3) & (d.surp > 0), 'M1 IWM>50d': lambda d: d.m1 == 1, 'M2 VIX<=median': lambda d: d.m2 == 1, 'M1+M2': lambda d: (d.m1 == 1) & (d.m2 == 1)}
rng = np.random.default_rng(3)
for pn, pm in (('SUB-$5', T.sub5), ('>= $5 (mid/small)', ~T.sub5)):
    for bn, bm in (('B vol>=1.5x', T.vr >= 1.5), ('C B+low-vol-day', (T.vr >= 1.5) & T.lv)):
        b = T[pm & bm]; r0, d0, n0 = sim(b); a0, c0 = sim(b, *H1)[0], sim(b, *H2)[0]
        print(f'\n=== {pn} | {bn}: {len(b)} flips, mean/trade {(b.ret - COST).mean() * 100:+.2f}% | book ann {ann(r0, 2.94):+.0f}% DD {d0 * 100:.0f}% | H1 {a0 * 100:+.0f}% H2 {c0 * 100:+.0f}%')
        print('   by DTE bucket (sessions to next earnings reaction):', {k: (int(len(x)), round((x.ret - COST).mean() * 100, 2)) for k, x in b.groupby(pd.cut(b.dte, [0, 3, 15, 1e9]))})
        print('   hold spans an earnings reaction:', {str(k): (int(len(x)), round((x.ret - COST).mean() * 100, 2)) for k, x in b.groupby('span')})
        print('   last surprise sign:', {k: (int(len(x)), round((x.ret - COST).mean() * 100, 2)) for k, x in b.groupby(np.sign(b.surp))})
        print(f"   {'rule':16s} {'kept':>5s} {'ann%':>6s} {'DD%':>5s} {'H1':>6s} {'H2':>6s} {'kept mean':>9s} {'skip mean':>9s} {'p(rand)':>8s}")
        for rn, f in RULES.items():
            kp = f(b).fillna(False).values; k_ = b[kp]; s_ = b[~kp]
            if len(k_) < 15: continue
            r, dd, n = sim(k_); ra = sim(k_, *H1)[0]; rb = sim(k_, *H2)[0]; km = (k_.ret - COST).mean() * 100; sm = (s_.ret - COST).mean() * 100 if len(s_) else np.nan
            rm = np.array([(b.ret.values[rng.permutation(len(b))[:len(k_)]] - COST).mean() for _ in range(2000)]) * 100; p = (rm >= km).mean()
            print(f'   {rn:16s} {len(k_):5d} {ann(r, 2.94):+6.0f} {dd * 100:5.0f} {ra * 100:+6.0f} {rb * 100:+6.0f} {km:+8.2f}% {sm:+8.2f}% {p:8.3f}')
for tk in ('NCMI', 'ZIP', 'ANGX', 'TIGR', 'FRMI'):
    z = T[T.tk == tk].sort_values('et').tail(3)[['et', 'ret', 'dte', 'dse', 'surp', 'span']]; print(tk); print(z.to_string())
