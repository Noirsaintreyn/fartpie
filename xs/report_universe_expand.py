"""WIDER UNIVERSE for the mid/small-cap screener: (S2) small-cap proxy for IWM = US NMS/NYQ stocks, market cap $0.3-5B, 3-month avg volume > 750k, price > $5 (Yahoo screener, today's list); (S3) sub-$5 stocks (price < $5 at the ENTRY bar, point-in-time, taken from the S2 list + a screener list of current sub-$5 names with avg vol > 1M, market cap > $150M; costs 40 bp round trip because spreads are wider); (S1) the original 44 as reference. The Russell 2000 holdings list itself is not reachable from here, so S2 is a proxy.
Same rules as midsmall_screener.py: 1h RTH vector bull flip, entry next open, exit next open after the bear flip. Filters: A = all bull flips, B = flip-bar volume >= 1.5x prior-20 mean, C = B + low-vol day (prev-day range bottom third of own history so far). Long-only, K slots equal weight, higher flip-volume first. Cost 15 bp round trip (40 bp for sub-$5 entries).
IMPORTANT: filter C was chosen on the original 44 names, so the new names are a fresh check of the filter itself (same period though). Survivorship caveat: lists are today's constituents."""
import sys, os, time, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import yfinance as yf
from yfinance import EquityQuery as EQ
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; CACHE = SC + 'uni_cache.pkl'
def screen(q, n):
    out = []
    for off in range(0, n, 250):
        r = yf.screen(q, size=250, offset=off, sortField='avgdailyvol3m', sortAsc=False); out += [x['symbol'] for x in r['quotes']]
        if len(r['quotes']) < 250: break
    return out
S2 = screen(EQ('and', [EQ('eq', ['region', 'us']), EQ('is-in', ['exchange', 'NMS', 'NYQ']), EQ('btwn', ['intradaymarketcap', 3e8, 5e9]), EQ('gt', ['avgdailyvol3m', 7.5e5]), EQ('gt', ['intradayprice', 5])]), 1000)
S3x = screen(EQ('and', [EQ('eq', ['region', 'us']), EQ('is-in', ['exchange', 'NMS', 'NYQ']), EQ('lt', ['intradayprice', 5]), EQ('gt', ['avgdailyvol3m', 1e6]), EQ('gt', ['intradaymarketcap', 1.5e8])]), 500)
S1 = M.DEFAULT; ALL = sorted(set(S1) | set(S2) | set(S3x)); ALL = [t for t in ALL if t.isalpha() and len(t) <= 5]
print('S1', len(S1), 'S2', len(S2), 'sub5 list', len(S3x), 'download', len(ALL), flush=True)
data = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}
todo = [t for t in ALL if t not in data]
for i in range(0, len(todo), 60):
    ch = todo[i:i + 60]
    for attempt in range(3):
        try:
            D = yf.download(ch, period='730d', interval='1h', prepost=False, group_by='ticker', threads=True, progress=False, auto_adjust=False); break
        except Exception as e: time.sleep(5); D = None
    if D is None: continue
    for tk in ch:
        try:
            Y = D[tk].dropna(subset=['Close']).rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']]; Y.index = Y.index.tz_convert(M.ET).tz_localize(None); Y = Y[~Y.index.duplicated()]
            if len(Y) > 600: data[tk] = Y
        except Exception: pass
    print(f'  downloaded {min(i + 60, len(todo))}/{len(todo)}  (have {len(data)})', flush=True); pickle.dump(data, open(CACHE, 'wb'))
TR = []
for k, (tk, Y) in enumerate(data.items()):
    try:
        Yc = M.closed_only(Y); F = M.features(Yc)
    except Exception: continue
    r = F.reg.values; n = len(F); op = Yc.open.values; cl = Yc.close.values; open_ = None
    for i in range(1, n):
        if r[i] == r[i - 1] or r[i] == 0: continue
        if r[i] == 1 and i + 1 < n and open_ is None: open_ = dict(tk=tk, ei=i + 1, epx=float(op[i + 1]), et=Yc.index[i + 1], vr=float(F.vrel.iat[i]) if F.vrel.iat[i] == F.vrel.iat[i] else np.nan, lv=bool(F.lv.iat[i]))
        elif r[i] == -1 and open_ is not None and i + 1 < n:
            o = open_; o['xpx'] = float(op[i + 1]); o['xt'] = Yc.index[i + 1]; o['open'] = False; TR.append(o); open_ = None
    if open_ is not None: open_['xpx'] = float(cl[-1]); open_['xt'] = Yc.index[-1]; open_['open'] = True; TR.append(open_)
T = pd.DataFrame(TR); T['ret'] = T.xpx / T.epx - 1; T = T[T.et >= pd.Timestamp('2023-12-01')]; print('trades', len(T), 'names', T.tk.nunique(), flush=True)
T['S1'] = T.tk.isin(S1); T['S2'] = T.tk.isin(S2); T['sub5'] = T.epx < 5
FILT = {'A all flips': lambda d: np.ones(len(d), bool), 'B vol>=1.5x': lambda d: (d.vr >= 1.5).values, 'C vol>=1.5x + low-vol day': lambda d: ((d.vr >= 1.5) & d.lv).values}
def sim(d, K, t0, t1, cost_fn):
    d = d[(d.et >= t0) & (d.et < t1)].sort_values(['et']); eq = 1.0; op = []; curve = [1.0]; n = w = 0; hold = []
    rows = d.to_dict('records'); rows.sort(key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0)))
    def close(p):
        nonlocal eq, n, w
        eq += p['sz'] * (p['ret'] - cost_fn(p)); curve.append(eq); n += 1; w += p['ret'] - cost_fn(p) > 0; hold.append((p['xt'] - p['et']) / np.timedelta64(1, 'D'))
    for r in rows:
        still = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: close(p)
    cv = np.array(curve); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min(), n, (w / n if n else np.nan), (np.mean(hold) if hold else np.nan)
cost = lambda p: .004 if p['epx'] < 5 else .0015
PER = {'Dec23-Dec24': (pd.Timestamp('2023-12-01'), pd.Timestamp('2025-01-01'), 13 / 12), '2025': (pd.Timestamp('2025-01-01'), pd.Timestamp('2026-01-01'), 1.0), '2026 YTD': (pd.Timestamp('2026-01-01'), pd.Timestamp('2027-01-01'), 314 / 365), 'FULL': (pd.Timestamp('2023-12-01'), pd.Timestamp('2027-01-01'), (pd.Timestamp('2026-11-10') - pd.Timestamp('2023-12-01')).days / 365)}
UNI = {'S1 original 44': T[T.S1], 'S2 small-cap proxy (IWM-like)': T[T.S2], 'S3 sub-$5 entries (all lists)': T[T.sub5], 'S4 everything (S1+S2+sub-$5)': T}
print('\n######## TRADE-LEVEL (mean % per trade after costs, win%, n) - is the filter real on NEW names?')
for un, d in UNI.items():
    print(f' {un}')
    for fn, f in FILT.items():
        s = d[f(d)]; c = s.ret - np.where(s.epx < 5, .004, .0015); print(f'    {fn:28s} n={len(s):5d}  mean {c.mean() * 100:+6.2f}%  median {c.median() * 100:+6.2f}%  win {100 * (c > 0).mean():4.1f}%')
for K in (8, 12):
    print(f'\n######## PORTFOLIO K={K} slots, equal weight, higher flip-volume first')
    for un, d in UNI.items():
        print(f' {un}')
        for fn, f in FILT.items():
            s = d[f(d)]; line = []
            for pn, (a, b, yrs) in PER.items():
                tr_, dd, n, wr, hd = sim(s, K, a, b, cost); ann = ((1 + tr_) ** (1 / yrs) - 1) * 100 if tr_ > -1 else -100
                line.append(f'{pn}: {tr_ * 100:+6.0f}% (DD {dd * 100:4.0f}%)' if pn != 'FULL' else f'FULL: {tr_ * 100:+6.0f}% = {ann:+5.0f}%/yr DD {dd * 100:4.0f}% n={n} win {wr * 100:4.1f}% hold {hd:4.1f}d')
            print(f'    {fn:28s} ' + ' | '.join(line), flush=True)
