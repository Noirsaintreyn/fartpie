"""Nov 12 2026: the stack by price band: <$1, $1-2, $2-5, <$5 all. DECLARED BEFORE RESULTS. Same definitions as report_12_mtf / report_12_smooth / report_12_tf_ema, nothing refit:
B = 1h bull flip, flip-bar volume >= 1.5x; D = daily vector bull on the last completed daily bar; K = causal Kalman slope > 0; A+ = B&D&K; TOP = A+ & flip above the 1h EMA200 and the 4h EMA200.
Entries since 2024-01, baseline close-based exit, 8-slot book, train < 2025-07-01 | test after. Cost 40 bp (and 150 bp for < $1, where spreads are wider). p = vs random same-size draws from the band's own B set."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); sys.path.insert(0, 'xs')
import midsmall_screener as M
from vector_pine import run_state_machine
sys.argv = sys.argv[:1]; exec(open('xs/report_12_smooth.py').read().split('def gen(tk):')[0].split('SC = ')[0].replace('import sys, pickle', 'import sys, pickle'), globals()) if False else None
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); COST = .004
_s = open('xs/report_12_smooth.py').read(); exec(_s[_s.index('def kalman(y):'):_s.index('def gen(tk):')])      # kalman()
def htf(Y, tf):
    day = Y.index.normalize(); G = Y.groupby([day, (Y.index.hour >= 13).astype(int)] if tf == '4h' else [day]); o = G.agg(close=('close', 'last'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), volume=('volume', 'sum'))
    last = G.apply(lambda x: x.index[-1]); lt = pd.DatetimeIndex(last.values); end = lt + pd.to_timedelta(np.where(lt.hour == 15, 30, 60), unit='m'); n = np.arange(1, len(o) + 1)
    reg = run_state_machine(o.assign(datetime=lt))[0] if tf == '1d' else np.zeros(len(o)); e200 = o.close.ewm(span=200, adjust=False).mean().where(n >= 200)
    return pd.DataFrame({'end': end, 'reg': reg, 'e200': e200.values}).sort_values('end')
def gen(tk):
    Y = M.closed_only(data[tk].dropna(subset=['open', 'close']))
    if len(Y) < 250: return []
    reg = run_state_machine(Y.assign(datetime=Y.index))[0]; c = Y.close.values; o = Y.open.values; v = Y.volume.values.astype(float); vrel = v / pd.Series(v).rolling(20).mean().shift().values; idx = Y.index
    s = pd.Series(c); e200 = s.ewm(span=200, adjust=False).mean().values; ly = np.log(c); sl = None; out = []; ent = None; hd = h4 = None
    cl = pd.DataFrame({'t': idx + pd.to_timedelta(np.where(idx.hour == 15, 30, 60), unit='m')})
    for i in range(1, len(Y) - 1):
        if reg[i] == reg[i - 1] or reg[i] == 0: continue
        if reg[i] == 1 and ent is None: ent = i
        elif reg[i] == -1 and ent is not None:
            fi = ent; ei = fi + 1; xi = i + 1
            if 0.2 <= c[fi] < 5 and vrel[fi] >= 1.5 and idx[ei] >= pd.Timestamp('2024-01-01') and fi >= 140:
                if sl is None: sl, _ = kalman(ly); hd, h4 = htf(Y, '1d'), htf(Y, '4h')
                t = cl.t.iat[fi]; d_ = hd[hd.end <= t].iloc[-1] if (hd.end <= t).any() else None; f4 = h4[h4.end <= t].iloc[-1] if (h4.end <= t).any() else None
                out.append(dict(tk=tk, et=idx[ei], xt=idx[xi], sig=c[fi], vr=vrel[fi], ret=o[xi] / o[ei] - 1, kal=sl[fi] > 0, ad=bool(d_ is not None and d_.reg == 1), a200=c[fi] > e200[fi], a4=bool(f4 is not None and f4.e200 == f4.e200 and c[fi] > f4.e200)))
            ent = None
        elif reg[i] == -1: ent = None
    return out
T = pd.DataFrame([x for tk in data for x in gen(tk)]); T.to_pickle(SC + 'bands_T.pkl'); print('B flips <$5:', len(T))
def book(Z, cost, K=8):
    eq = 1.0; op = []; curve = []
    for r in sorted(Z.to_dict('records'), key=lambda r: (r['et'], -r['vr'])):
        still = []
        for p in op:
            if p['xt'] <= r['et']: eq += p['sz'] * (p['ret'] - cost); curve.append((p['xt'], eq))
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: eq += p['sz'] * (p['ret'] - cost); curve.append((p['xt'], eq))
    return pd.Series(dict(curve)).sort_index()
rng = np.random.default_rng(7)
def rep(nm, Z, base, cost=COST):
    if len(Z) < 12: print(f'  {nm:<22} {len(Z)} trades (too few)'); return
    r = Z.ret - cost; a, b = r[Z.et < '2025-07-01'], r[Z.et >= '2025-07-01']; s = book(Z, cost); yrs = (s.index[-1] - Z.et.min()).days / 365.25; cg = s.iloc[-1] ** (1 / max(yrs, .5)) - 1
    rr = (base.ret - cost).values; dr = [rng.choice(rr, len(r), replace=False).mean() for _ in range(3000)] if len(r) < len(rr) else []; p = f' p={(np.array(dr) >= r.mean()).mean():.3f}' if dr else ''
    k = np.sort(r.values); x1 = f' ex-top1 {k[:-1].mean()*100:+.1f}%' if len(k) > 5 else ''
    print(f'  {nm:<22} {len(Z):>4} tr {len(Z)/yrs:>3.0f}/yr | mean {r.mean()*100:+.2f}% med {r.median()*100:+.1f}% win {(r>0).mean():.0%} | train {a.mean()*100:+.2f}% ({len(a)}) test {b.mean()*100:+.2f}% ({len(b)}) | book {cg:+.0%}/yr DD {(s/s.cummax()-1).min():.0%}{p}{x1}', flush=True)
for nm, lo, hi, cost in (('< $1', .2, 1, COST), ('< $1 @150bp', .2, 1, .015), ('$1-2', 1, 2, COST), ('$2-5', 2, 5, COST), ('< $5 (all)', .2, 5, COST)):
    X = T[(T.sig >= lo) & (T.sig < hi)]; print(f'== {nm}: {len(X)} B flips')
    rep('B', X, X, cost); rep('D (daily bull)', X[X.ad], X, cost); rep('K (Kalman up)', X[X.kal], X, cost); rep('A+ = B&D&K', X[X.ad & X.kal], X, cost); rep('TOP = A+ & 1h&4h EMA200', X[X.ad & X.kal & X.a200 & X.a4], X, cost)
