"""Nov 12 2026: (1) moving-average context at the flip, (2) EARLY exits before the 1h close. DECLARED BEFORE RESULTS. $1-2 names, 1h vector, vol>=1.5x flip (B) and B + daily bull (A), since 2024-01, 40 bp, train < 2025-07-01 | test after.
(1) context filters on the flip-bar close: > 1h EMA50, > EMA200, EMA50 > EMA200, > quarterly anchored VWAP, and 'all four'. Mechanism: a flip below the big averages is a bounce inside a downtrend (supply overhead).
(2) exit variants vs baseline (exit next open after the bear flip = close-based): LEVEL x: exit as soon as a bar trades x ATR below the PRIOR bar's vector level (fill = min(open, trigger) minus 20 bp extra stop slippage); x in 0, .25, .5, 1.0.
    SPEED k: exit as soon as a bar drops k ATR from its open (k in 1, 1.5, 2) at open - k ATR (-20 bp). Nothing fitted; every variant reported. Intrabar volume is not in 1h OHLCV, so volume is judged on the completed flip bar only (done in (1) via vol>=1.5x)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); sys.path.insert(0, 'xs')
import midsmall_screener as M
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); COST = .004; SLIP = .002
T0 = pd.read_pickle(SC + 'mtf_T.pkl'); keep = set(zip(T0.tk, T0.et))
def daily_bull_at(Y, i):
    return None
def gen(tk):
    Y = M.closed_only(data[tk].dropna(subset=['open', 'close']))
    if len(Y) < 250: return []
    reg, lev, atr, _ = run_state_machine(Y.assign(datetime=Y.index)); c = Y.close.values; o = Y.open.values; h = Y.high.values; l = Y.low.values; idx = Y.index
    s = pd.Series(c); e50 = s.ewm(span=50, adjust=False).mean().values; e200 = s.ewm(span=200, adjust=False).mean().values
    q = idx.to_period('Q'); tp = (Y.high + Y.low + Y.close) / 3; qv = ((tp * Y.volume).groupby(q).cumsum() / (Y.volume.groupby(q).cumsum() + 1e-9)).values; out = []; ent = None
    for i in range(1, len(Y) - 1):
        if reg[i] == reg[i - 1] or reg[i] == 0: continue
        if reg[i] == 1 and ent is None: ent = i
        elif reg[i] == -1 and ent is not None:
            fi = ent; ei = fi + 1; xi = i + 1
            if (idx[ei], ) and (tk, idx[ei]) in keep:
                base = o[xi] / o[ei] - 1; res = dict(tk=tk, et=idx[ei], xt=idx[xi], epx=o[ei], sig=c[fi], base=base, xtime={}, ret={'base': base})
                res.update(a50=c[fi] > e50[fi], a200=c[fi] > e200[fi], x5020=e50[fi] > e200[fi], avw=c[fi] > qv[fi])
                for nm, kind, x in [('L0', 'L', 0), ('L.25', 'L', .25), ('L.5', 'L', .5), ('L1', 'L', 1.0), ('S1', 'S', 1.0), ('S1.5', 'S', 1.5), ('S2', 'S', 2.0)]:
                    ex = None
                    for j in range(ei, xi):
                        a_ = atr[j - 1]
                        if not a_ == a_: continue
                        trig = (lev[j - 1] - x * a_) if kind == 'L' else (o[j] - x * a_)
                        if kind == 'L' and reg[j - 1] != 1: continue
                        if l[j] <= trig: ex = min(o[j], trig) * (1 - SLIP); break
                    res['ret'][nm] = (ex / o[ei] - 1) if ex else base
                    res['xtime'][nm] = ex is not None
                out.append(res)
            ent = None
        elif reg[i] == -1: ent = None
    return out
R = [x for tk in data for x in gen(tk)]; print('trades regenerated', len(R), 'of', len(T0))
T = pd.DataFrame([{k: v for k, v in r.items() if k not in ('ret', 'xtime')} | {f'r_{k}': v for k, v in r['ret'].items()} | {f'x_{k}': v for k, v in r['xtime'].items()} for r in R]).merge(T0[['tk', 'et', 'vr', 'lv', 'ad', 'a4']], on=['tk', 'et'])
T = T[(T.sig >= 1) & (T.sig < 2)].copy(); T.to_pickle(SC + 'ma_exit_T.pkl')
def book(T, col, K=8):
    eq = 1.0; op = []; curve = []; X = T.assign(ret=T[col]).to_dict('records')
    for r in sorted(X, key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))):
        still = []
        for p in op:
            if p['xt'] <= r['et']: eq += p['sz'] * (p['ret'] - COST); curve.append((p['xt'], eq))
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: eq += p['sz'] * (p['ret'] - COST); curve.append((p['xt'], eq))
    s = pd.Series(dict(curve)).sort_index(); return s
def rep(nm, Z, col='r_base'):
    if len(Z) < 15: print(f'  {nm:<26} {len(Z)} trades (too few)'); return
    r = Z[col] - COST; a, b = r[Z.et < '2025-07-01'], r[Z.et >= '2025-07-01']; s = book(Z, col); yrs = (s.index[-1] - Z.et.min()).days / 365.25; cg = s.iloc[-1] ** (1 / max(yrs, .5)) - 1
    print(f'  {nm:<26} {len(Z):>4} tr | mean {r.mean()*100:+.2f}% med {r.median()*100:+.1f}% win {(r>0).mean():.0%} p10 {r.quantile(.1)*100:+.1f}% | train {a.mean()*100:+.2f}% ({len(a)}) test {b.mean()*100:+.2f}% ({len(b)}) | book {cg:+.0%}/yr DD {(s/s.cummax()-1).min():.0%}', flush=True)
B = T[T.vr >= 1.5]; A = B[B.ad]
print('== (1) MA context at the flip, $1-2, vol>=1.5x (B)')
rep('B base', B)
for k, nm in (('a50', 'close > EMA50'), ('a200', 'close > EMA200'), ('x5020', 'EMA50 > EMA200'), ('avw', 'close > quarterly VWAP')):
    rep(f'B & {nm}', B[B[k]]); rep(f'B & NOT {nm}', B[~B[k]])
allm = B.a50 & B.a200 & B.x5020 & B.avw; rep('B & all four', B[allm]); rep('B & none of four', B[~B.a50 & ~B.a200 & ~B.x5020 & ~B.avw])
print('   on A (B + daily bull):'); rep('A base', A); rep('A & close > EMA200', A[A.a200]); rep('A & NOT close > EMA200', A[~A.a200]); rep('A & close > qVWAP', A[A.avw]); rep('A & NOT > qVWAP', A[~A.avw])
for nm, Z in (('B', B), ('A', A)):
    print(f'== (2) exits on {nm}: baseline (close-based flip) vs early exits (fraction of trades the early rule fired)')
    rep('baseline', Z)
    for v in ('L0', 'L.25', 'L.5', 'L1', 'S1', 'S1.5', 'S2'): rep(f'{v} fired {Z["x_" + v].mean():.0%}', Z, f'r_{v}')
