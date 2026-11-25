"""Report 487 (Nov 25 2026): Rey - 'run everything you said' (the follow-through / signal-exit test I proposed after report 486). Entry = next open after the daily flip close; baseline exit = the daily vector's bear flip, filled at the NEXT OPEN after that close; 30bp round trip. No time-based holds: every exit below is a signal.
DECLARED BEFORE RESULTS. Group: sub-$5 and sub-$10, 4-feature base5 minus VOLX (ATRP, F11, LTURN, -LMC; train-ECDF cuts, top 30% of the bucket's train rows), flips 2022+ (test). Mechanism: a flip that is real (news / squeeze) keeps printing above its flip-bar structure; a failing flip gives it back and sits under it until the vector finally breaks, losing 12-18%. Exit rules, each checked at every daily close from the entry day, fill next open, earliest of rule or baseline wins:
 X1 STRUCT: first close below the flip-day LOW.
 X2 ATRSTOP: first close below entry open x (1 - 1.5 x flip-day ATRP)   [ATR-scaled stop on close, not fixed].
 X3 DISTRIB: first down close on volume >= 2x its 20d mean while the trade has NOT yet touched +10%.
 X4 LOCK: once the high has touched +10%, exit on the first close below the prior day's low (protects winners; before +10% only the baseline applies).
 X5 = X1 + X4.
Read per rule vs baseline: mean, median, win, p10, share of baseline +30% winners still held to >= +30% (damage to the tail), era split (2022 | 2023-26), month-clustered paired t of mean difference. PASS = mean up in BOTH eras AND t >= 2 AND the +30% winner capture >= 80% of baseline."""
import numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d); O['tr'] = O.d.dt.year <= 2021; W = pd.read_pickle(SP + 'ext/wide_daily_ohlcv.pkl')
B = {'ATRP': 1, 'F11': 1, 'LTURN': 1, 'LMC': -1}; COST = .003
def score(D):
    R = []
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); q = np.where(x.isna(), np.nan, q); R.append(q if s > 0 else 1 - q)
    return np.nanmean(np.vstack(R), axis=0)
cache = {}
def sim(tk, d, atrp):
    if tk not in cache:
        df = W.get(tk)
        if df is None: cache[tk] = None; return None
        df = df[~df.index.duplicated()]; cache[tk] = (df, run_state_machine(df.assign(datetime=df.index))[0])
    if cache[tk] is None: return None
    df, reg = cache[tk]; ix = df.index; i = ix.searchsorted(d)
    if i + 2 >= len(ix) or ix[i] != d: return None
    o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values; v = df.volume.values.astype(float); vm = pd.Series(v).rolling(20).mean().shift(1).values; e = o[i + 1]; fl = l[i]; touched = False; ex = {}; n = len(ix)
    for k in range(i + 1, n - 1):
        touched_before = touched; touched = touched or h[k] >= 1.10 * e
        sig = {'BASE': reg[k] == -1, 'X1': c[k] < fl, 'X2': c[k] < e * (1 - 1.5 * atrp), 'X3': (c[k] < c[k - 1]) and v[k] >= 2 * vm[k] and not touched, 'X4': touched and c[k] < l[k - 1]}
        for nm, s_ in sig.items():
            if s_ and nm not in ex: ex[nm] = (o[k + 1] / e - 1 - COST)
        if 'BASE' in ex: break
    if 'BASE' not in ex: return None
    pk = h[i + 1:i + 1 + 400].max() / e - 1
    out = {'BASE': ex['BASE']}
    for nm in ('X1', 'X2', 'X3', 'X4'): out[nm] = min([ex[x] for x in ('BASE', nm) if x in ex], key=lambda _: 0) if nm not in ex else None
    return ex, e
def run(cap):
    S = O[O.px < cap].dropna(subset=['ATRP']).copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[(~S.tr) & (S.s_ >= cut)]; rows = []
    for r in G.itertuples():
        x = sim(r.tk, r.d, r.ATRP)
        if x is None: continue
        ex, e = x; row = dict(d=r.d, mo=str(r.d.to_period('M')), BASE=ex['BASE'])
        # earliest exit wins: the dict stores each rule's first signal in time order of first firing; recompute order by firing index is not kept, so take min exit TIME via a second pass
        rows.append((r.tk, r.d, ex, e))
    return rows
def sim2(tk, d, atrp, rule):
    df, reg = cache[tk]; ix = df.index; i = ix.searchsorted(d); o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values; v = df.volume.values.astype(float); vm = pd.Series(v).rolling(20).mean().shift(1).values; e = o[i + 1]; fl = l[i]; touched = False; n = len(ix); rules = rule.split('+')
    for k in range(i + 1, n - 1):
        touched = touched or h[k] >= 1.10 * e
        sig = reg[k] == -1 or ('X1' in rules and c[k] < fl) or ('X2' in rules and c[k] < e * (1 - 1.5 * atrp)) or ('X3' in rules and c[k] < c[k - 1] and v[k] >= 2 * vm[k] and not touched) or ('X4' in rules and touched and c[k] < l[k - 1])
        if sig: return o[k + 1] / e - 1 - COST
    return None
RULES = ['BASE', 'X1', 'X2', 'X3', 'X4', 'X1+X4']
for cap in (5, 10):
    S = O[O.px < cap].dropna(subset=['ATRP']).copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[(~S.tr) & (S.s_ >= cut)]; R = []
    for r in G.itertuples():
        sim(r.tk, r.d, r.ATRP)
        if cache.get(r.tk) is None: continue
        res = {k: sim2(r.tk, r.d, r.ATRP, k) for k in RULES}
        if res['BASE'] is None: continue
        R.append(dict(d=r.d, mo=str(r.d.to_period('M')), **res))
    R = pd.DataFrame(R).dropna(); print(f'\n######## sub-${cap} 4-feature group, flips with a resolved baseline exit: {len(R)} of {len(G)} | structure_small r vs sim BASE mean {G.r.mean():+.1%} vs {R.BASE.mean():+.1%}')
    big = R.BASE >= .30
    for k in RULES:
        x = R[k]; dd = (R.groupby('mo')[k].mean() - R.groupby('mo').BASE.mean()).dropna(); t = dd.mean() / (dd.std() / np.sqrt(len(dd))) if len(dd) > 2 and dd.std() > 0 else np.nan
        e22 = R[R.d.dt.year == 2022]; e23 = R[R.d.dt.year >= 2023]
        print(f'  {k:7s} mean {x.mean():+6.1%} med {x.median():+.1%} win {(x > 0).mean():.0%} p10 {x.quantile(.1):+.0%} | +30% winners kept >=+30%: {(x[big] >= .30).mean():.0%} of {big.sum()} | 2022 {e22[k].mean():+.1%} (base {e22.BASE.mean():+.1%}) 2023-26 {e23[k].mean():+.1%} (base {e23.BASE.mean():+.1%}) | diff vs BASE {dd.mean():+.2%} t {t:+.2f}')
