"""506c: prop eval as first passage. P(+900 before floor/daily breach), horizon 365d, fixed size sweep and equity-scaled size (size = f * cushion above floor, capped)."""
import sys
exec(open('xs/report_506_perps.py').read().split("if __name__")[0])
PERP = {'NVDA', 'MU', 'AMD', 'GOOGL', 'TSLA', 'COIN', 'META', 'AVGO', 'AAPL', 'HOOD', 'ORCL', 'PLTR', 'AMZN', 'MSFT'}
def coh(X, start, size_fn, K, horizon=365):
    X = X[(X.entry >= start) & (X.entry < start + pd.Timedelta(days=horizon))]
    ev = sorted([(r.entry, 0, i) for i, r in enumerate(X.itertuples())] + [(r.exit, 1, i) for i, r in enumerate(X.itertuples())])
    rets = X.ret.values; hold = ((X.exit - X.entry).dt.total_seconds() / 86400).values; sz = {}; op = set(); bal = 10000.0; cur = None; ds = bal
    pn = lambda i: sz[i] * (rets[i] - 2 * COM - SWAP * hold[i])
    for ts, k, i in ev:
        d = ts.normalize()
        if d != cur: cur = d; ds = bal + sum(min(0, pn(j)) for j in op)
        if k == 0:
            if len(op) >= K: continue
            sz[i] = size_fn(bal); op.add(i)
        else:
            if i not in sz or i not in op: continue
            op.discard(i); bal += pn(i)
            if bal >= 10900: return 'P', (ts - start).days
        eq = bal + sum(min(0, pn(j)) for j in op)
        if eq < 9700 or eq < ds - 300: return 'F', (ts - start).days
    return 'N', None
def evalset(T, size_fn, K, step=3):
    T = T.sort_values('entry'); days = pd.date_range('2019-02-01', '2025-10-20', freq='B')[::step]
    R = [coh(T, d, size_fn, K) for d in days]; p = [r for r in R if r[0] == 'P']; f = [r for r in R if r[0] == 'F']; n = [r for r in R if r[0] == 'N']
    return dict(n=len(R), pass_=round(100 * len(p) / len(R)), fail=round(100 * len(f) / len(R)), open_=round(100 * len(n) / len(R)), p_before_f=round(100 * len(p) / max(1, len(p) + len(f))),
                pass90=round(100 * np.mean([1 if (r[0] == 'P' and r[1] <= 90) else 0 for r in R])), med_pass_days=float(np.median([r[1] for r in p])) if p else None)
if __name__ == '__main__':
    rows = []
    for uni, f in (('perp14', lambda T: T[T.tk.isin(PERP)]),):
        for name in ('ALL flips', 'PCV', 'tagged (PCV,PCZ,SKW,CHM)', 'PCZ'):
            T = f(SETS[name])
            cfgs = [(f'fixed {S}x{K}', (lambda b, S=S: S), K) for S, K in ((1500, 5), (3000, 5), (4000, 5), (5000, 4), (6000, 4), (7500, 4), (10000, 3))]
            cfgs += [(f'cushion {c}x{K}', (lambda b, c=c, K=K: min(30000 / K, c * (b - 9700))), K) for c, K in ((6, 5), (10, 5), (15, 4))]
            for lab, fn, K in cfgs:
                r = evalset(T, fn, K); rows.append(dict(uni=uni, set=name, cfg=lab, **r)); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv('xs/report_506c_first_passage.csv', index=False)
