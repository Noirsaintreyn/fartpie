"""Nov 9 2026: SWAP FOR THE CHOSEN GAMMA COMBO (xs/scr_gamma_combo.py) - rough, Yahoo hourly prices, Dec 2 2024 - Dec 30 2025. DECLARED BEFORE RESULTS.
When all 6 slots are full and a new flip carries the flag, sell the weakest LOSING holding and take it. Flags: the chosen 'POS + near 52w high' (declared);
'POS + top-third ratio' (EXPLORATORY: it had the biggest edge in both periods but was not the 2019-22 pick)."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_hp_swap.py').read(); exec(_s[:_s.index("\nif __name__ == '__main__':")])
HI = {}
for tk in T0.tk.unique():
    try: Yd = pd.read_parquet(f'/home/claude/fartpie/data/yahoo/daily_v/{tk}.parquet')
    except Exception: continue
    h52 = (Yd.close / Yd.close.rolling(252, min_periods=200).max()); HI[tk] = pd.Series(h52.values, index=Yd.index).shift(1)   # prior close, used for the next session
T0['hi52'] = [HI[t].asof(d) if t in HI else np.nan for t, d in zip(T0.tk, T0.day)]; T0['f_hi52'] = (T0.gx > 0) & (T0.hi52 >= .95)
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[W.sp & (W.ratio >= 1.5) & (W.entry < '2023-01-01') & (W.entry >= '2019-01-01')]; raq = W.ratio.quantile(2 / 3)
T0['f_ratio'] = (T0.gx > 0) & (T0.ratio >= raq)
def taken_f(X, col, N=6, brk=.08):
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []; nrep = 0
    for r in X.itertuples():
        e = r.entry
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr, _h in done: real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
        op = [h for h in op if h['x'] > e]
        if len(op) >= N and col and getattr(r, col):
            cand = [(price_at(h['tk'], e) / h['px'] - 1, h) for h in op if h['e'] < e and np.isfinite(h['px'])]; cand = [c_ for c_ in cand if np.isfinite(c_[0]) and c_[0] < 0]
            if cand:
                cur, h = min(cand, key=lambda z: z[0]); op.remove(h); pend = [p for p in pend if p[2] is not h]; nrep += 1
                rr = (1 + cur) * (1 - 2e-4) - 1; out[h['k']] = (h['tk'], h['e'], e, rr); real += np.log1p(rr / N); hist.append((e, real)); win.append((e, real))
        if len(op) >= N: continue
        if brk and hist:
            t0 = e - pd.Timedelta(days=28)
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [hh for _, hh in win]) < np.log1p(-brk): continue
        h = dict(tk=r.tk, e=e, x=r.exit, px=r.px, k=len(out)); op.append(h); pend.append((r.exit, r.ret, h)); out.append((r.tk, e, r.exit, r.ret))
    return out, nrep
if __name__ == '__main__':
    X0 = order('ratio'); print(f'candidates {len(T0):,} | flagged: POS + near 52w high {T0.f_hi52.sum()}, POS + top-third ratio {T0.f_ratio.sum()} | Dec 2024 - Dec 2025', flush=True)
    for lab, col in (('no swapping (live)', None), ('swap for POS + near 52w high (declared)', 'f_hi52'), ('swap for POS + top-third ratio (exploratory)', 'f_ratio')):
        tk_, nrep = taken_f(X0, col); s_ = daily(tk_); e = (1 + s_).cumprod(); print(f'  {lab:<46} {e.iloc[-1] - 1:+6.1%} / DD {(e / e.cummax() - 1).min():6.1%} | swaps {nrep}', flush=True)
