"""Nov 9 2026: SWAP ONLY FOR 'HIGH-PROBABILITY' FLIPS (Rey) - rough, Yahoo hourly prices, Dec 2 2024 - Dec 30 2025 (per-stock GEX ends Dec 2025). DECLARED BEFORE RESULTS.
When all 6 slots are full and a new flip arrives that is high-probability, sell the weakest holding (lowest open P&L at that moment) and take the new one:
  S-NEG  new flip's stock in negative gamma (Rey's idea)            S-NEG-L  same, only if the weakest holding is losing
  S-POS  new flip's stock in positive gamma (did better per trade)   S-POS-L  same, only if the weakest holding is losing
Only the 18 names with GEX can trigger a swap. Live ratio order otherwise. Whole window only (too short to split)."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_intraday_1h.py').read(); exec(_s[:_s.index("\nif __name__ == '__main__':")])
GX = []
for f in glob.glob('/mnt/user-data/uploads/fartpie/xs/ext/gex/*.csv'):
    g = pd.read_csv(f, parse_dates=['date']).sort_values('date'); g['tk'] = f.split('/')[-1][:-4]; g['nxt'] = g.date.shift(-1); GX.append(g[['tk', 'nxt', 'net_gex']].dropna(subset=['nxt']))
GX = pd.concat(GX).set_index(['tk', 'nxt']).net_gex
T0['gx'] = [GX.get((t, d), np.nan) for t, d in zip(T0.tk, T0.day)]; T0 = T0[T0.entry < '2025-12-31']; D = D[D < '2025-12-31']; J = J.loc[D]
def taken_hp(X, rule=None, N=6, brk=.08):
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []; nrep = 0
    for r in X.itertuples():
        e = r.entry
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr, _h in done: real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
        op = [h for h in op if h['x'] > e]
        hp = np.isfinite(r.gx) and ((r.gx < 0) if rule and 'NEG' in rule else (r.gx > 0))
        if len(op) >= N and rule and hp:
            cand = [(price_at(h['tk'], e) / h['px'] - 1, h) for h in op if h['e'] < e and np.isfinite(h['px'])]; cand = [c_ for c_ in cand if np.isfinite(c_[0])]
            if rule.endswith('-L'): cand = [c_ for c_ in cand if c_[0] < 0]
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
    X0 = order('ratio'); print(f'candidates {len(T0):,}, with GEX {T0.gx.notna().mean():.0%} | Dec 2024 - Dec 2025: return / max DD', flush=True)
    for rule in (None, 'S-NEG', 'S-NEG-L', 'S-POS', 'S-POS-L'):
        tk_, nrep = taken_hp(X0, rule); s_ = daily(tk_); e = (1 + s_).cumprod()
        print(f'  {str(rule or "no swapping (live)"):<20} {e.iloc[-1] - 1:+6.1%} / DD {(e / e.cummax() - 1).min():6.1%} | swaps {nrep}', flush=True)
