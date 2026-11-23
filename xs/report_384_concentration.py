"""Nov 23 2026 report 384. Concentration stress on the live vector book (b302, 54 names, 6 slots, 8% breaker).
(a) remove the top-5 contributors (NVDA, CMG, AAPL, FDX, BA) -> re-run the full slot sim; (b) remove 5 random names x200 draws (null);
(c) leave-one-name-out over all 54.  Kill rule (pre-set): Sharpe < 1.0 without top five => edge is five stocks."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
def run(Bs):
    TK, _ = sim_swap(Bs.reset_index(drop=True), None); z = spread(TK); z = z[z.index >= '2019-01-01']; a, b = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']
    return dict(n=len(TK), cagr=cagr(z), dd=mdd(z), sh=sharpe(z.values), sh_tr=sharpe(a.values), sh_te=sharpe(b.values), cg_tr=cagr(a), cg_te=cagr(b))
def fmt(l, r): return f"{l:28s} trades {r['n']:4d} CAGR {r['cagr']:5.1%} DD {r['dd']:6.1%} Sharpe {r['sh']:.2f} | train CAGR {r['cg_tr']:5.1%} Sh {r['sh_tr']:.2f} | test CAGR {r['cg_te']:5.1%} Sh {r['sh_te']:.2f}"
base = run(B0); print(fmt('BASE (54 names)', base), flush=True)
top5 = ['NVDA', 'CMG', 'AAPL', 'FDX', 'BA']; print(fmt('minus top5', run(B0[~B0.tk.isin(top5)])), flush=True)
rng = np.random.default_rng(7); names = sorted(B0.tk.unique()); rest = [n for n in names if n not in top5]; res = []
for i in range(60):
    d = list(rng.choice(names, 5, replace=False)); res.append(run(B0[~B0.tk.isin(d)]))
R = pd.DataFrame(res); print('random-5 removal x60: Sharpe median %.2f [5%% %.2f, 95%% %.2f]; CAGR median %.1f%% [%.1f, %.1f]; test Sharpe median %.2f' % (R.sh.median(), R.sh.quantile(.05), R.sh.quantile(.95), R.cagr.median() * 100, R.cagr.quantile(.05) * 100, R.cagr.quantile(.95) * 100, R.sh_te.median()), flush=True)
lo = []
for n in names: r = run(B0[B0.tk != n]); lo.append((n, r['sh'], r['cagr'], r['sh_te'], r['sh_tr']))
L = pd.DataFrame(lo, columns=['tk', 'sh', 'cagr', 'sh_te', 'sh_tr']).sort_values('sh'); print('leave-one-out Sharpe: min %.2f (%s) max %.2f (%s) base %.2f; CAGR min %.1f%% max %.1f%%' % (L.sh.min(), L.tk.iloc[0], L.sh.max(), L.tk.iloc[-1], base['sh'], L.cagr.min() * 100, L.cagr.max() * 100))
print('lowest 5 (most important names):', L.head(5)[['tk', 'sh', 'cagr']].round(3).values.tolist()); print('highest 5 (names that hurt):', L.tail(5)[['tk', 'sh', 'cagr']].round(3).values.tolist())
