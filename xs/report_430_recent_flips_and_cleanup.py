"""Nov 24 2026 report 430.  (1) Re-state the report-429 candidate WITHOUT 2026 (gex files end 2025-12-30; earlier reports forward-filled stale gamma into 2026 = 72 of 2,134 entries).  (2) What 2026 / the last months look like for the live book (gated 15m flips, vector exit, 6 slots, true daily MTM) and for the same book with the per-name vol-relief gate N2 (report 404), by month, vs SPY.  No parameters changed; descriptive."""
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
T = pd.read_parquet(SP + 'g429.parquet'); T = T[T.entry < '2026-01-01']
for gk, xk in (('G20', 'X3'), ('G20', 'X1')):
    B = T[(T.gate == gk) & (T.exit == xk)]; on = B[~B.N2up]; print(f'[ex-2026] {gk} {xk}: base n{len(B)} win {(B.ret>0).mean()*100:.1f} med {B.ret.median()*100:.2f} mean {B.ret.mean()*100:.2f} | N2dn n{len(on)} win {(on.ret>0).mean()*100:.1f} med {on.ret.median()*100:.2f} mean {on.ret.mean()*100:.2f} | diff {(on.ret.mean()-B[B.N2up].ret.mean())*100:+.2f}pp')
_t = open('xs/report_404_name_vol_relief.py').read(); exec(_t[:_t.rindex("rows = []; TK0, s0")])
TK0, s0 = live_fn(B0); Bn = run_gate_name(N2); TKn, sn = live_fn(Bn)
M_ = pd.read_parquet(SP + 'macro_daily.parquet'); spy = M_[M_.symbol == 'SPY'].assign(d=lambda x: pd.to_datetime(x.ts).dt.normalize()).drop_duplicates('d').set_index('d').close.sort_index().pct_change().reindex(s0.index).fillna(0)
mo = lambda s: ((1 + s).resample('ME').prod() - 1) * 100
tab = pd.DataFrame({'live': mo(s0), 'live+N2': mo(sn), 'SPY': mo(spy)}); print('\nMonthly % (true MTM, 6 slots, 2bp each side):'); print(tab[tab.index >= '2025-07-01'].round(1).to_string())
for lab, a in (('2026 YTD', '2026-01-01'), ('last 3 months of book data', s0.index.max() - pd.Timedelta(days=92)), ('2025', None)):
    if lab == '2025': m = (s0.index >= '2025-01-01') & (s0.index < '2026-01-01')
    else: m = s0.index >= pd.Timestamp(a)
    f = lambda s: ((1 + s[m]).prod() - 1) * 100; dd = lambda s: ((1 + s[m]).cumprod() / (1 + s[m]).cumprod().cummax() - 1).min() * 100
    print(f"{lab:28s} live {f(s0):6.1f}% (MDD {dd(s0):5.1f}) | live+N2 {f(sn):6.1f}% (MDD {dd(sn):5.1f}) | SPY {f(spy):6.1f}% (MDD {dd(spy):5.1f})")
print('book data ends', s0.index.max().date())
tr = pd.DataFrame(TK0, columns=['tk', 'entry', 'exit', 'ret', 'ra']) if len(TK0[0]) == 5 else None
if tr is not None: pass
n = pd.DataFrame([dict(tk=t[0], entry=t[1], exit=t[2], ret=t[3]) for t in TK0]); n['n'] = n.ret - 4e-4
for lab, a, b in (('2019-25', '2019-01-01', '2026-01-01'), ('2026 YTD', '2026-01-01', '2027-01-01'), ('2026 May-Aug', '2026-05-01', '2027-01-01')): d = n[(n.entry >= a) & (n.entry < b)]; print(f"live-book trades {lab:13s} n{len(d):5d} win {(d.n>0).mean()*100:5.1f} med {d.n.median()*100:5.2f} mean {d.n.mean()*100:5.2f}")
