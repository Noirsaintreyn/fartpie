"""Nov 4 2026: Dow / Nasdaq-100 names competing for the SAME 6 slots (single book, ratio tie-break, live exits, 8% breaker). DECLARED BEFORE RESULTS; judged on 2022-26 AND no harm in 2019-21.
Live = S&P (sp flag) ratio >= 1.5. Candidates added (names compete by ratio, so low-ratio adds only fill free slots):
 UA live + Dow names with ratio >= 1.0 | UB live + Dow names with ratio >= 0.5 | UC Dow + Nasdaq-100 (ratio >= 1.5, single book) | UD live + Nasdaq-100 (>= 1.5) + Dow (>= 1.0) | UE live + Nasdaq-100 (>= 1.5) | UF live + Dow 15m-track trades (>= 1.0)
PLUS diagnostics: signals per year, qualifying names per year, share of days with >= 6 open signals (book full) in the live set."""
_s = open('xs/universe_test.py').read(); exec(_s[:_s.index("B0 = ev(W[W.sp]")].replace(" & (W.ratio >= 1.5)].copy()", "].copy()").replace("print('unclassified", "pass #").replace("print('parity", "pass #"))
W = W[W.ratio.notna()].reset_index(drop=True); LIVE = W[W.sp & (W.ratio >= 1.5)].reset_index(drop=True)
print('LIVE trades', len(LIVE), '(parity 17447)')
print('\nDIAGNOSTICS (live set): year | signals | tickers qualifying | days with >=6 signals open | avg open signals')
days = pd.bdate_range('2019-01-01', '2026-10-20'); op = np.zeros(len(days))
for e, x in zip(LIVE.entry.dt.normalize().values, LIVE.exit.dt.normalize().values): op[(days >= e) & (days <= x)] += 1
for y in range(2019, 2027):
    m = days.year == y; g = LIVE[LIVE.entry.dt.year == y]; print(f'  {y}: signals {len(g):>5} tickers {g.tk.nunique():>3}  days>=6 open {np.mean(op[m] >= 6):.0%}  avg open {op[m].mean():.1f}  avg ratio {g.ratio.mean():.1f}  avg trade {g.ret.mean()*100:+.2f}%')
def cat(*fs): return pd.concat(fs).drop_duplicates(['tk', 'entry', 'h1']).sort_values('entry').reset_index(drop=True)
D1 = W[W.dow & (W.ratio >= 1.0)]; D05 = W[W.dow & (W.ratio >= .5)]; N15 = W[W.ndx & (W.ratio >= 1.5)]; D15 = W[W.dow & ~W.h1 & (W.ratio >= 1.0)]
U = {'UA live + Dow(>=1.0)': cat(LIVE, D1), 'UB live + Dow(>=0.5)': cat(LIVE, D05), 'UC Dow + Nasdaq-100 (>=1.5)': cat(W[W.dow & (W.ratio >= 1.5)], N15), 'UD live + Nasdaq-100 + Dow(>=1.0)': cat(LIVE, N15, D1),
     'UE live + Nasdaq-100 (>=1.5)': cat(LIVE, N15), 'UF live + Dow 15m only(>=1.0)': cat(LIVE, D15)}
B0 = ev(LIVE); print('\n40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
for nm, Y in U.items():
    a = Y[Y.entry < '2022']; b = Y[Y.entry >= '2022']; print(f'  {nm}: {Y.tk.nunique()} tickers, {len(Y)} trades (live {len(LIVE)}), avg {a.ret.mean()*100:+.2f}% / {b.ret.mean()*100:+.2f}%', flush=True); line(nm, ev(Y), B0)
