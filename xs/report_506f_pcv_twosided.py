"""506f: PCV, cushion sizing (6x/10x, 5 slots), TWO-SIDED persistence: x0.5 when trailing-N mean < 0, x1.5 when > 0 (declared before looking). Persistence measured on (a) all universe flips, (b) PCV's own flips (all 54 names). N = 20/30/60. Train 2019-21 / test 2022-25 / by year."""
src = open('xs/report_506d_context_size.py').read()
exec(open('xs/report_506_perps.py').read().split("if __name__")[0])
PERP = {'NVDA', 'MU', 'AMD', 'GOOGL', 'TSLA', 'COIN', 'META', 'AVGO', 'AAPL', 'HOOD', 'ORCL', 'PLTR', 'AMZN', 'MSFT'}
exec('def coh2' + src.split('def coh2')[1].split('MULTS = ')[0])
T = SETS['PCV'].copy(); T = T[T.tk.isin(PERP)].sort_values('entry').reset_index(drop=True)
def mk(S):
    U = S.sort_values('exit'); return U.exit.values, np.cumsum(U.ret.values)
SRC = {'universe': mk(SETS['ALL flips']), 'pcv_own': mk(SETS['PCV'])}
def trail(ts, N, key):
    ex, cr = SRC[key]; j = np.searchsorted(ex, np.datetime64(ts), side='left'); return np.nan if j < 2 * N else (cr[j - 1] - cr[j - 1 - N]) / N
for key in SRC:
    for N in (20, 30, 60): T[f'{key}{N}'] = [trail(t, N, key) for t in T.entry]
cfgs = [('flat', None, None)] + [(f'{k} N={N}', k, N) for k in SRC for N in (20, 30, 60)]
def mf_for(k, N):
    if k is None: return lambda X: np.ones(len(X))
    return lambda X: np.where(X[f'{k}{N}'].values < 0, .5, 1.5)
rows = []
for era, (a, b) in (('train 2019-21', ('2019-02-01', '2021-12-31')), ('test 2022-25', ('2022-01-03', '2025-10-20'))):
    days = pd.date_range(a, b, freq='B')[::3]
    for lab, k, N in cfgs:
        for c, K in ((6, 5), (10, 5)):
            R = [coh2(T, d, mf_for(k, N), (lambda bl, c=c: c * (bl - 9700)), K) for d in days]
            p = [r for r in R if r[0] == 'P']; f = [r for r in R if r[0] == 'F']
            rows.append(dict(era=era, ctx=lab, cushion=f'{c}x{K}', n=len(R), passed=round(100 * len(p) / len(R)), failed=round(100 * len(f) / len(R)), p_before_f=round(100 * len(p) / max(1, len(p) + len(f))),
                             pass90=round(100 * np.mean([1 if (r[0] == 'P' and r[1] <= 90) else 0 for r in R])), med_days=float(np.median([r[1] for r in p])) if p else None)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('xs/report_506f_pcv_twosided.csv', index=False)
yr = []
for y in range(2019, 2026):
    days = pd.date_range(f'{y}-01-02', f'{y}-12-31', freq='B')[::3]; days = days[days <= pd.Timestamp('2025-10-20')]
    for lab, k, N in (('flat', None, None), ('universe N=30', 'universe', 30), ('pcv_own N=30', 'pcv_own', 30)):
        R = [coh2(T, d, mf_for(k, N), (lambda bl: 6 * (bl - 9700)), 5) for d in days]; p = [r for r in R if r[0] == 'P']; f = [r for r in R if r[0] == 'F']
        yr.append(dict(year=y, ctx=lab, passed=round(100 * len(p) / len(R)), failed=round(100 * len(f) / len(R)), pass90=round(100 * np.mean([1 if (r[0] == 'P' and r[1] <= 90) else 0 for r in R]))))
print(pd.DataFrame(yr).pivot_table(index='ctx', columns='year', values=['passed', 'failed'], aggfunc='first').to_string()); pd.DataFrame(yr).to_csv('xs/report_506f_by_year.csv', index=False)
