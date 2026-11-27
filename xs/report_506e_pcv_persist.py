"""506e: PCV flips on the 14 perp names, cushion sizing, with and without the persistence throttle (half size when the trailing-N closed flips across the universe averaged < 0). N in 20/30/60 declared up front.
Train cohorts 2019-21, test 2022-25; also by purchase year."""
import sys
src = open('xs/report_506d_context_size.py').read()
exec(open('xs/report_506_perps.py').read().split("if __name__")[0])
PERP = {'NVDA', 'MU', 'AMD', 'GOOGL', 'TSLA', 'COIN', 'META', 'AVGO', 'AAPL', 'HOOD', 'ORCL', 'PLTR', 'AMZN', 'MSFT'}
exec('def coh2' + src.split('def coh2')[1].split('MULTS = ')[0])
T = SETS['PCV'].copy(); T = T[T.tk.isin(PERP)].sort_values('entry').reset_index(drop=True)
U = SETS['ALL flips'].sort_values('exit'); ex = U.exit.values; cr = np.cumsum(U.ret.values)
def trail(ts, N):
    j = np.searchsorted(ex, np.datetime64(ts), side='left'); return np.nan if j < 2 * N else (cr[j - 1] - cr[j - 1 - N]) / N
rows = []
for N in (20, 30, 60):
    T[f'bad{N}'] = [trail(t, N) < 0 for t in T.entry]
cfgs = [('flat', None)] + [(f'persist{N}', N) for N in (20, 30, 60)]
for era, (a, b) in (('train 2019-21', ('2019-02-01', '2021-12-31')), ('test 2022-25', ('2022-01-03', '2025-10-20'))):
    days = pd.date_range(a, b, freq='B')[::3]
    for lab, N in cfgs:
        mf = (lambda X: np.ones(len(X))) if N is None else (lambda X, N=N: np.where(X[f'bad{N}'].values, .5, 1.0))
        for c, K in ((6, 5), (10, 5)):
            R = [coh2(T, d, mf, (lambda bl, c=c: c * (bl - 9700)), K) for d in days]
            p = [r for r in R if r[0] == 'P']; f = [r for r in R if r[0] == 'F']
            rows.append(dict(era=era, ctx=lab, cushion=f'{c}x{K}', n=len(R), passed=round(100 * len(p) / len(R)), failed=round(100 * len(f) / len(R)), p_before_f=round(100 * len(p) / max(1, len(p) + len(f))),
                             pass90=round(100 * np.mean([1 if (r[0] == 'P' and r[1] <= 90) else 0 for r in R])), med_days=float(np.median([r[1] for r in p])) if p else None)); print(rows[-1], flush=True)
D = pd.DataFrame(rows); D.to_csv('xs/report_506e_pcv_persist.csv', index=False)
# by purchase year, cushion 6x5, flat vs persist30
yr = []
for y in range(2019, 2026):
    days = pd.date_range(f'{y}-01-02', f'{y}-12-31', freq='B')[::3]; days = days[days <= pd.Timestamp('2025-10-20')]
    for lab, N in (('flat', None), ('persist30', 30)):
        mf = (lambda X: np.ones(len(X))) if N is None else (lambda X, N=N: np.where(X[f'bad{N}'].values, .5, 1.0))
        R = [coh2(T, d, mf, (lambda bl: 6 * (bl - 9700)), 5) for d in days]; p = [r for r in R if r[0] == 'P']; f = [r for r in R if r[0] == 'F']
        yr.append(dict(year=y, ctx=lab, n=len(R), passed=round(100 * len(p) / len(R)), failed=round(100 * len(f) / len(R)), open_=round(100 * (len(R) - len(p) - len(f)) / len(R)), pass90=round(100 * np.mean([1 if (r[0] == 'P' and r[1] <= 90) else 0 for r in R]))))
print(pd.DataFrame(yr).to_string()); pd.DataFrame(yr).to_csv('xs/report_506e_by_year.csv', index=False)
