"""506b: same sim, restricted to names that are on MyFundedPerps today (screenshot Nov 27) AND in the backtest, broken out by purchase year."""
import sys; sys.argv = ['x']
exec(open('xs/report_506_perps.py').read().split("if __name__")[0])
PERP = {'NVDA', 'MU', 'AMD', 'GOOGL', 'TSLA', 'COIN', 'META', 'AVGO', 'AAPL', 'HOOD', 'ORCL', 'PLTR', 'AMZN', 'MSFT'}
rows = []
for uni, f in (('perp14', lambda T: T[T.tk.isin(PERP)]), ('all54', lambda T: T)):
    for name in ('ALL flips', 'tagged (PCV,PCZ,SKW,CHM)', 'PCZ', 'SKW', 'PCV', 'CHM'):
        T = f(SETS[name])
        for S, K in ((2000, 5), (3000, 5), (4000, 4)):
            T2 = T.sort_values('entry'); days = pd.date_range('2019-02-01', '2026-04-20', freq='B')[::2]
            R = [(d, cohort(T2, d, S, K)) for d in days]
            for y in sorted({d.year for d, _ in R}) + ['ALL']:
                Q = [r for d, r in R if y == 'ALL' or d.year == y]
                P = np.array([r['passed'] if r['passed'] is not None else np.nan for r in Q], float); Y = np.array([r['pay'] if r['pay'] is not None else np.nan for r in Q], float); Fl = np.array([r['fail'] if r['fail'] is not None else np.nan for r in Q], float)
                rows.append(dict(uni=uni, set=name, size=S, slots=K, year=y, n=len(Q), trades=int(len(T2[T2.entry.dt.year == y])) if y != 'ALL' else len(T2), pass90=round(float(np.mean(P <= 90)) * 100), pay90=round(float(np.mean(Y <= 90)) * 100),
                                 pay180=round(float(np.mean(Y <= 180)) * 100), fail90=round(float(np.mean(Fl <= 90)) * 100), med_pay=float(np.nanmedian(Y)) if (~np.isnan(Y)).any() else None))
    print(uni, 'done', flush=True)
D = pd.DataFrame(rows); D.to_csv('xs/report_506b_perps_by_year.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
for uni in ('perp14', 'all54'):
    for m in ('pay180', 'fail90'):
        print('\n', uni, m); print(D[(D.uni == uni) & (D['size'] == 3000)].pivot_table(index='set', columns='year', values=m, aggfunc='first').to_string())
print(D[(D.uni == 'perp14') & (D.year == 'ALL')].to_string())
