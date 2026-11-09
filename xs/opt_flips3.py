"""Nov 9 2026: CALLS ON THE SCREENER'S FLIPS vs RANDOM ENTRIES WITH THE SAME HOLDING TIME. DECLARED BEFORE RESULTS.
xs/opt_flips2.py: calls / call spreads held to the screener's exit looked far better on 1h-track flips than 15m flips - but 1h trades are held longer and these names
rose a lot (hindsight universe), so a random long call held as long would also have made money. Fair test: for every flip, 3 random entry days in the SAME name and
SAME period (2019-21 / 2022-26) held for the SAME number of trading days. Edge = flip minus matched random.
Structures (real EOD bid / ask, xs/opt_flips.py): CALL (nearest strike at or above spot) and CDS (long that call, short the call at or above S (1 + sigma sqrt(td))),
WEEKLY and MONTHLY expiries, exit at the screener's exit close (or expiry). Every trade's risk is the premium paid (the stop is built in).
Report per track (15m / 1h) and fill (TOUCH / MID): mean, median, win rate, t (by month), flip - random; without NVDA and TSLA; by year; then a monthly book at
1 unit of premium per trade."""
_o = open('/home/claude/fartpie/xs/opt_flips.py').read(); exec(_o[:_o.rindex("if __name__ == '__main__':")])
if __name__ == '__main__':
    F = T[T.tk.isin(B)].copy(); F = F[F.d0 >= '2019-01-01']; rng = np.random.default_rng(1); rows = []
    for r_ in F.itertuples():
        bk = B[r_.tk]; d0, d1 = D(r_.d0), D(r_.d1); k0 = np.searchsorted(bk.days, d0); n = max(0, np.searchsorted(bk.days, d1) - k0)
        y0, y1 = (D('2019-01-01'), D('2022-01-01')) if r_.d0.year < 2022 else (D('2022-01-01'), D('2027-01-01'))
        pool = np.where((bk.days >= y0) & (bk.days < y1))[0]; pool = pool[pool + n < len(bk.days)]
        rows.append(('flip', r_.tk, d0, d1, bool(r_.h1), r_.d0.year))
        for k in rng.choice(pool, 3, replace=False): rows.append(('rand', r_.tk, bk.days[k], bk.days[k + n], bool(r_.h1), r_.d0.year))
    R = []
    for s, tk, d0, d1, h1, yr in rows:
        for ten, st, ex, fill, ret, win in entry_all(B[tk], d0, {'VEC': d1}):
            if st in ('CALL', 'CDS'): R.append((s, tk, d0, h1, yr, ten, st, fill, ret))
    R = pd.DataFrame(R, columns=['set', 'tk', 'd0', 'h1', 'yr', 'ten', 'st', 'fill', 'ret']); R['per'] = np.where(R.yr >= 2022, '22-26', '19-21')
    R['mo'] = pd.to_datetime(R.d0.values.astype('datetime64[D]')).to_period('M'); R.to_parquet('/tmp/claude-0/opt_flips3_R.parquet')
    def st_(t):
        m_ = t.groupby('mo').ret.mean(); tt = m_.mean() / m_.std() * np.sqrt(len(m_)) if len(m_) > 2 else np.nan
        return f'{t.ret.mean() * 100:+6.1f}% med {t.ret.median() * 100:+6.1f}% win {np.mean(t.ret > 0):3.0%} t {tt:+4.1f} n{len(t):5d}'
    for trk in (False, True):
        print(f'\n===== {"1h" if trk else "15m"}-track flips =====', flush=True)
        for fill in ('TOUCH', 'MID'):
            for ten in ('W', 'M'):
                for st in ('CALL', 'CDS'):
                    g = R[(R.h1 == trk) & (R.ten == ten) & (R.st == st) & (R.fill == fill)]
                    for per in ('19-21', '22-26'):
                        a = g[(g.set == 'flip') & (g.per == per)]; b = g[(g.set == 'rand') & (g.per == per)]
                        a2 = a[~a.tk.isin(['NVDA', 'TSLA'])]; b2 = b[~b.tk.isin(['NVDA', 'TSLA'])]
                        print(f'  {fill:<5} {ten} {st:<4} {per} | flip {st_(a)} | random {st_(b)} | edge {(a.ret.mean() - b.ret.mean()) * 100:+6.1f} '
                              f'| ex NVDA/TSLA edge {(a2.ret.mean() - b2.ret.mean()) * 100:+6.1f}', flush=True)
    print('\nedge by year (flip - matched random, % per trade), TOUCH:', flush=True)
    for trk in (False, True):
        for ten, st in (('W', 'CALL'), ('M', 'CALL'), ('W', 'CDS'), ('M', 'CDS')):
            g = R[(R.h1 == trk) & (R.ten == ten) & (R.st == st) & (R.fill == 'TOUCH')]
            e = g.groupby(['yr', 'set']).ret.mean().unstack(); ok_ = (e.flip - e.rand) * 100
            print(f'  {"1h " if trk else "15m"} {ten} {st:<4} ' + ' '.join(f'{y} {v:+5.1f}' for y, v in ok_.items()), flush=True)
    print('\nmonthly book, 1 unit of premium per trade, TOUCH (flips only):', flush=True)
    for trk in (False, True):
        for ten, st in (('W', 'CALL'), ('M', 'CALL'), ('W', 'CDS'), ('M', 'CDS')):
            g = R[(R.h1 == trk) & (R.ten == ten) & (R.st == st) & (R.fill == 'TOUCH') & (R.set == 'flip')]
            for per in ('19-21', '22-26'):
                m_ = g[g.per == per].groupby('mo').ret.agg(['sum', 'size'])
                print(f'  {"1h " if trk else "15m"} {ten} {st:<4} {per}: {m_["size"].mean():5.1f} trades/mo, avg month {m_["sum"].mean():+6.2f} units, months positive {np.mean(m_["sum"] > 0):3.0%}, '
                      f'worst {m_["sum"].min():+6.2f}, best {m_["sum"].max():+6.2f}', flush=True)
