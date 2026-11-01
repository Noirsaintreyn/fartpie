"""Nov 1 2026: which 'over-extended' measure best picks the short zone? Each is turned into its percentile vs its own past year (no look-ahead),
gate = top 10% / top 5% of days. Sources: SPY / QQQ / the stock's sector ETF / the stock itself, EMA z-score over 20 and 50 days; relative z
(stock z minus SPY z); 'herding' = average 20-day correlation between the 11 sector ETFs; and dispersion (cross-sector spread of 20d returns).
Shorts share the 6 long slots (no extra exposure). Avg short trade, then the combined book; top candidates through 40 matched runs."""
exec(open('/home/claude/fartpie/xs/baskets.py').read().split("L0 = dret(")[0])
import json
SEC = json.load(open('/home/claude/fartpie/sp500_sectors.json'))
ETF = {'Information Technology': 'XLK', 'Health Care': 'XLV', 'Financials': 'XLF', 'Consumer Discretionary': 'XLY', 'Communication Services': 'XLC',
       'Industrials': 'XLI', 'Consumer Staples': 'XLP', 'Energy': 'XLE', 'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB'}
def ed(t):
    if os.path.exists(f'{E}/{t}_yf.parquet'):
        x = pd.read_parquet(f'{E}/{t}_yf.parquet'); x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]; s = x['Close']; s.index = pd.to_datetime(s.index).normalize(); return s
    x = pd.read_parquet(f'{E}/{t}.parquet'); x['dt'] = pd.to_datetime(x.datetime).dt.normalize(); return x.set_index('dt').close
def pct(s): return s.rolling(252, min_periods=120).rank(pct=True)
C = {t: ed(t) for t in ETF.values()}; spy = ed('SPY'); qqq = ed('QQQ')
F = pd.DataFrame({'SPY z20': z(spy, 20), 'SPY z50': z(spy, 50), 'QQQ z20': z(qqq, 20), 'QQQ z50': z(qqq, 50)})
R20 = pd.DataFrame({t: c.pct_change() for t, c in C.items()}).dropna(how='all')
corr = R20.rolling(20).corr(); herd = corr.groupby(level=0).apply(lambda m: m.values[np.triu_indices(m.shape[0], 1)].mean() if m.notna().all().all() else np.nan)
F['herding'] = herd.shift(1); F['dispersion'] = -pd.DataFrame({t: c / c.shift(20) - 1 for t, c in C.items()}).std(axis=1).shift(1)   # low dispersion = high value
FP = F.apply(pct)
SZt = {t: pd.DataFrame({'z20': z(c, 20), 'z50': z(c, 50)}) for t, c in C.items()}
def per_trade(T):
    T = T.copy(); d = T.entry.dt.normalize()
    for c in FP: T[c] = FP[c].reindex(d.values, method='ffill').values
    for n in (20, 50):
        T[f'sector z{n}'] = [SZt[ETF[SEC.get(t)]][f'z{n}'].asof(x) if SEC.get(t) in ETF else np.nan for t, x in zip(T.tk, d)]
        T[f'stock z{n}'] = T[f'sz{n}']; T[f'rel z{n}'] = T[f'sz{n}'] - T[f'mz{n}']
    for c in ('sector z20', 'sector z50', 'stock z20', 'stock z50', 'rel z20', 'rel z50'):            # cross-sectional z -> percentile within each year
        T[c] = T[c].groupby(T.entry.dt.year).rank(pct=True)
    return T
Sh2 = per_trade(Sh)
COLS = list(FP.columns) + ['sector z20', 'sector z50', 'stock z20', 'stock z50', 'rel z20', 'rel z50']
L0 = dret(simw(Ls))
print(f'{"gate (top x% of its own history)":<34}{"avg short 19-21 / 22-26":>26}{"n":>12}   book (shared slots) 19-21 | 22-26 (return/DD)')
print(f'  {"live, no shorts":<32}{"":>38}   {fmt(stat(L0.cumsum()))}')
res = []
for c in COLS:
    for q in (.9, .95):
        g = (Sh2[c] >= q).fillna(False).values; s = Sh[g]
        a = s.groupby(np.where(s.entry.dt.year <= 2021, 0, 1)).ret.mean()
        st_ = stat(dret(simw(pd.concat([Ls, s]))).cumsum())
        res.append((c, q, st_)); print(f'  {c + f" top {1 - q:.0%}":<32}{a.get(0, np.nan) * 100:>+10.2f}% /{a.get(1, np.nan) * 100:>+6.2f}%{int(g.sum()):>10}   {fmt(st_)}', flush=True)
