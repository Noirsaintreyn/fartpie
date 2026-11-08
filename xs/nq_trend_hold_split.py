"""Nov 5 2026: where do the trend-hold points come from? DECOMPOSITION, not a new rule: D0 entries split by window (AM 09:45-11:30 / PM 11:45-15:30),
and the obvious combination: today's Rule A trades (AM, 75 / 50) + trend-hold on PM flips only. NQ points per contract after costs."""
_th = open('xs/nq_trend_hold.py').read(); exec(_th[:_th.index("rng = np.random.default_rng(9)")])
L = pd.read_csv('live_data/nq_rule_a_trades.csv', parse_dates=['day']); RA = L.groupby('day').net_pts.sum()
def show(nm, P):
    P = P.reindex(ALLD).fillna(0.0); M = P.groupby(P.index.to_period('M')).sum(); cum = P.cumsum(); yr = P.groupby(P.index.year).sum()
    print(f'  {nm:<34} {P.mean():+6.2f}/day | {M.mean():+6.1f}/month | months >= 50 {(M >= 50).mean():4.0%} | worst month {M.min():+5.0f} | max DD {(cum - cum.cummax()).min():+6.0f} | Sharpe {P.mean() / P.std() * np.sqrt(252):.2f} | losing years {int((yr < 0).sum())}', flush=True)
    print('      ' + ' '.join(f'{y} {v:+.0f}' for y, v in yr.items()), flush=True)
day_pts = lambda T: T.groupby('day').pts.sum()
AMf, PMf = FL[FL.me <= 690], FL[FL.me > 690]
show('D0 AM flips only (trend-hold)', day_pts(runs(AMf, one))); PMd = day_pts(runs(PMf, one)); show('D0 PM flips only (trend-hold)', PMd)
show('Rule A (current system)', RA); show('Rule A + PM trend-hold', RA.reindex(ALLD).fillna(0) + PMd.reindex(ALLD).fillna(0))
# Rule A entries (same 5m entry bars) held with the trend-hold exit instead of 75 / 50 + noon flat
LA = pd.read_csv('live_data/nq_rule_a_trades.csv', parse_dates=['day', 'entry']); keys = set(zip(LA.entry.values.astype('datetime64[ns]'), LA.dir.values))
RAx = FL[[(t5[j], d) in keys for j, d in zip(FL.j.values, FL.dir.values)]]
print(f'  matched {len(RAx)} of {len(LA)} Rule A entries to flips', flush=True)
show('Rule A entries, trend-hold exit', day_pts(runs(RAx, one))); show('Rule A entries, trend-hold + vol stop', day_pts(runs(RAx, one, True)))
