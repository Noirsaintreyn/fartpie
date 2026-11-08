"""Nov 8 2026: THE 'NEED A FEW POINTS' TRADE - take the vector flip with a take profit set by volatility. DECLARED BEFORE RESULTS.
Every RTH 15m flip (and the Rule A subset): target = f x 20-day average daily range (NQ points, known at the prior close), f = 0.10 / 0.15 / 0.20 / 0.30;
stop = 1x or 2x the target; exit early if the 15m vector flips back; flat 15:55; <= 4 a day, no overlap; 1 pt cost.
Part 1 per trade: win rate, average, by period. Part 2 as a quota filler on top of the best own-account book (C30 composite trend-hold + 4 x Rule A):
  on a day that starts with month-to-date below 7.5 a day x days so far, also trade the filler (1 / 2 contracts); vs never and vs always.
Floor stats as xs/nq_floor.py (months >= 7.5 a day, losing months, worst month, worst 6 months, worst year, max drawdown)."""
_f = open('xs/nq_floor.py').read(); exec(_f[:_f.rindex("if __name__ == '__main__':")])
ADR = X.adr.values; yy = pd.DatetimeIndex(X.day).year.values
def filler(f, sm, mask):
    rows = []
    for day, g in X[mask].groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            if n >= 4 or r.j <= busy or not np.isfinite(ADR[r.Index]) or ADR[r.Index] <= 0: continue
            T = f * ADR[r.Index]; res = sim(int(r.j), int(r.dir), T, sm * T, 955, True)
            if res is None: continue
            rows.append((day, res[0] - 1.0, r.Index)); busy = res[1]; n += 1
    return pd.DataFrame(rows, columns=['day', 'pts', 'i'])
if __name__ == '__main__':
    m22 = X.day >= '2022-01-01'; print(f'20-day ADR, NQ points: median 2018-21 {np.nanmedian(ADR[~m22.values]):.0f}, 2022-26 {np.nanmedian(ADR[m22.values]):.0f}', flush=True)
    print('\nPART 1 per trade (n, win, avg pts) | 2018-21 | 2022-26 | by year', flush=True)
    F = {}
    for lab, mask in (('every flip', np.ones(len(X), bool)), ('Rule A flips', (X.ruleA == 1).values)):
        for f in (.10, .15, .20, .30):
            for sm in (1, 2):
                T = filler(f, sm, mask); F[(lab, f, sm)] = T; ty = pd.DatetimeIndex(T.day).year
                a, b = T[ty < 2022].pts, T[ty >= 2022].pts
                print(f'  {lab:<12} target {f:.2f} x ADR, stop {sm}x: n{len(T):4d} win {np.mean(T.pts > 0):3.0%} avg {T.pts.mean():+5.1f} | {a.mean():+5.1f} | {b.mean():+5.1f} | ' +
                      ' '.join(f'{T[ty == y].pts.mean():+5.1f}' for y in range(2018, 2027)), flush=True)
    print('\nPART 2 quota filler on C30 + 4 x Rule A (every flip filler)', flush=True)
    MAIN = C30 + 4 * RA
    print(f'  {"version":<40} avg/day  months>=7.5/day  losing months  worst month  worst 6-mo/day  worst year/day  max DD  Sharpe', flush=True)
    def show_(nm, P):
        r = stats(P); print(f'  {nm:<40} {r["avg"]:+7.2f} {r["m75"]:15.0%} {r["mneg"]:14.0%} {r["wm"]:+12.0f} {r["w6"]:+15.2f} {r["wy"]:+10.2f} ({r["wyy"]}) {r["dd"]:+7.0f} {r["sh"]:7.2f}', flush=True)
    show_('MAIN (C30 + 4 x RA), no filler', MAIN)
    for f, sm in ((.10, 2), (.15, 2), (.20, 1), (.30, 1)):
        T = F[('every flip', f, sm)]; dp = T.groupby(pd.DatetimeIndex(T.day)).pts.sum().reindex(AD).fillna(0.0)
        for k in (1, 2):
            show_(f'+ filler always, {f:.2f} ADR / stop {sm}x, {k} ct', MAIN + k * dp)
            out = []; mtd = 0.0; nd = 0; cur = None
            for d in AD:
                if cur != d.to_period('M'): cur = d.to_period('M'); mtd = 0.0; nd = 0
                behind = mtd < 7.5 * nd; v = MAIN.loc[d] + (k * dp.loc[d] if behind else 0.0); out.append(v); mtd += v; nd += 1
            show_(f'+ filler when behind, {f:.2f} ADR / stop {sm}x, {k} ct', pd.Series(out, index=AD))
