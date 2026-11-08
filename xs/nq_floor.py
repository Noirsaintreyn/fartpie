"""Nov 8 2026: THE FLOOR - keep the bad stretches at or above 7.5 NQ points a day by making more on the good ones. DECLARED BEFORE RESULTS.
Own account (no prop rules), 2018 - Oct 2026, NQ points after costs. Streams:
  C30  composite trend-hold: contracts = checks - 3 (9 checks), stop 30, x1.67 contracts     C50  same, stop 50
  A30  six-check trend-hold (A6x3), stop 30, x1.67                                          RA   prop-style Rule A trades (75 / 50, flat 12:00), 1 contract
  combos: C30 + 2 x RA, C30 + 4 x RA, C50 + 2 x RA
Per stream: average a day; share of months averaging >= 7.5 a day; share of losing months; worst month; worst 3-month and 6-month average a day; worst calendar year
(average a day); max drawdown; Sharpe. Then the size multiple that lifts the worst 6-month stretch to 7.5 a day, and the drawdown that comes with it."""
_t = open('xs/nq_composite_th.py').read(); exec(_t[:_t.rindex("if __name__ == '__main__':")])
AD = ALLD[(ALLD >= '2018-01-01')]
def stream(sv, sz): return book(sv, sz, '2018-01-01', '2027-01-01')[0].reindex(AD).fillna(0.0)
RAt = pd.read_csv('live_data/nq_rule_a_trades.csv'); RA = RAt.groupby(pd.to_datetime(RAt.day)).net_pts.sum().reindex(AD).fillna(0.0)
C30 = stream(np.full(len(X), 30.0), comp_sz * 50 / 30); C50 = stream(np.full(len(X), 50.0), comp_sz); A30 = stream(np.full(len(X), 30.0), base * 50 / 30)
S = {'C30 composite, stop 30': C30, 'C50 composite, stop 50': C50, 'A30 six checks, stop 30': A30, 'RA Rule A 75/50, 1 contract': RA,
     'C30 + 2 x RA': C30 + 2 * RA, 'C30 + 4 x RA': C30 + 4 * RA, 'C50 + 2 x RA': C50 + 2 * RA}
def stats(P):
    mo = P.groupby(P.index.to_period('M')); mavg = mo.sum() / mo.size(); cum = P.cumsum(); yr = P.groupby(P.index.year).mean()
    return dict(avg=P.mean(), m75=(mavg >= 7.5).mean(), mneg=(mo.sum() < 0).mean(), wm=mo.sum().min(), w3=P.rolling(63).mean().min(), w6=P.rolling(126).mean().min(),
                wy=yr.min(), wyy=int(yr.idxmin()), dd=(cum - cum.cummax()).min(), sh=P.mean() / P.std() * np.sqrt(252), yr=yr)
if __name__ == '__main__':
    print('2018 - Oct 2026, own account, NQ points', flush=True)
    print(f'  {"stream":<28} avg/day  months>=7.5/day  losing months  worst month  worst 3-mo/day  worst 6-mo/day  worst year/day  max DD  Sharpe | by year (avg a day)', flush=True)
    R = {}
    for nm, P in S.items():
        r = stats(P); R[nm] = r
        print(f'  {nm:<28} {r["avg"]:+7.2f} {r["m75"]:15.0%} {r["mneg"]:14.0%} {r["wm"]:+12.0f} {r["w3"]:+15.2f} {r["w6"]:+15.2f} {r["wy"]:+10.2f} ({r["wyy"]}) {r["dd"]:+7.0f} {r["sh"]:7.2f} | '
              + ' '.join(f'{y} {v:+.1f}' for y, v in r['yr'].items()), flush=True)
    print('\nSIZE NEEDED so the worst 6-month stretch averages 7.5 a day (only possible when that stretch is positive)', flush=True)
    for nm, r in R.items():
        if r['w6'] > 0: m = 7.5 / r['w6']; print(f'  {nm:<28} x{m:.2f} -> average {r["avg"] * m:+.1f} a day, max drawdown {r["dd"] * m:+,.0f}, worst month {r["wm"] * m:+,.0f}', flush=True)
        else: print(f'  {nm:<28} worst 6 months are negative ({r["w6"]:+.2f} a day): no size fixes it', flush=True)
