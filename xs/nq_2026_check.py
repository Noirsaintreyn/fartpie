"""Nov 4 2026: why no 2026 payouts - Rule A trades in 2026 month by month (1-lot pts and $ at Rule A size), vs the base flips."""
_ss = open('xs/nq_structure_size.py').read(); exec(_ss[:_ss.rindex("if __name__ == '__main__':")])
XA = X[~cutA]; RA = run_l(XA); RB = run_l(X)
sz = np.array([CUT(tuple(bool(X.at[i, c]) for c in FL)) for i in RA.index]); RA['usd'] = RA.net * 20 * sz
for nm, R_ in (('Rule A (75/50)', RA), ('every flip (75/50)', RB.assign(usd=RB.net * 20))):
    y = R_[R_.day >= '2026-01-01']; m = y.groupby(y.day.dt.to_period('M')).agg(n=('net', 'size'), win=('net', lambda s: (s > 0).mean()), pts=('net', 'mean'), usd=('usd', 'sum'))
    print(f'\n{nm} 2026: trades {len(y)}, win {y.net.gt(0).mean():.0%}, avg {y.net.mean():+.1f} pts, total ${y.usd.sum():+,.0f} | 2022-25 avg {R_[(R_.day >= P0) & (R_.day < "2026")].net.mean():+.1f} pts, {len(R_[(R_.day >= P0) & (R_.day < "2026")]) / 48:.1f} trades/month', flush=True)
    print(m.round(2).to_string(), flush=True)
