"""Nov 2 2026: most booked P&L is overnight (2019-21: 6.4 of 7.0 log points; 2022-26: 3.7 of 5.6). Test checking stop / lock exits only at
the end of the day (15:45 bar) so trades are never cut by intraday noise and always keep the overnight. 40 matched runs vs live."""
exec(open('/home/claude/fartpie/xs/regime_exit_an.py').read().split("RX = prep2(")[0])
RX = prep2(glob.glob('rx2/*.parquet'), 1.5)
def runs(Z):
    rng = np.random.default_rng(101); out = []
    for i in range(41):
        dl = rng.random(len(Z)) < .05 if i else np.zeros(len(Z), bool)
        out.append([v for cd in stat(dret(simw(Z[~dl])).cumsum()) for v in (cd[0], cd[1], cd[0] / -cd[1])])
    return np.array(out)
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<30} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}  (return {(a[1:, 0] > b[1:, 0]).mean():.0%} / {(a[1:, 3] > b[1:, 3]).mean():.0%})'
    print(s, flush=True)
for v in ('live', 'eod'):
    Z = RX[RX['var'] == v]; print(f'  {v}: {len(Z)} trades, avg {Z.ret.mean():+.2%}, win {(Z.ret > 0).mean():.0%}, avg hold {(Z.exit - Z.entry).dt.total_seconds().mean() / 86400:.1f} days', flush=True)
B = runs(RX[RX['var'] == 'live'].reset_index(drop=True)); line('live', B)
line('exits checked only at 15:45', runs(RX[RX['var'] == 'eod'].reset_index(drop=True)), B)
