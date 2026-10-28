"""Oct 28 2026: the put wall (strike with the most put OI within +-15% of spot, opt_summary, 23 names 2018-26).
 A  does it act as support? daily: next-5-day low vs the wall when the close sits 0-2% above it, vs farther away; what happens after a break
 B  on the trades the live book takes (23-name book): trade return by distance to the put wall at entry, both halves
 C  as a stop floor: tighter info - trades whose vector stop is below the put wall vs above (stop distance from the 15m bars is not stored,
    so B uses the wall's distance only)"""
import os, glob, numpy as np, pandas as pd
EXT = '/mnt/user-data/uploads/fartpie/xs/ext'
A = []
for f in glob.glob(f'{EXT}/opt_summary/*.csv'):
    o = pd.read_csv(f, parse_dates=['date']).sort_values('date'); o['tk'] = os.path.basename(f)[:-4]
    s = o.spot; o['dist'] = s / o.put_wall - 1
    o['lo5'] = s[::-1].rolling(5, min_periods=5).min()[::-1].shift(-1)                 # lowest close of the next 5 days
    o['r5'] = s.shift(-5) / s - 1; o['r20'] = s.shift(-20) / s - 1
    o['held5'] = o.lo5 >= o.put_wall; o['brk'] = (s < o.put_wall) & (s.shift(1) >= o.put_wall.shift(1))
    A.append(o)
A = pd.concat(A, ignore_index=True).dropna(subset=['r20']).reset_index(drop=True); A['h'] = np.where(A.date.dt.year <= 2021, '2018-21', '2022-26')
A = A[~A.tk.isin(['SPY', 'QQQ', 'IWM'])].reset_index(drop=True)
print('A) support test: closes ABOVE the put wall, by distance; next-5-day: wall held (no close below), 5d / 20d return')
b = pd.cut(A.dist, [0, .01, .02, .04, .07, .1, .2]); X = A[A.dist > 0]
print(X.groupby([b[A.dist > 0], 'h']).agg(n=('r5', 'size'), held=('held5', 'mean'), r5=('r5', lambda v: v.mean() * 100), r20=('r20', lambda v: v.mean() * 100)).round(2).unstack('h').to_string())
print('\n   price BELOW the wall:', A[A.dist <= 0].groupby('h').agg(n=('r5', 'size'), r5=('r5', lambda v: v.mean() * 100), r20=('r20', lambda v: v.mean() * 100)).round(2).to_dict('index'))
print('   day of a fresh close below the wall:', A[A.brk].groupby('h').agg(n=('r5', 'size'), r5=('r5', lambda v: v.mean() * 100), r20=('r20', lambda v: v.mean() * 100)).round(2).to_dict('index'))
print('   all days baseline:', A.groupby('h').agg(r5=('r5', lambda v: v.mean() * 100), r20=('r20', lambda v: v.mean() * 100)).round(2).to_dict('index'))
# B: trades the 23-name live book takes
exec(open('/home/claude/fartpie/xs/rank_opts.py').read().split("FA = []")[0])
L2 = L[L.tk.isin(A.tk.unique())]; TK = taken(L2); TK['d'] = TK.entry.dt.normalize().astype('datetime64[ns]')
W = A[['tk', 'date', 'dist']].rename(columns={'date': 'd'}); W['d'] = W.d + pd.Timedelta(days=1)          # prior close's wall, known at entry
TK = pd.merge_asof(TK.sort_values('d'), W.sort_values('d').astype({'d': 'datetime64[ns]'}), on='d', by='tk', direction='backward', tolerance=pd.Timedelta(days=5))
TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print(f'\nB) trades the 23-name book takes: {len(TK)}, wall known for {TK.dist.notna().mean():.0%}. avg trade %, win%, by distance above the put wall at entry')
q = pd.cut(TK.dist, [-1, 0, .03, .06, .09, .12, 1], labels=['below wall', '0-3%', '3-6%', '6-9%', '9-12%', '12%+'])
print(TK.groupby([q, 'h']).ret.agg(n='size', avg=lambda v: v.mean() * 100, win=lambda v: (v > 0).mean() * 100).round(2).unstack('h').to_string())
print('\nC) book (23 names): skip flips that sit just above the put wall; yearly return / max DD')
W2 = W.sort_values('d').astype({'d': 'datetime64[ns]'}); Z = L2.copy(); Z['d'] = Z.entry.dt.normalize().astype('datetime64[ns]')
Z = pd.merge_asof(Z.sort_values('d'), W2, on='d', by='tk', direction='backward', tolerance=pd.Timedelta(days=5)).sort_values('entry').reset_index(drop=True)
P2 = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def sh(nm, Y): print(f'  {nm:<34}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Y, a, b) for a, b in P2)) + f'  ({len(Y)} signals)', flush=True)
sh('live (23 names)', Z)
for lo, hi in ((0, .01), (0, .02), (0, .03), (0, .05), (.01, .03)):
    sh(f'skip {lo:.0%}-{hi:.0%} above the wall', Z[~((Z.dist > lo) & (Z.dist <= hi))])
sh('only below the wall or 6%+ above', Z[(Z.dist <= 0) | (Z.dist > .06) | Z.dist.isna()])
rng = np.random.default_rng(2); R = []
for k in range(30):
    R.append([sim(Z.drop(index=rng.choice(Z.index, int(len(Z) * .08), replace=False)), a, b)[0] for a, b in P2])
R = np.array(R); print(f'  luck band: drop 8% of signals at random x30   2019-21 {R[:, 0].min():+.1%}..{R[:, 0].max():+.1%}   2022-26 {R[:, 1].min():+.1%}..{R[:, 1].max():+.1%}')
