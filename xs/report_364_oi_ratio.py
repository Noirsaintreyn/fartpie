"""Nov 22 2026 report 364. Rey: 'would the options volume ratio on the universe be better with open interest?'  Declared: OI RATIO = (call OI + put OI) x 100 / shares traded... but the live ratio is volume-based (contracts x 100 / share volume), so the declared comparison is on the same denominator:
 R_vol (live, = b302 ratio), R_oi = total OI (0-60 DTE, report at the open of the flip day, i.e. prior close) x 100 / share volume of the prior day, R_turn = option volume / OI (turnover).  LEVELS, not own-history z (the live gate and rank are levels).
 Tests: (1) within the live universe (ratio >= 1.5) top-minus-bottom tercile flip mean (train | test, week-clustered t) for R_oi and R_turn; (2) the 6-slot book with pick order R_oi high-first, R_turn high-first and R_vol (live), on the 35 names with OI; halves train 2019-21 | test 2022-26.  Pass for (2): beats live CAGR in both halves.  Expected (reports 294/338/355): the ratio is a liquidity threshold, so OI will behave the same."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
U = '/mnt/user-data/uploads/fartpie/'; SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O = pd.read_csv(U + 'xs/ext/optvol_ratio.csv', parse_dates=['date'])[['tk', 'date', 'sh']]
rows = []
for f in glob.glob(U + 'xs/ext/oi_daily/*.csv'):
    tk = os.path.basename(f)[:-4]; d = pd.read_csv(f, parse_dates=['date']); d['tk'] = tk; rows.append(d)
OD = pd.concat(rows).merge(O, on=['tk', 'date'], how='left'); OD['oi'] = OD.call_oi + OD.put_oi; OD['vol'] = OD.call_vol + OD.put_vol
OD['r_oi'] = OD.oi * 100 / OD.sh; OD['r_turn'] = OD.vol / OD.oi; OD = OD.sort_values(['tk', 'date']); OD[['r_oi', 'r_turn']] = OD.groupby('tk')[['r_oi', 'r_turn']].shift(1)
OD['date'] = OD.date.astype('datetime64[ns]'); B0['day'] = pd.to_datetime(B0.day).astype('datetime64[ns]'); B0 = B0.sort_values('day')
B0 = pd.merge_asof(B0, OD[['tk', 'date', 'r_oi', 'r_turn']].sort_values('date'), left_on='day', right_on='date', by='tk', direction='backward', tolerance=pd.Timedelta(days=5)).sort_values('entry').reset_index(drop=True)
print('flips %d, with OI ratio %d; median ratio live %.2f, OI-ratio %.2f, turnover %.2f; Spearman(live ratio, OI ratio) %.2f' % (len(B0), B0.r_oi.notna().sum(), B0.ratio.median(), B0.r_oi.median(), B0.r_turn.median(), B0[['ratio', 'r_oi']].corr('spearman').iloc[0, 1]), flush=True)
B0['wk'] = B0.entry.dt.to_period('W').astype(str); B0['half'] = np.where(B0.entry < '2022-01-01', 'train', 'test'); tr = B0[B0.half == 'train']
for col in ('r_oi', 'r_turn', 'ratio'):
    lo, hi = tr[col].quantile([1 / 3, 2 / 3]); out = []
    for h in ('train', 'test'):
        X = B0[(B0.half == h) & B0[col].notna()]; Z = pd.concat([X[X[col] >= hi].assign(x=1.0), X[X[col] <= lo].assign(x=0.0)])
        m = sm.OLS(Z.ret.values, sm.add_constant(Z.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Z.wk)[0]}); out.append('%+.2f%% (t %+.1f)' % (m.params[1] * 100, m.tvalues[1]))
    print('tercile top-bottom %-7s train %s | test %s' % (col, *out), flush=True)
def cg(TK):
    z = spread(TK); z = z[z.index >= '2019-01-01']; return cagr(z), mdd(z), sharpe(z.values), cagr(z[z.index < '2022-01-01']), cagr(z[z.index >= '2022-01-01'])
def mk(col, hi):
    def o(Bs):
        v = Bs[col].fillna(Bs[col].median()).values; k = -v if hi else v; return Bs.assign(_k=k).sort_values(['entry', '_k', 'tk'], kind='stable')
    return o
def live(Bs): return Bs.assign(_k=Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values).sort_values(['entry', '_k', 'tk'], kind='stable')
for lab, fn in (('LIVE order', live), ('R_oi high first', mk('r_oi', True)), ('R_oi low first', mk('r_oi', False)), ('R_turn high first', mk('r_turn', True)), ('R_turn low first', mk('r_turn', False))):
    order = fn; TK, _ = sim_swap(B0, None); c = cg(TK); print('%-18s CAGR %.1f%% DD %.1f%% Sharpe %.2f | train %.1f test %.1f' % (lab, c[0] * 100, c[1] * 100, c[2], c[3] * 100, c[4] * 100), flush=True)
