"""Nov 9 2026: BIG MOVES FOR OPTIONS - HIGH-VOL FLIPS INTO NEGATIVE GAMMA WITH CALL FLOW ALIGNED (Rey's discretionary setup). DECLARED BEFORE RESULTS.
Flips: the screener's bull flips in names with ThetaData option history (one per name per day, xs/opt_flips.py), 2019-21 / 2022-26. Real EOD bid / ask.
Flags at the PRIOR close:  NEG own net GEX < 0 (xs/ext/gex, 18 names) | POS own net GEX > 0 | HIVOL Yang-Zhang 20d in its top third (2019-21 cut) |
                           FLOW call / put volume (ThetaData, prior day) above its own 20-day median (calls heavier than usual)
Sets: all flips | NEG | NEG + HIVOL | NEG + FLOW | NEG + HIVOL + FLOW (the setup) | POS + HIVOL + FLOW (contrast) | HIVOL + FLOW (any gamma)
Measured:  stock: share of flips whose trade made >= +5% / >= +10% (big moves), mean;
           long CALL (nearest strike at / above spot), WEEKLY (~7 days) and MONTHLY (~30 days), held to the screener's exit or expiry, bought at the ASK and sold at
           the BID (TOUCH, what you would really pay): mean return on premium, median, win rate; and the same structure on RANDOM days in the same names with the
           same flags (does the flip add anything beyond the flags?).
PASS: the setup's calls beat all flips AND its own random days in BOTH periods at the TOUCH."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
U = '/mnt/user-data/uploads/fartpie'; R = pd.read_parquet('/tmp/claude-0/opt_flips_R.parquet'); R['d0'] = pd.to_datetime(R.d0)
T = pd.read_parquet(f'{U}/xs/ext/wide_T.parquet'); T = T[T.sp & (T.ratio >= 1.5)].copy(); T['d0'] = T.entry.dt.normalize()
T = T.sort_values(['tk', 'entry']).drop_duplicates(['tk', 'd0']).set_index(['tk', 'd0'])
def yz(Y, n=20):
    o, h, l, c = np.log(Y.open), np.log(Y.high), np.log(Y.low), np.log(Y.close); co = o - c.shift(1); oc = c - o; rs = (h - c) * (h - o) + (l - c) * (l - o); k = .34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt(co.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean())
CTX = []
for tk in R.tk.unique():
    c = pd.DataFrame(index=pd.DatetimeIndex([]))
    f = f'/home/claude/fartpie/data/yahoo/daily_v/{tk}.parquet'
    if os.path.exists(f): Y = pd.read_parquet(f); c = pd.DataFrame({'yz': yz(Y).shift(1)}, index=Y.index)
    g = f'{U}/xs/ext/gex/{tk}.csv'
    if os.path.exists(g): G = pd.read_csv(g, parse_dates=['date']).drop_duplicates('date').set_index('date'); c = c.join(G.net_gex.shift(1).rename('gam'), how='outer')
    q = f'{U}/data/theta_compact/{tk}.parquet'
    if os.path.exists(q):
        Q = pd.read_parquet(q); v = Q.groupby(['d', 'r']).vol.sum().unstack(); cp = (v.get(1) / v.get(-1)).replace([np.inf], np.nan); cp.index = pd.to_datetime(cp.index)
        c = c.join(((cp > cp.rolling(20, min_periods=10).median())).shift(1).rename('flow'), how='outer')
    c['tk'] = tk; CTX.append(c.rename_axis('d0').reset_index())
CTX = pd.concat(CTX)
R = R.merge(CTX, on=['tk', 'd0'], how='left')
cut = R[(R.per == '19-21')].drop_duplicates(['tk', 'd0']).yz.quantile(2 / 3)
R['NEG'] = R.gam < 0; R['POS'] = R.gam > 0; R['HIVOL'] = R.yz >= cut; R['FLOW'] = R.flow.fillna(False).astype(bool)
SETS = {'all flips': pd.Series(True, index=R.index), 'NEG': R.NEG, 'NEG + HIVOL': R.NEG & R.HIVOL, 'NEG + FLOW': R.NEG & R.FLOW, 'NEG + HIVOL + FLOW (setup)': R.NEG & R.HIVOL & R.FLOW,
        'POS + HIVOL + FLOW': R.POS & R.HIVOL & R.FLOW, 'HIVOL + FLOW (any gamma)': R.HIVOL & R.FLOW}
if __name__ == '__main__':
    F = R[(R.set == 'flip')].drop_duplicates(['tk', 'd0']).copy(); F['sret'] = [T.ret.get((t, d), np.nan) for t, d in zip(F.tk, F.d0)]
    print(f'option names {R.tk.nunique()}, with gamma {R[R.gam.notna()].tk.nunique()} | YZ top-third cut {cut:.4f}\n\n=== stock: big moves after the flip (share >= +5% / >= +10%, mean, n) | 2019-21 ; 2022-26 ===', flush=True)
    for nm, m_ in SETS.items():
        g_ = F[m_.loc[F.index]]; cells = []
        for p_ in ('19-21', '22-26'):
            z = g_[g_.per == p_].sret.dropna(); cells.append(f'{np.mean(z >= .05):4.0%} / {np.mean(z >= .10):4.0%}, mean {z.mean():+.2%} (n {len(z)})')
        print(f'  {nm:<28} ' + ' ; '.join(cells), flush=True)
    print('\n=== long calls, real fills (buy ask / sell bid): mean / median return on premium, win | flip vs RANDOM days with the same flags | 2019-21 ; 2022-26 ===', flush=True)
    for ten in ('W', 'M'):
        for ex in (('VEC', 'EXP') if ten == 'W' else ('VEC',)):
            print(f'  -- {"WEEKLY" if ten == "W" else "MONTHLY"} call, exit {"at the screener exit" if ex == "VEC" else "at expiry"} --', flush=True)
            for nm, m_ in SETS.items():
                cells = []
                for p_ in ('19-21', '22-26'):
                    a = R[m_ & (R.set == 'flip') & (R.ten == ten) & (R.st == 'CALL') & (R.ex == ex) & (R.fill == 'TOUCH') & (R.per == p_)].ret
                    b = R[m_ & (R.set == 'base') & (R.ten == ten) & (R.st == 'CALL') & (R.ex == ex) & (R.fill == 'TOUCH') & (R.per == p_)].ret
                    cells.append(f'flip {a.mean():+7.1%} / med {a.median():+6.1%} win {np.mean(a > 0):3.0%} (n {len(a):4d}) vs random {b.mean():+7.1%} (n {len(b):4d})')
                print(f'    {nm:<28} ' + ' ; '.join(cells), flush=True)
# --- follow-up: the one cell that passed (WEEKLY call held to expiry, the setup) - how lumpy is it? 1 unit of premium per trade
if __name__ == '__main__':
    print('\n=== the setup, WEEKLY call held to expiry, TOUCH: robustness ===', flush=True)
    for nm in ('NEG + HIVOL + FLOW (setup)', 'all flips'):
        m_ = SETS[nm]; A = R[m_ & (R.set == 'flip') & (R.ten == 'W') & (R.st == 'CALL') & (R.ex == 'EXP') & (R.fill == 'TOUCH')].copy(); A['mo'] = A.d0.dt.to_period('M')
        for p_ in ('19-21', '22-26'):
            z = A[A.per == p_]; mo = z.groupby('mo').ret.sum(); top = z.ret.sort_values(ascending=False)
            print(f'  {nm:<28} {p_}: {len(z) / max(z.mo.nunique(), 1):4.1f} trades in active months | month avg {mo.mean():+.2f} units, months positive {np.mean(mo > 0):.0%}, '
                  f'worst {mo.min():+.2f}, best {mo.max():+.2f} | mean without the top 5 trades {top.iloc[5:].mean():+.1%} (top 5: {", ".join(f"{x:+.0%}" for x in top.iloc[:5])})', flush=True)
        z = A[A.per == '22-26']; print('    by year (2022-26): ' + ' | '.join(f'{y}: {g_.ret.mean():+.0%} (n {len(g_)})' for y, g_ in z.groupby(z.d0.dt.year)), flush=True)
        print('    by name (2022-26, top / bottom 3 by total): ' + ', '.join(f'{k} {v:+.1f}u' for k, v in z.groupby('tk').ret.sum().sort_values(ascending=False).iloc[[0, 1, 2, -3, -2, -1]].items()), flush=True)
# --- follow-up 2 (EXPLORATORY: this was the declared CONTRAST set, not the setup): POS + HIVOL + FLOW calls closed at the screener's exit
if __name__ == '__main__':
    print('\n=== POS + HIVOL + FLOW, calls closed at the screener exit, TOUCH: robustness ===', flush=True)
    for ten in ('W', 'M'):
        A = R[SETS['POS + HIVOL + FLOW'] & (R.set == 'flip') & (R.ten == ten) & (R.st == 'CALL') & (R.ex == 'VEC') & (R.fill == 'TOUCH')].copy(); A['mo'] = A.d0.dt.to_period('M')
        for p_ in ('19-21', '22-26'):
            z = A[A.per == p_]; mo = z.groupby('mo').ret.sum(); top = z.ret.sort_values(ascending=False)
            print(f'  {"WEEKLY" if ten == "W" else "MONTHLY"} {p_}: n {len(z)}, mean {z.ret.mean():+.1%}, without top 5 {top.iloc[5:].mean():+.1%} | month avg {mo.mean():+.2f}u, months positive {np.mean(mo > 0):.0%}, worst {mo.min():+.2f}u', flush=True)
        z = A[A.per == '22-26']; print('    by year: ' + ' | '.join(f'{y}: {g_.ret.mean():+.0%} (n {len(g_)})' for y, g_ in z.groupby(z.d0.dt.year)) + ' | by name top / bottom: ' +
                                       ', '.join(f'{k} {v:+.1f}u' for k, v in z.groupby('tk').ret.sum().sort_values(ascending=False).iloc[[0, 1, 2, -2, -1]].items()), flush=True)
