"""Nov 4 2026: QQQ options context on the Rule A NQ trade - vanna, charm, dealer delta, 25-delta skew, put/call, ATM IV, IV trend, VRP. DECLARED BEFORE RESULTS.
Data at the PRIOR close: options_greeks_daily(_v2)_QQQ.csv (gex / dex / skew / OI / volume / vanna / charm, through Dec 15 2025) and xs/ext/gex/QQQ.csv (ATM IV, call / put volume, through Oct 8 2026).
Vonna and vega exposure (vex) are not in the files (no per-strike data), so they cannot be tested.
SIZE labels (1-year percentile terciles): VANNA, CHARM, DEX (call + put delta), SKEW (25-delta), PC_OI, PC_VOL, IV, IV5 (5-day IV change), VRP (IV^2 - HAR variance).
DIRECTION gates (trade direction must match): VANNA-FLOW = sign(net vanna) x -sign(5-day IV change) (positive vanna + falling IV -> dealers buy); CHARM = sign(net charm);
   SKEW-FADE = long only when skew is in its top third, short only in its bottom third; PC-FADE = long only when put/call volume is in its top third, short only in its bottom third.
A level counts if it beats the other trades in 2018-21, 2022-23 AND 2024-25 (n >= 30 each). Also YZ (Yang-Zhang 20d top third) by period, since it just replaced HAR.
Any counting SIZE label is added as a third size-down flag (1 / 0.5 / 0.3 NQ by YZ + VWAP stretched + label) in the Lucid 50K / Topstep sims, against YZ + VWAP alone."""
_vf = open('xs/nq_vol_flag.py').read(); exec(_vf[:_vf.rindex("if __name__ == '__main__':")])
O1 = pd.read_csv(f'{ROOT}/options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
O2 = pd.read_csv(f'{ROOT}/options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
O3 = pd.read_csv(f'{ROOT}/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
OP = pd.DataFrame({'VANNA': O2.net_vanna, 'CHARM': O2.net_charm, 'DEX': O1.call_dex + O1.put_dex, 'SKEW': O1.skew_25d, 'PC_OI': O1.put_oi / O1.call_oi,
                   'PC_VOL': O3.put_vol / O3.call_vol, 'IV': O3.atm_iv}).sort_index()
OP['IV5'] = OP.IV.diff(5); OP['VRP'] = OP.IV ** 2 - har.reindex(OP.index) * 252
RA = run_l(X[~cutA]); RA = RA.join(X[['dir']]); RA['per'] = np.select([RA.day < P0, RA.day < '2024-01-01', RA.day < '2026-01-01'], ['18-21', '22-23', '24-25'], '2026')
PERS = ('18-21', '22-23', '24-25'); prv = lambda s: pd.Series(prior(s, RA.day), index=RA.index)
def report(name, lab):
    out = []
    for v_ in [x for x in pd.unique(lab.dropna())]:
        m = lab == v_; ok = all(((m & (RA.per == p)).sum() >= 30 and RA.net[m & (RA.per == p)].mean() > RA.net[~m & lab.notna() & (RA.per == p)].mean()) for p in PERS)
        print(f'  {name:<10} {str(v_):<12} ' + ' | '.join(f'{p} {RA.net[m & (RA.per == p)].mean():+6.1f} (n{(m & (RA.per == p)).sum():>3}) vs rest {RA.net[~m & lab.notna() & (RA.per == p)].mean():+6.1f}' for p in PERS)
              + f' | 2026 {RA.net[m & (RA.per == "2026")].mean():+6.1f} (n{(m & (RA.per == "2026")).sum()})' + ('   <- beats the rest in all three' if ok else ''), flush=True)
        if ok: out.append(v_)
    return out
print('SIZE LABELS (Rule A trades, avg pts per 1-lot trade)', flush=True)
yzf = pd.Series(FLAG['YZ'].reindex(pd.DatetimeIndex(RA.day)).values, index=RA.index).map({True: 'YZ high', False: 'YZ not high'}); report('YZ', yzf)
GOOD = {}
for c in ('VANNA', 'CHARM', 'DEX', 'SKEW', 'PC_OI', 'PC_VOL', 'IV', 'IV5', 'VRP'):
    pc = OP[c].rolling(252, min_periods=120).rank(pct=True); v = prv(pc)
    lab = pd.Series(np.where(v.isna(), None, np.where(v < 1 / 3, 'low third', np.where(v > 2 / 3, 'high third', 'mid third'))), index=RA.index)
    for g in report(c, lab): GOOD[f'{c} {g}'] = (pc, g)
print('\nDIRECTION GATES (with vs against)', flush=True)
vf = np.sign(prv(OP.VANNA)) * -np.sign(prv(OP.IV5)); ch = np.sign(prv(OP.CHARM)); skp = prv(OP.SKEW.rolling(252, min_periods=120).rank(pct=True)); pcp = prv(OP.PC_VOL.rolling(252, min_periods=120).rank(pct=True))
sk = pd.Series(np.where(skp > 2 / 3, 1, np.where(skp < 1 / 3, -1, 0)), index=RA.index).where(skp.notna()); pcg = pd.Series(np.where(pcp > 2 / 3, 1, np.where(pcp < 1 / 3, -1, 0)), index=RA.index).where(pcp.notna())
for nm, gd in (('VANNA-FLOW', vf), ('CHARM', ch), ('SKEW-FADE', sk), ('PC-FADE', pcg)):
    lab = pd.Series(np.where(gd.isna() | (gd == 0), None, np.where(gd == RA.dir, 'with', 'against')), index=RA.index); report(nm, lab)
SYSV = {'YZ + VWAP (base)': SYSV['YZ']}
for nm, (pc, lvl) in GOOD.items():
    fl = pd.Series(prior(pc, D_), index=D_); f_ = (fl < 1 / 3) if lvl == 'low third' else ((fl > 2 / 3) if lvl == 'high third' else ((fl >= 1 / 3) & (fl <= 2 / 3)))
    bad = ~f_ & fl.notna()                                              # size DOWN outside the good level
    D = [[] for _ in D_]
    for k, lst in enumerate(base):
        v = int(bool(FLAG['YZ'].iloc[k])) + int(bool(bad.iloc[k]))
        for th, net, mae, p in lst: D[k].append((((1.0, .5, .3)[min(v + int(th[2]), 2)],), net, mae, p))
    SYSV[f'YZ + VWAP + not {nm}'] = D
print(f'\nsize labels that count: {list(GOOD) or "none"}', flush=True)
if __name__ == '__main__':
    jobs = [(m, fn) for m in SYSV for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job7, jobs)
    print('\nFIRM SIMS', flush=True)
    for r in RES:
        fu, y = r['full'], r['y26']
        print(f'  {r["nm"]:<34} {r["fn"]:<9} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} | 2022-Oct26 net {fu["net"]:+7,.0f} | 2026 net {y["net"]:+6,.0f}', flush=True)
