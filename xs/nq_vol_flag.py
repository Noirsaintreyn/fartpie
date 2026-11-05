"""Nov 4 2026: which volatility model should drive the Rule A size-down flag? HAR vs GJR-GARCH vs Yang-Zhang vs Garman-Klass (website) vs none. DECLARED BEFORE RESULTS.
Rule A trades (skip thin 60-day path / hard break / monthly LVN), fixed 50 / 75. Size 1 / 0.5 / 0.3 NQ by (VOL flag + VWAP stretched).
VOL flag = the model's 1-year percentile (252 sessions, min 120) at the PRIOR close is in its top third:
  HAR   log-HAR forecast of next-day 5m realized variance (refit monthly)  [current]
  GJR   GJR-GARCH(1,1)-t next-day variance (fit on 2015-21, parameters fixed after)
  YZ    Yang-Zhang 20-day realized volatility (daily QQQ OHLC)
  GK    Garman-Klass 60-day (the website's sigma)
  NONE  no volatility flag (VWAP flag only)
Per model: % of trades flagged, avg pts and std of flagged vs unflagged trades 2022-26; Lucid 50K ($88) and Topstep sims incl. 2026."""
_vx = open('xs/nq_vol_exits.py').read(); exec(_vx[:_vx.rindex("if __name__ == '__main__':")])
pr_ = lambda s: s.rolling(252, min_periods=120).rank(pct=True)
o_, h_, l_, c_ = Dq.Open, Dq.High, Dq.Low, Dq.Close; n = 20; kz = .34 / (1.34 + (n + 1) / (n - 1))
ov = np.log(o_ / c_.shift(1)); oc = np.log(c_ / o_); rs = np.log(h_ / c_) * np.log(h_ / o_) + np.log(l_ / c_) * np.log(l_ / o_)
yz = ov.rolling(n).var() + kz * oc.rolling(n).var() + (1 - kz) * rs.rolling(n).mean()
PCT = {'HAR': HP, 'GJR': pr_(gjr.dropna()), 'YZ': pr_(yz.dropna()), 'GK': pr_(gk.clip(lower=1e-12).rolling(60, min_periods=20).mean().dropna())}
FLAG = {m: pd.Series(prior(s, D_) > 2 / 3, index=D_) for m, s in PCT.items()}; FLAG['NONE'] = pd.Series(False, index=D_)
RA = run_l(X[~cutA]); RA = RA[RA.day >= P0]
print('flagged trades 2022-26: avg pts and std (1 lot)', flush=True)
for m, fl in FLAG.items():
    if m == 'NONE': continue
    f_ = fl.reindex(pd.DatetimeIndex(RA.day)).values.astype(bool); a, b = RA.net[f_], RA.net[~f_]
    print(f'  {m:<4} flagged {f_.mean():4.0%} | flagged avg {a.mean():+5.1f} std {a.std():4.1f} | unflagged avg {b.mean():+5.1f} std {b.std():4.1f} | flagged in 2026 {fl[fl.index >= "2026"].mean():.0%} of days', flush=True)
base = SYSX['cutA']; SYSV = {}
for m, fl in FLAG.items():
    D = [[] for _ in D_]
    for k, lst in enumerate(base):
        v = int(bool(fl.iloc[k]))
        for th, net, mae, p in lst: D[k].append((((1.0, .5, .3)[min(v + int(th[2]), 2)],), net, mae, p))
    SYSV[m] = D
if __name__ == '__main__':
    jobs = [(m, fn) for m in SYSV for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job7, jobs)
    print('\nFIRM SIMS (Rule A, size by VOL flag + VWAP stretched)', flush=True)
    for r in RES:
        fu, y = r['full'], r['y26']
        print(f'  {r["nm"]:<5} {r["fn"]:<9} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} '
              f'| 2022-Oct26 net {fu["net"]:+7,.0f} ({fu["payouts"]} payouts) | 2026 net {y["net"]:+6,.0f}', flush=True)
