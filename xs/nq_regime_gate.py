"""Nov 4 2026: a REGIME GATE that switches between the normal system and the '2026 mode' (half size against the 1h vector) using only data known before the trade. DECLARED BEFORE RESULTS.
Base = final system (Rule A trades, Yang-Zhang + VWAP sizing 1 / 0.5 / 0.3 NQ, 50 / 75, flat 12:00; eval skips flips against the 09:30 gap). Mode sizing applies in eval and funded.
DETECTORS (TREND mode when):
  PERF  among the last 40 closed Rule A trades, flips WITH the 1h vector averaged more than flips AGAINST it (>= 8 of each, else normal)
  VR    QQQ 60-day variance ratio VR(5) = var(5-day returns) / (5 x var(1-day)) at the prior close > 1
  AC    QQQ 60-day lag-1 autocorrelation of daily returns at the prior close > 0
ACTIONS:  -1 = TREND mode only: half size against the 1h vector, normal mode unchanged
          -2 = both ways: TREND mode half size against the 1h vector; NORMAL mode half size WITH the 1h vector (what worked in 2018-25)
References: always normal (final system) | always 2026 mode.
Report: % of trades in TREND mode by year, Rule A $ by year at final sizing, Lucid 50K ($88) + Topstep pass / payout rates, net per eval, 12-month median, 2022-26 and 2026 net."""
_fs = open('xs/nq_final_system.py').read(); exec(_fs[:_fs.rindex("if __name__ == '__main__':")])
RL = run_l(XA).join(XA[['dir', 'g_H1']]).sort_values(['day']); RL['with1h'] = (RL.g_H1 == RL.dir) & (RL.g_H1 != 0); RL['ag1h'] = (RL.g_H1 != RL.dir) & (RL.g_H1 != 0)
perf = pd.Series(False, index=XA.index); days_sorted = RL.day.values
for idx, day in zip(XA.index, XA.day.values):
    past = RL[RL.day.values < day].tail(40); w, a = past[past.with1h], past[past.ag1h]
    perf.loc[idx] = len(w) >= 8 and len(a) >= 8 and w.net.mean() > a.net.mean()
r_ = np.log(RTD.c).diff(); vr = (r_.rolling(5).sum().rolling(60).var() / (5 * r_.rolling(60).var())).shift(1); ac = r_.rolling(60).apply(lambda x: pd.Series(x).autocorr(), raw=True).shift(1)
TREND = {'PERF': perf, 'VR': pd.Series(vr.reindex(pd.DatetimeIndex(XA.day)).values > 1, index=XA.index), 'AC': pd.Series(ac.reindex(pd.DatetimeIndex(XA.day)).values > 0, index=XA.index)}
yr = XA.day.dt.year.clip(lower=2021).map(lambda y: '2018-21' if y == 2021 else str(y)); yr[XA.day < '2022-01-01'] = '2018-21'
print('% of Rule A trades in TREND mode', flush=True)
for nm, t in TREND.items(): print(f'  {nm:<5} ' + ' | '.join(f'{y} {t[yr == y].mean():.0%}' for y in ('2018-21', '2022', '2023', '2024', '2025', '2026')), flush=True)
ag1 = AG['H1']; w1 = pd.Series(((XA.g_H1 == XA.dir) & (XA.g_H1 != 0)).values, index=XA.index)
MULT = {'always normal': pd.Series(1.0, index=XA.index), 'always 2026 mode': pd.Series(np.where(ag1, .5, 1.0), index=XA.index)}
for nm, t in TREND.items():
    MULT[f'{nm}-1'] = pd.Series(np.where(t & ag1, .5, 1.0), index=XA.index)
    MULT[f'{nm}-2'] = pd.Series(np.where(t & ag1, .5, np.where(~t & w1, .5, 1.0)), index=XA.index)
base_sz = pd.Series([(1.0, .5, .3)[min(int(bool(YZF.reindex([pd.Timestamp(d)]).iloc[0])) + int(bool(X.at[i, 'VWAPX'])), 2)] if pd.Timestamp(d) in YZF.index else 1.0 for i, d in zip(XA.index, XA.day)], index=XA.index)
RLx = RL.join(base_sz.rename('bs')); RLx['y'] = yr.reindex(RLx.index)
print('\nRule A $ by year at final sizing x mode (1 contract-equivalent ladder; slot effects ignored)', flush=True)
for nm, m in MULT.items():
    usd = RLx.net * 20 * RLx.bs * m.reindex(RLx.index)
    print(f'  {nm:<17} ' + ' | '.join(f'{y} {usd[RLx.y == y].sum():+7,.0f}' for y in ('2018-21', '2022', '2023', '2024', '2025', '2026')), flush=True)
VAR = {}
for nm, m in MULT.items():
    DS[f'e_{nm}'] = build(~AG['GAP'], mult=m); DS[f'f_{nm}'] = build(mult=m); VAR[nm] = (f'e_{nm}', f'f_{nm}', f'f_{nm}')
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job10, jobs)
    print('\nREGIME-GATED SYSTEMS (gap-gated eval; mode sizing in eval and funded)', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<17} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | net/eval {r["net_eval"]:+6,.0f} '
              f'| 12m median {r["k_med"]:+6,.0f} (10th {r["k_p10"]:+5,.0f}) positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
