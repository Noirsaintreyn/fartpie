"""Nov 5 2026: NEWS GUARD, ENTRY-TIME SIZING, ADX SLOPE, EMA / SMA CROSSES, TIME-SEGMENT DRIFT, SMOOTH EQUITY SIZING. DECLARED BEFORE RESULTS.
Base = final system: Rule A trades, 75 / 50, flat 12:00, Yang-Zhang + VWAP ladder, eval gap-gated x0.3 cold streaks, funded x0.7 (cold = last 20 trades' 1-lot net < 0).
Labels (all known before entry):
  NEWS   CPI day (BLS release list, Feb seasonal-revision releases dropped) | NFP day (approximated as the first Friday) | FOMC statement day (14:00 ET, after we are flat) | day after FOMC
  TIME   entry 09:35-10:15 / 10:15-11:00 / 11:00-11:35
  ADX    daily QQQ ADX(14) and its 5-day slope at the prior close: trend (ADX >= 25) rising / falling, no trend (< 25) rising / falling
  EMA    15m EMA 9 vs 21 at the flip bar agrees / disagrees with the flip;  SMA  prior close above / below the daily SMA 20 and SMA 20 vs 50, agree / disagree with the flip
  DRIFT  the 60-session mean return from this 30-minute entry segment to 12:00 (prior days only) agrees / disagrees with the flip
Sizing variants, Lucid 50K ($88), skipped trades do not free the day's 2-trade slot:
  NEWS-SKIP 0 on CPI / NFP / FOMC days | NEWS-HALF x0.5 there | NEWS-UP x1.5 on CPI / NFP days | LATE-HALF x0.5 after 11:00 | LATE-SKIP | ADX-HALF x0.5 when ADX slope < 0 |
  EMA-HALF x0.5 when the 15m EMA cross disagrees | DRIFT-HALF x0.5 when drift disagrees |
  EWMA smooth cold-streak sizing: mult = floor + (1 - floor) x logistic(EWMA(1-lot net, halflife 8 trades) / 8 pts) (floor 0.3 eval, 0.7 funded) instead of the step |
  SHADOW step cut, but restore as soon as the 1-lot equity curve is at a 10-trade high."""
_rt = open('xs/nq_reach_tp.py').read(); exec(_rt[:_rt.rindex("if __name__ == '__main__':")])
CPI = pd.to_datetime('''2018-01-12 2018-02-14 2018-03-13 2018-04-11 2018-05-10 2018-06-12 2018-07-12 2018-08-10 2018-09-13 2018-10-11 2018-11-14 2018-12-12 2019-01-11 2019-02-13 2019-03-12
2019-04-10 2019-05-10 2019-06-12 2019-07-11 2019-08-13 2019-09-12 2019-10-10 2019-11-13 2019-12-11 2020-01-14 2020-02-13 2020-03-11 2020-04-10 2020-05-12 2020-06-10 2020-07-14 2020-08-12
2020-09-11 2020-10-13 2020-11-12 2020-12-10 2021-01-13 2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11 2021-09-14 2021-10-13 2021-11-10 2021-12-10 2022-01-12
2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13 2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13
2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12 2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13
2024-12-11 2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18 2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12
2026-06-10 2026-07-14 2026-08-12 2026-09-11 2026-10-14'''.split())
FOMC = pd.to_datetime('''2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19 2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30
2019-12-11 2020-01-29 2020-03-03 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16 2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13 2024-01-31
2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10 2026-01-28 2026-03-18
2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28'''.split())
days_all = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(d5)))).normalize().unique()
_fr = pd.Series(days_all)[days_all.dayofweek == 4]; NFP = pd.DatetimeIndex(_fr.groupby(_fr.dt.to_period('M')).first().values)
FOMC1 = pd.DatetimeIndex([days_all[np.searchsorted(days_all, d) + 1] for d in FOMC if np.searchsorted(days_all, d) + 1 < len(days_all)])
dX = pd.DatetimeIndex(XA.day).normalize()
NEWS = pd.Series(np.select([dX.isin(CPI), dX.isin(NFP), dX.isin(FOMC), dX.isin(FOMC1)], ['CPI day', 'NFP day', 'FOMC day', 'day after FOMC'], 'no event'), index=XA.index)
mm = m5[XA.j.values]; TIME = pd.Series(np.select([mm <= 615, mm <= 660], ['09:35-10:15', '10:15-11:00'], '11:00-11:35'), index=XA.index)
# daily RTH bars from the 5m arrays
rth = (m5 >= 570) & (m5 < 960); DB = pd.DataFrame({'d': pd.to_datetime(d5[rth]), 'o': o5[rth], 'h': h5[rth], 'l': l5[rth], 'c': c5[rth]}).groupby('d').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
pc = DB.c.shift(1); tr = pd.concat([DB.h - DB.l, (DB.h - pc).abs(), (DB.l - pc).abs()], axis=1).max(axis=1)
up_, dn_ = DB.h.diff(), -DB.l.diff(); pdm = np.where((up_ > dn_) & (up_ > 0), up_, 0.0); ndm = np.where((dn_ > up_) & (dn_ > 0), dn_, 0.0)
w = lambda s: pd.Series(s, index=DB.index).ewm(alpha=1 / 14, adjust=False).mean()
atr14 = w(tr); pdi = 100 * w(pdm) / atr14; ndi = 100 * w(ndm) / atr14; adx = w((100 * (pdi - ndi).abs() / (pdi + ndi)).fillna(0))
adx_p = pd.Series(prior(adx, XA.day), index=XA.index); slope_p = pd.Series(prior(adx.diff(5), XA.day), index=XA.index)
ADX = pd.Series(np.where(adx_p >= 25, 'trend ', 'no trend ') + np.where(slope_p >= 0, 'rising', 'falling'), index=XA.index)
sma20, sma50 = DB.c.rolling(20).mean(), DB.c.rolling(50).mean(); dr = XA.dir.values
abv = pd.Series(prior(DB.c - sma20, XA.day), index=XA.index); x2050 = pd.Series(prior(sma20 - sma50, XA.day), index=XA.index)
SMA = pd.Series(np.where(np.sign(abv) == dr, 'close vs SMA20 agrees', 'close vs SMA20 disagrees'), index=XA.index)
SMAX = pd.Series(np.where(np.sign(x2050) == dr, 'SMA20/50 agrees', 'SMA20/50 disagrees'), index=XA.index)
kfb = end_i.reindex(pd.DatetimeIndex(t5[XA.j.values])).values.astype(int) - 1
e9, e21 = H15.close.ewm(span=9, adjust=False).mean().values, H15.close.ewm(span=21, adjust=False).mean().values
EMA = pd.Series(np.where(np.sign(e9[kfb] - e21[kfb]) == dr, '15m EMA9/21 agrees', '15m EMA9/21 disagrees'), index=XA.index)
# time-segment drift: return from each 30-min segment start to the 12:00 open, mean over the prior 60 sessions
seg = ((m5 - 570) // 30); okseg = (m5 >= 570) & (m5 < 720); noon_o = pd.Series(o5[m5 == 720], index=pd.to_datetime(d5[m5 == 720]))
SG = pd.DataFrame({'d': pd.to_datetime(d5[okseg]), 's': seg[okseg], 'o': o5[okseg]}).groupby(['d', 's']).o.first().unstack()
RET = np.log(noon_o.reindex(SG.index).values[:, None] / SG.values); DRM = pd.DataFrame(RET, index=SG.index, columns=SG.columns).rolling(60, min_periods=40).mean().shift(1)
sx = np.clip((mm - 570) // 30, 0, 4); dv = np.array([DRM.at[d, s] if d in DRM.index and s in DRM.columns else np.nan for d, s in zip(dX, sx)])
DRIFT = pd.Series(np.where(np.isnan(dv), 'n/a', np.where(np.sign(dv) == dr, 'drift agrees', 'drift disagrees')), index=XA.index)
full = seq('T75'); nets = pd.Series({i: v[0] - 1.0 for i, v in full.items()}); per = pd.Series(PER(XA.loc[nets.index].day), index=nets.index)
print(f'Rule A trades {len(nets)}; per trade 1 lot (NQ pts) by label and period', flush=True)
for nm, L in (('NEWS', NEWS), ('TIME', TIME), ('ADX', ADX), ('EMA', EMA), ('SMA', SMA), ('SMAX', SMAX), ('DRIFT', DRIFT)):
    print(f'  -- {nm}', flush=True); lb = L.loc[nets.index]
    for v_ in sorted(lb.unique()):
        g = nets[lb == v_]; cells = ' | '.join(f'{p} {g[per[g.index] == p].mean():+6.1f} (n{(per[g.index] == p).sum()})' for p in ('18-21', '22-23', '24-25', '2026'))
        print(f'    {v_:<26} n{len(g):>4} win {g.gt(0).mean():4.0%} avg {g.mean():+6.1f} | {cells}', flush=True)
def eqm(cut, mode='step'):
    m = pd.Series(1.0, index=XA.index); hist = []; ew = 0.0; a = 1 - 0.5 ** (1 / 8)
    for idx in order:
        if idx not in nets.index: continue
        cold = len(hist) >= 20 and sum(hist[-20:]) < 0
        if mode == 'step': m.loc[idx] = cut if cold else 1.0
        elif mode == 'ewma': m.loc[idx] = (cut + (1 - cut) / (1 + np.exp(-ew / 8.0))) if len(hist) >= 10 else 1.0
        elif mode == 'shadow':
            cum = np.cumsum(hist); hi10 = len(cum) >= 10 and cum[-1] >= cum[-10:].max()
            m.loc[idx] = cut if (cold and not hi10) else 1.0
        x = nets.loc[idx]; hist.append(x); ew = a * x + (1 - a) * ew
    return m
ev_res = seq('T75', ~AG['GAP']); one = pd.Series(1.0, index=XA.index)
FM = {'BASE': one, 'NEWS-SKIP': pd.Series(np.where(NEWS.isin(['CPI day', 'NFP day', 'FOMC day']), 0.0, 1.0), index=XA.index),
      'NEWS-HALF': pd.Series(np.where(NEWS.isin(['CPI day', 'NFP day', 'FOMC day']), .5, 1.0), index=XA.index), 'NEWS-UP': pd.Series(np.where(NEWS.isin(['CPI day', 'NFP day']), 1.5, 1.0), index=XA.index),
      'LATE-HALF': pd.Series(np.where(TIME == '11:00-11:35', .5, 1.0), index=XA.index), 'LATE-SKIP': pd.Series(np.where(TIME == '11:00-11:35', 0.0, 1.0), index=XA.index),
      'ADX-HALF': pd.Series(np.where(slope_p < 0, .5, 1.0), index=XA.index), 'EMA-HALF': pd.Series(np.where(EMA.str.endswith('disagrees'), .5, 1.0), index=XA.index),
      'DRIFT-HALF': pd.Series(np.where(DRIFT == 'drift disagrees', .5, 1.0), index=XA.index)}
VAR = {}
for nm, fm in FM.items():
    DS[f'e_{nm}'] = to_D(ev_res, eqm(.3) * fm); DS[f'f_{nm}'] = to_D(full, eqm(.7) * fm); VAR[nm] = (f'e_{nm}', f'f_{nm}', f'f_{nm}')
for nm, mode in (('EWMA', 'ewma'), ('SHADOW', 'shadow')):
    DS[f'e_{nm}'] = to_D(ev_res, eqm(.3, mode)); DS[f'f_{nm}'] = to_D(full, eqm(.7, mode)); VAR[nm] = (f'e_{nm}', f'f_{nm}', f'f_{nm}')
if __name__ == '__main__':
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job10, [(nm, 'LUCID 50K') for nm in VAR])
    print('\nFINAL SYSTEM WITH EACH GUARD (Lucid 50K)', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<11} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
