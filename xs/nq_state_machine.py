"""Nov 4 2026: Hurst x volatility STATE MACHINE that decides the trade type, walk-forward, with a placebo. DECLARED BEFORE RESULTS.
States at the PRIOR close: TREND / MR = QQQ 120-day Hurst (R/S on daily log returns) above / below its trailing 1-year median; HIGH / LOW vol = Yang-Zhang 20d in / not in its top third.
Trade types: WITH = flip in the 1h vector's direction, AGAINST = flip against it (others unchanged). Base = final system (Rule A, Yang-Zhang + VWAP sizing, 50 / 75, gap-gated eval).
  THEORY   TREND states: AGAINST x0.5;  MR states: WITH x0.5   (fixed, not fitted)
  LEARN-X  per state, halve the type with the worse average among that state's EARLIER Rule A trades (all history before the trade; >= 10 of each, else unchanged)
  LEARN-24 same, earlier trades in the last 24 months only
  PLACEBO  LEARN-X on the state series circularly shifted by 1-6 years (20 draws): same machinery, no real regime information
Diagnostics: per state and year, WITH minus AGAINST avg pts (does a state's best trade type repeat?). Rule A $ by year; Lucid 50K ($88) + Topstep sims for normal / THEORY / LEARN-X / LEARN-24."""
_rg = open('xs/nq_regime_gate.py').read(); exec(_rg[:_rg.rindex("if __name__ == '__main__':")])
def hurst(x):
    x = x[np.isfinite(x)]
    if len(x) < 100: return np.nan
    ns, rs = [8, 16, 32, 64], []
    for n in ns:
        m = len(x) // n; seg = x[:m * n].reshape(m, n); z = np.cumsum(seg - seg.mean(1, keepdims=True), 1); r = z.max(1) - z.min(1); s = seg.std(1)
        rs.append(np.mean(r[s > 0] / s[s > 0]))
    return np.polyfit(np.log(ns), np.log(rs), 1)[0]
rr = np.log(RTD.c).diff().values; Hs = pd.Series([hurst(rr[max(0, i - 119):i + 1]) for i in range(len(rr))], index=RTD.index)
Htr = (Hs > Hs.rolling(252, min_periods=120).median()).shift(1)
yzp = yz.dropna().rolling(252, min_periods=120).rank(pct=True); Vhi = (yzp > 2 / 3).reindex(RTD.index).shift(1)
ST = pd.Series(np.where(Htr.fillna(False).astype(bool), 'TREND', 'MR'), index=RTD.index).astype(object) + '/' + pd.Series(np.where(Vhi.fillna(False).astype(bool), 'HIGH', 'LOW'), index=RTD.index); ST = ST.astype(object)
ST[Htr.isna()] = None
st_tr = pd.Series(ST.reindex(pd.DatetimeIndex(XA.day)).values, index=XA.index)
RLs = RL.join(st_tr.rename('st')); RLs['y'] = yr.reindex(RLs.index)
print('state mix of Rule A trades by year: ' + ' | '.join(f'{y} ' + str(RLs[RLs.y == y].st.value_counts(normalize=True).round(2).to_dict()) for y in ('2018-21', '2022', '2023', '2024', '2025', '2026')), flush=True)
print('\nWITH minus AGAINST the 1h vector (avg pts) by state and year', flush=True)
for s_ in sorted(RLs.st.dropna().unique()):
    row = []
    for y in ('2018-21', '2022', '2023', '2024', '2025', '2026'):
        g = RLs[(RLs.st == s_) & (RLs.y == y)]; w, a = g[g.with1h].net, g[g.ag1h].net
        row.append(f'{y} {w.mean() - a.mean():+6.1f} (n{len(w)}/{len(a)})' if len(w) >= 3 and len(a) >= 3 else f'{y}    n/a')
    print(f'  {s_:<10} ' + ' | '.join(row), flush=True)
def learn(states, window=None):
    out = pd.Series(1.0, index=XA.index); RLz = RL.join(states.rename('st'))
    for idx, day in zip(XA.index, XA.day.values):
        s_ = states.loc[idx]
        if s_ is None or (isinstance(s_, float) and np.isnan(s_)): continue
        past = RLz[(RLz.day.values < day) & (RLz.st == s_)]
        if window: past = past[past.day >= pd.Timestamp(day) - pd.DateOffset(months=window)]
        w, a = past[past.with1h].net, past[past.ag1h].net
        if len(w) < 10 or len(a) < 10: continue
        if w.mean() < a.mean() and w1.loc[idx]: out.loc[idx] = .5
        if a.mean() < w.mean() and ag1.loc[idx]: out.loc[idx] = .5
    return out
trend_ = st_tr.fillna('').str.startswith('TREND'); mr_ = st_tr.fillna('').str.startswith('MR')
MULT2 = {'always normal': MULT['always normal'], 'THEORY': pd.Series(np.where(trend_ & ag1, .5, np.where(mr_ & w1, .5, 1.0)), index=XA.index),
         'LEARN-X': learn(st_tr), 'LEARN-24': learn(st_tr, 24)}
def yearly(m): usd = RLx.net * 20 * RLx.bs * m.reindex(RLx.index); return {y: usd[RLx.y == y].sum() for y in ('2018-21', '2022', '2023', '2024', '2025', '2026')}
print('\nRule A $ by year (final sizing x state-machine sizing)', flush=True)
for nm, m in MULT2.items(): print(f'  {nm:<14} ' + ' | '.join(f'{y} {v:+7,.0f}' for y, v in yearly(m).items()) + f' | 2022-26 {sum(v for y, v in yearly(m).items() if y != "2018-21"):+8,.0f}', flush=True)
rng = np.random.default_rng(17); dl = ST.index; plc = []
for _ in range(20):
    off = int(rng.integers(252, 6 * 252)); sh = pd.Series(np.roll(ST.values, off), index=dl); s2 = pd.Series(sh.reindex(pd.DatetimeIndex(XA.day)).values, index=XA.index)
    yv = yearly(learn(s2)); plc.append((sum(v for y, v in yv.items() if y != '2018-21'), yv['2026']))
plc = np.array(plc); print(f'  PLACEBO LEARN-X (20 shifted state series): 2022-26 median {np.median(plc[:, 0]):+,.0f} (range {plc[:, 0].min():+,.0f} to {plc[:, 0].max():+,.0f}) | 2026 median {np.median(plc[:, 1]):+,.0f}', flush=True)
VAR = {}
for nm, m in MULT2.items():
    DS[f'e_{nm}'] = build(~AG['GAP'], mult=m); DS[f'f_{nm}'] = build(mult=m); VAR[nm] = (f'e_{nm}', f'f_{nm}', f'f_{nm}')
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job10, jobs)
    print('\nSTATE-MACHINE SYSTEMS (gap-gated eval; sizing in eval and funded)', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<14} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
