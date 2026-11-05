"""Nov 4 2026: 5m and 1m vectors for the NQ NY AM trade, any day vs negative-gamma days. DECLARED BEFORE RESULTS.
PART A  QQQ 5m / 15m bars 2018 - Oct 2026 (QQQ proxy -> NQ pts), vector state from premarket + RTH bars (live settings), signal bars closing 09:30-11:30, entry next bar open,
        both ways, <= 2 a day (next only after the previous closed), 1 NQ 75/50, flat 12:00, 1 pt cost.
   sets: F15 15m flip (the base) | F15-NEG 15m on negative-gamma days | F5 5m flip | F5-NEG | F5x15 5m flip only in the direction of the 15m vector | F5x15-NEG
   per set: flips/day, trades, avg pts 2018-21 / 2022-23 / 2024-26, coin-flip avg (10 draws), realistic months 2022 - Oct 2026,
   and the firm sims (nq_firms.py rules, real-time drawdown) for TOPSTEP and LUCID FLEX: flat 1 NQ (E1/F1) and the full sizing stack in eval + funded (EALL/FALL:
   THIN 60-day path + HAR high + VWAP extended; flags recomputed on each set, cut-offs from its own 2018-21).
PART B  real MNQ 1-minute prices Dec 2024 - Nov 2025 (last price + VWAP only: 1m bar = open at the prior close, high / low = max / min of open, close, VWAP),
        1m / 5m / 15m vectors built from the same minutes, same trade rules, path checked on 1m bars, any day vs negative gamma. One year only."""
_f2 = open('xs/nq_firms2.py').read(); exec(_f2[:_f2.rindex("if __name__ == '__main__':")])
G_ = G.sort_index()
def negday(d): g = G_[G_.index < d]; return bool(len(g) and g.iloc[-1] < 0)
NEG = {d: negday(d) for d in days}
def cset(reg, me_close, ent, dts_day):
    out = []
    for k in range(2, len(reg)):
        if reg[k] == reg[k - 1] or reg[k - 1] == 0 or not (570 < me_close[k] <= 690): continue
        j = ent[k]
        if j < 0 or j >= len(H5) or d5[j] != dts_day[k]: continue
        out.append((pd.Timestamp(dts_day[k]), int(j), int(reg[k])))
    X = pd.DataFrame(out, columns=['day', 'j', 'dir']); X['k'] = NQs.reindex(pd.DatetimeIndex(X.day) - pd.Timedelta(days=1), method='ffill').values / o5[X.j.values]
    X = X[np.isfinite(X.k)].reset_index(drop=True); X['neg'] = X.day.map(NEG).astype(bool); return X
reg5 = np.asarray(vec(H5[['open', 'high', 'low', 'close']].reset_index(drop=True)))
C5 = cset(reg5, m5 + 5, np.arange(len(H5)) + 1, d5)
s15 = st_at(H15.end.values, H15.reg.values, t5[C5.j.values]); C5x = C5[s15 == C5.dir.values].reset_index(drop=True)
C15 = C[['day', 'j', 'dir', 'k']].reset_index(drop=True); C15['neg'] = C15.day.map(NEG).astype(bool)
def flags_any(Cc):
    Cc = Cc.reset_index(drop=True).copy(); kq = np.searchsorted(H15.end.values, t5[Cc.j.values], side='right') - 1; vw = []; ah = []
    for j, d_, a_, day, k_ in zip(Cc.j.values, Cc.dir.values, atr15[np.maximum(kq, 0)], Cc.day.values, Cc.k.values):
        i0 = j
        while i0 > 0 and d5[i0 - 1] == d5[j] and m5[i0 - 1] >= 570: i0 -= 1
        w = vol5[i0:j]; v_ = (tp[i0:j] * w).sum() / w.sum() if j > i0 and w.sum() > 0 else np.nan; vw.append((o5[j] - v_) * d_ / a_)
        day = pd.Timestamp(day); key = (day, 60)
        if key not in cache: cache[key] = prof(day, 60)
        P_ = cache[key]; e = o5[j]; eb = int(np.floor(np.log(e) / BW)); tb = int(np.floor(np.log(e + d_ * 75 / k_) / BW))
        ah.append(np.nan if P_ is None else zone(P_[0], P_[1], eb + d_, tb))
    Cc['vwapd'] = vw; Cc['ahead60'] = ah; isb = Cc.day < P0
    Cc['THIN'] = Cc.ahead60 < Cc.loc[isb, 'ahead60'].quantile(1 / 3); Cc['HARH'] = prior(HP, Cc.day) > 2 / 3; Cc['VWAPX'] = Cc.vwapd > Cc.loc[isb, 'vwapd'].quantile(2 / 3); return Cc
SETS = {'F15 15m flip (base)': C15, 'F15-NEG': C15[C15.neg], 'F5 5m flip': C5, 'F5-NEG': C5[C5.neg], 'F5x15 5m flip with 15m': C5x, 'F5x15-NEG': C5x[C5x.neg]}
rng = np.random.default_rng(9); PER = ('18-21', '22-23', '24-26')
print('PART A: QQQ 2018 - Oct 2026', flush=True)
for nm, Cc in SETS.items():
    Cc = flags_any(Cc); SETS[nm] = Cc; R = run_l(Cc); R['per'] = np.where(R.day < P0, '18-21', np.where(R.day < '2024-01-01', '22-23', '24-26'))
    cf = [run_l(Cc.assign(dir=rng.choice([-1, 1], len(Cc)))).net.mean() for _ in range(10)]; M_ = months((R.net * 20).groupby(R.day).sum()); M6 = M_[M_.month >= '2026-01']
    pm = R.groupby('per').net.agg(['mean', 'size'])
    print(f'  {nm:<24} flips/day {len(Cc) / len(days):.2f} trades {len(R)} win {R.net.gt(0).mean():.0%} | avg ' + ' / '.join(f'{p} {pm.loc[p, "mean"]:+5.1f} (n{int(pm.loc[p, "size"])})' for p in PER) +
          f' | coin-flip {np.median(cf):+.1f} | months passed {M_.res.eq("passed").mean():4.0%} lost {M_.res.eq("lost").mean():4.0%} mean {M_.pnl.mean():+5,.0f} | 2026 passed {M6.res.eq("passed").sum()}/{len(M6)}', flush=True)
DAYS_SET = {nm: build_day(Cc) for nm, Cc in SETS.items()}
# ---------------------------------------------------------------- PART B: MNQ minutes
def partB():
    M = pd.read_parquet(f'{ROOT}/mnq_orderflow_1min.parquet'); M['dt'] = pd.to_datetime(M.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    M = M.dropna(subset=['last_price']).sort_values('dt').drop_duplicates('dt'); M['m'] = M.dt.dt.hour * 60 + M.dt.dt.minute
    M = M[(M.m >= 240) & (M.m < 960) & (M.dt.dt.dayofweek < 5)].reset_index(drop=True); M['d'] = M.dt.dt.normalize()
    c = M.last_price.values; o = np.r_[c[0], c[:-1]]; o = np.where(M.d.values != np.r_[M.d.values[0], M.d.values[:-1]], c, o); vw_ = M.vwap.fillna(M.last_price).values
    B1 = pd.DataFrame({'dt': M.dt, 'open': o, 'high': np.maximum.reduce([o, c, vw_]), 'low': np.minimum.reduce([o, c, vw_]), 'close': c})
    def agg(B, f):
        x = B.set_index('dt'); g = x.groupby([x.index.normalize(), pd.Grouper(freq=f, origin='start_day', offset='4h')]).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        g.index = g.index.get_level_values(1); g = g.reset_index(); return g
    t1 = B1.dt.values; d1 = M.d.values; m1 = M.m.values; h1, l1, c1, o1 = B1.high.values, B1.low.values, B1.close.values, B1.open.values
    def trades(B, mins):
        reg = np.asarray(vec(B[['open', 'high', 'low', 'close']])); end = (B.dt + pd.Timedelta(minutes=mins)).values; me = pd.DatetimeIndex(end).hour * 60 + pd.DatetimeIndex(end).minute
        cand = []
        for k in range(2, len(B)):
            if reg[k] == reg[k - 1] or reg[k - 1] == 0 or not (570 < me[k] <= 690): continue
            j = np.searchsorted(t1, end[k])
            if j < len(t1) and d1[j] == np.datetime64(pd.Timestamp(end[k]).normalize()): cand.append((pd.Timestamp(d1[j]), j, int(reg[k])))
        X = pd.DataFrame(cand, columns=['day', 'j', 'dir']); X['neg'] = X.day.map(lambda d: negday(d)); return X
    def run1(X):
        out = []
        for day, g in X.sort_values('j').groupby('day'):
            busy = -1; n = 0
            for j, d in zip(g.j.values, g.dir.values):
                if n >= 2 or j <= busy: continue
                e = o1[j]; res = None
                for q in range(j, len(t1)):
                    if d1[q] != d1[j] or m1[q] >= 720: res = ((c1[q - 1] - e) * d, q - 1); break
                    lo_ = ((l1[q] if d > 0 else h1[q]) - e) * d; hi_ = ((h1[q] if d > 0 else l1[q]) - e) * d
                    if lo_ <= -50: res = (-50, q); break
                    if hi_ >= 75: res = (75, q); break
                if res: out.append((day, res[0] - 1.0)); busy = res[1]; n += 1
        return pd.DataFrame(out, columns=['day', 'net'])
    print(f'\nPART B: real MNQ minutes {M.d.min():%Y-%m-%d} -> {M.d.max():%Y-%m-%d}, {M.d.nunique()} sessions (path on 1m bars)', flush=True)
    for lab, B, mins in (('1m vector', B1, 1), ('5m vector', agg(B1, '5min'), 5), ('15m vector', agg(B1, '15min'), 15)):
        X = trades(B, mins)
        for sub, nm in ((X, 'any day'), (X[X.neg], 'negative gamma')):
            R = run1(sub); h = R.day < '2025-06-01'; cf = [run1(sub.assign(dir=rng.choice([-1, 1], len(sub)))).net.mean() for _ in range(10)]
            print(f'  {lab:<11} {nm:<15} flips/day {len(sub) / M.d.nunique():5.2f} trades {len(R):>3} win {R.net.gt(0).mean():.0%} avg {R.net.mean():+5.1f} (Dec-May {R[h].net.mean():+5.1f} / Jun-Nov {R[~h].net.mean():+5.1f}) | coin-flip {np.median(cf):+.1f} | total {R.net.sum():+6.0f} pts', flush=True)
if __name__ == '__main__':
    partB()
    jobs = [(nm, fn, en, fsn) for nm in SETS for fn in ('TOPSTEP', 'LUCID FLEX') for en, fsn in (('E1', 'F1'), ('EALL', 'FALL'))]
    def runjob(a):
        global DAY
        nm, fn, en, fsn = a; DAY = DAYS_SET[nm]; return dict(job2((fn, en, fsn)), system=nm)
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(runjob, jobs)
    print('\nFIRM SIMS (real-time drawdown; starts 2022 - 2025)', flush=True)
    for r in RES:
        print(f'  {r["system"]:<24} {r["firm"]:<11} {r["eval"]:<4} {r["funded"]:<4} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m paid {r["k_paid"]:4.0%} median {r["k_med"]:+6,.0f}', flush=True)
