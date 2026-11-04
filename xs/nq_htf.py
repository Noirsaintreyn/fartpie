"""Nov 4 2026: higher-timeframe and cross-market vectors for the NQ NY AM trade (QQQ proxy, 150/75, 1 NQ, 2018 - Sep 2026). DECLARED BEFORE RESULTS.
Base = S0 (15m QQQ vector flip, bar closing 09:30-11:30, negative gamma, first signal of the day). Filters, each state taken at the signal time from COMPLETED bars only:
  H1   QQQ 1h vector (09:30-anchored hours, premarket included) agrees with the trade      H4  QQQ 4h vector (04:00-anchored) agrees
  SPY  SPY 15m vector agrees                                                             MEGA >= 5 of the 8 largest Nasdaq names (AAPL MSFT NVDA AMZN META GOOGL AVGO TSLA) 15m vectors agree
  ALL4 all four agree
Also as TRIGGERS instead of the 15m flip: T1H  QQQ 1h flip (bar closing 10:30 or 11:30), negative gamma;  TMEGA  the hour the megacap 15m breadth crosses 5 of 8 in one direction (09:45-11:30), negative gamma.
Each: trades, trades/week, win, avg, halves, coin-flip avg, eval pass vs coin-flip (rules as nq_prop_of.py)."""
_s = open('xs/nq_ctx.py').read(); exec(_s[:_s.index("for T, S in ((100, 50), (150, 75)):")])
def st_at(df_end, reg, when):
    """state of the last bar whose END <= when"""
    i = np.searchsorted(df_end, np.asarray(when, dtype='datetime64[ns]'), side='right') - 1
    return np.where(i >= 0, reg[np.clip(i, 0, None)], 0)
x = H5.set_index('dt')[['open', 'high', 'low', 'close']]
def aggr(freq, off):
    g = x.groupby([x.index.normalize(), pd.Grouper(freq=freq, origin='start_day', offset=off)]).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    g.index = g.index.get_level_values(1); g = g.reset_index(); g.columns = ['dt', 'open', 'high', 'low', 'close']; g['end'] = (g.dt + pd.Timedelta(freq)).values; return g
Q1, Q4 = aggr('1h', '9h30min'), aggr('4h', '4h')
for q in (Q1, Q4): q['reg'] = np.asarray(vec(q[['open', 'high', 'low', 'close']]))
def l15(path):
    H = load(path); H15_ = agg15(H); H15_['reg'] = np.asarray(vec(H15_[['open', 'high', 'low', 'close']])); H15_['end'] = (H15_.dt + pd.Timedelta('15min')).values; return H15_
SPY15 = l15(glob.glob(f'{ROOT}/lse_ohlcv_cache/SPY_15m_*.parquet')[0])
MEGA = {t: l15(glob.glob(f'{ROOT}/lse_clean/{t}_15m_*.parquet')[0]) for t in ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'AVGO', 'TSLA')}
print('loaded; SPY 15m through', SPY15.dt.max(), flush=True)
sig_t = t5[C2.j.values]                                                     # signal time = the 15m bar's end = entry bar open
C2['H1'] = st_at(Q1.end.values, Q1.reg.values, sig_t) == C2.dir.values; C2['H4'] = st_at(Q4.end.values, Q4.reg.values, sig_t) == C2.dir.values
C2['SPY'] = st_at(SPY15.end.values, SPY15.reg.values, sig_t) == C2.dir.values
mb = np.zeros(len(C2))
for t, B in MEGA.items(): mb += (st_at(B.end.values, B.reg.values, sig_t) == C2.dir.values).astype(int)
C2['MEGA'] = mb >= 5; C2['ALL4'] = C2.H1 & C2.H4 & C2.SPY & C2.MEGA
def report(nm, sub, T=150, S=75):
    R = run_q(sub, T, S)
    if len(R) < 20: print(f'  {nm:<44} n{len(R)} (too few)'); return
    cf, cp = [], []
    for _ in range(10):
        Z = sub.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); Rc = run_q(Z, T, S); cf.append(Rc.pts.mean()); cp.append(prop(Rc, days)[0])
    pv, mdn, _ = prop(R, days); a, b = R[R.day < '2022-01-01'], R[R.day >= '2022-01-01']
    print(f'  {nm:<44} n{len(R):>4} ({len(R) / len(days) * 5:.2f}/wk) win {R.pts.gt(0).mean():.0%} avg {R.pts.mean():+6.1f} (coin-flip {np.median(cf):+.1f}) | H1 {a.pts.mean():+6.1f} / H2 {b.pts.mean():+6.1f} | eval pass {pv:.0%} vs coin-flip {np.nanmedian(cp):.0%}, median {mdn:.0f} sessions', flush=True)
rng = np.random.default_rng(11)
print('\nFILTERS on S0 (150/75, 1 NQ)', flush=True)
report('S0', C2)
for k in ('H1', 'H4', 'SPY', 'MEGA', 'ALL4'): report(f'S0 + {k} agrees', C2[C2[k]]); report(f'S0 + {k} DISagrees', C2[~C2[k]])
print('\nTRIGGERS (negative gamma, first of the day)', flush=True)
Q1['me'] = pd.to_datetime(Q1.end).dt.hour * 60 + pd.to_datetime(Q1.end).dt.minute; r = Q1.reg.values; TT = []
for k in range(2, len(Q1)):
    if r[k] == r[k - 1] or r[k - 1] == 0 or Q1.me.iloc[k] not in (630, 690): continue
    day = Q1.dt.iloc[k].normalize(); g = G[G.index < day]
    if not len(g) or g.iloc[-1] >= 0: continue
    j = np.searchsorted(t5, np.datetime64(Q1.end.iloc[k]))
    if j < len(H5) and d5[j] == np.datetime64(day): TT.append((day, int(j), int(r[k])))
T1 = pd.DataFrame(TT, columns=['day', 'j', 'dir']).groupby('day').head(1).reset_index(drop=True)
T1['k'] = NQs.reindex(pd.DatetimeIndex(T1.day) - pd.Timedelta(days=1), method='ffill').values / o5[T1.j.values]; report('T1H QQQ 1h flip', T1)
grid = SPY15[['end']].copy(); grid = grid[(pd.to_datetime(grid.end).dt.hour * 60 + pd.to_datetime(grid.end).dt.minute).between(585, 690)]
bt = np.zeros(len(grid)); bb = np.zeros(len(grid))
for t, B in MEGA.items(): s_ = st_at(B.end.values, B.reg.values, grid.end.values); bt += (s_ == 1); bb += (s_ == -1)
grid['up'], grid['dn'] = bt >= 5, bb >= 5; grid['day'] = pd.to_datetime(grid.end).dt.normalize(); TM = []
for day, g in grid.groupby('day'):
    gg = G[G.index < day]
    if not len(gg) or gg.iloc[-1] >= 0: continue
    prev_u = prev_d = None
    for e_, u, dn in zip(g.end.values, g.up.values, g.dn.values):
        if prev_u is not None and ((u and not prev_u) or (dn and not prev_d)):
            j = np.searchsorted(t5, np.datetime64(e_))
            if j < len(H5) and d5[j] == np.datetime64(day): TM.append((day, int(j), 1 if u and not prev_u else -1)); break
        prev_u, prev_d = u, dn
TMg = pd.DataFrame(TM, columns=['day', 'j', 'dir']); TMg['k'] = NQs.reindex(pd.DatetimeIndex(TMg.day) - pd.Timedelta(days=1), method='ffill').values / o5[TMg.j.values]
report('TMEGA megacap breadth crosses 5 of 8', TMg)
