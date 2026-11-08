"""Nov 8 2026: SAVED THETADATA OPTIONS + MNQ ORDER FLOW on every NQ flip. DECLARED BEFORE RESULTS.
Flips: every RTH 15m flip 2018 - Oct 2026 (nq_flip_features), outcomes TH (held to the 15m vector exit) and F75 (75 / 50), NQ points per flip.
OPTIONS at the prior close (QQQ, ThetaData): own net GEX (xs/ext/gex/QQQ.csv, to Oct 2026) sign and 1-year percentile; call / put walls in NQ points (gex_walls_nq.csv, to Dec 2025):
  room to the wall the trade runs into (call wall for longs, put wall for shorts) in 20-day ADR, and whether price already sits beyond it; vanna flow (sign(net vanna) x -sign(5-day IV change))
  agrees with the trade; charm sign agrees; 25-delta skew, put/call volume, ATM IV and its 5-day change as 1-year percentiles (greeks to Dec 15 2025).
ORDER FLOW (MNQ 1-minute, Dec 2024 - Nov 2025, sign corrected: aggressor buys - sells): flip-bar delta agrees with the trade; session CVD (from 09:30) agrees; flip-bar delta / volume
  (aggression) top third; absorption (flip-bar volume top third but range bottom half).
A label counts when its best bucket beats its worst in every period with data (options: 2018-21 / 22-23 / 24-25 [+2026 for GEX]; order flow: Dec-May / Jun-Nov)."""
_pd = open('xs/nq_points_day.py').read(); exec(_pd[:_pd.index("L = pd.read_csv")])
ROOT = '/mnt/user-data/uploads/fartpie'
X = pd.read_parquet('/tmp/claude-0/ml/nq_flip_features.parquet'); X['t'] = t5[X.j.values]; X['e_nq'] = o5[X.j.values] * k5[X.j.values]
D = H.groupby('day').agg(h=('high', 'max'), l=('low', 'min')); adr_nq = ((D.h - D.l) * pd.Series(k5, index=H.index).groupby(H.day).first()).rolling(20).mean().shift(1)
X['adr'] = adr_nq.reindex(pd.DatetimeIndex(X.day)).values
O1 = pd.read_csv(f'{ROOT}/options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
O2 = pd.read_csv(f'{ROOT}/options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
O3 = pd.read_csv(f'{ROOT}/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
W = pd.read_csv(f'{ROOT}/gex_walls_nq.csv', parse_dates=['date']).set_index('date').sort_index()
pct = lambda s: s.rolling(252, min_periods=120).rank(pct=True)
def prior(s, days):
    s = s.dropna(); s.index = pd.DatetimeIndex(s.index).normalize(); i = s.index.searchsorted(pd.DatetimeIndex(days), side='left') - 1
    return np.where(i >= 0, s.values[np.clip(i, 0, None)], np.nan)
dd = pd.DatetimeIndex(X.day)
X['gex'] = prior(O3.net_gex, dd); X['gexpct'] = prior(pct(O3.net_gex), dd)
X['cw'] = prior(W.call_wall_nq, dd); X['pw'] = prior(W.put_wall_nq, dd)
wall = np.where(X.dir > 0, X.cw, X.pw); X['wall_room'] = (wall - X.e_nq) * X.dir / X.adr
iv5 = O3.atm_iv.diff(5); vflow = np.sign(O2.net_vanna) * -np.sign(iv5.reindex(O2.index))
X['vanna_flow'] = prior(vflow, dd) * X.dir; X['charm'] = np.sign(prior(O2.net_charm, dd)) * X.dir
X['skew25'] = prior(pct(O1.skew_25d), dd); X['pcvol'] = prior(pct(O3.put_vol / O3.call_vol), dd); X['iv'] = prior(pct(O3.atm_iv), dd); X['iv5'] = np.sign(prior(iv5, dd))
PER = ['18-21', '22-23', '24-25', '2026']
def table(name, lab, pers=PER, col='th', data=None):
    Z = X if data is None else data; out = []
    for v in sorted(lab.dropna().unique(), key=str):
        g = Z[lab == v]; out.append((v, len(g), g[col].mean(), [g[g.per == p][col].mean() if (g.per == p).sum() >= 8 else np.nan for p in pers]))
    if len(out) < 2: return None
    best, worst = max(out, key=lambda o: o[2]), min(out, key=lambda o: o[2])
    cons = all(np.sign(b - w) == np.sign(best[2] - worst[2]) for b, w in zip(best[3], worst[3]) if np.isfinite(b) and np.isfinite(w))
    print(f'  -- {name} [{col}] spread {best[2] - worst[2]:+.1f} (best {best[0]}, worst {worst[0]}){"  * every period" if cons else ""}', flush=True)
    for v, n, m, pp in out: print(f'    {str(v):<16} n{n:>4} {m:+6.1f} (' + ' / '.join(f'{x:+.1f}' for x in pp) + ')', flush=True)
    return (name, col, best[2] - worst[2], cons, str(best[0]))
T3 = lambda s: pd.qcut(s.rank(method='first'), 3, labels=['low', 'mid', 'high']).astype(str).where(s.notna())
RES = []
print(f'OPTIONS (per flip; periods 18-21 / 22-23 / 24-25 / 2026)', flush=True)
for col in ('th', 'f75'):
    RES += [table('GEX sign (own, to Oct 2026)', pd.Series(np.where(X.gex.isna(), None, np.where(X.gex > 0, 'positive', 'negative')), index=X.index), col=col),
            table('GEX percentile', T3(X.gexpct), col=col),
            table('room to wall (ADR)', pd.Series(np.select([X.wall_room.isna(), X.wall_room < 0, X.wall_room < .5, X.wall_room < 1.5], [None, 'beyond wall', '< 0.5', '0.5-1.5'], '> 1.5'), index=X.index), col=col),
            table('vanna flow agrees', X.vanna_flow.map({1.0: 'agrees', -1.0: 'against'}), col=col), table('charm agrees', X.charm.map({1.0: 'agrees', -1.0: 'against'}), col=col),
            table('skew 25-delta', T3(X['skew25']), col=col), table('put/call volume', T3(X.pcvol), col=col), table('ATM IV', T3(X.iv), col=col),
            table('IV 5-day change', X.iv5.map({1.0: 'rising', -1.0: 'falling'}), col=col)]
# ---- order flow
M = pd.read_parquet(f'{ROOT}/mnq_orderflow_1min.parquet'); M['dt'] = pd.to_datetime(M.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
M = M.dropna(subset=['last_price']).sort_values('dt').drop_duplicates('dt').set_index('dt'); M['delta'] = -M.signed_vol.fillna(M.buy_vol.fillna(0) - M.sell_vol.fillna(0))
M['vol'] = M.buy_vol.fillna(0) + M.sell_vol.fillna(0); M['m'] = M.index.hour * 60 + M.index.minute; rth = M[(M.m >= 570) & (M.m < 960)]
cvd = rth.delta.groupby(rth.index.normalize()).cumsum()
inwin = (X.t >= M.index.min() + pd.Timedelta(days=1)) & (X.t <= M.index.max()); Y = X[inwin].copy()
fb_d, fb_v, fb_r, cv = [], [], [], []
for t in Y.t:
    w = M.loc[t - pd.Timedelta('15min'): t - pd.Timedelta('1min')]; fb_d.append(w.delta.sum()); fb_v.append(w.vol.sum()); fb_r.append(w.last_price.max() - w.last_price.min() if len(w) else np.nan)
    c_ = cvd.loc[:t - pd.Timedelta('1min')]; cv.append(c_.iloc[-1] if len(c_) and c_.index[-1].normalize() == t.normalize() else np.nan)
Y['fbd'] = np.sign(fb_d) * Y.dir; Y['cvd'] = np.sign(cv) * Y.dir; Y['aggr'] = np.array(fb_d) * Y.dir / np.maximum(fb_v, 1)
Y['absorb'] = (pd.Series(fb_v, index=Y.index).rank(pct=True) >= 2 / 3) & (pd.Series(fb_r, index=Y.index).rank(pct=True) <= .5)
Y['per'] = np.where(Y.day < '2025-06-01', 'Dec-May', 'Jun-Nov'); P2 = ['Dec-May', 'Jun-Nov']
print(f'\nORDER FLOW: {len(Y)} flips Dec 2024 - Nov 2025 (halves Dec-May / Jun-Nov)', flush=True)
for col in ('th', 'f75'):
    RES += [table('flip-bar delta agrees', Y.fbd.map({1.0: 'agrees', -1.0: 'against'}), P2, col, Y), table('session CVD agrees', Y.cvd.map({1.0: 'agrees', -1.0: 'against'}), P2, col, Y),
            table('aggression (delta / volume)', T3(Y.aggr), P2, col, Y), table('absorption on the flip bar', Y.absorb.map({True: 'absorption', False: 'none'}), P2, col, Y)]
print('\nSUMMARY: * = best beats worst in every period', flush=True)
for r in [r for r in RES if r]: print(f'  {"*" if r[3] else " "} {r[0]:<30} {r[1]:<4} spread {r[2]:+6.1f} best {r[4]}', flush=True)
X.to_parquet('/tmp/claude-0/ml/nq_flip_features_opt.parquet'); Y.to_parquet('/tmp/claude-0/ml/nq_flip_features_of.parquet')
