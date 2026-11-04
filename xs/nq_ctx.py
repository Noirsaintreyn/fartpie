"""Nov 4 2026: does the screener's context (stress, SPY z-score, down markets, VIX, options) sharpen the NQ NY AM trade? DECLARED BEFORE RESULTS.
Trades = S0 from xs/nq_prop_long.py (QQQ 15m vector flip, bar closing 09:30-11:30, negative QQQ gamma at the prior close, first signal of the day, 5m path, NQ points, 1 pt cost),
targets/stops 100/50 and 150/75. Context, all known at the PRIOR close (or the 09:30 open for the gap):
  REG     SPY vs its 200-day x VIX/VIX3M > 1 (stress): bull calm / bull stress / bear calm / bear stress   (bull+stress was the screener's best cell)
  Z20     SPY 20-day z-score: < -1.5 stretched down / -1.5..1.5 / > 1.5 stretched up
  MKT20   SPY 20-day return < 0 (market down) vs >= 0
  VIXP    VIX 1-year percentile: low / mid / high terciles
  GEXQ    how negative the gamma is: deepest third / middle / shallowest third of the negative days
  PCR     QQQ put/call volume at the prior close: > 1.5 vs <= 1.5 (fear in the options tape)
  GAP     trade WITH the 09:30 gap (QQQ open vs prior close) vs against it
Each cell: n, win rate, avg pts, both halves (2018-21 / 2022-26). A cell counts only if it beats all S0 trades in BOTH halves with n >= 25 in each.
The best few then get the prop eval (pass rate vs coin-flip)."""
_s = open('xs/nq_prop_long.py').read(); exec(_s[:_s.index("rng = np.random.default_rng(0)")])
Y = yf.download(['SPY', '^VIX', '^VIX3M'], start='2016-01-01', interval='1d', progress=False, auto_adjust=True)['Close'].ffill(); Y.index = Y.index.normalize()
spy = Y['SPY']; X = pd.DataFrame({'bull': spy > spy.rolling(200).mean(), 'stress': Y['^VIX'] / Y['^VIX3M'] > 1, 'z20': (spy - spy.rolling(20).mean()) / spy.rolling(20).std(),
                                  'r20': spy / spy.shift(20) - 1, 'vixp': Y['^VIX'].rolling(252, min_periods=120).rank(pct=True)})
GX = pd.read_csv(f'{ROOT}/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); GX['pcr'] = GX.put_vol / GX.call_vol.replace(0, np.nan)
def prior(df, days): return df.reindex(pd.DatetimeIndex(days) - pd.Timedelta(days=1), method='ffill').reset_index(drop=True)
P = prior(X, C.day); Q = prior(GX[['net_gex', 'pcr']], C.day)
C2 = pd.concat([C.reset_index(drop=True), P, Q], axis=1)
pc = np.array([c5[np.searchsorted(t5, np.datetime64(d)) - 1] for d in C2.day])                          # prior 15:55 close (last 5m bar before the day)
op = np.array([o5[np.searchsorted(t5, np.datetime64(d + pd.Timedelta(hours=9, minutes=30)))] for d in C2.day]); C2['gap_dir'] = np.sign(op - pc)
C2['REG'] = np.where(C2.bull, 'bull ', 'bear ') + np.where(C2.stress, 'stress', 'calm')
C2['Z20'] = pd.cut(C2.z20, [-99, -1.5, 1.5, 99], labels=['stretched down', 'normal', 'stretched up']).astype(str)
C2['MKT20'] = np.where(C2.r20 < 0, 'SPY 20d down', 'SPY 20d up'); C2['VIXP'] = pd.cut(C2.vixp, [-1, 1 / 3, 2 / 3, 2], labels=['VIX low', 'VIX mid', 'VIX high']).astype(str)
C2['GEXQ'] = pd.qcut(C2.net_gex.rank(method='first'), 3, labels=['gamma deepest', 'gamma mid', 'gamma shallow']).astype(str)
C2['PCR'] = np.where(C2.pcr > 1.5, 'put/call > 1.5', 'put/call <= 1.5'); C2['GAP'] = np.where(C2.gap_dir == C2.dir, 'with the gap', 'against the gap')
C2['DIR'] = np.where(C2.dir > 0, 'long', 'short')
for T, S in ((100, 50), (150, 75)):
    R = run_q(C2, T, S).reset_index(drop=True); Z = pd.concat([C2.reset_index(drop=True), R[['pts']]], axis=1).dropna(subset=['pts']); Z['h'] = np.where(Z.day < '2022-01-01', 'H1', 'H2')
    base = {h: Z[Z.h == h].pts.mean() for h in ('H1', 'H2')}
    print(f'\n== {T}/{S}: all S0 trades n{len(Z)} win {Z.pts.gt(0).mean():.0%} avg {Z.pts.mean():+.1f} pts (H1 {base["H1"]:+.1f} / H2 {base["H2"]:+.1f})', flush=True)
    keep = []
    for col in ('DIR', 'REG', 'Z20', 'MKT20', 'VIXP', 'GEXQ', 'PCR', 'GAP'):
        for v, g in Z.groupby(col):
            a, b = g[g.h == 'H1'], g[g.h == 'H2']; ok = len(a) >= 25 and len(b) >= 25 and a.pts.mean() > base['H1'] and b.pts.mean() > base['H2']
            print(f'  {col:<6} {v:<17} n{len(g):>4} win {g.pts.gt(0).mean():.0%} avg {g.pts.mean():+6.1f} | H1 n{len(a):>3} {a.pts.mean():+6.1f} / H2 n{len(b):>3} {b.pts.mean():+6.1f}' + ('   <- both halves' if ok else ''), flush=True)
            if ok: keep.append((col, v))
    rng = np.random.default_rng(3)
    for col, v in keep:
        sub = C2[C2[col] == v]; R2 = run_q(sub, T, S); pv = prop(R2, days)[0]; cf = []
        for _ in range(10): Zc = sub.copy(); Zc['dir'] = rng.choice([-1, 1], len(Zc)); cf.append(prop(run_q(Zc, T, S), days)[0])
        print(f'    eval {col}={v}: n{len(R2)} ({len(R2) / len(days) * 5:.2f}/wk) pass {pv:.0%} vs coin-flip median {np.nanmedian(cf):.0%}', flush=True)
