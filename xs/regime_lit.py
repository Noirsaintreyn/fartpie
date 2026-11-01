"""Nov 1 2026: market-regime gates from the literature, fixed before looking (no thresholds tuned). All known at the prior close.
 A  Garg-Goulding-Harvey-Mazzoleni states: fast = sign(SPY 21d return), slow = sign(SPY 252d return) -> Bull / Correction / Rebound / Bear
 B  Daniel-Moskowitz panic: SPY 24-month return < 0 (bear) AND 21d realized vol above its 1-year median
 C  Shu-Mulvey statistical jump model regime (jm_regimes.csv, lambda 35; 1 vs 0)
 D  Stambaugh-Yu-Yuan sentiment proxy: VIX below its 1-year median = calm / high sentiment
For each: avg long trade (book's trades) and avg short trade by state; the book with shorts ONLY in the favored state (shared slots and own
6-slot basket) and longs 1.5x in the favored long state; 40 matched perturbed runs vs live on return per drawdown."""
exec(open('/home/claude/fartpie/xs/baskets.py').read().split("L0 = dret(")[0])
spy = S.close
st = pd.DataFrame(index=S.index)
f, s_ = np.sign(spy / spy.shift(21) - 1), np.sign(spy / spy.shift(252) - 1)
st['A'] = np.select([(s_ > 0) & (f > 0), (s_ > 0) & (f <= 0), (s_ <= 0) & (f > 0)], ['Bull', 'Correction', 'Rebound'], 'Bear')
rv = np.log(spy).diff().rolling(21).std(); st['B'] = np.where((spy / spy.shift(504) - 1 < 0) & (rv > rv.rolling(252).median()), 'Panic', 'Normal')
J = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/jm_regimes.csv', parse_dates=['Date']).set_index('Date').lam35
st['C'] = J.reindex(st.index).map({1.0: 'JM-1', 0.0: 'JM-0'}).fillna('na')
vx = pd.read_parquet('/tmp/claude-0/ml/vix_daily.parquet').Close; vx.index = pd.to_datetime(vx.index).tz_localize(None).normalize()
vx = vx.reindex(st.index, method='ffill'); st['D'] = np.where(vx < vx.rolling(252).median(), 'Calm', 'Fearful')
st = st.shift(1)
nx = spy.pct_change().shift(-1); print('jump-model sanity (next-day SPY return by regime, %):', (nx.groupby(st['C']).mean() * 100).round(3).to_dict())
for T in (Ls, Sh):
    d = T.entry.dt.normalize()
    for k in 'ABCD': T[k] = st[k].reindex(d.values, method='ffill').values
TK = taken(Ls).merge(Ls[['tk', 'entry'] + list('ABCD')], on=['tk', 'entry'])
for k in 'ABCD':
    print(f'\n{k}: avg trade %   (longs the book takes | all bear-flip shorts)   2019-21 / 2022-26')
    a = TK.groupby([k, TK.entry.dt.year <= 2021]).ret.agg(['mean', 'size']); b = Sh.groupby([k, Sh.entry.dt.year <= 2021]).ret.agg(['mean', 'size'])
    for s in sorted(set(TK[k].dropna())):
        g = lambda t, h: f"{t.loc[(s, h), 'mean'] * 100:+.2f}% ({int(t.loc[(s, h), 'size'])})" if (s, h) in t.index else '  -  '
        print(f'  {s:<11} long {g(a, True)} / {g(a, False)}    short {g(b, True)} / {g(b, False)}')
FAV = {'A': ('Bull', 'Rebound'), 'B': ('Normal', 'Panic'), 'C': ('JM-1', 'JM-0'), 'D': ('Calm', 'Fearful')}   # (short-favored state, long-favored state) - from the papers
def build(k, mode, dl, ds):
    l = Ls[~dl].copy(); s = Sh[~ds & (Sh[k] == FAV[k][0]).values]
    if mode == 'long1.5': l['w'] = np.where(l[k] == FAV[k][1], 1.5, 1.0); return dret(simw2(l))
    if mode == 'share': return dret(simw(pd.concat([l, s])))
    if mode == 'own6': return dret(simw(l)) + dret(simw(s, n=6))
def simw2(Z, n=6, brk=.08):                                     # per-trade weight w (x 1/6)
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r_, w in zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.w.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= n: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append(x); pend.append((x, r_ * w / n))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); return s.groupby(s.index.normalize()).last()
rng = np.random.default_rng(31); D = [(rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool), rng.random(len(Sh)) < .05 if i else np.zeros(len(Sh), bool)) for i in range(41)]
rd = lambda r: [c / -d for c, d in stat(r.cumsum())]
base = np.array([rd(dret(simw(Ls[~dl]))) for dl, ds in D]); print(f'\nlive return/DD: exact {base[0, 0]:.2f} / {base[0, 1]:.2f}, median {np.median(base[1:, 0]):.2f} / {np.median(base[1:, 1]):.2f}')
for k in 'ABCD':
    for mode in ('share', 'own6', 'long1.5'):
        r = np.array([rd(build(k, mode, dl, ds)) for dl, ds in D]); w = (r[1:] > base[1:]).mean(axis=0)
        lab = {'share': f'shorts only in {FAV[k][0]}, shared slots', 'own6': f'shorts only in {FAV[k][0]}, own 6-slot basket', 'long1.5': f'longs 1.5x in {FAV[k][1]}'}[mode]
        print(f'  {k} {lab:<44} exact {r[0, 0]:.2f} / {r[0, 1]:.2f}   median {np.median(r[1:, 0]):.2f} / {np.median(r[1:, 1]):.2f}   beats live in {w[0]:.0%} / {w[1]:.0%} of runs', flush=True)
