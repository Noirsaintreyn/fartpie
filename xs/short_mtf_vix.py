"""Nov 1 2026: shorts by (1) the other-timeframe auction - 1h / 4h / daily vector states at the bear flip (aligned = higher timeframes also bear),
and (2) VIX's own daily vector: short only when VIX's vector is BULL (fear trending up) or BEAR. Avg short trade per period, then the book
(long + own 6-slot short basket, exact) for any cell positive in both periods."""
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
src = open('/home/claude/fartpie/xs/mtf_vote.py').read(); exec(src[src.index('def et(df)'):src.index('out = []')])
Sh = Sh.reset_index(drop=True); out = []
for tk, g in Sh.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    X_ = et(pd.read_parquet(f[0])); m = X_.dt.dt.hour * 60 + X_.dt.dt.minute; R = X_[(m >= 570) & (m < 960)].reset_index(drop=True)
    S_ = {'v1h': states(agg(R, '1h', '9h30min'), pd.Timedelta('1h')), 'v4h': states(agg(R, '4h', '9h30min'), pd.Timedelta('4h'))}
    D_ = R.set_index('dt').resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index(); S_['vD'] = states(D_, pd.Timedelta('16h'))
    x = g[['entry']].copy()
    for k_, s in S_.items():
        s = s[~s.index.duplicated(keep='last')].sort_index(); ix = s.index.searchsorted(x.entry.values, side='right') - 1; x[k_] = np.where(ix >= 0, s.values[np.clip(ix, 0, None)], np.nan)
    out.append(x.drop(columns='entry')); print(tk, end=' ', flush=True)
Sh = Sh.join(pd.concat(out))
V = pd.read_parquet('/tmp/claude-0/ml/vix_ohlc.parquet'); V.index = pd.to_datetime(V.index).tz_localize(None).normalize()
Vd = V.reset_index().rename(columns={'Date': 'dt', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})[['dt', 'open', 'high', 'low', 'close']].assign(volume=1.0)
vreg = pd.Series(LS.vec(Vd).reg.values, index=Vd.dt).shift(1)
Sh['vixvec'] = vreg.reindex(Sh.entry.dt.normalize().values, method='ffill').map({1: 'VIX vector bull', -1: 'VIX vector bear'}).values
Sh['h'] = np.where(Sh.entry.dt.year <= 2021, '2019-21', '2022-26'); Sh['aligned'] = ((Sh.v1h == -1).astype(int) + (Sh.v4h == -1) + (Sh.vD == -1))
print('\n\navg SHORT trade % (n)   2019-21 / 2022-26')
def tab(c, Z=Sh, pre=''):
    t = Z.groupby([c, 'h']).ret.agg(['mean', 'size']).unstack('h')
    for k in t.index:
        print(f'  {pre}{c}={k}: ' + ' / '.join(f'{t.loc[k, ("mean", h)] * 100:+.2f}% ({int(t.loc[k, ("size", h)])})' if t.loc[k, ("size", h)] == t.loc[k, ("size", h)] else '-' for h in ('2019-21', '2022-26')))
for c in ('v1h', 'v4h', 'vD', 'aligned', 'vixvec'): tab(c)
print('  -- VIX vector x higher timeframes aligned (all 3 bear):')
tab('vixvec', Sh[Sh.aligned == 3], 'aligned3 & ')
L0 = dret(simw(Ls)); print(f'\nlong alone {fmt(stat(L0.cumsum()))}')
for nm, g in (('VIX vector bull', Sh.vixvec == 'VIX vector bull'), ('VIX vector bear', Sh.vixvec == 'VIX vector bear'), ('all 3 TFs bear', Sh.aligned == 3),
              ('all 3 TFs bear & VIX bull', (Sh.aligned == 3) & (Sh.vixvec == 'VIX vector bull'))):
    print(f'  long + own6 shorts [{nm}]: {fmt(stat((L0 + dret(simw(Sh[g.values], n=6))).cumsum()))}', flush=True)
