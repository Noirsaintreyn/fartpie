"""FINRA daily short-sale volume as an entry feature (point in time: only the PRIOR session's file is used; day-t file is published after the close). svr = ShortVolume/TotalVolume. DECLARED: H-A high svr_dev (prior-day svr minus own prior-20d mean; shorts piling in against the breakout) -> worse follow-through; H-B high chronic svr20 (persistent short pressure) -> fatter right tail (squeeze fuel). Features: svr1, svr20, svr_dev, svr3dev, svsurge (prior-day short volume / own 20d mean short volume). TRAIN <2025-07-01, TEST after; cuts frozen from TRAIN."""
exec(open('xs/report_sub5_discover2.py').read().split("FS = ['atr_pct'")[0])
S = pd.read_pickle(SC + 'finra_sv.pkl').sort_values(['Symbol', 'Date']); S['svr'] = S.ShortVolume / S.TotalVolume.replace(0, np.nan); F2 = {}
for tk, g in S.groupby('Symbol'):
    g = g.set_index('Date'); g['svr20'] = g.svr.rolling(20, min_periods=15).mean().shift(2); g['svr3'] = g.svr.rolling(3).mean(); g['sv20'] = g.ShortVolume.rolling(20, min_periods=15).mean().shift(2); F2[tk] = g
def feats(r):
    g = F2.get(r.tk)
    if g is None: return pd.Series(dtype=float)
    p = g.index.searchsorted(pd.Timestamp(r.et).normalize()) - 1       # prior session row
    if p < 20: return pd.Series(dtype=float)
    x = g.iloc[p]; return pd.Series({'svr1': x.svr, 'svr20': x.svr20, 'svr_dev': x.svr - x.svr20, 'svr3dev': x.svr3 - x.svr20, 'svsurge': x.ShortVolume / x.sv20 if x.sv20 > 0 else np.nan})
add = T.apply(feats, axis=1); T = pd.concat([T, add], axis=1); tr = T[T.et < '2025-07-01']; te = T[T.et >= '2025-07-01']; print('coverage', add.svr1.notna().mean().round(3))
FS = ['svr_dev', 'svr3dev', 'svr20', 'svr1', 'svsurge']
print('\nUNIVARIATE Spearman IC vs net return'); 
for f in FS:
    a = tr[[f, 'net']].dropna(); b = te[[f, 'net']].dropna(); r1, p1 = spearmanr(a[f], a.net); r2, p2 = spearmanr(b[f], b.net); print(f'   {f:9s} train {r1:+.3f} (p {p1:.3f}, n {len(a)})  test {r2:+.3f} (p {p2:.3f}, n {len(b)})')
cuts = {}
for f in FS[:3]:
    q = tr[f].dropna().quantile([1 / 3, 2 / 3]).values; cuts[f] = q; print(f'\n{f}  (train tercile cuts {q.round(3)})'); print(f"   {'':6s}{'period':7s}{'n':>6s}{'mean':>8s}{'median':>8s}{'win':>6s}{'P>10%':>7s}{'P<-8%':>7s}")
    for nm, d in (('train', tr), ('test', te)):
        bk = pd.cut(d[f], [-np.inf, q[0], q[1], np.inf], labels=['low', 'mid', 'high'])
        for lab, x in d.groupby(bk): n = x.net; print(f'   {lab:6s}{nm:7s}{len(x):6d}{n.mean() * 100:+7.2f}%{n.median() * 100:+7.2f}%{(n > 0).mean() * 100:5.0f}%{(n > .1).mean() * 100:6.1f}%{(n < -.08).mean() * 100:6.1f}%')
def sim(d):
    rws = sorted(d.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))); eq = 1.0; op = []; cv = [1.0]
    def close(p):
        nonlocal eq
        eq += p['sz'] * (p['ret'] - COST); cv.append(eq)
    for r in rws:
        st = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: st.append(p)
        op = st
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: close(p)
    cv = np.array(cv); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min()
yrs = lambda d: max((d.et.max() - d.et.min()).days / 365, .5); EV = lambda d: (d.vr >= 1.5) & ~(d.dte <= 3) & ~(d.vwx > 1)
RULES = {'B vol>=1.5x': lambda d: d.vr >= 1.5, 'B & E+V': EV, 'E+V & svr_dev not high': lambda d: EV(d) & (d.svr_dev <= cuts['svr_dev'][1]), 'E+V & svr_dev low': lambda d: EV(d) & (d.svr_dev <= cuts['svr_dev'][0]), 'E+V & svr3dev not high': lambda d: EV(d) & (d.svr3dev <= cuts['svr3dev'][1]), 'E+V & svr20 high (squeeze fuel)': lambda d: EV(d) & (d.svr20 > cuts['svr20'][1]), 'E+V & svr20 not high': lambda d: EV(d) & (d.svr20 <= cuts['svr20'][1]), 'B & svr_dev not high': lambda d: (d.vr >= 1.5) & (d.svr_dev <= cuts['svr_dev'][1])}
print('\n8-slot books (cuts frozen from TRAIN):'); print(f"{'rule':34s}{'TRAIN n':>8s}{'ann':>7s}{'DD':>5s} |{'TEST n':>7s}{'ann':>7s}{'DD':>5s}{'mean/tr':>9s}")
for nm, f in RULES.items():
    a = tr[f(tr).fillna(False)]; b = te[f(te).fillna(False)]; ra, da = sim(a); rb, db = sim(b); an = lambda r, d: ((1 + r) ** (1 / yrs(d)) - 1) * 100 if r > -1 else -100
    print(f'{nm:34s}{len(a):8d}{an(ra, tr):+6.0f}%{da * 100:5.0f} |{len(b):7d}{an(rb, te):+6.0f}%{db * 100:5.0f}{(b.ret - COST).mean() * 100:+8.2f}%')
