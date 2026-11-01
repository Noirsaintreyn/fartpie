"""Nov 1 2026 (shorts): auction-market context at each BEAR flip - same measures as xs/amt_ctx.py, for the bear-flip shorts.
Original note: auction-market context at each flip (volume profile from 15m bars: each bar's volume spread evenly over its high-low range).
 pva     flip close vs the PRIOR day's value area (70% of volume): below VAL / inside / above VAH
 cva     vs the 5-day composite value area (balance of the last week)
 migr    value migration: prior day's POC vs the POC two days back (value moving up / down)
 dpoc    today's developing POC at the flip vs the prior day's POC
 vwap    flip close above / below today's session VWAP (premarket flips: n/a)
 ib      after 10:30: flip close above the initial-balance high (range extension) / inside / below
Per-trade avg by category for the trades the book takes (both periods), then the categories as slot priority, 40 matched runs."""
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
def profile(h, l, v, nb=120):
    lo, hi = l.min(), h.max()
    if not hi > lo: return np.nan, np.nan, np.nan
    edges = np.linspace(lo, hi, nb + 1); mid = (edges[:-1] + edges[1:]) / 2; vol = np.zeros(nb)
    for a, b, w in zip(l, h, v):
        i0, i1 = np.searchsorted(edges, a, 'right') - 1, np.searchsorted(edges, b, 'left')
        i0, i1 = max(i0, 0), min(max(i1, i0 + 1), nb); vol[i0:i1] += w / (i1 - i0)
    p = int(vol.argmax()); tot = vol.sum(); a = b = p; acc = vol[p]
    while acc < .7 * tot and (a > 0 or b < nb - 1):
        up = vol[b + 1] if b < nb - 1 else -1; dn = vol[a - 1] if a > 0 else -1
        if up >= dn: b += 1; acc += vol[b]
        else: a -= 1; acc += vol[a]
    return mid[p], edges[a], edges[b + 1]
rows = []
for tk, g in Sh.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 240) & (m < 960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True); H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute
    H['d'] = H.dt.dt.normalize(); R = H[H.m >= 570]
    days = {d: (x.high.values, x.low.values, x.volume.values.astype(float), x.close.values, x.m.values) for d, x in R.groupby('d')}
    D = sorted(days); P = {d: profile(*days[d][:3]) for d in D}
    for i0, e, h1 in zip(g.index, g.entry, g.h1):
        fb = e - pd.Timedelta('15min'); d = fb.normalize(); k = np.searchsorted(D, d)                  # flip bar = bar before the entry
        if k < 6: continue
        prev = D[k - 1] if (k < len(D) and D[k] == d) or k == len(D) else D[k - 1]
        pk = D.index(prev)
        if pk < 6: continue
        row = H.loc[H.dt == fb]
        if not len(row): continue
        c = float(row.close.iloc[0]); poc1, val1, vah1 = P[prev]; poc2 = P[D[pk - 1]][0]
        hh = np.concatenate([days[D[j]][0] for j in range(pk - 4, pk + 1)]); ll = np.concatenate([days[D[j]][1] for j in range(pk - 4, pk + 1)]); vv = np.concatenate([days[D[j]][2] for j in range(pk - 4, pk + 1)])
        _, cval, cvah = profile(hh, ll, vv)
        r = dict(i=i0, pva='below' if c < val1 else 'above' if c > vah1 else 'inside', cva='below' if c < cval else 'above' if c > cvah else 'inside', migr='up' if poc1 > poc2 else 'down')
        if d in days and fb.hour * 60 + fb.minute >= 570:
            h_, l_, v_, c_, m_ = days[d]; s = m_ <= fb.hour * 60 + fb.minute
            dp, _, _ = profile(h_[s], l_[s], v_[s]); r['dpoc'] = 'above prior POC' if dp > poc1 else 'below prior POC'
            tp = (h_[s] + l_[s] + c_[s]) / 3; r['vwap'] = 'above' if c > (tp * v_[s]).sum() / max(v_[s].sum(), 1) else 'below'
            if fb.hour * 60 + fb.minute >= 630:
                ib = m_ < 630; ibh, ibl = h_[ib].max(), l_[ib].min(); r['ib'] = 'above IB high' if c > ibh else 'below IB low' if c < ibl else 'inside IB'
        rows.append(r)
    print(tk, end=' ', flush=True)
A = pd.DataFrame(rows).set_index('i'); S2 = Sh.join(A); S2.to_parquet('/tmp/claude-0/ml/amt_short.parquet')
S2['h'] = np.where(S2.entry.dt.year <= 2021, '2019-21', '2022-26')
print('\n\navg SHORT trade % (n), all bear flips   2019-21 / 2022-26   (positive = the short made money)')
for c in ('pva', 'cva', 'migr', 'dpoc', 'vwap', 'ib'):
    t = S2.groupby([c, 'h']).ret.agg(['mean', 'size']).unstack('h')
    print(f'  {c}: ' + '   '.join(f'{k}: {t.loc[k, ("mean", "2019-21")] * 100:+.2f}% ({int(t.loc[k, ("size", "2019-21")])}) / {t.loc[k, ("mean", "2022-26")] * 100:+.2f}% ({int(t.loc[k, ("size", "2022-26")])})' for k in t.index if k == k))
z = (S2.mz20 > 1.75) & (S2.vp_ok if 'vp_ok' in S2 else True)
print('\n  same, only when SPY z20 > 1.75 (stretched market):')
for c in ('pva', 'dpoc', 'vwap', 'ib'):
    t = S2[S2.mz20 > 1.75].groupby([c, 'h']).ret.agg(['mean', 'size']).unstack('h')
    print(f'  {c}: ' + '   '.join(f'{k}: ' + ' / '.join(f'{t.loc[k, ("mean", h)] * 100:+.2f}% ({int(t.loc[k, ("size", h)])})' if (("mean", h) in t.columns and t.loc[k, ("size", h)] == t.loc[k, ("size", h)]) else '-' for h in ('2019-21', '2022-26')) for k in t.index))
