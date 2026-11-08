"""Nov 8 2026: the GEX regime and the A6 trend-hold inside Lucid. DECLARED BEFORE RESULTS (GEX found today on 2018-2026, so the 2022+ prop window is not fully clean for it).
Engine as nq_th_prop / nq_account_sizes (real-time trailing drawdown, $100 lock, 90% of min(50% of profit, cap), 5 qualifying days, 5 payouts per account, day stop,
5 accounts on GREEN days, copied trades, eval x0.3 / funded x0.7 cold streaks). QQQ net GEX at the prior close (xs/ext/gex/QQQ.csv, to Oct 2026).
  P0         current prop system: Rule A, 75 / 50, flat 12:00, 10 / 5 / 3 micros
  P0 GEXSKIP skip Rule A trades when GEX is positive          P0 GEXHALF  half size when GEX is positive
  P0 GEXUP   half size when positive, 1.5x when negative AND 4+ of the 5 A+ conditions (leverage at great trades)
  TH-A6 3 / 5  every flip held to the vector exit, A6 sizing 0 / 1 / 2, 3 or 5 micros a unit
  TH-A6x3 3    A6x3 sizing 0 / 1 / 2 / 3, 3 micros a unit (max 9 micros)
Lucid 50K and 150K; single evals every 5th session 2022-2025; points = $ / 20; 'a year' = full-run payouts / 4.77 years."""
_as = open('xs/nq_account_sizes.py').read(); exec(_as[:_as.rindex("if __name__ == '__main__':")])
XF = pd.read_parquet('/tmp/claude-0/ml/nq_flip_features_opt.parquet')
Hq = load(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/QQQ_5m_*.parquet')[0]); Hq = Hq[(Hq.m >= 570) & (Hq.m < 960)].reset_index(drop=True); XF['t'] = Hq.dt.values[XF.j.values]
P = XF[XF.day < '2022-01-01']; cut = {c: P[c].quantile(2 / 3) for c in ('rvol', 'rv5', 'vwd', 'comp')}
XF['a5'] = (XF.rvol >= cut['rvol']).astype(int) + (XF.rv5 >= cut['rv5']).astype(int) + (XF.vwd <= cut['vwd']).astype(int) + (XF.me <= 615).astype(int) + (XF.comp >= cut['comp']).astype(int)
KEY = lambda t, d: (np.datetime64(t, 'ns'), int(d))
GX = dict(zip([KEY(t, d) for t, d in zip(XF.t.values, XF.dir.values)], XF.gex.values)); A5 = dict(zip([KEY(t, d) for t, d in zip(XF.t.values, XF.dir.values)], XF.a5.values))
gex_ra = pd.Series([GX.get(KEY(t5[int(j)], d), np.nan) for j, d in zip(XA.j.values, XA.dir.values)], index=XA.index)
a5_ra = pd.Series([A5.get(KEY(t5[int(j)], d), np.nan) for j, d in zip(XA.j.values, XA.dir.values)], index=XA.index)
print(f'Rule A trades matched to GEX: {gex_ra.notna().mean():.0%}', flush=True)
def fm_of(kind):
    g, a = gex_ra.values, a5_ra.values
    if kind == 'SKIP': v = np.where(g > 0, 0.0, 1.0)
    elif kind == 'HALF': v = np.where(g > 0, .5, 1.0)
    elif kind == 'UP': v = np.where(g > 0, .5, np.where((g < 0) & (a >= 4), 1.5, 1.0))
    else: v = np.ones(len(g))
    return pd.Series(np.where(np.isnan(g), 1.0, v), index=XA.index)
TT = pd.read_parquet('/tmp/claude-0/ml/nq_gex_trades.parquet'); dposm = {pd.Timestamp(d): i for i, d in enumerate(D_)}
def lists(var, per_unit):
    g = TT[TT['var'] == var].sort_values(['day', 'j']); E = [[] for _ in D_]; F = [[] for _ in D_]; hist = []
    for r in g.itertuples():
        cold = len(hist) >= 20 and sum(hist[-20:]) < 0; k = dposm.get(pd.Timestamp(r.day)); hist.append(r.net)
        if k is None: continue
        u = r.size * per_unit / 10.0
        E[k].append(((u * (.3 if cold else 1.0),), r.net, r.mae, None)); F[k].append(((u * (.7 if cold else 1.0),), r.net, r.mae, None))
    return E, F
V = {'P0 current prop system': (DS['e_BASE'], DS['f_BASE'])}
for kind, lab in (('SKIP', 'P0 GEXSKIP skip positive GEX'), ('HALF', 'P0 GEXHALF half size in positive GEX'), ('UP', 'P0 GEXUP half in positive, 1.5x at negative + 4 A+')):
    fm = fm_of(kind); V[lab] = (to_D(ev_res, eqm(.3) * fm), to_D(full, eqm(.7) * fm))
V['TH-A6 trend-hold, 3 micros a unit'] = lists('A6', 3); V['TH-A6 trend-hold, 5 micros a unit'] = lists('A6', 5); V['TH-A6x3 trend-hold, 3 micros a unit'] = lists('A6x3', 3)
def single(A, s_list):
    o = []
    for s in s_list:
        res, k = ev(A, s, NS); paid = 0.0; npay = 0
        if res == 'pass': fr, k2, pays = fu(A, k + 1, NS, 5); paid = sum(a for _, a in pays); npay = len(pays)
        o.append((res == 'pass', npay > 0, paid - A['fee']))
    o = np.array(o, dtype=float); return o[:, 0].mean(), o[:, 1].mean(), o[:, 2].mean()
if __name__ == '__main__':
    s0 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))); last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); yrs = (D_[NS - 1] - D_[s0]).days / 365.25
    starts = [int(np.searchsorted(D_, pd.Timestamp(x))) for x in ('2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01', '2024-01-01', '2024-07-01')]
    print(f'Lucid, 5 accounts on GREEN days, Jan 2022 -> {D_[NS - 1]:%b %d %Y} ({yrs:.2f} years); points = $ / 20', flush=True)
    for nm, (de, df) in V.items():
        DE, DF = de, df; print(f'\n  {nm}', flush=True)
        for a in ('50K', '150K'):
            A = ACC[a]; E, r = book(A, s0, NS, 5); W = [book(A, s, min(s + 504, NS), 5)[1] for s in starts]; pe, pp, ne = single(A, list(range(s0, last, 5)))
            yr = []
            for y in (2022, 2023, 2024, 2025, 2026):
                lo, hi = int(np.searchsorted(D_, pd.Timestamp(f'{y}-01-01'))), int(np.searchsorted(D_, pd.Timestamp(f'{y + 1}-01-01'))); g = E[(E.k >= lo) & (E.k < min(hi, NS))]; yr.append(f'{y} {g.amt.sum() / 20:+,.0f}')
            print(f'    {a:<5} eval pass {pe:.0%}, evals paid {pp:.0%}, net per eval {ne / 20:+,.0f} pts | 5 accounts {r["net"] / 20:+,.0f} pts = {r["net"] / 20 / yrs:+,.0f} a year, payouts {r["npay"]}, '
                  f'dry {r["gap"]} mo | 24-mo median {np.median([w["net"] for w in W]) / 20:+,.0f} | ' + ' '.join(yr), flush=True)
