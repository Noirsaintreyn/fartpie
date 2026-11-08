"""Nov 8 2026: LOCATION GATES on the prop system, and STAGGERED account starts. DECLARED BEFORE RESULTS.
Gates were picked from nq_location.py using 2018-2021 prop trades only (each bucket was the worst or near-worst there); 2022 - Oct 2026 is the out-of-sample check.
  G1 NO-MANS-LAND  skip when the entry is in the premium half of the prior day's range (long: above the midpoint but below the prior-day high; mirrored for shorts)
  G2 UPPER VALUE   skip when the entry is inside the prior-day value area on the trade side of POC (long: between POC and VAH)
  G3 VWAP 0-1      skip when the entry is 0 to 1 session-VWAP standard deviations past VWAP in the trade's direction
  G4 Z21 HIGH      skip when the daily 21-day z-score (trade direction) is in its 2018-21 top third (already stretched with the trade)
  G5 G1 or G3      G6 G1 or G2 or G3
STAGGER: the same system and gates, but account i (0-4) may not buy its first eval until 0 / 21 / 42 / 63 sessions x i after Jan 2022 (one new account a month / 2 months /
  quarter) so the 5 copied accounts stop passing, paying and blowing up in the same weeks.
Report: 1 contract per trade (2018-21 kept vs skipped, 2022-26 kept vs skipped, totals by year), Lucid 50K and 150K 5 accounts (eval pass, paid, points a year, payouts by year,
months with a payout, longest dry spell). Points = $ / 20."""
_l = open('xs/nq_location.py').read(); exec(_l[:_l.rindex("if __name__ == '__main__':")])
XF = pd.read_parquet('/tmp/claude-0/ml/nq_flip_features_opt.parquet'); XF['t'] = Hq.dt.values[XF.j.values]
ZK = dict(zip([(np.datetime64(t, 'ns'), int(d)) for t, d in zip(XF.t.values, XF.dir.values)], XF.z21.values))
F = []
for j, d in zip(XA.j.values, XA.dir.values):
    i = qpos.get(np.datetime64(t5[int(j)], 'ns')); f = loc(i, int(d)) if i is not None else None; f = f or dict(vz=np.nan, value=None, broke=None, rng=None)
    f['z21'] = ZK.get((np.datetime64(t5[int(j)], 'ns'), int(d)), np.nan); F.append(f)
F = pd.DataFrame(F, index=XA.index); yrA = pd.DatetimeIndex(XA.day).year
z21cut = F.z21[(yrA < 2022) & XA.index.isin(nets.index)].quantile(2 / 3)
g1 = F.rng == 'premium half'; g2 = F['value'] == 'value, trade side of POC'; g3 = (F.vz >= 0) & (F.vz < 1); g4 = F.z21 >= z21cut
G = {'P0 as is': pd.Series(False, index=XA.index), 'G1 no-mans-land (premium half)': g1, 'G2 upper value (POC to VAH)': g2, 'G3 VWAP 0-1 sd': g3,
     'G4 z21 top third': g4, 'G5 G1 or G3': g1 | g3, 'G6 G1 or G2 or G3': g1 | g2 | g3}
def book_st(A, s0, end, maxpay, off, n=5):
    E = pd.DataFrame([(t, k, a, i) for i in range(n) for t, k, a in slot2(A, min(s0 + i * off, end - 1), end, GREEN, maxpay)], columns=['t', 'k', 'amt', 'slot'])
    cash = np.zeros(NS + 1); np.add.at(cash, E.k.clip(upper=NS).values, E.amt.values)
    mo = pd.Series(cash[s0:end], index=D_[s0:end]).groupby(D_[s0:end].to_period('M')).sum(); pm = (mo > 0).values; gap = run = 0
    for v in pm: run = 0 if v else run + 1; gap = max(gap, run)
    return E, dict(net=cash[s0:end].sum(), npay=(E.t == 'pay').sum(), mpay=pm.sum(), nmo=len(pm), gap=gap, evals=(E.t == 'buy').sum())
def single(A, s_list):
    o = []
    for s in s_list:
        res, k = ev(A, s, NS); npay = 0
        if res == 'pass': fr, k2, pays = fu(A, k + 1, NS, 5); npay = len(pays)
        o.append((res == 'pass', npay > 0))
    o = np.array(o, dtype=float); return o[:, 0].mean(), o[:, 1].mean()
if __name__ == '__main__':
    s0 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))); last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); yrs = (D_[NS - 1] - D_[s0]).days / 365.25
    n1 = nets; y1 = pd.DatetimeIndex(XA.loc[n1.index, 'day']).year.values
    print(f'z21 top-third cut from 2018-21 prop trades: {z21cut:+.2f}; prop trades with location {F.loc[n1.index].rng.notna().mean():.0%}', flush=True)
    print('\n1 CONTRACT, prop trades (Rule A, 75 / 50 noon), NQ points', flush=True)
    print('  gate                               2018-21 kept / skipped (n)        2022-26 kept / skipped (n)        totals by year 2022 2023 2024 2025 2026 (base in brackets)', flush=True)
    base = pd.Series(n1.values, index=y1).groupby(level=0).sum()
    for nm, g in G.items():
        sk = g.loc[n1.index].values; a, b = (y1 < 2022), (y1 >= 2022)
        kept = pd.Series(np.where(sk, 0, n1.values), index=y1).groupby(level=0).sum()
        f = lambda m: f'{n1.values[m & ~sk].mean():+6.1f} / {n1.values[m & sk].mean() if (m & sk).any() else np.nan:+6.1f} ({(m & sk).sum():3d})'
        print(f'  {nm:<34} {f(a):<34} {f(b):<34} ' + ' '.join(f'{kept.get(y, 0):+5.0f}({base.get(y, 0):+.0f})' for y in range(2022, 2027)), flush=True)
    print('\nLUCID, 5 accounts on GREEN days, Jan 2022 -> Oct 2026', flush=True)
    for nm, g in G.items():
        fm = pd.Series(np.where(g.values, 0.0, 1.0), index=XA.index); DE = to_D(ev_res, eqm(.3) * fm); DF = to_D(full, eqm(.7) * fm)
        print(f'\n  {nm}', flush=True)
        for a in ('50K', '150K'):
            A = ACC[a]; pe, pp = single(A, list(range(s0, last, 5)))
            for off in ((0, 42) if nm != 'P0 as is' else (0, 21, 42, 63)):
                E, r = book_st(A, s0, NS, 5, off); E = E.assign(yr=[D_[int(min(k, NS - 1))].year for k in E.k])
                py = E[E.t == 'pay'].groupby('yr').amt.sum() / 20
                print(f'    {a:<5} stagger {off:>2}: eval pass {pe:.0%}, evals paid {pp:.0%} | {r["net"] / 20:+,.0f} pts = {r["net"] / 20 / yrs:+,.0f} a year ({r["net"] / 20 / r["nmo"]:+.0f} a month) | '
                      f'evals {r["evals"]}, payouts {r["npay"]}, months paid {r["mpay"]}/{r["nmo"]}, longest dry {r["gap"]} mo | payouts by year ' + ' '.join(f'{y} {py.get(y, 0):,.0f}' for y in range(2022, 2027)), flush=True)
