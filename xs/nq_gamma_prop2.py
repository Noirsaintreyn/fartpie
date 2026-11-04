_s = open("xs/nq_gamma_prop.py").read(); exec(_s[:_s.rindex("if __name__ == ")])
H5 = bars('QQQ'); H15 = agg15(H5); days = pd.DatetimeIndex(sorted(H5[(H5.m >= 570)].dt.dt.normalize().unique()))
print('PROP with 5 MNQ ($10/pt) and coin-flip baseline (same trades, random direction, 30 draws: median pass rate)', flush=True)
for bn, H in (('15m', H15), ('5m', H5)):
    for both in (False, True):
        T = trades(H, both, False); Tn = T[T.gex.values < 0]; tag = f"{bn} {'both' if both else 'long'} NEG"
        p = daily(Tn, days); out = []
        for usd in (20, 10):
            pr, md, n = prop(p, usd); out.append(f'{usd}$/pt pass {pr:.0%} med {md:.0f}d')
        rng = np.random.default_rng(1); cf = []
        for _ in range(30):
            Z = Tn.copy(); Z['pts'] = (Z.pts + 1.0) * rng.choice([-1, 1], len(Z)) - 1.0; cf.append(prop(daily(Z, days), 20)[0])
        print(f'  {tag:<16} ' + ' | '.join(out) + f' | coin-flip 1 NQ pass median {np.median(cf):.0%} (range {min(cf):.0%}-{max(cf):.0%})', flush=True)
print('\nCHECK on real MNQ 1-min prints, 15m bars', flush=True)
HM = agg15(bars('MNQ'))
for both in (False, True):
    T = trades(HM, both, True); Tq = trades(H15[(H15.dt >= HM.dt.min()) & (H15.dt <= HM.dt.max())].reset_index(drop=True), both, False)
    for gn, f in (('ALL', lambda X: np.ones(len(X), bool)), ('NEG gamma', lambda X: X.gex.values < 0)):
        a, b = T[f(T)], Tq[f(Tq)]
        print(f"  {'both' if both else 'long'} {gn:<10} MNQ: n{len(a)} {a.pts.sum():+.0f} pts avg {a.pts.mean():+.1f} win {a.pts.gt(0).mean():.0%} | QQQ proxy same months: n{len(b)} {b.pts.sum():+.0f} pts", flush=True)
T = trades(H15, True, False); Tn = T[T.gex.values < 0].copy(); Tn['yr'] = Tn.day.dt.year
print('\n15m both NEG by year (pts):', Tn.groupby('yr').pts.sum().round(0).to_dict(), flush=True)
