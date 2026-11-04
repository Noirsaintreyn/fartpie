"""Nov 4 2026: follow-up to nq_greeks.py (declared before running): SIZE UP when the calm-and-negative-vanna cell is on, instead of filtering.
2022-2025 (Greeks end Dec 2025). ALLx2 and S0x2, 1 NQ base, 75/50. Z5: 2 NQ when vanna is in its low tercile AND HAR is in its low tercile, else 1. Z6: 2 NQ when vanna low OR HAR low, else 1.
One-month eval pass (starts 2022-25) with coin-flip direction, monthly $ (2022-25)."""
_s = open('xs/nq_greeks.py').read(); exec(_s[:_s.index("SZ = {")])
G2 = pd.read_csv(f'{ROOT}/options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); VP_ = G2.net_vanna.rolling(252, min_periods=120).rank(pct=True)
rng = np.random.default_rng(61); END = '2026-01-01'
def dollars(R, rule):
    hl = prior(HP, R.day) < 1 / 3; vl = prior(VP_, R.day) < 1 / 3
    mult = np.where(rule == 'Z5', np.where(hl & vl, 2, 1), np.where(rule == 'Z6', np.where(hl | vl, 2, 1), 1))
    return ((R.pts - 1.0) * 20 * mult).groupby(R.day).sum()
for sn, (Cc, pdn) in (('ALLx2', (AL, 2)), ('S0x2', (AL[AL.neg], 2))):
    R = run_n(Cc, 75, 50, pdn)
    for rule in ('Z0', 'Z5', 'Z6'):
        dv = dollars(R, rule); e = evald(dv, '2022-01-01', END); x = dv[(dv.index >= P0) & (dv.index < END)]
        mo = x.groupby(x.index.to_period('M')).sum().reindex(pd.period_range(P0, '2025-12', freq='M')).fillna(0); cf = []
        for _ in range(10): Z = Cc.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); cf.append(evald(dollars(run_n(Z, 75, 50, pdn), rule), '2022-01-01', END)['pass'])
        print(f'  {sn:<6} {rule}: pass {e["pass"]:.0%} fail {e["fail"]:.0%} (coin-flip {np.median(cf):.0%}) | month $ mean {mo.mean():+,.0f} median {mo.median():+,.0f} up {mo.gt(0).mean():.0%} worst {mo.min():+,.0f}', flush=True)
