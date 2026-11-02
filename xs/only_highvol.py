"""Nov 2 2026: take ONLY the trades that happen in high volatility. Market: stress (A+ / Rebound / jump-model bear), VIX above its 1y median;
stock: its own 20d realized vol in the top half / top third of its past year. 40 matched runs vs live (fixed budget, 6 slots)."""
exec(open('/home/claude/fartpie/xs/flip_move_model.py').read().split("PK = np.full")[0].replace("print(", "(lambda *a, **k: None)("))
Ls['fear'] = (st['D'].reindex(Ls.d.values, method='ffill').values == 'Fearful')
SV = {}
for tk in Ls.tk.unique():
    b = bars(tk)
    if b is None: continue
    r = b[1].pct_change(); v = r.rolling(20).std(); SV[tk] = v.rolling(252, min_periods=120).rank(pct=True).shift(1)
Ls['svp'] = [SV[t].asof(d) if t in SV else np.nan for t, d in zip(Ls.tk, Ls.d)]
Ls['stress'] = (Ls.dn > 0) | (Ls.reb > 0) | (Ls.jm0 > 0)
rng = np.random.default_rng(127); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(Z0): return np.array([[v_ for cd in stat(dret(simw(Z0[~dl[Z0.index]])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in [pd.Series(d, Ls.index) for d in DR]])
def line(nm, a, n, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<36} keeps {n:.0%}   ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
B0 = ev(Ls); line('live (all flips)', B0, 1.0)
for nm, m in (('only market stress', Ls.stress), ('only VIX above median', Ls.fear), ('only stock vol top half', Ls.svp > .5), ('only stock vol top third', Ls.svp > 2 / 3),
              ('only stress OR stock vol top third', Ls.stress | (Ls.svp > 2 / 3))):
    m = m.fillna(False).astype(bool); line(nm, ev(Ls[m]), m.mean(), B0)
