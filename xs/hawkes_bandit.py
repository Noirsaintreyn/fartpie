"""Nov 2 2026:
 HAWKES  jump-contagion intensity (Ait-Sahalia, Cacho-Diaz & Laeven 2015 style, kernel fixed in advance - no fitting): each stock's intensity of
         big UP days = sum over its own past +2-sigma up days of exp(-age / tau) + 0.3 x the same over its sector peers' (S&P sectors), tau = 5
         and 10 trading days; known at the prior close. Rank flips by it (alone / + ratio).
 BANDIT  adaptive choice of ranker (adaptive-markets): arms = ratio, ratio+contracting vol (vhar), ratio+news share (on_share), ratio+stress first,
         ratio+Hawkes; each month follow the arm with the best trailing-12-month return/drawdown (of its own book), and Thompson sampling.
40 matched runs for the rankers; the bandit is judged on the stitched monthly path."""
import json
exec(open('/home/claude/fartpie/xs/flip_move_model.py').read().split("PK = np.full")[0].replace("print(", "(lambda *a, **k: None)("))
SEC = json.load(open('/home/claude/fartpie/sp500_sectors.json'))
CL = pd.DataFrame({tk: bars(tk)[1] for tk in Ls.tk.unique() if bars(tk) is not None}).sort_index(); R_ = CL.pct_change()
BIG = (R_ > 2 * R_.rolling(20).std().shift(1)).astype(float)
def intensity(tau):
    k = np.exp(-1 / tau); own = BIG.copy() * 0.0; acc = np.zeros(BIG.shape[1]); vals = BIG.values; out = np.zeros_like(vals)
    for t in range(len(vals)): acc = acc * k + np.nan_to_num(vals[t]); out[t] = acc
    own = pd.DataFrame(out, index=BIG.index, columns=BIG.columns)
    sec = pd.Series({c: SEC.get(c, 'x') for c in BIG.columns}); peer = own.copy() * 0
    for s_, cols in sec.groupby(sec).groups.items():
        tot = own[list(cols)].sum(axis=1)
        for c in cols: peer[c] = tot - own[c]
    return (own + .3 * peer).shift(1)                                                   # known at the prior close
for tau in (5, 10):
    I = intensity(tau); Ls[f'hk{tau}'] = [I[t].asof(d) if t in I else np.nan for t, d in zip(Ls.tk, Ls.d)]
TK = taken(Ls).merge(Ls[['tk', 'entry', 'hk5', 'hk10']], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print('HAWKES: avg trade % by quintile of jump intensity (book trades)')
for c in ('hk5', 'hk10'):
    q = pd.qcut(TK[c].rank(method='first'), 5, labels=False) + 1; t = (TK.groupby([q, 'h']).ret.mean() * 100).unstack('h')
    print(f'  {c}: ' + '  '.join(f'q{i}: {t.loc[i, "2019-21"]:+.2f}/{t.loc[i, "2022-26"]:+.2f}' for i in t.index), flush=True)
pr = lambda s: s.groupby(Ls.entry.dt.year).rank(pct=True)
rng = np.random.default_rng(137); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(key_):
    Z0 = Ls.copy(); Z0['ratio'] = key_.fillna(key_.median()).values
    return np.array([[v_ for cd in stat(dret(simw(Z0[~dl])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<26} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
B0 = ev(Ls.ratio); line('live (ratio)', B0)
for c in ('hk5', 'hk10'): line(f'{c} high first', ev(Ls[c]), B0); line(f'ratio + {c}', ev(pr(Ls.ratio) + pr(Ls[c])), B0)
print('\nBANDIT: monthly arm selection')
stress = ((Ls.dn > 0) | (Ls.reb > 0) | (Ls.jm0 > 0)).astype(float)
ARMS = {'ratio': Ls.ratio, 'ratio+calm vol': pr(Ls.ratio) + (Ls.vhar == -1) * .5, 'ratio+news share': pr(Ls.ratio) + pr(Ls.on_share),
        'ratio+stress first': pr(Ls.ratio) + stress * 10, 'ratio+hawkes': pr(Ls.ratio) + pr(Ls.hk10)}
RET = {}
for nm, k in ARMS.items():
    Z0 = Ls.copy(); Z0['ratio'] = k.fillna(k.median()).values; RET[nm] = dret(simw(Z0))
RET = pd.DataFrame(RET).fillna(0); M = RET.groupby(RET.index.to_period('M')).sum()          # monthly log returns per arm
def pick_ftl(M):
    ch = []
    for i in range(len(M)):
        if i < 12: ch.append('ratio'); continue
        w = M.iloc[i - 12:i]; eq = w.cumsum(); dd = (eq - eq.cummax()).min().replace(0, -1e-4); ch.append((w.sum() / -dd).idxmax())
    return ch
def pick_ts(M, seed):
    r = np.random.default_rng(seed); ch = []
    for i in range(len(M)):
        if i < 12: ch.append('ratio'); continue
        w = M.iloc[max(0, i - 24):i]; draw = {a: r.normal(w[a].mean(), w[a].std() / np.sqrt(len(w))) for a in M.columns}; ch.append(max(draw, key=draw.get))
    return ch
def perf(path):
    out = []
    for a, b in (('2020-01', '2021-12'), ('2022-01', '2026-12')):
        s = path[(path.index >= pd.Period(a)) & (path.index <= pd.Period(b))]; eq = s.cumsum(); c = np.exp(s.sum()) ** (12 / len(s)) - 1; dd = np.exp((eq - eq.cummax()).min()) - 1; out.append((c, dd))
    return '   '.join(f'{c:+.1%}/yr DD {d:.0%} ({c / -d:.2f})' for c, d in out)
for a in M.columns: print(f'  arm {a:<20} {perf(M[a])}')
ch = pick_ftl(M); path = pd.Series([M[c].iloc[i] for i, c in enumerate(ch)], index=M.index); print(f'  follow-the-leader        {perf(path)}   picks: {pd.Series(ch[12:]).value_counts().to_dict()}')
ts = [pd.Series([M[c].iloc[i] for i, c in enumerate(pick_ts(M, s))], index=M.index) for s in range(20)]
print(f'  Thompson (median of 20)  {perf(pd.concat(ts, axis=1).median(axis=1))}')
