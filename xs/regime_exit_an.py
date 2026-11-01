"""Nov 1 2026: theory-driven, fixed-budget tests (rules fixed in advance). Stress = SPY downtrend (A+) OR Rebound OR jump-model bear;
calm bull = Garg 'Bull' state and not stress.
 1 regime exits      stress -> 'loose' (no lock, 1h stop); calm bull -> 'tight' (lock +3%, 15m stop); else live. Also each half alone.
 2 wider universe    in stress, also take S&P names with options ratio 1.0-1.5 (or 1.2-1.5)
 3 calm leash        in calm bull, exit if not up +2% after 2 sessions
 combos of the winners. 40 matched runs, return per drawdown vs live."""
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
_W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); _W['d'] = _W.entry.dt.normalize()
RT = _W.groupby(['tk', 'd']).agg(ratio=('ratio', 'first'), sp=('sp', 'first')).reset_index().sort_values('d')
def prep2(fs, cut):
    S_ = pd.concat(pd.read_parquet(f) for f in fs); S_['d'] = S_.entry.dt.normalize()
    S_ = S_[(S_.entry >= '2019-01-01') & ((S_.entry.dt.hour * 60 + S_.entry.dt.minute) < 840)].sort_values('d')
    S_ = pd.merge_asof(S_, RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S_ = S_[(S_.sp == True) & (S_.ratio >= cut)]; S_['w'] = 1.0; return S_.reset_index(drop=True)
RX = prep2(glob.glob('rx/*.parquet'), 1.0)
d = RX.entry.dt.normalize()
dn = mk.dn.reindex(d.values, method='ffill').fillna(False).astype(bool).values
Ast = st['A'].reindex(d.values, method='ffill').values; Cst = st['C'].reindex(d.values, method='ffill').values
RX['stress'] = dn | (Ast == 'Rebound') | (Cst == 'JM-0'); RX['calm'] = (Ast == 'Bull') & ~RX.stress
V = lambda v: RX[RX['var'] == v]
core = lambda Z: Z[Z.ratio >= 1.5]
live = core(V('live'))
print(f'live reproduction: {len(live)} trades, avg {live.ret.mean():+.2%}; stress {live.stress.mean():.0%}, calm bull {live.calm.mean():.0%} of trades')
for v in ('live', 'loose', 'tight', 'leash'):
    Z = core(V(v)); print(f'  {v:<6} avg trade by regime: ' + '  '.join(f'{nm} {Z[m].ret.mean():+.2%} ({int(m.sum())})' for nm, m in (('stress', Z.stress), ('calm', Z.calm), ('other', ~Z.stress & ~Z.calm))))
def mix(stress_v='live', calm_v='live', other_v='live', wide=None, base_v=None):
    a = core(V(stress_v)); a = a[a.stress]; b = core(V(calm_v)); b = b[b.calm]; c = core(V(other_v)); c = c[~c.stress & ~c.calm]
    parts = [a, b, c]
    if wide is not None:
        w_ = V(stress_v); parts.append(w_[(w_.ratio >= wide) & (w_.ratio < 1.5) & w_.stress])
    return pd.concat(parts).sort_values('entry').reset_index(drop=True)
rng = np.random.default_rng(71)
def ev(Z):
    out = []
    for i in range(41):
        dl = rng.random(len(Z)) < .05 if i else np.zeros(len(Z), bool)
        out.append([v_ for cd in stat(dret(simw(Z[~dl])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])])
    return np.array(out)
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<46} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}  (return {(a[1:, 0] > b[1:, 0]).mean():.0%} / {(a[1:, 3] > b[1:, 3]).mean():.0%})'
    print(s, flush=True)
print('\n40 matched runs (same random drop pattern per run index), 2019-21 / 2022-26')
rng = np.random.default_rng(71); B0 = ev(mix()); line('live', B0)
T = {'1 stress loose + calm tight': mix('loose', 'tight'), '1a stress loose only': mix('loose'), '1b calm tight only': mix(calm_v='tight'),
     '2 wider universe in stress (ratio >= 1.0)': mix(wide=1.0), '2a wider universe in stress (ratio >= 1.2)': mix(wide=1.2),
     '3 calm leash (2 sessions, +2%)': mix(calm_v='leash'), '1a + 2a': mix('loose', wide=1.2), '1 + 2a + 3': None}
for nm, Z in T.items():
    if Z is None:
        a = core(V('loose')); a = a[a.stress]; b = core(V('leash')); b = b[b.calm]; c = core(V('live')); c = c[~c.stress & ~c.calm]; w_ = V('loose'); w_ = w_[(w_.ratio >= 1.2) & (w_.ratio < 1.5) & w_.stress]
        Z = pd.concat([a, b, c, w_]).sort_values('entry').reset_index(drop=True)
    rng = np.random.default_rng(71); line(nm, ev(Z), B0)
