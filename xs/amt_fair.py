"""Nov 1 2026: the one auction-market trait that held in both periods - developing POC above the prior day's POC at the flip
(+1.08% / +0.38% vs +0.39% / +0.23%) - as slot priority, and 'skip flips with developing POC below the prior POC', 40 matched runs."""
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
A = pd.read_parquet('/tmp/claude-0/ml/amt.parquet'); Ls = Ls.join(A[['dpoc', 'vwap']]) if 'dpoc' not in Ls else Ls
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<44} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}  (return {(a[1:, 0] > b[1:, 0]).mean():.0%} / {(a[1:, 3] > b[1:, 3]).mean():.0%})'
    print(s, flush=True)
rng = np.random.default_rng(59); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
pr = Ls.ratio.groupby(Ls.entry.dt.year).rank(pct=True)
def ev(Z0): return np.array([[v for cd in stat(dret(simw(Z0[~dl[Z0.index.isin(Ls.index[~dl]) | True][:0].index.size and Z0.index.isin(Ls.index[~dl])] if False else Z0[Z0.index.isin(Ls.index[~dl])])).cumsum()) for v in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
B0 = ev(Ls); line('live', B0)
up = (Ls.dpoc == 'above prior POC'); dn = (Ls.dpoc == 'below prior POC')
Z = Ls.copy(); Z['ratio'] = (pr + up * 10 - dn * 10).values; line('developing POC above prior POC first', ev(Z), B0)
line('skip flips with developing POC below prior POC', ev(Ls[~dn]), B0)
