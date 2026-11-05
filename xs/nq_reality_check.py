"""Nov 4 2026: REALITY CHECK of the final NQ prop system. DECLARED BEFORE RESULTS.
Final system: Rule A trades (15m flip 09:30-11:30; skip thin 60-day path / hard break / monthly LVN), Yang-Zhang + VWAP ladder 10 / 5 / 3 MNQ, x0.7 while the last 20 trades are net
negative, gap-gated eval, 50 / 75, flat 12:00, Lucid 50K ($88), drawdown in real time.
(1) PLACEBO 2022-26: same entries, filters and sizing rules, direction by coin flip (gap gate and cold-streak rule recomputed on the random trades), 10 draws.
(2) EARLIER YEARS 2018-21: the firm sims so far all ran 2022-26, where the Yang-Zhang / gap / x0.7 choices were made. Same rules on accounts started Jul 2018 - Dec 2020
    (12-month runs end by Dec 2021), vs every flip at flat 1 NQ and Rule A at flat 1 NQ. Per-trade label choices did look at 2018-21, so this is only partly out-of-sample.
Plus per-trade t-stats: every flip, Rule A, and the edge over the coin-flip placebo."""
_ad = open('xs/nq_adaptive_size.py').read(); exec(_ad[:_ad.rindex("if __name__ == '__main__':")])
order = XA.sort_values(['day', 'j']).index
def seq_net(dirs):
    out = {}
    for day, g in XA.loc[order].groupby('day', sort=False):
        busy = -1; nn = 0
        for idx, r in g.iterrows():
            if nn >= 2 or r.j <= busy: continue
            res = sim2(int(r.j), int(dirs.loc[idx]), r.k, 75.0, 50.0)
            if res is None: continue
            out[idx] = res[0] - 1.0; busy = res[1]; nn += 1
    return pd.Series(out)
def eq07(nets):
    m = pd.Series(1.0, index=XA.index); hist = []
    for idx in order:
        if idx in nets.index:
            m.loc[idx] = .7 if len(hist) >= 20 and sum(hist[-20:]) < 0 else 1.0; hist.append(nets.loc[idx])
    return m
def build_dir(dirs, allow, mult):
    D = [[] for _ in D_]; sub = XA.loc[order][allow.loc[order].values]
    for day, g in sub.groupby('day', sort=False):
        busy = -1; nn = 0
        for idx, r in g.iterrows():
            if nn >= 2 or r.j <= busy: continue
            res = sim2(int(r.j), int(dirs.loc[idx]), r.k, 75.0, 50.0)
            if res is None: continue
            net, qe, p, how = res; busy = qe; nn += 1
            if day not in dpos: continue
            k = dpos[day]; v = int(bool(YZF.iloc[k])) + int(bool(X.at[idx, 'VWAPX'])); size = (1.0, .5, .3)[min(v, 2)] * float(mult.loc[idx])
            p = np.array(p) - 1.0; D[k].append(((size,), net - 1.0, p.min(), p))
    return D
def gap_against(dirs): v = XA['g_GAP']; return pd.Series((v.notna() & (v != 0) & (v != dirs)).values, index=XA.index)
TR_ = RL.net; t_all = None
def stats(nm, D_e, D_f, s0, s1, end):
    DS['_e'], DS['_f'] = D_e, D_f; v = ('_e', '_f', '_f'); f = LUC; A = []
    for s in range(s0, s1):
        res, k, fees = evS(f, s, end, DS['_e']); r = dict(res=res, fees=fees, npay=0, got=0.0)
        if res == 'pass': fr, k2, pays = fuS(f, k + 1, end, DS['_f'], DS['_f']); r.update(npay=len(pays), got=sum(a for _, a in pays))
        A.append(r)
    A = pd.DataFrame(A); mstarts = [i for i in range(s0, s1) if i == s0 or D_[i].month != D_[i - 1].month]
    K = np.array([carS(f, s, min(s + 252, end), v)['net'] for s in mstarts if s + 252 <= end or end == NS])
    return dict(nm=nm, passed=(A.res == 'pass').mean(), paid_e=(A.npay > 0).mean(), net_eval=(A.got - A.fees).mean(), k_med=np.median(K) if len(K) else np.nan, k_neg=np.mean(K < 0) if len(K) else np.nan, nK=len(K))
def placebo(seed):
    rng = np.random.default_rng(seed); dirs = pd.Series(rng.choice([-1, 1], len(XA)), index=XA.index); nets = seq_net(dirs); m = eq07(nets)
    last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); return stats(f'placebo {seed}', build_dir(dirs, ~gap_against(dirs), m), build_dir(dirs, pd.Series(True, index=XA.index), m), 0, last, NS)
if __name__ == '__main__':
    nets0 = seq_net(XA.dir); m0 = eq07(nets0); last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01')))
    real = stats('REAL final system', build_dir(XA.dir, ~gap_against(XA.dir), m0), build_dir(XA.dir, pd.Series(True, index=XA.index), m0), 0, last, NS)
    with mp.get_context('fork').Pool(2) as pool: PL = pool.map(placebo, list(range(10)))
    print('(1) PLACEBO 2022-26, Lucid 50K: real vs coin-flip direction (10 draws)', flush=True)
    print(f'  real    eval pass {real["passed"]:.0%} | evals -> payout {real["paid_e"]:.0%} | net per eval {real["net_eval"]:+,.0f} | 12m median {real["k_med"]:+,.0f} | 12m negative {real["k_neg"]:.0%}', flush=True)
    P = pd.DataFrame(PL)
    print(f'  placebo eval pass {P.passed.median():.0%} | evals -> payout {P.paid_e.median():.0%} | net per eval median {P.net_eval.median():+,.0f} (range {P.net_eval.min():+,.0f} to {P.net_eval.max():+,.0f}) | 12m median {P.k_med.median():+,.0f} (best draw {P.k_med.max():+,.0f}) | 12m negative {P.k_neg.median():.0%}', flush=True)
    al = run_l(X); ra = nets0; t1 = al.net.mean() / al.net.std() * np.sqrt(len(al)); t2 = ra.mean() / ra.std() * np.sqrt(len(ra))
    print(f'\nper trade (1 lot, 2018 - Oct 2026): every flip n{len(al)} avg {al.net.mean():+.1f} pts t={t1:.1f} | Rule A n{len(ra)} avg {ra.mean():+.1f} t={t2:.1f} | Rule A by year ' +
          ', '.join(f'{y} {ra[XA.loc[ra.index].day.dt.year == y].mean():+.1f}' for y in range(2018, 2027)), flush=True)
    # ---- (2) earlier years: rebuild the engine's calendar from Jul 2018
    D_ = days[(days >= '2018-07-01') & (days < '2022-01-01')]; dpos = {d: i for i, d in enumerate(D_)}; NS = len(D_)
    YZF = pd.Series(prior(yz.dropna().rolling(252, min_periods=120).rank(pct=True), D_) > 2 / 3, index=D_)
    s1 = int(np.searchsorted(D_, pd.Timestamp('2021-01-01')))
    Dfin_e, Dfin_f = build_dir(XA.dir, ~gap_against(XA.dir), m0), build_dir(XA.dir, pd.Series(True, index=XA.index), m0)
    one = pd.Series(1.0, index=XA.index)
    def flat_all():
        Tt = run_p(X, 75, 50, 2); D = [[] for _ in D_]
        for day, idx, net, mae, p in Tt:
            if day in dpos: D[dpos[day]].append(((1.0,), net, mae, p))
        return D
    def flat_ruleA():
        D = [[] for _ in D_]
        for day, g in XA.loc[order].groupby('day', sort=False):
            busy = -1; nn = 0
            for idx, r in g.iterrows():
                if nn >= 2 or r.j <= busy: continue
                res = sim2(int(r.j), int(r.dir), r.k, 75.0, 50.0)
                if res is None: continue
                net, qe, p, how = res; busy = qe; nn += 1
                if day in dpos: D[dpos[day]].append(((1.0,), net - 1.0, (np.array(p) - 1.0).min(), np.array(p) - 1.0))
        return D
    Dall, DrA = flat_all(), flat_ruleA()
    print('\n(2) EARLIER YEARS: accounts started Jul 2018 - Dec 2020, everything ends by Dec 2021 (Lucid 50K rules and fees)', flush=True)
    for nm, De, Df in (('every flip, flat 1 NQ', Dall, Dall), ('Rule A, flat 1 NQ', DrA, DrA), ('FINAL system', Dfin_e, Dfin_f)):
        r = stats(nm, De, Df, 0, s1, NS)
        print(f'  {nm:<24} eval pass {r["passed"]:.0%} | evals -> payout {r["paid_e"]:.0%} | net per eval {r["net_eval"]:+,.0f} | 12m median {r["k_med"]:+,.0f} | 12m negative {r["k_neg"]:.0%} (n{r["nK"]})', flush=True)
