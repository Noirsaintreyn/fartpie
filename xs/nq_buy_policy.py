"""Nov 8 2026: WHEN TO BUY ACCOUNTS (cascade vs all at once vs context) and the SIZING LADDER (Yang-Zhang) and MOMENTUM, inside Lucid. DECLARED BEFORE RESULTS.
Trade systems (all Rule A, 75 / 50, flat 12:00, eval x0.3 / funded x0.7 cold streaks, gap-gated eval):
  S0 current: ladder 10 / 5 / 3 micros by YZH (prior-day Yang-Zhang vol top third) and VWAPX (entry stretched past VWAP)
  S1 S0 + skip entries inside value between POC and the trade-side edge (G2, recommended Nov 8)
  S2 S1 with a Yang-Zhang-only ladder: 10 micros, 5 when YZH (VWAPX trades were better and stopped out less: +12.7 / 27% vs +9.1 / 34%)
  S3 S1 + half size when the trade goes WITH the 50-day trend (with: -2.0 a trade in 2018-21 vs +14.4 against)
  S4 S2 + S3
Buying policies (cap 5 accounts alive at once, evals + funded; a dead account frees its slot; buys only on gate days):
  ALL5   fill every free slot on a gate day (current)            MONTH  one more slot opens every 21 sessions
  CASC1  one eval at a time: the next eval is bought when the current one passes (or fails)        CASC2  two evals at a time
  gates: GREEN (current account-start signal) | ALWAYS | VOLHI (Yang-Zhang 5-day percentile >= 2/3) | GEXNEG (QQQ net GEX < 0) | BELOW50 (QQQ under its 50-day average)
Part A: every system with ALL5-GREEN and CASC1-GREEN. Part B: every policy on the system with the best median 24-month 150K result in Part A.
Each line: full run Jan 2022 - Oct 2026 (points a year and a month, months with a payout, longest dry spell, deepest cash hole, evals bought, payouts by year) and all 34
24-month windows starting each month Jan 2022 - Oct 2024 (median and worst net points, median months with a payout). Points = $ / 20, payouts net of eval fees."""
_l = open('xs/nq_location_gates.py').read(); exec(_l[:_l.rindex("if __name__ == '__main__':")])
dqc = Hq.groupby('day').close.last(); a50 = (dqc / dqc.rolling(50).mean() - 1).shift(1)
w50 = pd.Series(np.sign(a50.reindex(pd.DatetimeIndex(XA.day).normalize()).values) * XA.dir.values > 0, index=XA.index)
def to_D2(res, mult, lad):
    D = [[] for _ in D_]
    for idx, (net, qe, p, how) in res.items():
        day = pd.Timestamp(XA.at[idx, 'day'])
        if day not in dpos: continue
        k = dpos[day]; yzh = bool(YZF.iloc[k]); vx = bool(X.at[idx, 'VWAPX'])
        size = ((1.0, .5, .3)[min(int(yzh) + int(vx), 2)] if lad == 'YZV' else (.5 if yzh else 1.0)) * float(mult.loc[idx])
        p = np.array(p) - 1.0; D[k].append(((size,), net - 1.0, p.min(), p))
    return D
one = pd.Series(1.0, index=XA.index); skip2 = pd.Series(np.where(G['G2 upper value (POC to VAH)'].values, 0.0, 1.0), index=XA.index); half50 = pd.Series(np.where(w50.values, .5, 1.0), index=XA.index)
SYS = {'S0 current ladder': (one, 'YZV'), 'S1 + skip POC-to-VAH': (skip2, 'YZV'), 'S2 S1, YZ-only ladder': (skip2, 'YZ'), 'S3 S1 + half size with 50-day trend': (skip2 * half50, 'YZV'),
       'S4 S2 + half with 50-day trend': (skip2 * half50, 'YZ')}
O3 = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
gex_s = prior(O3.net_gex, D_); b50 = a50.reindex(D_).values
GATES = {'GREEN': GREEN, 'ALWAYS': np.ones(NS + 1, bool), 'VOLHI': np.r_[yzp_s >= 2 / 3, True], 'GEXNEG': np.r_[np.nan_to_num(gex_s, nan=1) < 0, True], 'BELOW50': np.r_[np.nan_to_num(b50, nan=1) < 0, True]}
def make_get(A):
    C = {}
    def get(k):
        if k not in C:
            res, k2 = ev(A, k, NS)
            if res == 'pass': fr, k3, pays = fu(A, k2 + 1, NS, 5); last = NS if fr == 'open' else k3
            else: k3, pays, last = k2, [], (NS if res == 'open' else k2)
            C[k] = (res, NS if res == 'open' else k2, last, pays)
        return C[k]
    return get
def run_pol(get, A, s, e, gate, mode, cap=5):
    accts = []; cash = np.zeros(NS + 1); nbuy = 0; maxev = {'CASC1': 1, 'CASC2': 2}.get(mode, 99)
    for k in range(s, e):
        if not gate[k]: continue
        active = [a for a in accts if a[2] >= k]; nev = sum(1 for a in active if k <= a[1]); capk = min(cap, 1 + (k - s) // 21) if mode == 'MONTH' else cap
        while len(active) < capk and nev < maxev:
            res, k2, last, pays = get(k); a = (k, k2, last); accts.append(a); active.append(a); nev += 1; nbuy += 1; cash[k] -= A['fee']
            for kp, amt in pays: cash[min(kp, NS)] += amt
    return cash, nbuy
def stats(cash, s, e, nbuy):
    c = cash[s:e]; mo = pd.Series(c, index=D_[s:e]).groupby(D_[s:e].to_period('M')).sum(); pm = (mo > 0).values; gap = run = 0
    for v in pm: run = 0 if v else run + 1; gap = max(gap, run)
    return dict(net=c.sum() / 20, nmo=len(mo), mpay=int(pm.sum()), gap=gap, hole=np.cumsum(c).min(), evals=nbuy)
s0 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))); yrs = (D_[NS - 1] - D_[s0]).days / 365.25
WS = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in (2022, 2023, 2024) for m in range(1, 13)][:34]
def line(get, A, gate, mode):
    cash, nb = run_pol(get, A, s0, NS, gate, mode); r = stats(cash, s0, NS, nb)
    py = pd.Series(cash[s0:NS], index=D_[s0:NS]); py = py[py > 0].groupby(py[py > 0].index.year).sum() / 20
    W = []
    for w in WS:
        e = min(w + 504, NS); c, nb = run_pol(get, A, w, e, gate, mode); W.append(stats(c, w, e, nb))
    wn = [x['net'] for x in W]; wm = [x['mpay'] for x in W]
    return r, py, np.median(wn), min(wn), np.median(wm)
def show(lab, a, r, py, med, worst, mm):
    print(f'    {lab:<16} {a:<5} {r["net"] / yrs:+6,.0f}/yr {r["net"] / r["nmo"]:+5.0f}/mo | paid {r["mpay"]:2d}/{r["nmo"]} mo, dry {r["gap"]:2d} mo, cash hole ${-r["hole"]:,.0f}, evals {r["evals"]:2d} | '
          f'24-mo median {med:+6,.0f} worst {worst:+6,.0f}, paid {mm:.0f} mo | ' + ' '.join(f'{y} {py.get(y, 0):,.0f}' for y in range(2022, 2027)), flush=True)
if __name__ == '__main__':
    print(f'trades with the 50-day trend: {w50.loc[nets.index].mean():.0%}; G2 skips {(skip2.loc[nets.index] == 0).mean():.0%}', flush=True)
    best, bestv = None, -1e9
    print('\nPART A: systems', flush=True)
    for nm, (mult, lad) in SYS.items():
        DE = to_D2(ev_res, eqm(.3) * mult, lad); DF = to_D2(full, eqm(.7) * mult, lad); print(f'\n  {nm}', flush=True)
        for a in ('50K', '150K'):
            A = ACC[a]; get = make_get(A); s_list = list(range(s0, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), 5))
            o = [get(s) for s in s_list]; pe = np.mean([x[0] == 'pass' for x in o]); pp = np.mean([len(x[3]) > 0 for x in o])
            print(f'    {a}: single eval pass {pe:.0%}, evals paid {pp:.0%}', flush=True)
            for mode in ('ALL5', 'CASC1'):
                r, py, med, worst, mm = line(get, A, GREEN, mode); show(f'{mode}-GREEN', a, r, py, med, worst, mm)
                if a == '150K' and mode == 'ALL5' and med > bestv: best, bestv = nm, med
    print(f'\nPART B: buying policies on {best}', flush=True)
    mult, lad = SYS[best]; DE = to_D2(ev_res, eqm(.3) * mult, lad); DF = to_D2(full, eqm(.7) * mult, lad)
    for a in ('50K', '150K'):
        A = ACC[a]; get = make_get(A); print(f'\n  {a}', flush=True)
        for mode, g in (('ALL5', 'GREEN'), ('ALL5', 'ALWAYS'), ('ALL5', 'VOLHI'), ('ALL5', 'GEXNEG'), ('ALL5', 'BELOW50'), ('MONTH', 'GREEN'), ('CASC1', 'GREEN'), ('CASC1', 'ALWAYS'),
                        ('CASC1', 'VOLHI'), ('CASC2', 'GREEN'), ('CASC2', 'VOLHI')):
            r, py, med, worst, mm = line(get, A, GATES[g], mode); show(f'{mode}-{g}', a, r, py, med, worst, mm)
