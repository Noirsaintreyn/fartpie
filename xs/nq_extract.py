"""Nov 4 2026, POST-HOC (built from nq_dir_gates.log / nq_options_ctx.log): take the parts that work - gates that pass evals and protect funded accounts (1h vector, VXN trend, gap)
used by STAGE, or as SIZE cuts instead of skips. Base = Rule A trades, fixed 50 / 75, size 1 / 0.5 / 0.3 NQ by Yang-Zhang high + VWAP stretched (both stages). Lucid 50K ($88) + Topstep.
  A  base
  B  eval 1h-gated / funded base          C  eval VXN-gated / funded base          D  eval gap-gated / funded base
  E  eval 1h-gated / funded gap-gated     F  eval VXN-gated / funded VXN-gated until the 1st payout, then base
  G  eval 1h-gated / funded 1h-gated until the 1st payout, then base
  H  x0.5 size against VXN    I  x0.5 against 1h    J  x0.5 against gap    K  size by how many of 1h / VXN / gap are against: 0 x1, 1 x0.75, 2 x0.5, 3 x0.3
  L  x0.5 against vanna flow (data through Dec 2025; 2026 unchanged)      M  eval 1h-gated / funded K-sized"""
_dg = open('xs/nq_dir_gates.py').read(); exec(_dg[:_dg.rindex("if __name__ == '__main__':")])
o_, h_, l_, c_ = Dq.Open, Dq.High, Dq.Low, Dq.Close; n_ = 20; kz = .34 / (1.34 + (n_ + 1) / (n_ - 1))
yz = np.log(o_ / c_.shift(1)).rolling(n_).var() + kz * np.log(c_ / o_).rolling(n_).var() + (1 - kz) * (np.log(h_ / c_) * np.log(h_ / o_) + np.log(l_ / c_) * np.log(l_ / o_)).rolling(n_).mean()
YZF = pd.Series(prior(yz.dropna().rolling(252, min_periods=120).rank(pct=True), D_) > 2 / 3, index=D_)
O2 = pd.read_csv(f'{ROOT}/options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); O3 = pd.read_csv(f'{ROOT}/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').sort_index()
vfs = (np.sign(O2.net_vanna) * -np.sign(O3.atm_iv.diff(5).reindex(O2.index))).dropna()
XA['g_VFLOW'] = prior(vfs, XA.day)
def against(g): v = XA[f'g_{g}']; return (v.notna() & (v != 0) & (v != XA.dir)).values
AG = {g: pd.Series(against(g), index=XA.index) for g in ('H1', 'VXN', 'GAP', 'VFLOW')}
def build(allow=None, mult=None):
    D = [[] for _ in D_]; sub = XA if allow is None else XA[allow.values]
    for day, g in sub.sort_values('j').groupby('day'):
        busy = -1; nn = 0
        for idx, r in g.iterrows():
            if nn >= 2 or r.j <= busy: continue
            res = sim2(int(r.j), int(r.dir), r.k, 75.0, 50.0)
            if res is None: continue
            net, qe, p, how = res; busy = qe; nn += 1
            if day not in dpos: continue
            k = dpos[day]; v = int(bool(YZF.iloc[k])) + int(bool(X.at[idx, 'VWAPX'])); size = (1.0, .5, .3)[min(v, 2)] * (1.0 if mult is None else float(mult.loc[idx]))
            p = np.array(p) - 1.0; D[k].append(((size,), net - 1.0, p.min(), p))
    return D
cnt = AG['H1'].astype(int) + AG['VXN'].astype(int) + AG['GAP'].astype(int); Kmult = cnt.map({0: 1.0, 1: .75, 2: .5, 3: .3})
DS = {'base': build(), 'H1g': build(~AG['H1']), 'VXNg': build(~AG['VXN']), 'GAPg': build(~AG['GAP']),
      'xVXN': build(mult=np.where(AG['VXN'], .5, 1.0) * pd.Series(1.0, index=XA.index)), 'xH1': build(mult=pd.Series(np.where(AG['H1'], .5, 1.0), index=XA.index)),
      'xGAP': build(mult=pd.Series(np.where(AG['GAP'], .5, 1.0), index=XA.index)), 'K': build(mult=Kmult), 'xVF': build(mult=pd.Series(np.where(AG['VFLOW'], .5, 1.0), index=XA.index))}
VAR = {'A base': ('base', 'base', 'base'), 'B eval 1h-gated / funded base': ('H1g', 'base', 'base'), 'C eval VXN-gated / funded base': ('VXNg', 'base', 'base'),
       'D eval gap-gated / funded base': ('GAPg', 'base', 'base'), 'E eval 1h-gated / funded gap-gated': ('H1g', 'GAPg', 'GAPg'),
       'F eval VXN / funded VXN to 1st payout': ('VXNg', 'VXNg', 'base'), 'G eval 1h / funded 1h to 1st payout': ('H1g', 'H1g', 'base'),
       'H x0.5 against VXN': ('xVXN', 'xVXN', 'xVXN'), 'I x0.5 against 1h': ('xH1', 'xH1', 'xH1'), 'J x0.5 against gap': ('xGAP', 'xGAP', 'xGAP'),
       'K size by gates against': ('K', 'K', 'K'), 'L x0.5 against vanna flow': ('xVF', 'xVF', 'xVF'), 'M eval 1h-gated / funded K-sized': ('H1g', 'K', 'K')}
def evS(f, s, end, De):
    global DAY
    DAY = De; return evq(f, s, end)
def fuS(f, s, end, Dpre, Dpost):
    global DAY
    st = dict(bal=0.0, floor=-2000.0, peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0
    for k in range(s, end):
        DAY = Dpre if not pays else Dpost; pnl, br = dayq(k, st, f)
        if br: return 'lost', k, pays
        st['bal'] += pnl; since += pnl; wins += pnl >= f['win_thr']; b = st['bal']
        if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - 2000), f['lock'])
        if wins >= 5 and b > 0 and (not pays or since > 0):
            amt = min(.5 * b, f['cap'])
            if amt >= f['minpay']: pays.append((k, f['split'] * amt - f['pfee'])); st['bal'] -= amt; st['floor'] = float(f['post_floor']); fixed = True; wins = 0; since = 0.0
    return 'open', end, pays
def carS(f, s, end, v):
    De, Dpre, Dpost = (DS[x] for x in v); k = s; fees = got = 0.0; npay = 0
    while k < end:
        res, k2, fe = evS(f, k, end, De); fees += fe
        if res == 'fail': k = k2 + 1; continue
        if res == 'open': break
        fees += f['act']; fr, k3, pays = fuS(f, k2 + 1, end, Dpre, Dpost); got += sum(a for _, a in pays); npay += len(pays)
        if fr == 'lost': k = k3 + 1; continue
        break
    return dict(net=got - fees, payouts=npay)
def job9(a):
    nm, fn = a; v = VAR[nm]; f = TOP if fn == 'TOPSTEP' else LUC; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); A = []
    De, Dpre, Dpost = (DS[x] for x in v)
    for s in range(last):
        res, k, fees = evS(f, s, NS, De); r = dict(res=res, fees=fees + (f['act'] if res == 'pass' else 0), npay=0, got=0.0, fres=None)
        if res == 'pass': fr, k2, pays = fuS(f, k + 1, NS, Dpre, Dpost); r.update(fres=fr, npay=len(pays), got=sum(a_ for _, a_ in pays))
        A.append(r)
    A = pd.DataFrame(A); Fd = A[A.res == 'pass']; p = (A.npay > 0).mean()
    starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
    K = pd.DataFrame([carS(f, s, min(s + 252, NS), v) for s in starts])
    return dict(nm=nm, fn=fn, passed=(A.res == 'pass').mean(), blown0=((Fd.fres == 'lost') & (Fd.npay == 0)).mean(), per_pay=1 / p if p else np.inf, net_eval=(A.got - A.fees).mean(),
                k_med=K.net.median(), k_pos=(K.net > 0).mean(), k_p10=K.net.quantile(.1), full=carS(f, 0, NS, v)['net'], y26=carS(f, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), NS, v)['net'])
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job9, jobs)
    print('\nEXTRACTED VARIANTS (base = Rule A + Yang-Zhang / VWAP sizing)', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<40} {r["fn"]:<9} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} (10th {r["k_p10"]:+6,.0f}) positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
