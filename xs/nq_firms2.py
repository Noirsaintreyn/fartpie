"""Nov 4 2026: the FULL sizing stack through the firm sims (nq_firms.py rules: real-time drawdown on 5m paths, no new trade after -$1,000 on the day). DECLARED BEFORE RESULTS.
Flags per trade, all known before entry:
  THIN  60-day volume profile thin on the path to the 75-pt target (bottom third of 2018-21)
  HARH  HAR volatility forecast in its top third (1-year percentile at the prior close)
  VWAPX entry extended past session VWAP in the trade's direction (distance in 15m ATR, top third of 2018-21) - declared as 'chasing = smaller'
SIZING (fraction of 1 NQ):  F1 flat | FT 0.5 on THIN | FVH 1 / 0.5 / 0.3 by the count of THIN + HARH | FALL 1 / 0.5 / 0.3 by the count of THIN + HARH + VWAPX
  eval at 1 NQ (E1) or with the same FALL sizing (EALL)
SYSTEMS: live vector buffer 0.35 | buffer 0.25 (post-hoc pick from nq_buffer_neighbours.log; its flags are recomputed on its own trades, cut-offs from its 2018-21)
FIRMS: TOPSTEP, LUCID FLEX, MFFU BUILDER (the three best in nq_firms.log)."""
_f = open('xs/nq_firms.py').read(); exec(_f[:_f.index("if __name__ == '__main__':")])
src = open('xs/signal_lab.py').read(); vs = src[src.index('def vec(df, **kw):'):src.index('def donch(')]
vs = vs.replace('def vec(df, **kw):', 'def vec_ad(df, BAarr, **kw):').replace('level - a * BA', 'level - a * BAarr[i]').replace('level + a * BA', 'level + a * BAarr[i]'); exec(vs)
hh, ll, cc_ = H15.high.values, H15.low.values, H15.close.values; pc_ = np.r_[cc_[0], cc_[:-1]]
atr15 = pd.Series(np.maximum(hh - ll, np.maximum(abs(hh - pc_), abs(ll - pc_)))).ewm(alpha=1 / 14, adjust=False).mean().values
end_i = pd.Series(np.arange(len(H15)), index=pd.DatetimeIndex(H15.end))
def flags(Cc, thin_col=None):
    Cc = Cc.reset_index(drop=True).copy(); kk = end_i.reindex(pd.DatetimeIndex(t5[Cc.j.values])).values.astype(int); vw = []; ah = []
    for j, d_, a_, day, k_ in zip(Cc.j.values, Cc.dir.values, atr15[np.maximum(kk - 1, 0)], Cc.day.values, Cc.k.values):
        i0 = j
        while i0 > 0 and d5[i0 - 1] == d5[j] and m5[i0 - 1] >= 570: i0 -= 1
        w = vol5[i0:j]; v_ = (tp[i0:j] * w).sum() / w.sum() if j > i0 and w.sum() > 0 else np.nan; vw.append((o5[j] - v_) * d_ / a_)
        if thin_col is None:
            day = pd.Timestamp(day); key = (day, 60)
            if key not in cache: cache[key] = prof(day, 60)
            P_ = cache[key]; e = o5[j]; eb = int(np.floor(np.log(e) / BW)); tb = int(np.floor(np.log(e + d_ * 75 / k_) / BW))
            ah.append(np.nan if P_ is None else zone(P_[0], P_[1], eb + d_, tb))
    Cc['vwapd'] = vw; isb = Cc.day < P0
    if thin_col is None: Cc['ahead60'] = ah; Cc['THIN'] = Cc.ahead60 < Cc.loc[isb, 'ahead60'].quantile(1 / 3)
    else: Cc['THIN'] = Cc[thin_col] == 'LVN path ahead'
    Cc['HARH'] = prior(HP, Cc.day) > 2 / 3; Cc['VWAPX'] = Cc.vwapd > Cc.loc[isb, 'vwapd'].quantile(2 / 3); return Cc
def build_day(Cc):
    T_ = run_p(Cc); D = [[] for _ in D_]; fl = Cc.set_index(Cc.index)[['THIN', 'HARH', 'VWAPX']]
    for day, idx, net, mae, p in T_:
        if day in dpos: D[dpos[day]].append((tuple(bool(x) for x in fl.loc[idx].values), net, mae, p))
    return D
C35 = flags(C, 'AHEAD_60D'); C35.index = C.index
reg25 = np.asarray(vec_ad(H15[['open', 'high', 'low', 'close']].reset_index(drop=True), np.full(len(H15), .25))); me = H15.me.values; ends = H15.end.values; dts = H15.dt.values; out = []
for k in range(2, len(H15)):
    if reg25[k] == reg25[k - 1] or reg25[k - 1] == 0 or not (570 < me[k] <= 690): continue
    day = pd.Timestamp(dts[k]).normalize(); j = np.searchsorted(t5, np.datetime64(ends[k]))
    if j < len(H5) and d5[j] == np.datetime64(day): out.append((day, int(j), int(reg25[k])))
C25 = pd.DataFrame(out, columns=['day', 'j', 'dir']); C25['k'] = NQs.reindex(pd.DatetimeIndex(C25.day) - pd.Timedelta(days=1), method='ffill').values / o5[C25.j.values]
C25 = flags(C25[np.isfinite(C25.k)])
SYS = {'buffer 0.35 (live)': build_day(C35), 'buffer 0.25': build_day(C25)}
for nm, Cc in (('0.35', C35), ('0.25', C25)): print(f'flags {nm}: ' + ', '.join(f'{c} {Cc[Cc.day >= P0][c].mean():.0%}' for c in ('THIN', 'HARH', 'VWAPX')), flush=True)
def nflag(th, n): return sum(th[:n])
SZ2 = {'F1': lambda th: 1.0, 'FT': lambda th: .5 if th[0] else 1.0, 'FVH': lambda th: (1.0, .5, .3)[min(nflag(th, 2), 2)], 'FALL': lambda th: (1.0, .5, .3)[min(nflag(th, 3), 2)]}
SZE2 = {'E1': lambda th: 1.0, 'EALL': SZ2['FALL']}
COMBOS = [('E1', 'F1'), ('E1', 'FT'), ('E1', 'FVH'), ('E1', 'FALL'), ('EALL', 'FALL')]
def job2(arg):
    fn, en, fsn = arg; f = F[fn]; es, fs = SZE2[en], SZ2[fsn]; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); A = []
    for s in range(last):
        res, k, fees = eval_phase(f, s, NS, es); r = dict(res=res, fees=fees, npay=0, got=0.0, fres=None)
        if res == 'pass':
            r['fees'] += f['act']; fr, k2, pays = funded(f, k + 1, NS, fs); r.update(fres=fr, npay=len(pays), got=sum(a for _, a in pays))
        A.append(r)
    A = pd.DataFrame(A); Fd = A[A.res == 'pass']
    starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
    K = pd.DataFrame([career(f, s, min(s + 252, NS), es, fs) for s in starts]); p = (A.npay > 0).mean()
    return dict(firm=fn, eval=en, funded=fsn, passed=(A.res == 'pass').mean(), blown0=((Fd.fres == 'lost') & (Fd.npay == 0)).mean(), per_pay=1 / p if p else np.inf,
                net_eval=(A.got - A.fees).mean(), k_paid=K.paid.mean(), k_med=K.net.median(), k_p10=K.net.quantile(.1), k_mean=K.net.mean())
if __name__ == '__main__':
    ALL = []
    for sn, Dd in SYS.items():
        DAY = Dd; jobs = [(fn, en, fsn) for fn in ('TOPSTEP', 'LUCID FLEX', 'MFFU BUILDER') for en, fsn in COMBOS]
        with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job2, jobs)
        print(f'\n== {sn}', flush=True)
        for r in RES:
            ALL.append(dict(r, system=sn))
            print(f'  {r["firm"]:<13} {r["eval"]:<4} {r["funded"]:<4} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m paid {r["k_paid"]:4.0%} median {r["k_med"]:+6,.0f} 10th {r["k_p10"]:+6,.0f} mean {r["k_mean"]:+6,.0f}', flush=True)
    pd.DataFrame(ALL).to_csv('/tmp/claude-0/ml/nq_firms2.csv', index=False)
