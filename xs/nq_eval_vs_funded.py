"""Nov 4 2026: separate EVAL and FUNDED systems - flip system vs the vector-gated noise-area model, in every pairing. DECLARED BEFORE RESULTS.
FLIP = final flip system (Rule A, Yang-Zhang + VWAP ladder, x0.7 while the last 20 trades are net negative, gap-gated in the eval), 50 / 75, flat 12:00.
FLIP-0.3 = same with x0.3 in cold streaks (highest pass rate earlier).
NA = Zarattini-Aziz-Barbon noise area, entry only when the 15m vector agrees, exit when the 15m vector turns against (checked at :00 / :30), HARD STOP 50 pts (added for prop risk),
     flat at the 16:00 close, 10 MNQ (5 when Yang-Zhang is high), x0.7 while its own last 20 trades are net negative, 1 pt cost.
Pairings (eval -> funded): FLIP/FLIP | FLIP/NA | NA/FLIP | NA/NA | FLIP-0.3/FLIP | FLIP-0.3/NA. Lucid 50K ($88) + Topstep. Also calendar-year runs 2022 - 2026 (Lucid)."""
_rt = open('xs/nq_reach_tp.py').read(); exec(_rt[:_rt.rindex("if __name__ == '__main__':")])
rt_ = (m5 >= 570) & (m5 < 960); s15n = st_at(H15.end.values, H15.reg.values, t5 + np.timedelta64(5, 'm'))
bars = {pd.Timestamp(d): np.where(rt_ & (d5 == d))[0] for d in np.unique(d5[rt_])}; dl = sorted(bars)
mv = {d: pd.Series(np.abs(c5[ii] / o5[ii[0]] - 1), index=m5[ii] + 5) for d, ii in bars.items() if len(ii) >= 70}
SIG = pd.DataFrame(mv).T.sort_index().rolling(14, min_periods=14).mean().shift(1); pcl = {d: c5[bars[d][-1]] for d in dl}
def na_trades(hard=50.0):
    out = []
    for n_, d in enumerate(dl):
        if n_ == 0 or d not in SIG.index: continue
        ii = bars[d]; o0 = o5[ii[0]]; pc = pcl[dl[n_ - 1]]; sg = SIG.loc[d]
        if sg.isna().all(): continue
        k = NQs.reindex([d - pd.Timedelta(days=1)], method='ffill').values[0] / o0; ub0, lb0 = max(o0, pc), min(o0, pc); pos = 0; cv = cvv = 0.0; e = je = None; path = []
        for q in ii:
            cv += tp[q] * vol5[q]; cvv += vol5[q]
            if pos and q >= je:
                hi_ = ((h5[q] if pos > 0 else l5[q]) - e) * pos * k; lo_ = ((l5[q] if pos > 0 else h5[q]) - e) * pos * k
                if lo_ <= -hard: path += [hi_, -hard]; out.append((d, pos, je, -hard - 1.0, path)); pos = 0; continue
                path += [hi_, lo_]
            me = m5[q] + 5
            if q == ii[-1]:
                if pos: path.append((c5[q] - e) * pos * k); out.append((d, pos, je, path[-1] - 1.0, path)); pos = 0
                break
            if me % 30 or me < 600: continue
            s_ = sg.get(me, np.nan)
            if not np.isfinite(s_): continue
            ub, lb, px = ub0 * (1 + s_), lb0 * (1 - s_), c5[q]
            if pos and s15n[q] == -pos:
                x = (o5[q + 1] - e) * pos * k; path.append(x); out.append((d, pos, je, x - 1.0, path)); pos = 0
            if pos == 0:
                want = 1 if px > ub else (-1 if px < lb else 0)
                if want and s15n[q] == want: pos, e, je, path = want, o5[q + 1], q + 1, []
    return out
NT = na_trades(); NN = pd.Series([t[3] for t in NT]); Nd = pd.DatetimeIndex([t[0] for t in NT])
per_ = np.select([Nd < P0, Nd < pd.Timestamp('2024-01-01'), Nd < pd.Timestamp('2026-01-01')], ['18-21', '22-23', '24-25'], '2026')
print(f'NA (vector entry + vector exit + 50 hard stop): {len(NT)} trades ({len(NT) / len(dl):.2f}/day) win {NN.gt(0).mean():.0%} stopped {np.mean([t[3] <= -50.5 for t in NT]):.0%} | ' +
      ' / '.join(f'{p} {NN[per_ == p].mean():+5.1f} (n{(per_ == p).sum()})' for p in ('18-21', '22-23', '24-25', '2026')), flush=True)
hist = []; DN = [[] for _ in D_]
for d, pos, je, net, path in NT:
    mlt = .7 if len(hist) >= 20 and sum(hist[-20:]) < 0 else 1.0; hist.append(net)
    if d in dpos:
        k = dpos[d]; size = (.5 if bool(YZF.iloc[k]) else 1.0) * mlt; p = np.array(path) - 1.0; DN[k].append(((size,), net, p.min(), p))
DS['NA'] = DN
dP = lambda D: pd.Series([sum(net * 20 * th[0] for th, net, mae, p in x) for x in D], index=D_)
pf, pn = dP(DS['f_T75']), dP(DN); print(f'daily P&L correlation FLIP vs NA {np.corrcoef(pf, pn)[0, 1]:+.2f}, monthly {np.corrcoef(pf.groupby(D_.to_period("M")).sum(), pn.groupby(D_.to_period("M")).sum())[0, 1]:+.2f}', flush=True)
VAR = {'eval FLIP -> funded FLIP': ('e_T75', 'f_T75', 'f_T75'), 'eval FLIP -> funded NA': ('e_T75', 'NA', 'NA'), 'eval NA -> funded FLIP': ('NA', 'f_T75', 'f_T75'),
       'eval NA -> funded NA': ('NA', 'NA', 'NA'), 'eval FLIP-0.3 -> funded FLIP': ('e_EQ20 x0.3', 'f_T75', 'f_T75'), 'eval FLIP-0.3 -> funded NA': ('e_EQ20 x0.3', 'NA', 'NA')}
yrs_ = [(y, int(np.searchsorted(D_, pd.Timestamp(f'{y}-01-01'))), int(np.searchsorted(D_, pd.Timestamp(f'{y + 1}-01-01')))) for y in range(2022, 2027)]
def job13(a):
    r = job10(a); f = TOP if a[1] == 'TOPSTEP' else LUC; r['cal'] = {y: carS(f, s, e, VAR[a[0]])['net'] for y, s, e in yrs_}; return r
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job13, jobs)
    print('\nEVAL SYSTEM -> FUNDED SYSTEM', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<30} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | net/eval {r["net_eval"]:+6,.0f} '
              f'| 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} | calendar years ' + ' '.join(f'{y} {v:+,.0f}' for y, v in r['cal'].items()), flush=True)
