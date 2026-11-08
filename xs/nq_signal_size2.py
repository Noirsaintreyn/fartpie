"""Nov 8 2026: SIGNAL SIZING THAT RESPECTS THE DRAWDOWN. DECLARED BEFORE RESULTS. Round 1 (nq_signal_size.py): the 4-pillar Q separates trades (strong +24 / middle +2 per
trade 2022-26) but sizing up lost payouts in Lucid, because base size already risks about half the max loss on one full stop. So:
  D1 DOWN-ONLY   Q 0-2 at 0.5x, Q 3-4 at 1x (cut size when the signal disagrees, never above base)
  D2 CUSHION-UP  Q 3-4 at 1.5x only while the account has a cushion >= 1x the max loss above its floor (house money), else 1x
  D3 D1 + D2     D4 CUSHION-UP 1.75x at cushion >= 1.5x max loss, plus D1
Cushion = balance (incl. today's trades so far) - drawdown floor, checked before each trade. Same in eval and funded."""
_s = open('xs/nq_signal_size.py').read(); exec(_s[:_s.rindex("if __name__ == '__main__':")])
def to_D3(res, mult, boost):
    D = to_D2(res, mult, 'YZV'); out = [[] for _ in D_]; bmap = {}
    for idx, (net, qe, p, how) in res.items():
        day_ = pd.Timestamp(XA.at[idx, 'day'])
        if day_ in dpos: bmap.setdefault(dpos[day_], []).append(float(boost.loc[idx]))
    for k, lst in enumerate(D):
        bl = bmap.get(k, [1.0] * len(lst)); out[k] = [((th[0], b), net, mae, p) for (th, net, mae, p), b in zip(lst, bl)]
    return out
THR = {'v': 1.0}
def day(D, k, st, A):
    cum = 0.0; bal = st['bal']
    for th, net, mae, p in D[k]:
        if cum <= -1000 * A['scale']: break
        b = th[1] if len(th) > 1 and (bal + cum - st['floor']) >= THR['v'] * A['mll'] else 1.0
        mic = max(1, int(round(10 * th[0] * A['scale'] * b))); m = 2.0 * mic
        if bal + cum + mae * m <= st['floor']: return st['floor'] - bal, True
        cum += net * m
    return cum, False
down = pd.Series(np.where(Q >= 3, 1.0, .5), index=XA.index); hiB = lambda x: pd.Series(np.where(Q >= 3, x, 1.0), index=XA.index); oneB = pd.Series(1.0, index=XA.index)
V2 = {'S1 as is': (skip2, oneB, 1.0), 'D1 down-only (Q0-2 at 0.5x)': (skip2 * down, oneB, 1.0), 'D2 cushion-up 1.5x (cushion >= 1x max loss)': (skip2, hiB(1.5), 1.0),
      'D3 D1 + D2': (skip2 * down, hiB(1.5), 1.0), 'D4 D1 + 1.75x at cushion >= 1.5x max loss': (skip2 * down, hiB(1.75), 1.5)}
if __name__ == '__main__':
    AD = pd.DatetimeIndex(D_[s0:NS])
    for nm, (mult, boost, thr) in V2.items():
        THR['v'] = thr; idx = [i for i in full if pd.Timestamp(XA.at[i, 'day']) >= pd.Timestamp('2022-01-01')]
        pts = pd.Series([(full[i][0] - 1.0) * mult.loc[i] for i in idx], index=pd.DatetimeIndex(XA.loc[idx, 'day']).normalize()); pday = pts.groupby(level=0).sum().reindex(AD).fillna(0.0)
        print(f'\n  {nm}: 1 contract x size (no boost, no prop rules) {pts[pts != 0].mean():+.1f} per trade, {pday.mean():+.2f} a day', flush=True)
        DE = to_D3(ev_res, eqm(.3) * mult, boost); DF = to_D3(full, eqm(.7) * mult, boost)
        for a in ('50K', '150K'):
            A = ACC[a]; get = make_get(A); o = [get(s) for s in range(s0, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), 5)]
            pe = np.mean([x[0] == 'pass' for x in o]); pp = np.mean([len(x[3]) > 0 for x in o]); fund = [x for x in o if x[0] == 'pass']
            npay = np.mean([len(x[3]) for x in fund]); avgp = np.mean([amt for x in fund for _, amt in x[3]]) / 20
            r, py, med, worst, mm = line(get, A, GREEN, 'ALL5')
            print(f'    {a}: single eval pass {pe:.0%}, evals paid {pp:.0%}, funded account: {npay:.1f} payouts of {avgp:.0f} pts', flush=True); show('ALL5-GREEN', a, r, py, med, worst, mm)
