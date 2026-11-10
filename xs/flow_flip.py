"""Nov 9 2026: OPTIONS-FLOW DYNAMICS, RELATIVE STOCK VOLUME AND IMPLIED vs REALIZED VOL ON THE SCREENER'S FLIPS (Rey: 'relative stock trading volume, more on the
options vol ratio if a big move is coming - besides IV / OI / volume / greeks: changes, more money coming in, recent changes; the biggest unusual should be rated
highest'; 'YZ or HAR with vanna or vega, or individual-stock DEX'). DECLARED BEFORE RESULTS. Same candidates / outcomes / tests / periods as xs/chop_trend.py.
All from the PRIOR close (applies to the next session's flips). Needs the Mac: data/theta_vol (contracts, premium, calls per day), data/theta_feat (call / put volume,
signed volume, short-term OTM calls, open interest; ~22 of the names), xs/ext/gex (ATM IV; ~15 names), 15m bars (stock volume, realized vol).
FEATURES: PREM premium (dollars) / its 252-day median - money, not contracts | DR5 today's contract ratio / its average over the prior 5 days - acceleration |
 UNUS today's contracts as a percentile of the stock's own last 252 days | CALLSH call share of contracts minus its 20-day average |
 RVOL stock volume / its 20-day average | SGN signed call minus signed put volume / total (aggressor flow) | OTMC short-term OTM call volume / call volume |
 DOI total open interest vs 5 sessions earlier - new positions | IVCH5 ATM IV minus 5 sessions earlier | IVYZ ATM IV / 20-day Yang-Zhang vol | IVHAR ATM IV / HAR vol
 (equal-weight 1 / 5 / 22-day realized vol from 15m returns) | ROUGH, Sep 2024 on only: VANNA sign of net dealer vanna (xs/ext/gexprof2), reported by half-years.
 Per-stock DEX is not testable: no per-contract open interest + delta history on disk.
TEST 1 per trade (terciles cut on 2019-22), TEST 2 the 6-slot book (skip worst / order ratio + feature), chosen on 2019-22, judged on 2023-26."""
import os, glob, numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
_c = open(os.path.join(_HERE, 'chop_trend.py')).read(); exec(_c[:_c.index("\nif __name__ == '__main__':")], globals())
def prior(s, days):                                                    # daily series known at its date's close -> value for the next session
    s = s.sort_index(); i = s.index.searchsorted(days, 'left') - 1
    return np.where(i >= 0, s.values[np.maximum(i, 0)], np.nan)
EF = {}; OI = {}
for f, D_ in (('data/theta_feat/all_eod.csv', EF), ('data/theta_feat/all_oi.csv', OI)):
    if os.path.exists(P(f)):
        z = pd.read_csv(P(f)); dc = z.columns[0]; z[dc] = pd.to_datetime(z[dc])
        for tk, g in z.groupby('tk'): D_[tk] = g.set_index(dc).sort_index()
ROWS = []
for tk in TKS:
    Tk = T[T.tk == tk]; days = pd.DatetimeIndex(Tk.day.values); out = pd.DataFrame(index=Tk.index)
    f = P(f'data/theta_vol/{tk}.csv')
    if os.path.exists(f):
        v = pd.read_csv(f, parse_dates=['date']).set_index('date').sort_index(); c = v.opt_contracts.astype(float)
        med = c.rolling(252, min_periods=120).median(); ratio = c / med
        out['RATIO_'] = prior(ratio, days); out['PREM'] = prior(v.opt_premium / v.opt_premium.rolling(252, min_periods=120).median(), days)
        out['DR5'] = prior(ratio / ratio.shift(1).rolling(5).mean(), days)
        out['UNUS'] = prior(c.rolling(252, min_periods=120).rank(pct=True), days)
        cs = v.call_c / c.where(c > 0); out['CALLSH'] = prior(cs - cs.rolling(20).mean(), days)
    H = bars15(tk)
    if H is not None:
        dn = H.index.normalize(); g = H.groupby(dn); dv = g.volume.sum(); out['RVOL'] = prior(dv / dv.shift(1).rolling(20).mean(), days)
        o, h, l, cl = g.open.first(), g.high.max(), g.low.min(), g.close.last()
        lo_, lc_, lh_, ll_ = np.log(o / cl.shift(1)), np.log(cl / o), np.log(h / o), np.log(l / o); n = 20; k_ = .34 / (1.34 + (n + 1) / (n - 1))
        yz = np.sqrt(252 * (lo_.rolling(n).var() + k_ * lc_.rolling(n).var() + (1 - k_) * (lh_ * (lh_ - lc_) + ll_ * (ll_ - lc_)).rolling(n).mean()))
        r = np.log(H.close).diff(); r[dn != pd.Series(dn, index=H.index).shift(1).values] = np.nan; rv = (r ** 2).groupby(dn).sum()
        har = np.sqrt(252 * (rv + rv.rolling(5).mean() + rv.rolling(22).mean()) / 3)
    gf = P(f'xs/ext/gex/{tk}.csv')
    if os.path.exists(gf) and H is not None:
        gx = pd.read_csv(gf, parse_dates=['date']).set_index('date').sort_index(); iv = gx.atm_iv.where(gx.atm_iv > 0)
        out['IVCH5'] = prior(iv - iv.shift(5), days); out['IVYZ'] = prior(iv / yz.reindex(iv.index), days); out['IVHAR'] = prior(iv / har.reindex(iv.index), days)
    if tk in EF:
        e = EF[tk]; tot = (e.cvol + e.pvol).where(lambda x: x > 0)
        out['SGN'] = prior((e.sgn_c - e.sgn_p) / tot, days); out['OTMC'] = prior(e.st_otm_c / e.cvol.where(e.cvol > 0), days)
    if tk in OI:
        oi = OI[tk].oi_c + OI[tk].oi_p; out['DOI'] = prior(oi / oi.shift(5) - 1, days)
    vf = sorted(glob.glob(P(f'xs/ext/gexprof2/{tk}_*.parquet')))
    if vf:
        vv = pd.concat([pd.read_parquet(x) for x in vf]); vv['date'] = pd.to_datetime(vv.date); nv = vv.groupby('date').vanna.sum()
        out['VANNA'] = np.where(np.isnan(prior(nv, days)), None, np.where(prior(nv, days) > 0, 'pos vanna', 'neg vanna'))
    ROWS.append(out)
FL = pd.concat(ROWS); T = T.join(FL)
FEATO = [f for f in ['PREM', 'DR5', 'UNUS', 'CALLSH', 'RVOL', 'SGN', 'OTMC', 'DOI', 'IVCH5', 'IVYZ', 'IVHAR'] if f in T]
if __name__ == '__main__':
    print(f'candidates {len(T):,} | coverage: ' + ', '.join(f'{f} {T[f].notna().mean():.0%}' for f in FEATO + (['VANNA'] if 'VANNA' in T else [])), flush=True)
    if 'RATIO_' in T: print(f'  check: rebuilt contract ratio vs the live ratio - Spearman {T[["RATIO_", "ratio"]].corr("spearman").iloc[0, 1]:.2f}', flush=True)
    print('\n=== TEST 1 per trade: bucket | 2019-22 (CHOOSE): n, ret, win, chop-out, runner | 2023-26 (JUDGE) ===', flush=True); SUM = []; SC = {}
    for f in FEATO:
        b = bucket(f); ok = b.notna()
        if ok.sum() < 300: continue
        gA, gB = T[ok & (T.per == 'A')].groupby(b[ok & (T.per == 'A')]), T[ok & (T.per == 'B')].groupby(b[ok & (T.per == 'B')])
        mA, mB, cA, cB = gA.ret.mean(), gB.ret.mean(), gA.chop.mean(), gB.chop.mean()
        if len(mA) < 2 or len(mB) < 2: continue
        print(f'  {f}  (cuts {", ".join(f"{q:.3g}" for q in T.loc[T.per == "A", f].quantile([1 / 3, 2 / 3]))})', flush=True)
        for k in sorted(mA.index):
            a, bb = T[ok & (T.per == 'A') & (b == k)], T[ok & (T.per == 'B') & (b == k)]
            print(f'    {k:<8} n {len(a):5,} {a.ret.mean():+.2%} win {a.win.mean():3.0%} chop {a.chop.mean():3.0%} run {a.run.mean():3.0%} | '
                  f'n {len(bb):5,} {bb.ret.mean():+.2%} win {bb.win.mean():3.0%} chop {bb.chop.mean():3.0%} run {bb.run.mean():3.0%}', flush=True)
        hi_, lo_ = mA.idxmax(), mA.idxmin(); jB = mB.get(hi_, np.nan) - mB.get(lo_, np.nan); jC = cB.get(lo_, np.nan) - cB.get(hi_, np.nan)
        SUM.append((f, hi_, lo_, mA[hi_] - mA[lo_], jB, cA[lo_] - cA[hi_], jC)); SC[f] = (b, mA)
    print('\n  SUMMARY (best / worst bucket chosen on 2019-22): ret spread 2019-22 -> same buckets 2023-26 | chop-out gap (worst - best) 2019-22 -> 2023-26', flush=True)
    for f, hi_, lo_, sA, sB, ccA, ccB in sorted(SUM, key=lambda z: -np.nan_to_num(z[4], nan=-9)):
        print(f'    {f:<7} best {str(hi_):<8} worst {str(lo_):<8} {sA:+.2%} -> {sB:+.2%} | chop {ccA:+.0%} -> {ccB:+.0%}  {"HOLDS" if sB > 0 and ccB >= 0 else ""}', flush=True)
    if 'VANNA' in T:
        print('\n  ROUGH (no 2019-22 data): net dealer vanna sign, Sep 2024 - Sep 2025 | Oct 2025 on', flush=True)
        for k in ('neg vanna', 'pos vanna'):
            a, bb = T[(T.VANNA == k) & (T.entry < '2025-10-01')], T[(T.VANNA == k) & (T.entry >= '2025-10-01')]
            print(f'    {k:<10} n {len(a):4,} {a.ret.mean():+.2%} win {a.win.mean():3.0%} chop {a.chop.mean():3.0%} run {a.run.mean():3.0%} | '
                  f'n {len(bb):4,} {bb.ret.mean():+.2%} win {bb.win.mean():3.0%} chop {bb.chop.mean():3.0%} run {bb.run.mean():3.0%}', flush=True)
    print('\n=== TEST 2 the 6-slot book: CAGR / max DD 2019-22 (CHOOSE) | 2023-26 (JUDGE) ===', flush=True)
    live = daily(taken(T)); print(f'  {"live (ratio order, no skip)":<34} {st2(live)}', flush=True); RES = {}
    for f, (b, mA) in SC.items():
        RES[f'{f}: skip {mA.idxmin()}'] = daily(taken(T[b != mA.idxmin()]))
        sc = b.map(mA).astype(float).fillna(float(mA.median())); X = T.assign(_k=T.groupby('entry').ratio.rank(ascending=False) + sc.groupby(T.entry).rank(ascending=False))
        RES[f'{f}: order ratio + {f}'] = daily(taken(X.sort_values(['entry', '_k', 'tk'], kind='stable')))
        if f in ('PREM', 'UNUS', 'DR5'):                                # 'the biggest unusual should be rated highest': the feature alone as the pick order
            X = T.assign(_k=T.groupby('entry')[f].rank(ascending=False, na_option='bottom')); RES[f'{f}: order by {f} alone'] = daily(taken(X.sort_values(['entry', '_k', 'tk'], kind='stable')))
    ca = {k: cagr(s, '2019-01-01', '2023-01-01')[0] for k, s in RES.items()}; cl = cagr(live, '2019-01-01', '2023-01-01')[0]; cb = cagr(live, '2023-01-01', '2027-01-01')[0]
    for k in sorted(RES, key=lambda k_: -ca[k_]): print(f'  {k:<34} {st2(RES[k])}', flush=True)
    nb = sum(1 for k, s in RES.items() if ca[k] > cl and cagr(s, '2023-01-01', '2027-01-01')[0] > cb)
    best = max(RES, key=lambda k_: ca[k_]); jb = cagr(RES[best], '2023-01-01', '2027-01-01')[0]
    print(f'  -> {len(RES)} rules; {nb} beat live in BOTH periods | chosen on 2019-22: {best} -> 2023-26 {jb:.1%} vs live {cb:.1%} ({"beats" if jb > cb else "does not beat"} live)', flush=True)
    yr = lambda s: (1 + s).groupby(s.index.year).prod() - 1; yl, yb = yr(live), yr(RES[best])
    print('  by year, live vs chosen: ' + ' | '.join(f'{y} {yl[y]:+.0%} vs {yb[y]:+.0%}' for y in yl.index), flush=True)
