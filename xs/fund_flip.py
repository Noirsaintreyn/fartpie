"""Nov 9 2026: FUNDAMENTALS TO RANK THE SCREENER'S FLIPS (Rey: 'P/E ratios, stuff like that, fundamental factors to help rank; earnings has a lot of vol').
DECLARED BEFORE RESULTS. Same candidates / outcomes / tests / periods as xs/chop_trend.py (exec'd for the data and the book).
Point in time: SEC XBRL company facts (a value is usable only from the session after its 10-Q / 10-K filing date; Q4 = annual - first nine months) and Yahoo's
earnings history (release time, consensus EPS, reported adjusted EPS, surprise; usable from the reaction session). Prices: Yahoo daily prior close (split-adjusted);
SEC share counts brought to today's share units with Yahoo's split history.
FEATURES: EY earnings yield = last 4 reported adjusted EPS / price (inverse P/E; losses sort low) | SY sales yield = TTM revenue / market cap (inverse P/S) |
 BM book / market | MCAP log market cap | REVG latest quarter's revenue vs the same quarter a year earlier | GM TTM gross margin | NM TTM net margin |
 ROE TTM net income / equity | SURP last earnings surprise % | EREACT the stock's return on the last earnings reaction day |
 DSE sessions since the last earnings reaction (0-4 / 5-20 / 21+) | DTE sessions until the next earnings reaction (1-3 / 4-15 / 16+; dates are announced weeks ahead)
TEST 1 per trade (terciles cut on 2019-22), TEST 2 the 6-slot book (skip worst / order ratio + feature): chosen on 2019-22, judged on 2023-26."""
import os, json, numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
_c = open(os.path.join(_HERE, 'chop_trend.py')).read(); exec(_c[:_c.index("\nif __name__ == '__main__':")], globals())
FD = P('data/fund')
REV = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'RevenueFromContractWithCustomerIncludingAssessedTax', 'SalesRevenueNet',
       'RevenuesNetOfInterestExpense', 'SalesRevenueGoodsNet']
def facts(tk):
    f = f'{FD}/sec/{tk}.json'
    return json.load(open(f)).get('facts', {}) if os.path.exists(f) else {}
def flows(F, tags, unit='USD'):
    """3-month values per fiscal quarter, point in time (earliest filing, first tag in the list); Q4 derived from the annual value"""
    R = []
    for pri, tag in enumerate(tags):
        for x in F.get('us-gaap', {}).get(tag, {}).get('units', {}).get(unit, []):
            if 'start' not in x or x.get('form', '')[:4] not in ('10-Q', '10-K'): continue
            s, e = pd.Timestamp(x['start']), pd.Timestamp(x['end']); R.append((pri, e, (e - s).days, float(x['val']), pd.Timestamp(x['filed'])))
    if not R: return None
    df = pd.DataFrame(R, columns=['pri', 'end', 'dur', 'val', 'filed'])
    pick = lambda z: z.sort_values(['pri', 'filed']).groupby('end').first()
    q = pick(df[df.dur.between(80, 100)])[['val', 'filed']]; a = pick(df[df.dur.between(350, 380)]); n9 = pick(df[df.dur.between(260, 285)]); add = []
    for e, r in a.iterrows():
        if len(q) and (abs((q.index - e).days) <= 7).any(): continue
        m9 = [v for ee, v in n9.val.items() if abs((e - ee).days - 91) <= 12]
        if m9: add.append((e, r.val - m9[0], r.filed)); continue
        qq = q[(q.index > e - pd.Timedelta(days=365)) & (q.index <= e - pd.Timedelta(days=80))]
        if len(qq) == 3: add.append((e, r.val - qq.val.sum(), r.filed))
    if add: q = pd.concat([q, pd.DataFrame(add, columns=['end', 'val', 'filed']).set_index('end')])
    return q.sort_index()
def instants(F, tag, unit='USD'):
    R = [(pd.Timestamp(x['end']), float(x['val']), pd.Timestamp(x['filed'])) for x in F.get('us-gaap', {}).get(tag, {}).get('units', {}).get(unit, [])
         if 'start' not in x and x.get('form', '')[:4] in ('10-Q', '10-K')]
    return pd.DataFrame(R, columns=['end', 'val', 'filed']).sort_values('filed').groupby('end').first().sort_index() if R else None
def known(q, days, how):
    """value of a quarterly series as known before each day (filed strictly earlier): how = 'ttm' | 'last' | 'yoy'"""
    if q is None or not len(q): return pd.Series(np.nan, index=days)
    ev = sorted(q.filed.unique()); vals = []
    for t in ev:
        k = q[q.filed <= t].sort_index(); e = k.index[-1]
        if how == 'ttm': l4 = k.iloc[-4:]; v = l4.val.sum() if len(l4) == 4 and (l4.index[-1] - l4.index[0]).days <= 300 else np.nan
        elif how == 'last': v = k.val.iloc[-1]
        else:
            pv = k[(k.index >= e - pd.Timedelta(days=380)) & (k.index <= e - pd.Timedelta(days=350))]
            v = k.val.iloc[-1] / pv.val.iloc[-1] - 1 if len(pv) and pv.val.iloc[-1] > 0 else np.nan
        vals.append(v)
    S = pd.Series(vals, index=pd.DatetimeIndex(ev)); i = S.index.searchsorted(days, 'left') - 1
    return pd.Series(np.where(i >= 0, S.values[np.maximum(i, 0)], np.nan), index=days)
def fund_panel(tk, Y):
    F = facts(tk); days = Y.index; px = Y.close.shift(1)                 # prior close
    rev = flows(F, REV); ni = flows(F, ['NetIncomeLoss', 'ProfitLoss']); gp = flows(F, ['GrossProfit'])
    if gp is None and rev is not None:
        cg = flows(F, ['CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold'])
        if cg is not None: j = rev.join(cg, rsuffix='_c', how='inner'); gp = pd.DataFrame({'val': j.val - j.val_c, 'filed': j[['filed', 'filed_c']].max(axis=1)})
    sh = flows(F, ['WeightedAverageNumberOfDilutedSharesOutstanding'], 'shares'); eq = instants(F, 'StockholdersEquity')
    if eq is None: eq = instants(F, 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')
    sp_f = f'{FD}/earn/{tk}_splits.parquet'; spl = pd.read_parquet(sp_f).ratio if os.path.exists(sp_f) else pd.Series(dtype=float)
    if sh is not None and len(sh):
        fac = np.array([spl[spl.index > e].prod() if len(spl) else 1.0 for e in sh.index]); sh = sh.assign(val=sh.val * fac)
        med = sh.val.rolling(5, center=True, min_periods=1).median(); sh = sh[(sh.val / med).between(.5, 2)]
    R = known(rev, days, 'ttm'); NI = known(ni, days, 'ttm'); G = known(gp, days, 'ttm'); SH = known(sh, days, 'last'); EQ = known(eq.assign(val=eq.val) if eq is not None else None, days, 'last')
    mc = px * SH; out = pd.DataFrame(index=days)
    out['SY'] = R / mc; out['BM'] = EQ / mc; out['MCAP'] = np.log(mc); out['REVG'] = known(rev, days, 'yoy'); out['GM'] = G / R; out['NM'] = NI / R
    out['ROE'] = NI / EQ.where(EQ > 0)
    # earnings (Yahoo)
    ef = f'{FD}/earn/{tk}.parquet'
    for c_ in ('EY', 'SURP', 'EREACT', 'DSE', 'DTE'): out[c_] = np.nan
    if os.path.exists(ef):
        E = pd.read_parquet(ef).sort_values('when').drop_duplicates('when'); t = E.when; d0 = t.dt.normalize(); pre = (t.dt.hour * 60 + t.dt.minute) < 570
        r = np.where(pre, days.searchsorted(d0, 'left'), days.searchsorted(d0, 'right')); E = E.assign(r=r)
        Rp = E.dropna(subset=['rep']); Rp = Rp[Rp.r < len(days)].copy(); Rp['ttm'] = Rp.rep.rolling(4).sum(); C = Y.close.values
        Rp['react'] = [C[k] / C[k - 1] - 1 if 0 < k < len(C) else np.nan for k in Rp.r]
        idx = np.arange(len(days)); rr = Rp.r.values
        j = np.searchsorted(rr, idx, 'right') - 1                         # last release with reaction session <= day
        ok = j >= 0; jj = np.maximum(j, 0)
        out['EY'] = np.where(ok, Rp.ttm.values[jj], np.nan) / px.values
        out['SURP'] = np.where(ok, Rp.surp.clip(-100, 100).values[jj], np.nan); out['DSE'] = np.where(ok, idx - rr[jj], np.nan)
        j2 = np.searchsorted(rr, idx, 'left') - 1; ok2 = j2 >= 0                # reaction day itself not yet closed -> the release before
        out['EREACT'] = np.where(ok2, Rp.react.values[np.maximum(j2, 0)], np.nan)
        ra = np.sort(E.r.values); n = np.searchsorted(ra, idx, 'right')        # next reaction session strictly after the day (scheduled rows count)
        out['DTE'] = np.where(n < len(ra), ra[np.minimum(n, len(ra) - 1)] - idx, np.nan)
    return out
FEATF = ['EY', 'SY', 'BM', 'MCAP', 'REVG', 'GM', 'NM', 'ROE', 'SURP', 'EREACT', 'DSE', 'DTE']
CATS['DSE'] = ([-.5, 4.5, 20.5, 1e9], ['0-4', '5-20', '21+']); CATS['DTE'] = ([.5, 3.5, 15.5, 1e9], ['1-3', '4-15', '16+'])
FP = []
for tk in TKS:
    Y, _ = dbars(tk, None)
    if Y is None: continue
    Fp = fund_panel(tk, Y); Tk = T[T.tk == tk]; X = Fp.reindex(Tk.day.values); X.index = Tk.index; FP.append(X)
T = T.drop(columns=[c for c in FEATF if c in T]).join(pd.concat(FP)[FEATF])
if __name__ == '__main__':
    print(f'candidates {len(T):,} | coverage: ' + ', '.join(f'{f} {T[f].notna().mean():.0%}' for f in FEATF), flush=True)
    print('medians 2023-26: ' + ', '.join(f'{f} {T.loc[T.per == "B", f].median():.3g}' for f in FEATF), flush=True)
    print('\n=== TEST 1 per trade: bucket | 2019-22 (CHOOSE): n, ret, win, chop-out, runner | 2023-26 (JUDGE) ===', flush=True); SUM = []; SC = {}
    for f in FEATF:
        b = bucket(f); ok = b.notna()
        if ok.sum() < 300: continue
        gA, gB = T[ok & (T.per == 'A')].groupby(b[ok & (T.per == 'A')]), T[ok & (T.per == 'B')].groupby(b[ok & (T.per == 'B')])
        mA, mB, cA, cB = gA.ret.mean(), gB.ret.mean(), gA.chop.mean(), gB.chop.mean()
        if len(mA) < 2 or len(mB) < 2: continue
        print(f'  {f}' + ('' if f in CATS else f'  (cuts {", ".join(f"{q:.3g}" for q in T.loc[T.per == "A", f].quantile([1 / 3, 2 / 3]))})'), flush=True)
        for k in sorted(mA.index):
            a, bb = T[ok & (T.per == 'A') & (b == k)], T[ok & (T.per == 'B') & (b == k)]
            print(f'    {k:<8} n {len(a):5,} {a.ret.mean():+.2%} win {a.win.mean():3.0%} chop {a.chop.mean():3.0%} run {a.run.mean():3.0%} | '
                  f'n {len(bb):5,} {bb.ret.mean():+.2%} win {bb.win.mean():3.0%} chop {bb.chop.mean():3.0%} run {bb.run.mean():3.0%}', flush=True)
        hi_, lo_ = mA.idxmax(), mA.idxmin(); jB = mB.get(hi_, np.nan) - mB.get(lo_, np.nan); jC = cB.get(lo_, np.nan) - cB.get(hi_, np.nan)
        SUM.append((f, hi_, lo_, mA[hi_] - mA[lo_], jB, cA[lo_] - cA[hi_], jC)); SC[f] = (b, mA)
    print('\n  SUMMARY (best / worst bucket chosen on 2019-22): ret spread 2019-22 -> same buckets 2023-26 | chop-out gap (worst - best) 2019-22 -> 2023-26', flush=True)
    for f, hi_, lo_, sA, sB, ccA, ccB in sorted(SUM, key=lambda z: -np.nan_to_num(z[4], nan=-9)):
        print(f'    {f:<7} best {str(hi_):<8} worst {str(lo_):<8} {sA:+.2%} -> {sB:+.2%} | chop {ccA:+.0%} -> {ccB:+.0%}  {"HOLDS" if sB > 0 and ccB >= 0 else ""}', flush=True)
    print('\n=== TEST 2 the 6-slot book: CAGR / max DD 2019-22 (CHOOSE) | 2023-26 (JUDGE) ===', flush=True)
    live = daily(taken(T)); print(f'  {"live (ratio order, no skip)":<34} {st2(live)}', flush=True); RES = {}
    for f, (b, mA) in SC.items():
        RES[f'{f}: skip {mA.idxmin()}'] = daily(taken(T[b != mA.idxmin()]))
        sc = b.map(mA).astype(float).fillna(float(mA.median())); X = T.assign(_k=T.groupby('entry').ratio.rank(ascending=False) + sc.groupby(T.entry).rank(ascending=False))
        RES[f'{f}: order ratio + {f}'] = daily(taken(X.sort_values(['entry', '_k', 'tk'], kind='stable')))
    ca = {k: cagr(s, '2019-01-01', '2023-01-01')[0] for k, s in RES.items()}; cl = cagr(live, '2019-01-01', '2023-01-01')[0]; cb = cagr(live, '2023-01-01', '2027-01-01')[0]
    for k in sorted(RES, key=lambda k_: -ca[k_])[:12]: print(f'  {k:<34} {st2(RES[k])}', flush=True)
    nb = sum(1 for k, s in RES.items() if ca[k] > cl and cagr(s, '2023-01-01', '2027-01-01')[0] > cb)
    best = max(RES, key=lambda k_: ca[k_]); jb = cagr(RES[best], '2023-01-01', '2027-01-01')[0]
    print(f'  -> {len(RES)} rules; {nb} beat live in BOTH periods | chosen on 2019-22: {best} -> 2023-26 {jb:.1%} vs live {cb:.1%} ({"beats" if jb > cb else "does not beat"} live)', flush=True)
    yr = lambda s: (1 + s).groupby(s.index.year).prod() - 1; yl, yb = yr(live), yr(RES[best])
    print('  by year, live vs chosen: ' + ' | '.join(f'{y} {yl[y]:+.0%} vs {yb[y]:+.0%}' for y in yl.index), flush=True)
