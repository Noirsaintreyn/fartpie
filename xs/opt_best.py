"""Nov 9 2026: CALLS ONLY ON THE HIGHEST-PROBABILITY, BEST-RATED FLIPS (Rey: 'options on high-probability, best-rated trades'). DECLARED BEFORE RESULTS.
Flips: live candidates in the 23 names with ThetaData quotes, one per name per day, entered at the flip's 15m entry (Mac prices), exited at the screener's own exit.
TIERS (cut points from 2019-22 flips only, so both periods are out of sample for the options question):
  ALL       every flip
  RATIO     options ratio in its top third (the screener's own rating)
  CORE      entry >= the top-third cut above yesterday's VWAP (PVWD) AND above the 5-session composite value AND not below today's VWAP (09:30 entries pass)
  CORE+FLOW CORE and short-dated OTM call share in its top third (OTMC; ~22 names)
  CORE+RATIO CORE and ratio top third
  BEST      CORE and ratio top third and OTMC top third
CONTRACTS: M30 ATM (21-45 days, first strike at or above the entry) | M60 ATM (45-75 days) | M60 ITM (45-75 days, strike ~0.5 sigma x sqrt(days) below)
PRICES: entry = Black-Scholes at the 15m entry price with the contract's implied vol from the prior close mid + half the prior close spread; exit = Black-Scholes at the
  stock's exit price with the exit day's closing implied vol (entry vol if no quote) - half spread; past expiry = intrinsic. Sessions as time. $0.10 a contract a side.
REPORT per tier x contract, 2019-22 | 2023-26: n, win rate, average and median return on premium; the stock's own average return and win rate on the same flips;
  median cost of one contract (2023-26) and the share of flips where one contract costs <= $692 (the whole account)."""
import numpy as np, pandas as pd, io, contextlib
_q = open('/home/claude/fartpie/xs/opt_0dte.py').read()
with contextlib.redirect_stdout(io.StringIO()): exec(_q[:_q.index("VAR = {")])
FX = pd.read_csv('/mnt/user-data/uploads/Downloads/claude_work/flip_features.csv', parse_dates=['entry']).drop(columns=['ratio']).drop_duplicates(['tk', 'entry'])
F = F.merge(FX, on=['tk', 'entry'], how='left'); A_ = F.entry < '2023-01-01'
cut = {f: F.loc[A_, f].quantile(2 / 3) for f in ('PVWD', 'OTMC', 'ratio')}
core = (F.PVWD >= cut['PVWD']) & (F.ECOMP == 'above') & (F.VREC.fillna('n/a') != 'below VWAP')
TIER = {'ALL': F.index == F.index, 'RATIO': F.ratio >= cut['ratio'], 'CORE': core, 'CORE+FLOW': core & (F.OTMC >= cut['OTMC']),
        'CORE+RATIO': core & (F.ratio >= cut['ratio']), 'BEST': core & (F.ratio >= cut['ratio']) & (F.OTMC >= cut['OTMC'])}
CON = {'M30 ATM': (21, 45, 30, 'ATM'), 'M60 ATM': (45, 75, 60, 'ATM'), 'M60 ITM': (45, 75, 60, 'ITM')}
def price(r, con):
    bk = B[r.tk]; d0, d1 = DI(r.entry), DI(r.exit)
    if d0 not in bk.rng: return None
    k0 = np.searchsorted(bk.days, d0)
    if k0 == 0: return None
    dp = bk.days[k0 - 1]; lo, hi, tg, mny = CON[con]; a, b = bk.rng[d0]; dte = bk.dte[a:b]; ok_ = (dte >= lo) & (dte <= hi)
    if not ok_.any() or not np.isfinite(bk.sig.get(d0, np.nan)): return None
    dsel = dte[ok_][np.argmin(np.abs(dte[ok_] - tg))]; S_c = bk.S[d0]; S_e = S_c * r.px_e / r.px_c
    ix = a + np.where((dte == dsel) & (bk.r[a:b] == 1) & (bk.ask[a:b] > 0))[0]
    if not len(ix): return None
    e = bk.e[ix[0]]; K_ = bk.K[ix]; w = bk.sig[d0] * np.sqrt(max(1, sess(bk, d0, e)))
    c = ix[K_ <= S_e * (1 - .5 * w)] if mny == 'ITM' else ix[K_ >= S_e]
    if not len(c): return None
    j = c[np.argmax(bk.K[c])] if mny == 'ITM' else c[np.argmin(bk.K[c])]; K = float(bk.K[j]); qp = quote(bk, dp, e, K)
    if not qp or qp[0] <= 0 or qp[1] - qp[0] > .3 * (qp[0] + qp[1]) / 2: return None
    tod = (960 - (r.entry.hour * 60 + r.entry.minute)) / 390; v0 = iv((qp[0] + qp[1]) / 2, bk.S[dp], K, max(sess(bk, dp, e), 1) / 252)
    if not np.isfinite(v0): return None
    p0 = bs(S_e, K, (sess(bk, d0, e) + tod) / 252, v0) + (qp[1] - qp[0]) / 2; S_x = S_e * (1 + r.ret)
    if d1 >= e: p1 = max((bk.spot_at(e) if d1 > e else S_x) - K, 0.0)
    else:
        q1 = quote(bk, d1, e, K); v1 = iv((q1[0] + q1[1]) / 2, bk.S.get(d1, S_x), K, max(sess(bk, d1, e), 1) / 252) if q1 and q1[1] > 0 else np.nan
        hs = (q1[1] - q1[0]) / 2 if q1 else (qp[1] - qp[0]) / 2; p1 = max(bs(S_x, K, (sess(bk, d1, e) + .5) / 252, v1 if np.isfinite(v1) else v0) - hs, 0.0)
    return (p1 * 100 - .10) / (p0 * 100 + .10) - 1, p0 * 100
if __name__ == '__main__':
    print(f'flips {len(F):,} | cuts from 2019-22: PVWD >= {cut["PVWD"]:.2f} ATR, OTMC >= {cut["OTMC"]:.2f}, ratio >= {cut["ratio"]:.2f} | tier sizes: '
          + ', '.join(f'{k} {m.sum():,}' for k, m in TIER.items()), flush=True)
    RES = {c: pd.Series([price(r, c) for r in F.itertuples()], index=F.index) for c in CON}
    for c in CON:
        print(f'\n=== {c}: tier | 2019-22: n, win, avg, median (stock avg / win on the same flips) | 2023-26: same | one contract 2023-26: median cost, share <= $692 ===', flush=True)
        R = RES[c]; ok = R.notna().values; rr = R[ok].map(lambda z: z[0]); cc = R[ok].map(lambda z: z[1]); A2 = A_.values
        for k, m in TIER.items():
            L = []
            for lab, pm in (('2019-22', A2), ('2023-26', ~A2)):
                ix = F.index[np.asarray(m) & ok & pm]; z = rr.reindex(ix); st = F.loc[ix, 'ret']
                L.append(f'n {len(z):5,} win {np.mean(z > 0):3.0%} avg {z.mean():+6.1%} med {z.median():+6.1%} (stock {st.mean():+.2%} / {np.mean(st > 0):.0%})' if len(z) else 'n     0')
            cb = cc.reindex(F.index[np.asarray(m) & ok & ~A2])
            print(f'  {k:<11} ' + ' | '.join(L) + f' | ${cb.median():,.0f}, {np.mean(cb <= 692):.0%}', flush=True)
