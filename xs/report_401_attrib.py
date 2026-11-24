"""Nov 23 2026 report 401. Rey: 'stress test, run everything: how much is vector, how much is universe, how much is beta; the vector's job is drawdown, options ratio picks.'
DECLARED: layers on daily returns, all EXPOSURE-MATCHED to the live book's daily gross weight w_t (so cash drag is not mistaken for edge): SPY*w | 54-name EW*w (UNIVERSE) | eligible EW*w (S&P PIT + ratio>=1.5 at prior close; GATE) | live book (VECTOR).
Differences are additive in mean daily return. Also: beta/alpha of the book vs SPY; random-timing control (same names, same hold lengths, same gate, entry shifted U[-120,120] days; 300 draws) = vector TIMING value;
Rey's proposal: ratio-top6 always (R6), ratio-top6 only while vector regime bull (R6V, vector as pure classifier), vs the live flip book.  Drawdown: MDD, Ulcer, SPY stress windows.  Split 2019-21 | 2022-Aug 2026 (SPY ends 2026-08-14)."""
import sys, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs')
_s = open('xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
UP = '/mnt/user-data/uploads/fartpie/'; END2 = pd.Timestamp('2026-08-14'); P = pd.read_parquet(SP + 'panel401.parquet')
C = P.pivot(index='day', columns='tk', values='close'); G = P.pivot(index='day', columns='tk', values='reg'); idx = C.index[(C.index >= '2019-01-02') & (C.index <= END2)]; C = C.reindex(C.index[(C.index >= '2018-12-01') & (C.index <= END2)])
spy = C.pop('SPY'); G = G.drop(columns='SPY'); rets = C.pct_change(); rS = spy.pct_change().reindex(idx); NM = list(C.columns); rN = rets[NM].reindex(idx)
# gate: PIT membership + ratio median shifted
M = json.load(open('sp500_pit_membership.json')); fx = lambda t: 'META' if t == 'FB' else t
mem = pd.DataFrame({tk: [tk in {fx(t) for t in M.get(d.strftime('%Y-%m'), [])} for d in idx] for tk in NM}, index=idx)
_L = pd.read_csv(UP + 'xs/ext/optvol_ratio_live.csv', parse_dates=['date']); _L['r'] = 100 * _L.opt_contracts / _L.sh  # same definition as optvol_ratio.opt_ratio (checked on AAPL)
_r = pd.concat([pd.read_csv(UP + f'xs/ext/{f}.csv', parse_dates=['date']).rename(columns={'opt_ratio': 'r'})[['tk', 'date', 'r']] for f in ('optvol_ratio_sp',)] + [_L[['tk', 'date', 'r']]]).drop_duplicates(['tk', 'date']); Rr = _r.pivot_table(index='date', columns='tk', values='r').sort_index(); print('ratio names missing:', [t for t in NM if t not in Rr.columns])
med = Rr.rolling(252, min_periods=126).median().shift(1).reindex(Rr.index.union(idx)).ffill().reindex(idx)[NM]
elig = mem & (med >= 1.5); regP = G.shift(1).reindex(idx)  # vector regime at PRIOR close
rU = rN.mean(axis=1); rGt = rN.where(elig).mean(axis=1).fillna(0)
TK, _ = sim_swap(B0, None); TK = [t for t in TK if t[1] <= END2]
Bk = spread(TK).reindex(idx).fillna(0); w = pd.Series(0.0, index=idx)
for tk, e, x, r, ra in TK:
    a, b = idx.searchsorted(e.normalize()), idx.searchsorted(x.normalize()); w.iloc[a:min(b, len(idx) - 1) + 1] += 1 / 6
w = w.clip(upper=1)
Cv = C.reindex(idx).ffill().values; ci = {t: i for i, t in enumerate(C.columns)}
def mtm(TK, cost=2e-4):  # true daily mark-to-market: entry px -> daily closes -> exit px, 1/6 fixed weight, 2bp each side
    R = np.zeros(len(idx))
    for tk, e, x, r, ra in TK:
        c = ci[tk]; a = idx.searchsorted(e.normalize()); b = min(idx.searchsorted(x.normalize()), len(idx) - 1); pe = px_at(tk, e); pxx = px_at(tk, x)
        if a >= len(idx): continue
        if a >= b: R[a] += (pxx / pe - 1 - 2 * cost) / 6; continue
        R[a] += (Cv[a, c] / pe - 1 - cost) / 6; R[a + 1:b] += (Cv[a + 1:b, c] / Cv[a:b - 1, c] - 1) / 6; R[b] += (pxx / Cv[b - 1, c] - 1 - cost) / 6
    return pd.Series(R, index=idx)
Bs = Bk; Bk = mtm(TK); print('book: smoothed-spread CAGR %.1f%% Sh %.2f MDD %.1f%% | TRUE daily MTM CAGR %.1f%% Sh %.2f MDD %.1f%%' % (cagr(Bs) * 100, sharpe(Bs.values), mdd(Bs) * 100, cagr(Bk) * 100, sharpe(Bk.values), mdd(Bk) * 100))
def st(s, lab, sub=None):
    z = s if sub is None else s[sub]; e = (1 + z).cumprod(); dd = e / e.cummax() - 1; ul = np.sqrt((dd ** 2).mean()); y = (z.index[-1] - z.index[0]).days / 365.25
    return dict(lab=lab, cagr=e.iloc[-1] ** (1 / y) - 1, mu=z.mean() * 252, sh=sharpe(z.values), mdd=dd.min(), ulcer=ul)
def table(rows, title):
    print('\n' + title); print(f"{'':32s}{'CAGR':>7s}{'mean/yr':>9s}{'Sharpe':>8s}{'MDD':>8s}{'Ulcer':>7s}  | 2019-21 CAGR Sh | 2022-26 CAGR Sh")
    for lab, s in rows:
        a = st(s, lab); h1 = st(s, lab, s.index < '2022-01-01'); h2 = st(s, lab, s.index >= '2022-01-01')
        print(f"{lab:32s}{a['cagr']:7.1%}{a['mu']:9.1%}{a['sh']:8.2f}{a['mdd']:8.1%}{a['ulcer']:7.3f}  |   {h1['cagr']:6.1%} {h1['sh']:5.2f} |   {h2['cagr']:6.1%} {h2['sh']:5.2f}")
print('avg exposure w:', round(w.mean(), 3), '| eligible names/day:', round(elig.sum(axis=1).mean(), 1), '| days w>0:', round((w > 0).mean(), 3))
L = [('SPY (full exposure)', rS), ('SPY x w (beta, same exposure)', rS * w), ('54-name EW x w (UNIVERSE)', rU * w), ('eligible EW x w (GATE)', rGt * w), ('LIVE BOOK (VECTOR)', Bk)]
table(L, 'EXPOSURE-MATCHED LAYERS')
d = lambda a, b: ((a - b).mean() * 252)
print(f"\nANNUAL MEAN-RETURN STEPS (additive): exposure/cash drag {d(rS * w, rS):+.1%} | universe over SPY {d(rU * w, rS * w):+.1%} | S&P+ratio gate over universe {d(rGt * w, rU * w):+.1%} | vector book over gate {d(Bk, rGt * w):+.1%} | total book {Bk.mean() * 252:.1%}")
import statsmodels.api as sm
for nm, y in (('book', Bk), ('gate x w', rGt * w)):
    X = sm.add_constant(rS.fillna(0)); f = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5}); print(f"{nm} on SPY: beta {f.params.iloc[1]:.2f} alpha {f.params.iloc[0]*252:+.1%}/yr (t {f.tvalues.iloc[0]:.2f}) R2 {f.rsquared:.2f}")
X = sm.add_constant(pd.concat([rS.fillna(0).rename('spy'), (rU - rS).fillna(0).rename('uni'), (rGt - rU).rename('gate')], axis=1)); f = sm.OLS(Bk, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5}); print('book on SPY, universe-SPY, gate-universe: betas', f.params.drop('const').round(2).to_dict(), 'alpha %+.1f%%/yr t %.2f' % (f.params['const'] * 252 * 100, f.tvalues['const']))
# ---- random timing control
rn = rets[NM].reindex(idx).fillna(0).values; col = {t: i for i, t in enumerate(NM)}; el = elig.values; nD = len(idx); rng = np.random.default_rng(401)
tr = [(col[t[0]], idx.searchsorted(t[1].normalize()), min(idx.searchsorted(t[2].normalize()), nD - 1)) for t in TK if t[0] in col]
def book(shift, keep_gate):
    R = np.zeros(nD)
    for (c, a, b), dl in zip(tr, shift):
        n = b - a
        for _ in range(25 if keep_gate else 1):
            a2 = a + dl if _ == 0 else a + int(rng.integers(-120, 121))
            if a2 < 1 or a2 + n >= nD: a2 = None; continue
            if keep_gate and not el[a2, c]: a2 = None; continue
            break
        if a2 is None: a2 = a
        R[a2 + 1:a2 + n + 1] += rn[a2 + 1:a2 + n + 1, c] / 6; R[a2] -= 4e-4 / 6  # enter at the close of day a2 (no entry-day look-ahead)
    return pd.Series(R, index=idx)
act = book(np.zeros(len(tr), int), False)
res = {}
for kg in (False, True):
    S = []
    for k in range(300):
        s = book(rng.integers(-120, 121, len(tr)), kg); S.append((st(s, '')['cagr'], st(s, '')['sh'], st(s, '')['mdd']))
    res[kg] = np.array(S)
sa = st(act, ''); print(f"\nRANDOM-TIMING CONTROL (daily close-to-close, gross of cost; actual flips entered at the entry-day CLOSE: CAGR {sa['cagr']:.1%} Sh {sa['sh']:.2f} MDD {sa['mdd']:.1%}; actual book net-of-cost {st(Bk,'')['cagr']:.1%})")
for kg, nm in ((False, 'same names+durations, random entry'), (True, 'same + must be gate-eligible at entry')):
    A = res[kg]; print(f"  {nm:42s} CAGR median {np.median(A[:,0]):.1%} [5-95: {np.percentile(A[:,0],5):.1%},{np.percentile(A[:,0],95):.1%}] | Sharpe med {np.median(A[:,1]):.2f} | MDD med {np.median(A[:,2]):.1%} [5-95: {np.percentile(A[:,2],5):.1%},{np.percentile(A[:,2],95):.1%}] | actual CAGR pct-rank {np.mean(A[:,0] < sa['cagr']):.0%}, MDD better-than-control {np.mean(A[:,2] < sa['mdd']):.0%}")
# ---- Rey's proposal: ratio picks, vector classifies
def top6(need_vec, N=6, cost=2e-4):
    hold = []; R = np.zeros(nD); regv = regP.values; medv = med.values; prev = set()
    for i in range(nD):
        ok = [c for c in range(len(NM)) if el[i, c] and (not need_vec or regv[i, c] == 1)]
        hold = [c for c in hold if c in ok]; rest = sorted([c for c in ok if c not in hold], key=lambda c: -medv[i, c])
        hold = (hold + rest)[:N]; cur = set(hold); turn = len(cur ^ prev) / N; prev = cur
        R[i] = sum(rn[i, c] for c in hold) / N - turn * cost
    return pd.Series(R, index=idx)
R6 = top6(False); R6V = top6(True)
table([('SPY', rS), ('Ratio-top6 ALWAYS (no vector)', R6), ('Ratio-top6 + vector bull only', R6V), ('LIVE BOOK (flip entry)', Bk), ('Gate EW x w (matched)', rGt * w)], 'REY PROPOSAL: ratio selects, vector only classifies')
print('R6V avg exposure', round((R6V != 0).mean(), 3))
# ---- stress windows
W = {'Covid Feb19-Mar23 2020': ('2020-02-19', '2020-03-23'), '2022 bear Jan3-Oct12': ('2022-01-03', '2022-10-12'), 'Apr 2025 Feb19-Apr8': ('2025-02-19', '2025-04-08'), 'Dec 2018 (partial)': ('2019-01-02', '2019-01-04')}
print('\nSTRESS WINDOWS (cumulative return)'); print(f"{'':32s}" + ''.join(f'{k:>26s}' for k in list(W)[:3]))
for lab, s in [('SPY', rS), ('Universe EW (full)', rU), ('Gate EW (full)', rGt), ('Ratio-top6 always', R6), ('Ratio-top6 + vector bull', R6V), ('LIVE BOOK', Bk)]:
    print(f'{lab:32s}' + ''.join(f"{((1 + s[a:b].fillna(0)).prod() - 1):26.1%}" for a, b in list(W.values())[:3]))
print('\nbook avg exposure inside windows vs outside:', {k: round(w[a:b].mean(), 2) for k, (a, b) in list(W.items())[:3]}, 'overall', round(w.mean(), 2))

# ---- extra 1: is the 54-name set special? same-period EW of S&P members among the OTHER 108 pool names (daily candles; |daily ret|>50% treated as split artefact)
pool = pd.read_parquet(SP + 'lse_daily_pool.parquet'); pool['day'] = pd.to_datetime(pool.ts).dt.normalize(); PC = pool.pivot_table(index='day', columns='symbol', values='close').reindex(C.index).ffill(limit=3)
PR = PC.pct_change(); PR = PR.where(PR.abs() < .5); PR = PR.reindex(idx); other = [t for t in PR.columns if t not in NM]
memO = pd.DataFrame({tk: [tk in {fx(t) for t in M.get(d.strftime('%Y-%m'), [])} for d in idx] for tk in other}, index=idx)
rO = PR[other].where(memO).mean(axis=1); r54m = rN.where(mem).mean(axis=1)
print('\nUNIVERSE SELECTION CHECK (EW of S&P members, daily rebalanced, full exposure; no ratio gate):')
for lab, s_ in (('SPY', rS), ('54 research names', r54m), (f'other {len(other)} pool names', rO)): a_ = st(s_, lab); h1 = st(s_, lab, s_.index < '2022-01-01'); h2 = st(s_, lab, s_.index >= '2022-01-01'); print(f"  {lab:24s} CAGR {a_['cagr']:6.1%} Sh {a_['sh']:.2f} MDD {a_['mdd']:6.1%} | 2019-21 {h1['cagr']:6.1%} | 2022-26 {h2['cagr']:6.1%}")
# ---- extra 2: where does the book's drawdown protection come from? breaker off; slots 6 vs unlimited-ish
for lab, kw in (('book, breaker OFF', dict(brk=None)), ('book, 8% breaker (live)', dict())):
    TK2, _ = sim_swap(B0, None, **kw); TK2 = [t for t in TK2 if t[1] <= END2]; s_ = mtm(TK2); a_ = st(s_, lab); print(f"  {lab:24s} trades {len(TK2)} CAGR {a_['cagr']:6.1%} Sh {a_['sh']:.2f} MDD {a_['mdd']:6.1%} Ulcer {a_['ulcer']:.3f}")
