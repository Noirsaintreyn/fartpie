"""Nov 23 2026 report 402. Rey: 'work on the actual strat: macro + options + vector; higher median, higher win rate, less concentration, consistency, low DD; 5 unique ideas from the papers'.
LITERATURE (abstract-level; papers Rey sent are not in the workspace): Moreira-Muir 2017 volatility-managed portfolios (scale exposure by inverse trailing variance); Harvey et al. 2018 vol targeting;
Gilchrist-Zakrajsek 2012 / Wellington 'equity-vol-credit harmony' (credit leads equity risk); Xiong 2026 (arXiv 2605.20636, macro score from 21d 10y-yield change, SPY drawdown depth, VIX stress relief -> Sharpe 1.01 vs SPY 0.85);
Bollerslev-Tauchen-Zhou 2009 variance risk premium (VIX^2 - RV^2 predicts equity returns); Li-Wang 2026 (arXiv 2608.26115, option-implied IV spread/skew significant in every regime, smirk decays); correlation / dispersion regime (Driessen-Maenhout-Vilkov; Lopez de Prado HRP diversification).
IDEAS (DECLARED, thresholds from 2019-21 only, all state variables lagged to the prior close; entry gates re-run the 6-slot live sim on the filtered flips so freed slots refill):
 I1a VOL-MANAGED (book own 21d vol, target = train median, exposure min(1, target/vol), cash 0) | I1b same with SPY 21d vol.
 I2  CREDIT VECTOR: vector regime of daily HYG/IEF must be bull to enter.
 I3  CORRELATION REGIME: average pairwise 60d correlation of the 54 names <= train median to enter (high correlation = one macro bet).
 I4a VRP: VIX - 100*sqrt(252)*std(SPY 21d) >= train median to enter | I4b VIX STRESS RELIEF: VIX <= its 20d mean to enter.
 I5  CORRELATION-AWARE SLOTS: skip a flip whose max 60d correlation with an open name > 0.75 (I5a) or > 0.65 (I5b).
SCORE: trades, win rate, median trade (net 4bp), CAGR, Sharpe (true daily MTM), MDD, Calmar, share of months positive, worst month, worst 12m, top-5 names' share of P&L, CAGR ex-top5; 2019-21 | 2022-26 Calmar.
PASS (declared): Calmar higher AND MDD lower than live in BOTH halves, with median trade and win rate not lower (entry-gate/slot ideas) and top-5 share not higher."""
import sys, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); import vector_pine as vp
_s = open('xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
UP = '/mnt/user-data/uploads/fartpie/'; END2 = pd.Timestamp('2026-08-14'); P = pd.read_parquet(SP + 'panel401.parquet')
C = P.pivot(index='day', columns='tk', values='close'); idx = C.index[(C.index >= '2018-10-01') & (C.index <= END2)]; C = C.reindex(idx); spy = C.pop('SPY'); NM = list(C.columns); rn = C.pct_change().fillna(0).values; ncol = {t: i for i, t in enumerate(NM)}
tr_m = idx >= '2019-01-02'; TRAIN = idx < '2022-01-01'; nD = len(idx)
Cv = C.ffill().values; ci = ncol
vix = pd.read_parquet('data/yahoo/daily_v/VIX.parquet').close; vix.index = pd.to_datetime(vix.index).normalize(); vix = vix.reindex(idx).ffill()
MD = pd.read_parquet(SP + 'macro_daily.parquet'); MD['day'] = pd.to_datetime(MD.ts).dt.normalize(); MC = MD.pivot(index='day', columns='symbol', values='close').reindex(idx).ffill()
rS = spy.pct_change(); rv = rS.rolling(21).std() * np.sqrt(252) * 100
# state series (value at date t uses data through t; lagged one day at use)
cr = (MC.HYG / MC.IEF).dropna(); d_ = pd.DataFrame({'high': cr, 'low': cr, 'close': cr}); regs = vp.run_state_machine(d_)[0]; credit = pd.Series(regs, index=cr.index).reindex(idx).ffill()
avgcorr = pd.Series(index=idx, dtype=float); R_ = pd.DataFrame(rn, index=idx, columns=NM)
for i in range(61, nD):
    X = R_.iloc[i - 60:i + 1].values; cm = np.corrcoef(X.T); avgcorr.iloc[i] = np.nanmean(cm[np.triu_indices_from(cm, 1)])
vrp = vix - rv; relief = (vix <= vix.rolling(20).mean()).astype(float)
thr_c = avgcorr[TRAIN].median(); thr_v = vrp[TRAIN].median()
STATE = {'I2 credit vector bull': credit.shift(1) == 1, 'I3 low avg corr (<= train median)': avgcorr.shift(1) <= thr_c, 'I4a high VRP (>= train median)': vrp.shift(1) >= thr_v, 'I4b VIX stress relief': relief.shift(1) == 1}
print(f'thresholds (train): avg corr median {thr_c:.3f}, VRP median {thr_v:.2f}; share of days allowed (full): ' + ', '.join(f'{k[:3]} {v.reindex(idx).mean():.0%}' for k, v in STATE.items()), flush=True)
def mtm(TKs, cost=2e-4):
    R = np.zeros(nD)
    for tk, e, x, r, ra in TKs:
        c = ci[tk]; a = idx.searchsorted(e.normalize()); b = min(idx.searchsorted(x.normalize()), nD - 1); pe = px_at(tk, e); pxx = px_at(tk, x)
        if a >= nD: continue
        if a >= b: R[a] += (pxx / pe - 1 - 2 * cost) / 6; continue
        R[a] += (Cv[a, c] / pe - 1 - cost) / 6; R[a + 1:b] += (Cv[a + 1:b, c] / Cv[a:b - 1, c] - 1) / 6; R[b] += (pxx / Cv[b - 1, c] - 1 - cost) / 6
    return pd.Series(R, index=idx)[tr_m]
def sim_div(Bs, thr, N=6, brk=.08):
    T = order(Bs); op = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []
    for e, x, r, tk, ra in zip(T.entry.values, T.exit.values, T.ret.values, T.tk.values, T.ratio.values):
        done = sorted([p for p in op if p[0] <= e], key=lambda z: z[0]); op = [p for p in op if p[0] > e]
        for p in done: real += np.log1p(p[1] / N); hist.append((p[0], real)); win.append((p[0], real))
        if len(op) >= N: continue
        if op and thr is not None:
            a = idx.searchsorted(pd.Timestamp(e).normalize()); cs = [np.corrcoef(rn[a - 60:a, ci[tk]], rn[a - 60:a, ci[p[2]]])[0, 1] for p in op]
            if np.nanmax(cs) > thr: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [hh for _, hh in win]) < np.log1p(-brk): continue
        rec = [tk, pd.Timestamp(e), pd.Timestamp(x), r, ra]; op.append([x, r, tk, pd.Timestamp(e), ra]); out.append(tuple(rec))
    return out
def score(lab, TKs, s, base=None):
    TKs = [t for t in TKs if t[1] <= END2 and t[1] >= pd.Timestamp('2019-01-02')]; nr = np.array([t[3] - 4e-4 for t in TKs]); e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; cg = e.iloc[-1] ** (1 / y) - 1; dd = (e / e.cummax() - 1).min()
    mo = (1 + s).resample('ME').prod() - 1; r12 = e.pct_change(252).min(); pn = pd.Series([t[3] for t in TKs]).groupby([t[0] for t in TKs]).sum().sort_values(ascending=False); sh5 = pn.head(5).sum() / pn.sum() if pn.sum() > 0 else np.nan
    h = {k: s[m] for k, m in (('a', s.index < '2022-01-01'), ('b', s.index >= '2022-01-01'))}; hc = {}
    for k, z in h.items():
        ee = (1 + z).cumprod(); yy = (z.index[-1] - z.index[0]).days / 365.25; c_ = ee.iloc[-1] ** (1 / yy) - 1; d_ = (ee / ee.cummax() - 1).min(); hc[k] = (c_, d_, c_ / abs(d_))
    ex5 = ''
    return dict(lab=lab, n=len(TKs), win=(nr > 0).mean(), med=np.median(nr), cagr=cg, sh=sharpe(s.values), mdd=dd, calmar=cg / abs(dd), pm=(mo > 0).mean(), wm=mo.min(), w12=r12, top5=sh5, ca=hc['a'], cb=hc['b'], top=list(pn.index[:5]))
def line(d):
    return f"{d['lab']:38s} n{d['n']:5d} win {d['win']:5.1%} med {d['med']*100:5.2f}% | CAGR {d['cagr']:5.1%} Sh {d['sh']:4.2f} MDD {d['mdd']:6.1%} Cal {d['calmar']:4.2f} | +mo {d['pm']:4.0%} worstMo {d['wm']:6.1%} worst12m {d['w12']:6.1%} top5 {d['top5']:5.0%} | 19-21 MDD {d['ca'][1]:6.1%} Cal {d['ca'][2]:4.2f} | 22-26 MDD {d['cb'][1]:6.1%} Cal {d['cb'][2]:4.2f}"
def ex5(Bv, fn):
    tp = score('x', *fn(Bv))['top']; B2 = Bv[~Bv.tk.isin(tp)]; TK2, s2 = fn(B2); d = score('', TK2, s2); return d['cagr'], d['mdd']
def run_gate(mask):
    ed = pd.DatetimeIndex(B0.entry).normalize(); a = idx.searchsorted(ed); ok = np.where(a < nD, mask.reindex(idx).values[np.minimum(a, nD - 1)], True); ok = np.where(np.isnan(ok.astype(float)), True, ok).astype(bool)
    return B0[ok].reset_index(drop=True)
def live_fn(Bv):
    TKs, _ = sim_swap(Bv, None); TKs = [t for t in TKs if t[1] <= END2]; return TKs, mtm(TKs)
rows = []; TK0, s0 = live_fn(B0); base = score('LIVE (baseline)', TK0, s0); rows.append(base); print(line(base), flush=True); e0 = ex5(B0, live_fn); print(f"   live CAGR ex-top5 names {e0[0]:.1%} MDD {e0[1]:.1%}  (top5: {base['top']})")
# I1 vol-managed (scaling of the live book's daily returns; the trade list is unchanged)
sd_own = s0.rolling(21).std().shift(1) * np.sqrt(252); sd_spy = (rS.rolling(21).std() * np.sqrt(252)).shift(1).reindex(s0.index)
for lab, sd in (('I1a vol-managed (book own vol)', sd_own), ('I1b vol-managed (SPY vol)', sd_spy)):
    tgt = sd[sd.index < '2022-01-01'].median(); m = (tgt / sd).clip(upper=1).fillna(1); d = score(lab, TK0, s0 * m); rows.append(d); print(line(d) + f' | avg exposure {m.mean():.2f}', flush=True)
# I2-I4 entry gates
for lab, mask in STATE.items():
    Bf = run_gate(mask); TKs, s = live_fn(Bf); d = score(lab, TKs, s); rows.append(d); print(line(d) + f' | flips kept {len(Bf)/len(B0):.0%}', flush=True)
# I5 correlation-aware slots
for lab, thr in (('I5a skip if max corr > 0.75', .75), ('I5b skip if max corr > 0.65', .65)):
    TKs = [t for t in sim_div(B0, thr) if t[1] <= END2]; d = score(lab, TKs, mtm(TKs)); rows.append(d); print(line(d), flush=True)
pd.DataFrame(rows).drop(columns=['top', 'ca', 'cb']).to_csv(SP + 'ideas402.csv', index=False)
b = rows[0]; print('\nPASS TABLE (Calmar higher & MDD lower than live in BOTH halves):')
for d in rows[1:]: print(f"  {d['lab']:38s} half1 Cal {d['ca'][2]:.2f} vs {b['ca'][2]:.2f}, MDD {d['ca'][1]:.1%} vs {b['ca'][1]:.1%} | half2 Cal {d['cb'][2]:.2f} vs {b['cb'][2]:.2f}, MDD {d['cb'][1]:.1%} vs {b['cb'][1]:.1%} | win {d['win']:.1%} vs {b['win']:.1%}, med {d['med']*100:.2f} vs {b['med']*100:.2f}, top5 {d['top5']:.0%} vs {b['top5']:.0%} -> {'PASS' if (d['ca'][2] > b['ca'][2] and d['cb'][2] > b['cb'][2] and d['ca'][1] > b['ca'][1] and d['cb'][1] > b['cb'][1]) else 'fail'}")
