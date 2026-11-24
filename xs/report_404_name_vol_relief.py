"""Nov 23 2026 report 404. Rey: 'run the VIX-relief logic on the individual assets, or use the other vol models we tested'.
DECLARED (thresholds fixed now, nothing tuned; state at the prior close, enter only if the NAME is in relief):
 N1  close-to-close: 10d realized vol <= its own 20d mean.
 N2  15m realized vol (sum of squared 15m returns per RTH day, Andersen-Bollerslev): 5d mean <= 20d mean.
 N3  HAR-RV (Corsi 2009; log RV on lag day/week/month, pooled OLS fit on 2019-21 only): next-day forecast <= 22d mean RV.
 M1  market: SPY 10d realized vol <= its 20d mean (no VIX).
 C1  VIX relief 20d (report 402) AND N2 | C2  VIX relief 20d AND N1.
 NULL: 60 random block gates per (name, 10-day block) at the same keep share, 6-slot live sim re-run each time.
Same scorecard and PASS rule as report 402.  (Yang-Zhang / GJR-GARCH not run: daily low/open not in the 15m price cache used by the book, and Yang-Zhang was already negative as a flip filter in earlier reports.)"""
import numpy as np
_t = open('xs/report_402_five_ideas.py').read(); exec(_t[:_t.index("rows = []; TK0, s0")])
# daily 15m realized vol per name
RV = {}
for tk in NM:
    ts, o_, h_, c_ = PX[tk]; t = pd.DatetimeIndex(ts); mm = t.hour * 60 + t.minute; ok = (mm >= 570) & (mm < 960); t = t[ok]; c = np.log(c_[ok]); day = t.normalize(); r = np.r_[np.nan, np.diff(c)]; r[np.r_[True, day[1:] != day[:-1]]] = np.nan
    RV[tk] = pd.Series(r ** 2, index=day).groupby(level=0).sum(min_count=1) ** .5
RVd = pd.DataFrame(RV).reindex(idx); RVd = RVd.where(RVd > 0)
ret = pd.DataFrame(rn, index=idx, columns=NM)
rv10 = ret.rolling(10).std(); N1 = rv10 <= rv10.rolling(20).mean()
N2 = RVd.rolling(5).mean() <= RVd.rolling(20).mean()
L = np.log(RVd); X = pd.concat({'d': L, 'w': np.log(RVd.rolling(5).mean()), 'm': np.log(RVd.rolling(22).mean())}, axis=1); y = L.shift(-1)
trn = idx < '2022-01-01'; Xf = np.column_stack([X[k].values[trn].ravel() for k in ('d', 'w', 'm')]); yf = y.values[trn].ravel(); okf = np.isfinite(Xf).all(1) & np.isfinite(yf)
A_ = np.column_stack([np.ones(okf.sum()), Xf[okf]]); beta = np.linalg.lstsq(A_, yf[okf], rcond=None)[0]; print('HAR pooled coefs (const, d, w, m):', beta.round(3), flush=True)
F = np.exp(beta[0] + beta[1] * X['d'] + beta[2] * X['w'] + beta[3] * X['m']); N3 = F <= RVd.rolling(22).mean()
rv_s = rS.rolling(10).std(); M1 = pd.Series((rv_s <= rv_s.rolling(20).mean()).values, index=idx)
VX = (vix <= vix.rolling(20).mean())
def run_gate_name(Mdf):
    S = Mdf.shift(1); ed = pd.DatetimeIndex(B0.entry).normalize(); a = np.minimum(idx.searchsorted(ed), nD - 1); c = np.array([ncol[t] for t in B0.tk]); v = S.values[a, c].astype(float); ok = np.where(np.isnan(v), True, v > 0)
    return B0[ok].reset_index(drop=True)
rows = []; TK0, s0 = live_fn(B0); base = score('LIVE', TK0, s0); print(line(base)); print(f"   CAGR halves {base['ca'][0]:.1%} | {base['cb'][0]:.1%}")
VARS = {'N1 name close-RV10 relief': N1, 'N2 name 15m-RV5 relief': N2, 'N3 name HAR forecast <= 22d mean': N3}
for lab, Mdf in VARS.items():
    Bf = run_gate_name(Mdf); TKs, s = live_fn(Bf); d = score(lab, TKs, s); rows.append(d); print(line(d) + f' | kept {len(Bf)/len(B0):.0%} | CAGR halves {d["ca"][0]:.1%} | {d["cb"][0]:.1%}', flush=True)
Bf = run_gate(M1.shift(1).astype(bool)); TKs, s = live_fn(Bf); d = score('M1 SPY RV10 relief (no VIX)', TKs, s); rows.append(d); print(line(d) + f' | kept {len(Bf)/len(B0):.0%} | CAGR halves {d["ca"][0]:.1%} | {d["cb"][0]:.1%}', flush=True)
for lab, Mdf in (('C1 VIX relief AND N2', N2 & VX.values[:, None]), ('C2 VIX relief AND N1', N1 & VX.values[:, None])):
    Bf = run_gate_name(Mdf); TKs, s = live_fn(Bf); d = score(lab, TKs, s); rows.append(d); print(line(d) + f' | kept {len(Bf)/len(B0):.0%} | CAGR halves {d["ca"][0]:.1%} | {d["cb"][0]:.1%}', flush=True)
b = base; print('\nPASS TABLE:')
for d in rows: print(f"  {d['lab']:34s} half1 Cal {d['ca'][2]:.2f} (live {b['ca'][2]:.2f}) MDD {d['ca'][1]:.1%} (live {b['ca'][1]:.1%}) | half2 Cal {d['cb'][2]:.2f} ({b['cb'][2]:.2f}) MDD {d['cb'][1]:.1%} ({b['cb'][1]:.1%}) | win {d['win']:.1%} med {d['med']*100:.2f}% top5 {d['top5']:.0%} -> {'PASS' if (d['ca'][2] > b['ca'][2] and d['cb'][2] > b['cb'][2] and d['ca'][1] > b['ca'][1] and d['cb'][1] > b['cb'][1]) else 'fail'}")
# null for the best name-level gate by Sharpe among N1-N3
nm = max(VARS, key=lambda k: [r for r in rows if r['lab'] == k][0]['sh']); real = [r for r in rows if r['lab'] == nm][0]; keep = float(np.nanmean(VARS[nm].shift(1).values)); rng = np.random.default_rng(404); out = []
for k in range(60):
    blk = rng.random((nD // 10 + 2, len(NM))) < keep; Mrand = pd.DataFrame(np.repeat(blk, 10, axis=0)[:nD], index=idx, columns=NM); Bf = run_gate_name(Mrand); TKs, s = live_fn(Bf); d = score('', TKs, s); out.append((d['mdd'], d['w12'], d['calmar'], d['cagr'], d['sh']))
A = np.array(out); print(f"\nNULL for '{nm}' (60 random per-name block gates, keep {keep:.0%}): MDD median {np.median(A[:,0]):.1%} [5-95 {np.percentile(A[:,0],5):.1%}, {np.percentile(A[:,0],95):.1%}] | Calmar median {np.median(A[:,2]):.2f} | Sharpe median {np.median(A[:,4]):.2f}")
print(f"real: MDD {real['mdd']:.1%} better than {np.mean(A[:,0] < real['mdd']):.0%} of null; Calmar higher than {np.mean(A[:,2] < real['calmar']):.0%}; Sharpe higher than {np.mean(A[:,4] < real['sh']):.0%}")
