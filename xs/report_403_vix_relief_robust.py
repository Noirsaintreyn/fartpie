"""Nov 23 2026 report 403: stress test of the one near-miss from report 402 (I4b VIX stress relief gate: enter only if prior-close VIX <= its 20d mean).
DECLARED: (a) neighbours VIX <= 10d mean / 50d mean / VIX <= VIX 5d ago / VIX<20 level (contrast): lead is robust only if the three relief variants all cut MDD in BOTH halves; (b) null: 100 random block gates (10-day blocks, on-probability = the real kept share 0.48-0.58) -> percentile of the real gate's MDD / worst-12m / Calmar; (c) combination with I5a correlation-aware slots; (d) CAGR by half."""
import numpy as np
_t = open('xs/report_402_five_ideas.py').read(); exec(_t[:_t.index("rows = []; TK0, s0")])
TK0, s0 = live_fn(B0); base = score('LIVE', TK0, s0); print(line(base)); print(f"   CAGR halves {base['ca'][0]:.1%} | {base['cb'][0]:.1%}")
V = {'relief 20d (the 402 gate)': (vix <= vix.rolling(20).mean()), 'relief 10d': (vix <= vix.rolling(10).mean()), 'relief 50d': (vix <= vix.rolling(50).mean()), 'VIX <= VIX 5d ago': (vix <= vix.shift(5)), 'VIX level < 20 (contrast)': (vix < 20)}
res = {}
for lab, m in V.items():
    Bf = run_gate(m.shift(1).astype(bool)); TKs, s = live_fn(Bf); d = score(lab, TKs, s); res[lab] = d; print(line(d) + f' | kept {len(Bf)/len(B0):.0%} | CAGR halves {d["ca"][0]:.1%} | {d["cb"][0]:.1%}', flush=True)
Bf = run_gate((vix <= vix.rolling(20).mean()).shift(1).astype(bool)); TKs = [t for t in sim_div(Bf, .75) if t[1] <= END2]; d = score('relief20 + I5a corr-aware slots', TKs, mtm(TKs)); print(line(d) + f' | CAGR halves {d["ca"][0]:.1%} | {d["cb"][0]:.1%}', flush=True)
# null: random block gates
rng = np.random.default_rng(403); real = res['relief 20d (the 402 gate)']; keep = (vix <= vix.rolling(20).mean()).mean(); out = []
for k in range(100):
    blk = rng.random(nD // 10 + 2) < keep; m = pd.Series(np.repeat(blk, 10)[:nD], index=idx); Bf = run_gate(m.shift(1).fillna(True).astype(bool)); TKs, s = live_fn(Bf); d = score('', TKs, s); out.append((d['mdd'], d['w12'], d['calmar'], d['cagr'], d['sh']))
A = np.array(out); print(f"\nRANDOM BLOCK-GATE NULL (n=100, keep share ~{keep:.0%}): MDD median {np.median(A[:,0]):.1%} [5-95 {np.percentile(A[:,0],5):.1%}, {np.percentile(A[:,0],95):.1%}] | worst12m median {np.median(A[:,1]):.1%} | Calmar median {np.median(A[:,2]):.2f} | CAGR median {np.median(A[:,3]):.1%} | Sharpe median {np.median(A[:,4]):.2f}")
print(f"real gate percentile: MDD better than {np.mean(A[:,0] < real['mdd']):.0%} of null, worst12m better than {np.mean(A[:,1] < real['w12']):.0%}, Calmar higher than {np.mean(A[:,2] < real['calmar']):.0%}, Sharpe higher than {np.mean(A[:,4] < real['sh']):.0%}")
