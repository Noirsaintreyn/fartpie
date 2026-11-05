"""Nov 4 2026: TAKE PROFIT from the site's REACH model (reach_model_v2.json, 'touched by 12:00' logistic fit on NQ): target where the model gives a 50% touch chance. DECLARED BEFORE RESULTS.
Inputs at entry, as the site computes them: distance in bps, minutes to noon (1..145), sigma = Garman-Klass sigma of the last 60 5m bars (bps), RV = std of the last 6 5m log returns / sqrt(5) (bps/min);
the v2 path-shape features at their training means (neutral). Target = model distance for touch probability p, in NQ pts, floored at 20 and capped at 200.
  T75 fixed (current) | R50 p = 0.5 | R40 p = 0.4 (farther) | R60 p = 0.6 (closer) | R50 capped at 150.  Stop 50, flat 12:00, 1 pt cost.
Final system around it: Rule A trades, Yang-Zhang + VWAP ladder (whole micros), x0.7 while the last 20 trades (of that exit) are net negative, gap-gated eval. Lucid 50K ($88) + Topstep."""
_ad = open('xs/nq_adaptive_size.py').read(); exec(_ad[:_ad.rindex("if __name__ == '__main__':")])
RM = json.load(open('reach_model_v2.json')) if 'json' in globals() else __import__('json').load(open('reach_model_v2.json'))
mu_, sd_, cf_, b0_ = np.array(RM['mu']), np.array(RM['sd']), np.array(RM['coef']), RM['intercept']
def dist_bps(prob, tl, sig, rv):
    rest = (np.log(sig) - mu_[2]) / sd_[2] * cf_[2] + (np.log(rv) - mu_[3]) / sd_[3] * cf_[3] + (tl - mu_[1]) / sd_[1] * cf_[1]
    z0 = (np.log(prob / (1 - prob)) - b0_ - rest) / cf_[0]; return np.exp(z0 * sd_[0] + mu_[0])
gkv = .5 * np.log(h5 / l5) ** 2 - (2 * np.log(2) - 1) * np.log(c5 / o5) ** 2; gk60 = pd.Series(gkv).rolling(60, min_periods=30).mean().shift(1).values
lc = np.log(c5); rv6 = pd.Series(np.diff(lc, prepend=lc[0])).rolling(6).std().shift(1).values / np.sqrt(5)
order = XA.sort_values(['day', 'j']).index; TG = pd.DataFrame(index=XA.index)
j_ = XA.j.values; sig_b = np.sqrt(np.maximum(gk60[j_], 1e-12)) * 1e4; rv_b = np.maximum(rv6[j_], 1e-6) * 1e4; tl_ = np.clip(720 - m5[j_], 1, 145); pxn = o5[j_] * XA.k.values
for p_, nm in ((.5, 'R50'), (.4, 'R40'), (.6, 'R60')): TG[nm] = np.clip(dist_bps(p_, tl_, sig_b, rv_b) / 1e4 * pxn, 20, 200)
TG['R50 cap150'] = TG.R50.clip(upper=150); TG['T75'] = 75.0
print(f'sigma median {np.median(sig_b):.1f} bps, RV median {np.median(rv_b):.2f} bps/min (model training means {np.exp(mu_[2]):.1f} / {np.exp(mu_[3]):.2f})', flush=True)
mm = m5[j_]
for nm in ('R50', 'R40', 'R60'):
    print(f'  {nm} target pts: median {TG[nm].median():.0f} | entries 09:45-10:15 {TG[nm][mm <= 615].median():.0f} / 10:15-11:00 {TG[nm][(mm > 615) & (mm <= 660)].median():.0f} / 11:00-11:30 {TG[nm][mm > 660].median():.0f} | 2022 {TG[nm][XA.day.dt.year == 2022].median():.0f} / 2026 {TG[nm][XA.day.dt.year == 2026].median():.0f}', flush=True)
def seq(col, allow=None):
    out = {}; sub = XA.loc[order] if allow is None else XA.loc[order][allow.loc[order].values]
    for day, g in sub.groupby('day', sort=False):
        busy = -1; nn = 0
        for idx, r in g.iterrows():
            if nn >= 2 or r.j <= busy: continue
            res = sim2(int(r.j), int(r.dir), r.k, float(TG.at[idx, col]), 50.0)
            if res is None: continue
            out[idx] = res; busy = res[1]; nn += 1
    return out
def eq07(nets):
    m = pd.Series(1.0, index=XA.index); hist = []
    for idx in order:
        if idx in nets.index: m.loc[idx] = .7 if len(hist) >= 20 and sum(hist[-20:]) < 0 else 1.0; hist.append(nets.loc[idx])
    return m
def to_D(res, mult):
    D = [[] for _ in D_]
    for idx, (net, qe, p, how) in res.items():
        day = pd.Timestamp(XA.at[idx, 'day'])
        if day not in dpos: continue
        k = dpos[day]; v = int(bool(YZF.iloc[k])) + int(bool(X.at[idx, 'VWAPX'])); size = (1.0, .5, .3)[min(v, 2)] * float(mult.loc[idx])
        p = np.array(p) - 1.0; D[k].append(((size,), net - 1.0, p.min(), p))
    return D
PER = lambda d: np.select([d < P0, d < pd.Timestamp('2024-01-01'), d < pd.Timestamp('2026-01-01')], ['18-21', '22-23', '24-25'], '2026')
VAR = {}; print('\nRule A trades by take profit (1 lot, NQ pts)', flush=True)
for col in ('T75', 'R50', 'R40', 'R60', 'R50 cap150'):
    full = seq(col); nets = pd.Series({i: v[0] - 1.0 for i, v in full.items()}); hows = pd.Series({i: v[3] for i, v in full.items()}); per = PER(XA.loc[nets.index].day)
    print(f'  {col:<11} n{len(nets)} win {nets.gt(0).mean():.0%} target hit {np.mean(hows == "target"):.0%} stop {np.mean(hows == "stop"):.0%} noon {np.mean(hows == "noon"):.0%} | ' +
          ' / '.join(f'{p} {nets[per == p].mean():+5.1f}' for p in ('18-21', '22-23', '24-25', '2026')), flush=True)
    m = eq07(nets); DS[f'e_{col}'] = to_D(seq(col, ~AG['GAP']), m); DS[f'f_{col}'] = to_D(full, m); VAR[col] = (f'e_{col}', f'f_{col}', f'f_{col}')
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job10, jobs)
    print('\nFINAL SYSTEM WITH EACH TAKE PROFIT', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<11} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
