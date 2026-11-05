"""Nov 4 2026: MICROS FROM AN EQUATION instead of a fixed ladder. DECLARED BEFORE RESULTS.
micros = clip(round(R x cushion / $102), 1, 15)   cushion = balance - max-loss floor at the day's start; $102 = one micro at the 51-pt stop incl. cost.
  R is the share of the cushion risked on the trade:
  L0  current fixed ladder (10 / 5 / 3 MNQ by Yang-Zhang high + VWAP stretched; eval x0.3 / funded x0.7 cold streaks) - reference
  C25 / C50 / C75   R = 0.25 / 0.50 / 0.75 flat
  CV50  R = 0.50 x vol scale; vol scale = clip(1-year median Yang-Zhang sigma / current, 0.5, 1.5)
  K     R = 2.5 x Kelly from the last 40 trades (Kelly = p - (1 - p) / (avg win / avg loss), clipped 0 - 0.5), x vol scale  (2.5x because ruin only costs the eval fee)
  KS    'everything': K x 0.6 if VWAP stretched x 0.8 if HMM bear x 0.8 if Hurst < 0.45, vol scale from HAR instead of Yang-Zhang
Rule A trades, 75 / 50, flat 12:00, eval gap-gated. Lucid 50K ($88) + Topstep."""
_rt = open('xs/nq_reach_tp.py').read(); exec(_rt[:_rt.rindex("if __name__ == '__main__':")])
LBL = pd.read_parquet('/tmp/claude-0/ml/nq_flip_types_C.parquet')[['day', 'j', 'HMM', 'hurst']]
XL = XA[['day', 'j']].rename_axis('ix').reset_index().merge(LBL, on=['day', 'j'], how='left').set_index('ix')
full = seq('T75'); nets = pd.Series({i: v[0] - 1.0 for i, v in full.items()}); ordn = [i for i in order if i in nets.index]
kel = pd.Series(np.nan, index=XA.index); hist = []
for idx in ordn:
    if len(hist) >= 40:
        h = np.array(hist[-40:]); w, l = h[h > 0], -h[h <= 0]; p = len(w) / 40
        if len(w) and len(l): kel.loc[idx] = np.clip(p - (1 - p) / (w.mean() / l.mean()), 0, .5)
    hist.append(nets.loc[idx])
kel = kel.fillna(.15)
sig_yz = np.sqrt(yz.dropna()); vs_yz = (sig_yz.rolling(252, min_periods=120).median() / sig_yz).clip(.5, 1.5)
sig_har = np.sqrt(har.dropna()); vs_har = (sig_har.rolling(252, min_periods=120).median() / sig_har).clip(.5, 1.5)
vsy = pd.Series(prior(vs_yz, XA.day), index=XA.index).fillna(1.0); vsh = pd.Series(prior(vs_har, XA.day), index=XA.index).fillna(1.0)
vwx = pd.Series(X.loc[XA.index, 'VWAPX'].astype(bool).values, index=XA.index); bear = (XL.HMM == 'HMM bear').reindex(XA.index).fillna(False); mr = (XL.hurst < .45).reindex(XA.index).fillna(False)
RFR = {'C25': pd.Series(.25, index=XA.index), 'C50': pd.Series(.5, index=XA.index), 'C75': pd.Series(.75, index=XA.index), 'CV50': .5 * vsy,
       'K': 2.5 * kel * vsy, 'KS': 2.5 * kel * vsh * np.where(vwx, .6, 1.0) * np.where(bear, .8, 1.0) * np.where(mr, .8, 1.0)}
print('risk fraction R (share of cushion) - median / 10th / 90th pct: ' + ' | '.join(f'{k} {v.median():.2f} / {v.quantile(.1):.2f} / {v.quantile(.9):.2f}' for k, v in RFR.items()), flush=True)
print(f'Kelly from the last 40 trades: median {kel.median():.2f}, range {kel.min():.2f} - {kel.max():.2f}', flush=True)
def toR(res, rf):
    D = [[] for _ in D_]
    for idx, (net, qe, p, how) in res.items():
        day = pd.Timestamp(XA.at[idx, 'day'])
        if day in dpos: k = dpos[day]; pa = np.array(p) - 1.0; D[dpos[day]].append(((float(rf.loc[idx]),), net - 1.0, pa.min(), pa))
    return D
ev_res = seq('T75', ~AG['GAP'])
for nm, rf in RFR.items(): DS[f'e_{nm}'] = toR(ev_res, rf); DS[f'f_{nm}'] = toR(full, rf)
dayq_ladder = dayq; MICS = []
def dayq_R(k, st, f):
    cum = 0.0; bal = st['bal']; cush = max(bal - st['floor'], 0.0)
    for th, net, mae, p in DAY[k]:
        if cum <= -f['pstop']: break
        mic = int(np.clip(round(th[0] * cush / 102.0), 1, 15)); m = 2.0 * mic; MICS.append(mic)
        if bal + cum + mae * m <= st['floor']: return st['floor'] - bal, True
        if f['dll'] and cum + mae * m <= -f['dll']: cum = -float(f['dll']); break
        cum += net * m
    return cum, False
VAR = {'L0 fixed ladder (current)': ('e_EQ20 x0.3', 'f_T75', 'f_T75')}
VAR.update({nm: (f'e_{nm}', f'f_{nm}', f'f_{nm}') for nm in RFR})
def job17(a):
    global dayq, MICS
    dayq = dayq_ladder if a[0].startswith('L0') else dayq_R; MICS = []; r = job10(a); r['mic'] = np.mean(MICS) if MICS else np.nan; return r
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job17, jobs)
    print('\nSIZING BY EQUATION', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<26} {r["fn"]:<9} | avg micros {r["mic"]:4.1f} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} positive {r["k_pos"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
