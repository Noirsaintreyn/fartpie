"""Nov 12 2026: do DENOISED trend filters agree with the good vector flips? DECLARED BEFORE RESULTS. Rey: EMA adds information beyond consolidation; try Kalman / wavelet / EWMA smoothing so chop is out and the big moves line up.
$1-2 names, 1h vector bull flips with flip-bar volume >= 1.5x (B) and B + daily bull (A), since 2024-01, baseline exit (close-based flip), 40 bp, train < 2025-07-01 | test after. ALL CAUSAL (value at the flip bar uses only bars <= flip bar):
 KAL: local-linear-trend Kalman filter on log close, fixed noise ratios (obs var = var of 1-bar returns, level q = .05x, slope q = .001x - nothing fitted); feature = filtered slope > 0, and slope t-stat > 1.
 WAV: db4 level-3 soft-threshold (universal threshold) denoise of the trailing 128 log closes; feature = denoised level now > denoised 5 bars ago AND close > denoised level.
 EWM: EWMA(10) > EWMA(30) and EWMA(10) rising.  RAW: close > EMA200 (the earlier winner) for reference.
Also: of the trades that made >= +10% net, what share is 'aligned' per filter vs share of all trades aligned."""
import sys, pickle, numpy as np, pandas as pd, pywt, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); sys.path.insert(0, 'xs')
import midsmall_screener as M
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); COST = .004
T0 = pd.read_pickle(SC + 'mtf_T.pkl')[['tk', 'et', 'ad', 'lv']]
def kalman(y):
    r = np.diff(y); R = max(np.nanvar(r[:500]), 1e-8); qL, qS = .05 * R, .001 * R; x = np.array([y[0], 0.]); P = np.diag([R, R * .01]); F = np.array([[1, 1], [0, 1]]); Q = np.diag([qL, qS]); sl = np.zeros(len(y)); ts = np.zeros(len(y))
    for i in range(1, len(y)):
        x = F @ x; P = F @ P @ F.T + Q; S = P[0, 0] + R; K = P[:, 0] / S; x = x + K * (y[i] - x[0]); P = P - np.outer(K, P[0]); sl[i] = x[1]; ts[i] = x[1] / np.sqrt(max(P[1, 1], 1e-18))
    return sl, ts
def wav(ly):
    c = pywt.wavedec(ly, 'db4', level=3, mode='symmetric'); sig = np.median(np.abs(c[-1])) / .6745; thr = sig * np.sqrt(2 * np.log(len(ly)))
    c[1:] = [pywt.threshold(x, thr, 'soft') for x in c[1:]]; return pywt.waverec(c, 'db4', mode='symmetric')[:len(ly)]
def gen(tk):
    Y = M.closed_only(data[tk].dropna(subset=['open', 'close']))
    if len(Y) < 250: return []
    reg = run_state_machine(Y.assign(datetime=Y.index))[0]; c = Y.close.values; o = Y.open.values; v = Y.volume.values.astype(float); vrel = v / pd.Series(v).rolling(20).mean().shift().values; idx = Y.index
    ly = np.log(c); s = pd.Series(c); e200 = s.ewm(span=200, adjust=False).mean().values; ew10 = s.ewm(span=10, adjust=False).mean().values; ew30 = s.ewm(span=30, adjust=False).mean().values; sl = ts = None; out = []; ent = None
    for i in range(1, len(Y) - 1):
        if reg[i] == reg[i - 1] or reg[i] == 0: continue
        if reg[i] == 1 and ent is None: ent = i
        elif reg[i] == -1 and ent is not None:
            fi = ent; ei = fi + 1; xi = i + 1
            if 1 <= c[fi] < 2 and vrel[fi] >= 1.5 and idx[ei] >= pd.Timestamp('2024-01-01') and fi >= 140:
                if sl is None: sl, ts = kalman(ly)
                w = wav(ly[fi - 127:fi + 1])
                out.append(dict(tk=tk, et=idx[ei], xt=idx[xi], sig=c[fi], vr=vrel[fi], ret=o[xi] / o[ei] - 1, kal=sl[fi] > 0, kalt=ts[fi] > 1, wav=(w[-1] > w[-6]) and (ly[fi] > w[-1]), wavs=w[-1] > w[-6], ewm=(ew10[fi] > ew30[fi]) and (ew10[fi] > ew10[fi - 1]), a200=c[fi] > e200[fi]))
            ent = None
        elif reg[i] == -1: ent = None
    return out
T = pd.DataFrame([x for tk in data for x in gen(tk)]).merge(T0, on=['tk', 'et'], how='left'); T['ad'] = T.ad.fillna(False).astype(bool); T.to_pickle(SC + 'smooth_T.pkl'); print('B trades', len(T))
def book(Z, K=8):
    eq = 1.0; op = []; curve = []
    for r in sorted(Z.to_dict('records'), key=lambda r: (r['et'], -r['vr'])):
        still = []
        for p in op:
            if p['xt'] <= r['et']: eq += p['sz'] * (p['ret'] - COST); curve.append((p['xt'], eq))
            else: still.append(p)
        op = still
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: eq += p['sz'] * (p['ret'] - COST); curve.append((p['xt'], eq))
    return pd.Series(dict(curve)).sort_index()
def rep(nm, Z):
    if len(Z) < 15: print(f'  {nm:<30} {len(Z)} trades (too few)'); return
    r = Z.ret - COST; a, b = r[Z.et < '2025-07-01'], r[Z.et >= '2025-07-01']; s = book(Z); yrs = (s.index[-1] - Z.et.min()).days / 365.25; cg = s.iloc[-1] ** (1 / max(yrs, .5)) - 1
    print(f'  {nm:<30} {len(Z):>4} tr | mean {r.mean()*100:+.2f}% med {r.median()*100:+.1f}% win {(r>0).mean():.0%} | train {a.mean()*100:+.2f}% ({len(a)}) test {b.mean()*100:+.2f}% ({len(b)}) | book {cg:+.0%}/yr DD {(s/s.cummax()-1).min():.0%}', flush=True)
rng = np.random.default_rng(3)
def pval(Z, m):
    r = (Z.ret - COST).values; k = r[m.values]; dr = [rng.choice(r, len(k), replace=False).mean() for _ in range(4000)]; return (np.array(dr) >= k.mean()).mean()
B = T; print('== B ($1-2, vol>=1.5x)'); rep('base', B)
for k, nm in (('a200', 'close > EMA200 (ref)'), ('ewm', 'EWMA10>30 & rising'), ('kal', 'Kalman slope > 0'), ('kalt', 'Kalman slope t > 1'), ('wavs', 'Wavelet level rising'), ('wav', 'Wavelet rising & close above')):
    rep(f'{nm}', B[B[k]]); rep(f'  NOT {nm}', B[~B[k]]); print(f'      p vs random skips {pval(B, B[k]):.3f}')
print('== A (B + daily bull)'); A = B[B.ad]; rep('base', A)
for k, nm in (('a200', 'close > EMA200'), ('ewm', 'EWMA10>30 & rising'), ('kal', 'Kalman slope > 0'), ('wavs', 'Wavelet level rising')): rep(nm, A[A[k]]); rep(f'  NOT {nm}', A[~A[k]])
print('== do the big winners align?  share of trades >= +10% net that are aligned vs share of ALL trades aligned')
big = (B.ret - COST) >= .10; print(f'  big winners {int(big.sum())} of {len(B)}')
for k, nm in (('a200', 'EMA200'), ('ewm', 'EWMA'), ('kal', 'Kalman'), ('kalt', 'Kalman t>1'), ('wavs', 'Wavelet'), ('wav', 'Wavelet+above')): print(f'  {nm:<12} aligned: big winners {B[big][k].mean():.0%} | all {B[k].mean():.0%} | P(big | aligned) {big[B[k]].mean():.0%} vs P(big | not) {big[~B[k]].mean():.0%}')
