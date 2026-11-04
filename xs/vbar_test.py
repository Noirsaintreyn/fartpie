"""Nov 4 2026: volume bars instead of 15m time bars for the vector's 15m track. DECLARED BEFORE RESULTS. 54 names, live rules otherwise, 40 matched runs vs live, 2019-21 / 2022-26.
Volume bars are built from the 15m bars (premarket+RTH, as live): consecutive 15m bars are merged until their volume reaches V* = (mean total daily volume of the
PRIOR 20 sessions) / K; a bar never spans two sessions (the last partial bar of a day closes at 16:00); a single 15m bar above V* stays one bar (15m data can't split it).
  VB48  K = 48 (about one bar per 15 minutes on an average day; quiet premarket bars merge, busy open bars stay 15m)
  VB24  K = 24 (about 30 minutes on average)        VB96  K = 96 (mostly merges thin premarket bars only)
The vector runs on the volume bars; its state is known at the close of the volume bar's last 15m bar. Entries/exits stay at RTH 15m opens, sticky 1h stop, +5% lock,
the 1h track and its trades are unchanged. Parity check: variant L rebuilt here must equal the stored live set."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/stop_1h.py').read(); exec(src[:src.index("if __name__ == '__main__':")])
def vbars(H, K):
    d = H.dt.dt.normalize().values; v = H.volume.values.astype(float)
    dv = pd.Series(v).groupby(d).sum(); vstar = (dv.rolling(20, min_periods=10).mean().shift(1) / K)
    vs = pd.Series(d).map(vstar).values; o, h, l, c = H.open.values, H.high.values, H.low.values, H.close.values
    rows = []; cum = 0.0; s = 0
    for i in range(len(H)):
        cum += v[i]; last_of_day = i == len(H) - 1 or d[i + 1] != d[i]
        if (vs[i] == vs[i] and cum >= vs[i]) or last_of_day or vs[i] != vs[i]:
            rows.append((o[s], h[s:i + 1].max(), l[s:i + 1].min(), c[i], i)); cum = 0.0; s = i + 1
    B = pd.DataFrame(rows, columns=['open', 'high', 'low', 'close', 'end']); return B
def vreg(H, K):
    B = vbars(H, K); r = np.asarray(vec(B[['open', 'high', 'low', 'close']])); out = np.zeros(len(H), int); k = 0
    ends = B.end.values
    for j in range(len(H)):
        while k + 1 < len(ends) and ends[k + 1] <= j: k += 1
        out[j] = r[k] if ends[k] <= j else 0
    return out, len(B)
if __name__ == '__main__':
    Xo = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet'); tks = sorted(Xo.tk.unique()); rows = []; nb = {}
    for tk in tks:
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: continue
        H = load(f[0]); reg0, s10 = states(H, vec)
        rows += [(tk, 'L', False, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run2(H, reg0, s10, 'L')]
        for K in (48, 24, 96):
            rv, n = vreg(H, K); nb.setdefault(K, []).append(n / len(H))
            rows += [(tk, f'VB{K}', False, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run2(H, rv, s10, 'L')]
        for a, b, r in run(H, reg0, s10, True)[0]: rows.append((tk, 'H1', True, pd.Timestamp(a), pd.Timestamp(b), r))
        print(tk, flush=True)
    print('volume bars per 15m bar:', {K: round(float(np.mean(v)), 2) for K, v in nb.items()}, flush=True)
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet('/tmp/claude-0/ml/vbar_V.parquet')
    # ---- evaluate with the yardstick (stop_1h_ev's machinery on this file)
    e = open('xs/stop_1h_ev.py').read().replace("/tmp/claude-0/ml/stop1h_V.parquet", "/tmp/claude-0/ml/vbar_V.parquet")
    exec(e[:e.index("for nm in ('A3'")])
    for nm in ('VB48', 'VB24', 'VB96'):
        Xv = build(nm); h = Xv[~Xv.h1]
        print(f'  {nm}: trades {len(Xv)}, 15m-track avg {h.ret.mean():+.3%} (live {XL[~XL.h1].ret.mean():+.3%}), win {h.ret.gt(0).mean():.0%} (live {XL[~XL.h1].ret.gt(0).mean():.0%}), runners>=15% {(h.ret >= .15).sum()} (live {(XL[~XL.h1].ret >= .15).sum()})', flush=True)
        line(nm, ev(Xv), B0)
