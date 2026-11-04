"""Nov 4 2026: ATR smoothing inside the vector. DECLARED BEFORE RESULTS. Live = Wilder RMA(14) (an EWMA with alpha 1/14). Variants, regenerated through the live engine (signal_lab states/run, 15m + 1h tracks):
 sma    simple moving average of true range, 14 | ewma  EWMA of true range, span 14 (alpha 2/15, faster than Wilder) | sma7 SMA 7 (fast)
Usage: python3 xs/atr_smooth.py SHARD NSHARD ; then xs/atr_smooth_ev.py (book test vs live)."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SH, NS = int(sys.argv[1]), int(sys.argv[2]); sys.argv = sys.argv[:1] + ['x']; HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
_orig = compute_atr_series
def _tr(h, l, c):
    pc = np.r_[c[0], c[:-1]]; t = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); t[0] = h[0] - l[0]; return t
def f_sma(n): return lambda h, l, c, L: pd.Series(_tr(h, l, c)).rolling(n, min_periods=n).mean().values
def f_ewma(n): return lambda h, l, c, L: pd.Series(_tr(h, l, c)).ewm(span=n, adjust=False, min_periods=n).mean().values
VARS = {'sma': f_sma(14), 'ewma': f_ewma(14), 'sma7': f_sma(7)}
OUT = '/tmp/claude-0/ml/atrsm'; os.makedirs(OUT, exist_ok=True)
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/*_15m_*.parquet')}
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); rows = []
    for nm, fn in VARS.items():
        globals()['compute_atr_series'] = fn
        reg, s1 = states(H, lambda d: vec(d))
        for h1t in (False, True): rows += [(tk, nm, h1t) + t for t in run(H, reg, s1, h1t)[0]]
    globals()['compute_atr_series'] = _orig
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
