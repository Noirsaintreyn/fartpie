"""Nov 4 2026: volatility-estimator swaps inside the vector. DECLARED BEFORE RESULTS. The vector's ATR (price units) is replaced by an ATR-equivalent from another estimator, rescaled by ONE constant per ticker so its median equals Wilder ATR14's median
(so BREAK_ATR 0.35 etc. keep their meaning; only the shape/speed of the estimate changes). Regenerated through the live engine, 15m + 1h tracks.
 natr   Wilder RMA(14) of TR/prev-close (percent range), times the current close (ATR as a percent of price, normalised)
 park   Parkinson sigma over 14 bars: sqrt(mean(ln(H/L)^2)/(4 ln2)) x close
 gk     Garman-Klass over 14 bars (open approximated by the previous close): sqrt(mean(.5 ln(H/L)^2 - (2ln2-1) ln(C/O)^2)) x close
 gjr    GJR-GARCH(1,1) on 15m close-to-close returns, FIXED params alpha .05 gamma .08 beta .90 omega set to the sample variance (no fit, no look-ahead): sigma_t from info through bar t-1 ... x close
Usage: python3 xs/vol_scale.py SHARD NSHARD ; then xs/vol_scale_ev.py"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SH, NS = int(sys.argv[1]), int(sys.argv[2]); sys.argv = sys.argv[:1] + ['x']; HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
_orig = compute_atr_series
def _wild(x, n):
    out = np.full(len(x), np.nan); 
    if len(x) < n: return out
    out[n - 1] = np.nanmean(x[:n]); a = 1.0 / n
    for i in range(n, len(x)): out[i] = out[i - 1] * (1 - a) + x[i] * a
    return out
def _tr(h, l, c):
    pc = np.r_[c[0], c[:-1]]; t = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); t[0] = h[0] - l[0]; return t
def v_natr(h, l, c, L): pc = np.r_[c[0], c[:-1]]; return _wild(_tr(h, l, c) / pc, L) * c
def v_park(h, l, c, L): return np.sqrt(pd.Series(np.log(h / l) ** 2).rolling(L, min_periods=L).mean().values / (4 * np.log(2))) * c
def v_gk(h, l, c, L):
    o = np.r_[c[0], c[:-1]]; x = .5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2; return np.sqrt(np.clip(pd.Series(x).rolling(L, min_periods=L).mean().values, 1e-12, None)) * c
def v_gjr(h, l, c, L):
    r = np.r_[0, np.diff(np.log(c))]; a, g, b = .05, .08, .90; var0 = np.nanvar(r); om = var0 * (1 - a - g / 2 - b); s2 = np.empty(len(r)); s2[0] = var0
    for i in range(1, len(r)): s2[i] = om + (a + g * (r[i - 1] < 0)) * r[i - 1] ** 2 + b * s2[i - 1]
    out = np.sqrt(s2) * c; out[:L] = np.nan; return out
VARS = {'natr': v_natr, 'park': v_park, 'gk': v_gk, 'gjr': v_gjr}
def mk(fn):
    def f(h, l, c, L):
        base = _orig(h, l, c, L); v = fn(h, l, c, L); ok = ~np.isnan(base) & ~np.isnan(v) & (v > 0); k = np.median(base[ok]) / np.median(v[ok]); return v * k
    return f
OUT = '/tmp/claude-0/ml/volsc'; os.makedirs(OUT, exist_ok=True)
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/*_15m_*.parquet')}
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); rows = []
    for nm, fn in VARS.items():
        globals()['compute_atr_series'] = mk(fn)
        reg, s1 = states(H, lambda d: vec(d))
        for h1t in (False, True): rows += [(tk, nm, h1t) + t for t in run(H, reg, s1, h1t)[0]]
    globals()['compute_atr_series'] = _orig
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
