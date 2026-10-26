"""Oct 26 2026: pullback strategies built on the vector, alone and paired with the HAR expected range.
All long only, inside a 1h vector uptrend, on the 54 live names (ratio >= 1.5 S&P), entries 10:00-14:00 at the next 15m open.
 A  level retest   a 15m bar dips to the 1h vector level (low within 0.25 x 1h ATR of it) and closes back above it, green
 B  failed flip    the 15m flips back bull within 8 bars of a 15m bear flip, while the 1h stayed bull the whole time
 C  HAR pullback   price has pulled back >= 0.75 of today's expected range (HAR budget 1.7 sigma) from the day's high, green 15m bar
HAR pairing for A / B: also require the pullback from the day's high to be >= 0.5 / 0.75 expected ranges.
Exit for all: a 15m close under the 1h vector's break line (level - BREAK_ATR x 1h ATR, ratchets up with the level) or the 1h flips bear
-> next open; LOCK5/0.5 profit lock as live. One open trade per name per variant.
python3 xs/pullback_lab.py BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, re, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A_ = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
src = open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]; exec(src)
v = src[src.index('def vec('):src.index('def donch(')].replace('def vec(', 'def vec_lv(')
v = v.replace("regs = np.zeros(n, int)", "regs = np.zeros(n, int); LV = np.full(n, np.nan); AT = np.full(n, np.nan)")
v = v.replace("regs[i] = regime; continue", "regs[i] = regime; LV[i] = level; AT[i] = a; continue").replace("        regs[i] = regime\n    return regs", "        regs[i] = regime; LV[i] = level; AT[i] = a\n    return regs, LV, AT")
exec(v)
BARS, OUT, WT = A_; os.makedirs(OUT, exist_ok=True); BA = BASE['BREAK_ATR']
def har(x): return 1.7 * (np.exp(-0.6186 + 0.2939 * x + 0.3661 * x.rolling(5).mean() + 0.1783 * x.rolling(22).mean()) / 1.665).shift(1)

def sim_var(H, sig, s1, brk):
    o, c, hi, d, m = H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values; T = []; pos = False
    for j in range(1, len(H)):
        if not 570 <= m[j] < 960: continue
        if not pos:
            if sig[j - 1] and 600 <= m[j] < 840: pos, e, px, peak = True, j, o[j], o[j]
            continue
        if j == e: continue
        k = j - 1; peak = max(peak, hi[k]) if k >= e else peak; pg = peak / px - 1
        ex = (m[k] >= 570 and c[k] < brk[k]) or s1[k] != 1 or (pg >= .05 and c[k] / px - 1 <= .5 * pg)
        if ex: T.append((d[e], d[j], o[j] / px - 1 - C2)); pos = False
    return T

W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
SH, NS = int(os.getenv('SHARD', 0)), int(os.getenv('NSHARD', 1))
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); reg = vec(H)
    r = agg(H, '1h', '9h30min'); g1, L1, A1 = vec_lv(r)
    s1, lv1, at1 = (known(H, r, x, '1h') for x in (g1, L1, A1))
    o, c, h, l, m = H.open.values, H.close.values, H.high.values, H.low.values, H.m.values
    brk = lv1 - BA * at1
    R_ = H[m >= 570]; dd = R_.dt.dt.normalize(); D = R_.groupby(dd).agg(h=('high', 'max'), l=('low', 'min'))
    b = har(np.log(np.log(D.h / D.l).clip(lower=1e-5))).reindex(H.dt.dt.normalize()).values
    dhi = pd.Series(np.where(m >= 570, h, np.nan)).groupby(H.dt.dt.normalize().values).cummax().values
    depth = np.log(dhi / c) / b; rth = m >= 570; green = c > o
    sA = rth & (s1 == 1) & (l <= lv1 + .25 * at1) & (c > lv1) & green
    rb = pd.Series(reg); fb = (rb == 1) & (rb.shift(1) != 1); fdn = (rb == -1) & (rb.shift(1) != -1)
    lastdn = pd.Series(np.where(fdn, np.arange(len(H)), np.nan)).ffill().values
    s1min = pd.Series(s1).rolling(9, min_periods=1).min().values
    sB = rth & fb.values & (np.arange(len(H)) - lastdn <= 8) & (s1min == 1)
    sC = rth & (s1 == 1) & (depth >= .75) & green
    rows = []
    V = {'A_retest': sA, 'A_retest_har50': sA & (depth >= .5), 'A_retest_har75': sA & (depth >= .75),
         'B_failflip': sB, 'B_failflip_har50': sB & (depth >= .5), 'B_failflip_har75': sB & (depth >= .75), 'C_har75': sC, 'C_har100': sC & (depth >= 1.0)}
    for nm, sg in V.items(): rows += [(tk, nm, True) + t for t in sim_var(H, np.nan_to_num(sg).astype(bool), s1, brk)]
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
