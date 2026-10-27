"""Oct 26 2026: model the trade after entry. Track the open trade's own swing structure on 15m bars (pivot lows, 2 bars each side,
confirmed 2 bars later) and change the exit by it - the information about winners arrives after the flip, so model it there.
 live           the live engine (15m + 1h tracks, sticky 1h stop after +2% with the 1h bull, LOCK5/0.5)
 promote        also switch a 15m trade to the 1h stop as soon as it prints 2 higher pivot lows after entry (trend structure confirmed)
 cut            exit a trade below +2% when a pivot low prints under the previous pivot low after entry (structure failed)
 promote_cut    both
 trail_hl       once 2 higher lows print, the stop becomes a 15m close under the latest higher low (replaces the vector exit)
python3 xs/trade_structure.py BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A_ = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
BARS, OUT, WT = A_; os.makedirs(OUT, exist_ok=True); LP = 2
def run_s(H, reg, s1, h1track, rule):
    o, c, hi, lo, d, m = H.open.values, H.close.values, H.high.values, H.low.values, H.dt.values, H.m.values
    pv = np.full(len(H), np.nan)                                   # pivot low value confirmed at index i (the low at i-LP)
    for i in range(2 * LP, len(H)):
        x = lo[i - LP]
        if x < lo[i - 2 * LP:i - LP].min() and x <= lo[i - LP + 1:i + 1].min(): pv[i] = x
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            if (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1):
                pos, e, px, b1, peak, h1, PL, hl = True, j, o[j], False, o[j], False, [], 0
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        for q in range(g[p - 2] + 1 if p >= 2 else k, k + 1):           # pivots confirmed since the last check
            if q > e + LP and pv[q] == pv[q]:
                if PL and pv[q] > PL[-1]: hl += 1
                elif PL and pv[q] < PL[-1]: hl = -1 if hl <= 0 else 0
                PL.append(pv[q])
        b1 = b1 or KS[p] == 1; gain = c[j - 1] / px - 1; now = gain >= .02 and b1
        if rule in ('promote', 'promote_cut') and hl >= 2 and b1: now = True
        h1 = h1 or now
        ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1)); pg = peak / px - 1
        if rule == 'trail_hl' and hl >= 2: ex = c[j - 1] < PL[-1]
        if pg >= .05 and gain <= .5 * pg: ex = True
        if rule in ('cut', 'promote_cut') and hl < 0 and gain < .02: ex = True
        if ex: T.append((d[e], d[j], o[j] / px - 1 - C2)); pos = False
    return T
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
SH, NS = int(os.getenv('SHARD', 0)), int(os.getenv('NSHARD', 1))
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); reg, s1 = states(H, vec); rows = []
    for rule in ('live', 'promote', 'cut', 'promote_cut', 'trail_hl'):
        for h1t in (False, True): rows += [(tk, rule, h1t) + t for t in run_s(H, reg, s1, h1t, rule)]
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
