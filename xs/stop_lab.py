"""Wide stops (Oct 26 2026): stop distance at entry = entry price vs the vector stop of the track that bought (15m: 15m stop; 1h: 1h stop).
Variants on the live engine + book: hard cap on the loss (exit on a close X% under entry), skip wide-stop trades, size by stop distance
(equal risk per trade, clipped 0.5x-1.5x and normalized to an average of 1x).  python3 xs/stop_lab.py BARS_DIR OUT_DIR WIDE_T [TK..]"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'structure_lab.py')).read().split("\nW = pd.read_parquet(WT)")[0])
from vector_pine import run_state_machine
def stops(B):
    reg, lv, atr, _ = run_state_machine(B[['open', 'high', 'low', 'close']].assign(datetime=B.dt).reset_index(drop=True))
    return np.asarray(reg), np.asarray(lv, float) - BASE['BREAK_ATR'] * np.asarray(atr, float)
def run_cap(H, reg, s1, st15, st1, h1track, cap=None):
    o, c, hi, d, m = H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            sig = (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1)
            if sig: pos, e, px, b1, peak, h1 = True, j, o[j], False, o[j], False; sd = 1 - (st1[j - 1] if h1track else st15[j - 1]) / px
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or KS[p] == 1; now = c[j - 1] / px - 1 >= .02 and b1; h1 = h1 or now
        ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1)); pg = peak / px - 1
        if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex = True
        if cap is not None and c[j - 1] <= px * (1 - cap): ex = True
        if ex: T.append((d[e], d[j], o[j] / px - 1 - 2e-4, sd)); pos = False
    return T
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
for tk in (sys.argv[4:] or CORE):
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); reg, st15 = stops(H); hb = agg(H, '1h', '9h30min'); r1, s1h = stops(hb)
    s1 = known(H, hb, r1, '1h'); st1 = pd.merge_asof(pd.DataFrame({'t': H.dt + pd.Timedelta('15min')}), pd.DataFrame({'k': hb.dt + pd.Timedelta('1h'), 'v': s1h}),
                                                      left_on='t', right_on='k', direction='backward').v.values
    rows = []
    for nm, cap in (('cap_none', None), ('cap_3', .03), ('cap_4', .04), ('cap_6', .06)):
        for h1t in (False, True): rows.extend((tk, nm, h1t) + t for t in run_cap(H, reg, s1, st15, st1, h1t, cap))
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret', 'sd']).to_parquet(fo); print(tk, len(rows), flush=True)
