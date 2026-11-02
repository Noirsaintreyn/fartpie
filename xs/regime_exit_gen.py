"""Nov 2 2026 add: eod = stop / lock exits checked only on the 15:45 bar (exit at its open, ~the close): hold through intraday noise, keep the overnight.
Nov 1 2026: exit variants of the live engine (same entries), for regime-dependent exits.
 loose  no +5% lock, exit only on the 1h vector stop (both tracks)      tight  lock at +3% (give back half), 15m stop only (no sticky 1h switch)
 leash  live rules + exit if 2 sessions (52 RTH bars) after entry the trade is up < +2%
python3 xs/regime_exit_gen.py BARS_DIR OUT_DIR WIDE_T (SHARD/NSHARD)"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
BARS, OUT, WT = A; os.makedirs(OUT, exist_ok=True)
def run_v(H, reg, s1, h1track, mode):
    o, c, hi, d, m = H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            sig = (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1)
            if sig: pos, e, px, b1, peak, h1, pe = True, j, o[j], False, o[j], False, p
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or KS[p] == 1; now = c[j - 1] / px - 1 >= .02 and b1
        if mode != 'tight': h1 = h1 or now
        if mode == 'loose': h1 = True
        ex = KS[p] == -1 if (h1track or h1) else (KR[p] == -1); pg = peak / px - 1
        if mode != 'loose':
            lk = .03 if mode == 'tight' else .05
            if pg >= lk and (c[j - 1] / px - 1) <= .5 * pg: ex = True
        if mode == 'eod' and m[j] != 945: ex = False
        if mode == 'leash' and p - pe >= 52 and c[j - 1] / px - 1 < .02: ex = True
        if ex: T.append((d[e], d[j], o[j] / px - 1 - C2)); pos = False
    return T
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.0)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
SH, NS = int(os.getenv('SHARD', 0)), int(os.getenv('NSHARD', 1))
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); reg, s1 = states(H, vec); rows = []
    for mode in (os.getenv('MODES', 'live,loose,tight,leash').split(',')):
        for h1t in (False, True): rows += [(tk, mode, h1t) + t for t in (run(H, reg, s1, h1t)[0] if mode == 'live' else run_v(H, reg, s1, h1t, mode))]
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
