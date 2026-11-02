"""Nov 2 2026: 'buy everything, then cull' exits. Live engine entries/exits, plus a 10:00 cull check (exit at the 10:00 bar open):
 c1_1   first morning after entry: exit unless the trade is up >= +1%
 c1_2   first morning after entry: exit unless up >= +2%
 c2_2   second morning after entry: exit unless up >= +2%
 cbig   every morning after entry: exit unless TODAY is a big day so far (prior close -> 10:00 >= +1 daily sigma) or the trade is up >= +3%
 cbig1  first morning only: exit unless today big so far (>= +1 sigma) or up >= +3%
python3 xs/cull_gen.py BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
BARS, OUT, WT = A; os.makedirs(OUT, exist_ok=True)
MODES = {'c1_1': (1, .01, False, True), 'c1_2': (1, .02, False, True), 'c2_2': (2, .02, False, True), 'cbig': (1, .03, True, False), 'cbig1': (1, .03, True, True)}
def run_c(H, reg, s1, h1track, k_, x_, big, once, pc, sg):
    o, c, hi, d, m = H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values; dd = H.day.values
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            sig = (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1)
            if sig: pos, e, px, b1, peak, h1, nd, lastd = True, j, o[j], False, o[j], False, 0, dd[j]
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        if dd[j] != lastd: nd += 1; lastd = dd[j]
        b1 = b1 or KS[p] == 1; now = c[j - 1] / px - 1 >= .02 and b1; h1 = h1 or now
        ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1)); pg = peak / px - 1
        if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex = True
        if m[j] == 600 and (nd == k_ if once else nd >= k_):
            cum = c[j - 1] / px - 1; keep = cum >= x_
            if big and pc[j] > 0 and sg[j] > 0: keep = keep or (c[j - 1] / pc[j] - 1 >= sg[j])
            if not keep: ex = True
        if ex: T.append((d[e], d[j], o[j] / px - 1 - C2)); pos = False
    return T
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.0)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
SH, NS = int(os.getenv('SHARD', 0)), int(os.getenv('NSHARD', 1))
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); reg, s1 = states(H, vec); H['day'] = H.dt.dt.normalize()
    R = H[(H.m >= 570) & (H.m < 960)]; D = R.groupby('day').close.last(); dr = D.pct_change(); sd = dr.rolling(20).std()
    pcs = D.shift(1); H['pc'] = H.day.map(pcs).fillna(0).values; H['sg'] = H.day.map(sd.shift(1)).fillna(0).values
    rows = [(tk, 'live', h1t) + t for h1t in (False, True) for t in run(H, reg, s1, h1t)[0]]
    for nm, (k_, x_, big, once) in MODES.items():
        for h1t in (False, True): rows += [(tk, nm, h1t) + t for t in run_c(H, reg, s1, h1t, k_, x_, big, once, H.pc.values, H.sg.values)]
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
