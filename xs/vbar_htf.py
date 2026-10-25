"""Higher-timeframe volume bars as the slow track (Oct 25 2026). Bars per day: 6.5 (~1h), 3.25 (~2h), 1.6 (~4h), 1 (~daily).
 r15_vK   regular 15m vector + volume-bar slow track (K bars a day)
 vF_vK    volume-bar fast track (F a day) + volume-bar slow track (K a day)
python3 xs/vbar_htf.py BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'structure_lab.py')).read().split("W = pd.read_parquet(WT)")[0])
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
for tk in CORE:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); rows = []; reg, s1 = states(H, vec); VB = {}
    for k in (6.5, 3.25, 1.6, 1.0):
        b = vol_bars(H, k); VB[k] = on_15m(H, b, vec(b))
    pairs = [(f'r15_v{k}', reg, VB[k]) for k in (3.25, 1.6, 1.0)] + [('v6.5_v3.25', VB[6.5], VB[3.25]), ('v6.5_v1.6', VB[6.5], VB[1.6]), ('v3.25_v1.0', VB[3.25], VB[1.0])]
    for nm, f_, s_ in pairs:
        for h1t in (False, True): rows.extend((tk, nm, h1t, 'trade') + t for t in run(H, f_, s_, h1t)[0])
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'kind', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
