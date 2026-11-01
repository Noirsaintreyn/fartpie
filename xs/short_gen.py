"""Nov 1 2026: the short side - the live engine mirrored on BEAR flips (15m + 1h tracks, sticky 1h stop after +2% in favor, LOCK5/0.5).
Run on the inverted price series (1/price, high<->low) with the vector states negated; trade return converted back to a short's return.
python3 xs/short_gen.py BARS_DIR OUT_DIR WIDE_T (SHARD/NSHARD)"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
BARS, OUT, WT = A; os.makedirs(OUT, exist_ok=True)
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
SH, NS = int(os.getenv('SHARD', 0)), int(os.getenv('NSHARD', 1))
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); reg, s1 = states(H, vec)
    I = H.copy(); I['open'], I['close'], I['high'], I['low'] = 1 / H.open, 1 / H.close, 1 / H.low, 1 / H.high
    rows = []
    for h1t in (False, True):
        for e, x, r in run(I, -np.asarray(reg), -np.asarray(s1), h1t)[0]:
            rows.append((tk, 'short', h1t, e, x, 1 - 1 / (1 + r + C2) - C2))
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
