"""Timeframe pairs for the two tracks (fast entries + stop until +2%, slow entries + sticky stop): 45m+1h, 30m+1h, 30m+45m
(15m+30m / 15m+45m are tf30 / tf45 in xs/vbar_lab.py; 15m+1h is live). python3 xs/tf_pairs.py BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'structure_lab.py')).read().split("W = pd.read_parquet(WT)")[0])
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
for tk in CORE:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); rows = []; S = {}
    for nm, fq in (('30', '30min'), ('45', '45min'), ('60', '1h')):
        r = agg(H, fq, '9h30min'); S[nm] = known(H, r, vec(r), fq)
    for fast, slow in (('45', '60'), ('30', '60'), ('30', '45')):
        for h1t in (False, True): rows.extend((tk, f'tf{fast}_{slow}', h1t, 'trade') + t for t in run(H, S[fast], S[slow], h1t)[0])
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'kind', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
