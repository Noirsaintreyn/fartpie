"""Volume-bar and slow-track follow-ups (Oct 25 2026)
 vb40 / vb60       the vector on volume bars, 40 or 60 a day (the slow track uses 4x bigger bars)
 mix_r15_v1h       regular 15m vector with a VOLUME-bar slow track (6.5 bars a day) for the 1h entries and the sticky stop
 mix_v15_r1h       volume-bar 15m track (26 a day) with the regular 1h vector
 tf30 / tf45       the slow track on 30-minute or 45-minute bars instead of 1 hour (its entries and the sticky stop)
python3 xs/vbar_lab.py BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'structure_lab.py')).read().split("W = pd.read_parquet(WT)")[0])
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
for tk in CORE:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); rows = []
    def add(var, T, h1): rows.extend((tk, var, h1, 'trade') + t for t in T)
    reg, s1 = states(H, vec); V = {}
    for k in (26, 40, 60):
        b15, b1 = vol_bars(H, k), vol_bars(H, k / 4); V[k] = (on_15m(H, b15, vec(b15)), on_15m(H, b1, vec(b1)))
    S = {}
    for nm, fq in (('tf30', '30min'), ('tf45', '45min')):
        r = agg(H, fq, '9h30min'); S[nm] = known(H, r, vec(r), fq)
    for h1t in (False, True):
        for k in (40, 60): add(f'vb{k}', run(H, V[k][0], V[k][1], h1t)[0], h1t)
        add('mix_r15_v1h', run(H, reg, V[26][1], h1t)[0], h1t)
        add('mix_v15_r1h', run(H, V[26][0], s1, h1t)[0], h1t)
        for nm in S: add(nm, run(H, reg, S[nm], h1t)[0], h1t)
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'kind', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
