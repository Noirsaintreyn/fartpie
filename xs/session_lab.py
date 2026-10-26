"""Which bars build the vector (Oct 26 2026): RTH only (09:30-16:00) vs premarket+RTH (04:00-16:00, live) vs extended (04:00-20:00,
after-hours included). Entries/exits still fill only at RTH opens; live engine and book otherwise.
python3 xs/session_lab.py BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'structure_lab.py')).read().split("\nW = pd.read_parquet(WT)")[0])
def load_w(f, m0, m1):
    H = pd.read_parquet(f); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= m0) & (m < m1) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; return H
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
for tk in (sys.argv[4:] or CORE):
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    rows = []
    for nm, m0, m1 in (('sess_rth', 570, 960), ('sess_pre', 240, 960), ('sess_ext', 240, 1200)):
        H = load_w(fs[tk], m0, m1); reg, s1 = states(H, vec)
        for h1t in (False, True): rows.extend((tk, nm, h1t, 'trade') + t for t in run(H, reg, s1, h1t)[0])
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'kind', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
