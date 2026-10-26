"""5-minute entries and Hawkes exits (Oct 26 2026), live engine otherwise. Only the 25 core names with 5m history (LSE cache).
 e5_1h      buy early when the 5m vector flips bull while the 1h vector is bull (15m still bear): 15m state = 15m bull OR (5m AND 1h bull)
 e5_any     same without the 1h condition: 15m bull OR 5m bull
 a5_strict  15m flips only count once the 5m vector agrees (entry waits for 5m bull; exits as live)
 hk_live    live + Hawkes early exit (net sell-burst intensity >= 1.5, tau 8 bars) while the trade is on the 15m stop
 hk_e5_1h   e5_1h + the Hawkes exit          cp_hk   change-point early entry (0.5 / 0.8) + the Hawkes exit
 vec        live reproduction on the same names
python3 xs/five_min_lab.py BARS15_DIR BARS5_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
B5 = sys.argv[2]; sys.argv = [sys.argv[0], sys.argv[1], sys.argv[3], sys.argv[4]]
src = open(os.path.join(HERE, 'cycle_cp_lab.py')).read(); exec(src.split("\nW = pd.read_parquet(WT)")[0])
def hawkes_net(H, tau=8):
    c, o, h, l, v = H.close.values, H.open.values, H.high.values, H.low.values, H.volume.values.astype(float)
    pc = np.r_[np.nan, c[:-1]]; tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))); atr = pd.Series(tr).rolling(14).mean().shift(1).values
    rv = (pd.Series(v) / pd.Series(v).groupby(H.m.values).transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())).values
    d = c - pc; sell = (d < -.5 * atr) & (rv >= 1.5); buy = (d > .5 * atr) & (rv >= 1.5); k = np.exp(-1 / tau); S = B = 0.0; net = np.zeros(len(c))
    for i in range(len(c)): S = S * k + sell[i]; B = B * k + buy[i]; net[i] = S - B
    return net
def hawkes_exit(state, net, T=1.5):
    out = state.copy(); s = None
    for i in range(len(state)):
        if s == -1 and state[i] != 1: s = None
        if s is None and state[i] == 1 and net[i] >= T: s = -1
        if s is not None: out[i] = s
    return out
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
f5 = {os.path.basename(f).split('_5m_')[0]: f for f in glob.glob(f'{B5}/*_5m_2018-01-01_*.parquet')}
for tk in CORE:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs or tk not in f5: continue
    H = load(fs[tk]); F = load(f5[tk]); rows = []
    def add(var, a, b):
        for h1t in (False, True): rows.extend((tk, var, h1t, 'trade') + t for t in run(H, a, b, h1t)[0])
    reg, s1 = states(H, vec)
    v5 = known(H, F[['dt', 'open', 'high', 'low', 'close']], vec(F[['dt', 'open', 'high', 'low', 'close']]), '5min').astype(int)
    e1 = np.where((reg == 1) | ((v5 == 1) & (s1 == 1)), 1, -1); ea = np.where((reg == 1) | (v5 == 1), 1, -1)
    st_ = np.where((reg == 1) & (v5 == 1), 1, np.where(reg == -1, -1, 0))
    net = hawkes_net(H)
    r = np.log(H.close).diff(); x = (r / r.rolling(260, min_periods=60).std().shift(1)).values; cp, ms, mf = bocpd(x); ce = cp_states(reg, cp, ms, mf, 'entry', .5, .8)
    for nm, a in (('vec', reg), ('e5_1h', e1), ('e5_any', ea), ('a5_strict', st_), ('hk_live', hawkes_exit(reg, net)), ('hk_e5_1h', hawkes_exit(e1, net)), ('cp_hk', hawkes_exit(ce, net)), ('cp_e', ce)):
        add(nm, a, s1)
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'kind', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
