"""Nov 3 2026: should the vector be allowed to SELL overnight (premarket and after-hours bars) for every trade, instead of waiting for the next regular-session bar?
Data: 15m bars 04:00-20:00. Entries stay on regular-session bars (09:30-16:00) exactly as live. The vector state is computed on the extended bars.
 C   control: extended-bar vector, exits only on regular-session bars (what live does, but the state includes after-hours bars)
 PM  exits also allowed on premarket bars (04:00-09:30)      ON  exits allowed on every extended bar (premarket + after-hours)
Exit fill = open of the bar after the trigger; extended-hours fills pay 15 bp extra. Sticky 1h stop after +2% (1h state), +5% lock, as live.
Scored vs the CURRENT model (original data) and vs C."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
def load2(f):
    H = pd.read_parquet(f); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= 240) & (m < 1200) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True); H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; return H
def run5(H, reg, s1, mode):
    o, c, hi, d, m = H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values; n = len(H); out = []; pos = False; prev_rth_reg = 0; ext_n = 0
    for j in range(2, n):
        rth = 570 <= m[j] < 960
        allow = rth or (mode in ('PM', 'ON') and m[j] < 570) or (mode == 'ON' and m[j] >= 960)
        if not pos:
            if rth:
                if reg[j - 1] == 1 and prev_rth_reg != 1: pos, e, px, b1, peak, h1 = True, j, o[j], False, o[j], False
                prev_rth_reg = reg[j - 1]
            continue
        if rth: prev_rth_reg = reg[j - 1]
        if j == e or not allow: continue
        k = j - 1
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or s1[j - 1] == 1; cur = c[j - 1] / px - 1; h1 = h1 or (cur >= .02 and b1)
        ex = (s1[j - 1] == -1) if h1 else (reg[j - 1] == -1)
        pg = peak / px - 1
        if pg >= .05 and cur <= .5 * pg: ex = True
        if ex:
            cost = C2 + (0 if rth else .0015); out.append((d[e], d[j], o[j] / px - 1 - cost, int(not rth))); pos = False
    return out
if __name__ == '__main__':
    Xo = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet'); rows = []
    for tk in sorted(Xo.tk.unique()):
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: continue
        H = load2(f[0]); reg0, s10 = states(H, vec)
        for nm in ('C', 'PM', 'ON'): rows += [(tk, nm, False, pd.Timestamp(a), pd.Timestamp(b), r, x_) for a, b, r, x_ in run5(H, reg0, s10, nm)]
        rows += [(tk, 'H1', True, pd.Timestamp(a), pd.Timestamp(b), r, 0) for a, b, r in run(H, reg0, s10, True)[0]]
        print(tk, flush=True)
    V = pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret', 'extexit']); V0 = pd.read_parquet('/tmp/claude-0/ml/stop1h_V.parquet'); V0 = V0[V0['var'] == 'L']; V0['extexit'] = 0
    V0 = V0[['tk', 'var', 'h1', 'entry', 'exit', 'ret', 'extexit']]; pd.concat([V0, V]).to_parquet('/tmp/claude-0/ml/hours_V.parquet')
