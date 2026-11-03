"""Nov 3 2026: cleaner overnight test. The vector keeps its normal bars (premarket + regular session). AFTER-HOURS (16:00-20:00) can only SELL: if a 15m after-hours bar closes below the vector's
stop test (last regular bar's level - BREAK_ATR x ATR) the position is sold at the next after-hours bar open (+15 bp extended cost). 15m track before the +2% sticky-1h stop only; everything else as live.
 AH  that rule      Variants of the trigger depth: AH (as above), AH2 (needs a close 2 ATR below the level: only a real gap/crash), same fill rule."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE)
src = open(HERE + '/vec_state.py').read().split("if __name__ == '__main__':")[0]; exec(src)
ex = open(HERE + '/hours_exit.py').read(); exec("def load2" + ex.split("def load2")[1].split("def run5")[0])
BA_ = BASE['BREAK_ATR']
def run6(H, S, reg, s1, AH, extra):
    o, c, hi, d, m = H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; out = []; pos = False
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]; lvl, atr = S.lvl.values, S.atr.values
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            if KR[p] == 1 and KR[p - 1] != 1: pos, e, px, b1, peak, h1 = True, j, o[j], False, o[j], False
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        # after-hours check between the previous regular bar and this one (only when the previous bar was the day's last regular bar)
        if m[k] == 945 and k >= e and not h1 and KR[p - 1] == 1 and not np.isnan(lvl[k]):
            day = pd.Timestamp(d[k]).normalize(); A = AH.get(day)
            if A is not None:
                thr = lvl[k] - (BA_ + extra) * atr[k]; hit = np.where(A[1] < thr)[0]
                if len(hit) and hit[0] + 1 < len(A[0]):
                    q = hit[0] + 1; out.append((d[e], A[0][q], A[2][q] / px - 1 - C2 - .0015)); pos = False; continue
        b1 = b1 or KS[p] == 1; cur = c[j - 1] / px - 1; h1 = h1 or (cur >= .02 and b1)
        ex = (KS[p] == -1) if h1 else (KR[p] == -1)
        pg = peak / px - 1
        if pg >= .05 and cur <= .5 * pg: ex = True
        if ex: out.append((d[e], d[j], o[j] / px - 1 - C2)); pos = False
    return out
if __name__ == '__main__':
    Xo = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet'); rows = []
    for tk in sorted(Xo.tk.unique()):
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
        if not f: continue
        H = load(f[0]); S = vec_i(H[['dt', 'open', 'high', 'low', 'close']]); reg0, s10 = states(H, vec); E2 = load2(f[0]); E2 = E2[E2.m >= 960]
        AH = {dd: (g.dt.values, g.close.values, g.open.values) for dd, g in E2.groupby(E2.dt.dt.normalize())}
        for nm, ext in (('AH', 0.0), ('AH2', 2.0)): rows += [(tk, nm, False, pd.Timestamp(a), pd.Timestamp(b), r) for a, b, r in run6(H, S, reg0, s10, AH, ext)]
        print(tk, flush=True)
    V = pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']); V0 = pd.read_parquet('/tmp/claude-0/ml/stop1h_V.parquet'); V0 = V0[V0['var'].isin(['L', 'H1'])]
    pd.concat([V0, V]).to_parquet('/tmp/claude-0/ml/level_V.parquet')
