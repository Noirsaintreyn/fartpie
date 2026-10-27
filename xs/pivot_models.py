"""Oct 26 2026: advanced models built from the vector's parts (fractal pivots, ATR, clustering).
Stage T (trades through the live engine, per name):
 dow5 / dow3 / dow5_atr   Dow structure detector: bull on higher pivot high + higher pivot low, bear on lower both (ATR-filtered swings)
 pivclust                 pivot-cluster levels: the last 30 pivots grouped into zones (within 0.5 ATR, >= 2 touches); bull when a close
                          clears the nearest resistance zone by 0.25 ATR, bear when a close loses the nearest support zone by 0.25 ATR
 fib_pullback             inside a 1h vector uptrend: green 15m bar that dipped into the 38.2-78.6% retracement of the impulse leg
                          (1h pivot low before the flip -> high since) and closed back above 61.8%; stop = close under 78.6%
 fib_tp_half              live trades, half sold at the 1.618 extension of (pivot low before entry -> entry-day high)
Stage F (features of every live trade, known before entry):
 swing language   1h zig-zag (1.5 ATR reversal): last 3 completed swings (size in ATR, bars, retracement of the prior swing) + current swing
 duration         completed up-swings since the 1h vector flipped bull (semi-Markov trend age in swings)
 nesting          Dow structure state at pivot scales 2 / 5 / 12 / 30 on 15m bars
 analog path      last 32 15m closes, ATR-normalized, for a nearest-neighbour motif / analog model
python3 xs/pivot_models.py T|F BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A_ = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
MODE, BARS, OUT, WT = A_; os.makedirs(OUT, exist_ok=True); LPv = BASE['PIVOT_LEN']

def dow(df, L=5, katr=0.0):
    h, l, c = df.high.values, df.low.values, df.close.values; n = len(c); atr = compute_atr_series(h, l, c, 14)
    ph, pl = _piv(h, L, True), _piv(l, L, False); H_, L_ = [], []; st = np.zeros(n, int); s = 0
    for i in range(n):
        if ph[i] == ph[i] and (not katr or not L_ or ph[i] - L_[-1] >= katr * atr[i]): H_.append(ph[i])
        if pl[i] == pl[i] and (not katr or not H_ or H_[-1] - pl[i] >= katr * atr[i]): L_.append(pl[i])
        if len(H_) >= 2 and len(L_) >= 2:
            if H_[-1] > H_[-2] and L_[-1] > L_[-2]: s = 1
            elif H_[-1] < H_[-2] and L_[-1] < L_[-2]: s = -1
        st[i] = s
    return st

def pivclust(df, L=5, tol=.5, brk=.25, keep=30):
    h, l, c = df.high.values, df.low.values, df.close.values; n = len(c); atr = compute_atr_series(h, l, c, 14)
    ph, pl = _piv(h, L, True), _piv(l, L, False); pv = []; Z = []; st = np.zeros(n, int); s = 0
    for i in range(1, n):
        a = atr[i]
        if a != a: continue
        new = False
        if ph[i] == ph[i]: pv.append(ph[i]); new = True
        if pl[i] == pl[i]: pv.append(pl[i]); new = True
        if new:
            pv = pv[-keep:]; xs = sorted(pv); Z = []; grp = [xs[0]]
            for x in xs[1:]:
                if x - grp[-1] <= tol * a: grp.append(x)
                else:
                    if len(grp) >= 2: Z.append((grp[0], grp[-1]))
                    grp = [x]
            if len(grp) >= 2: Z.append((grp[0], grp[-1]))
        ref = c[i - 1]; up = [z for z in Z if z[0] > ref]; dn = [z for z in Z if z[1] < ref]
        R = min(up, key=lambda z: z[0]) if up else None; S = max(dn, key=lambda z: z[1]) if dn else None
        if s != 1 and R is not None and c[i] > R[1] + brk * a: s = 1
        elif s != -1 and S is not None and c[i] < S[0] - brk * a: s = -1
        st[i] = s
    return st

def zigzag(r, k=1.5):
    """1h zig-zag turning points: list of (confirm_index, turn_index, price, +1 top / -1 bottom)"""
    h, l, c = r.high.values, r.low.values, r.close.values; atr = compute_atr_series(h, l, c, 14); out = []
    d = 0; ext = c[0]; ei = 0
    for i in range(1, len(c)):
        a = atr[i]
        if a != a: continue
        if d >= 0:
            if h[i] > ext or d == 0 and h[i] >= ext: ext, ei = h[i], i
            if ext - l[i] >= k * a and (d == 1 or d == 0): out.append((i, ei, ext, 1)); d = -1; ext, ei = l[i], i; continue
            if d == 0 and l[i] < c[0] - k * a: d = -1; ext, ei = l[i], i
        if d == -1:
            if l[i] < ext: ext, ei = l[i], i
            if h[i] - ext >= k * a: out.append((i, ei, ext, -1)); d = 1; ext, ei = h[i], i
    return out, atr

W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
SH, NS = int(os.getenv('SHARD', 0)), int(os.getenv('NSHARD', 1))
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); reg, s1 = states(H, vec); o, c, hi, lo, d, m = H.open.values, H.close.values, H.high.values, H.low.values, H.dt.values, H.m.values; n = len(H)
    r = agg(H, '1h', '9h30min'); pl1 = _piv(r.low.values, LPv, False); plk = known(H, r, np.nan_to_num(pl1, nan=0.0), '1h')
    last_pl = pd.Series(np.where(plk == 0, np.nan, plk)).ffill().values; f1 = np.r_[False, (s1[1:] == 1) & (s1[:-1] != 1)]
    LIVE = {h1t: run(H, reg, s1, h1t)[0] for h1t in (False, True)}
    if MODE == 'T':
        rows = []
        for nm, f in (('dow5', lambda q: dow(q, 5)), ('dow3', lambda q: dow(q, 3)), ('dow5_atr', lambda q: dow(q, 5, 1.0)), ('pivclust', pivclust)):
            rg, sx = states(H, f)
            for h1t in (False, True): rows += [(tk, nm, h1t) + t for t in run(H, rg, sx, h1t)[0]]
        for h1t in (False, True): rows += [(tk, 'live', h1t) + t for t in LIVE[h1t]]
        pos = False; anchor = np.nan; top = -np.inf
        for j in range(1, n):
            k = j - 1
            if f1[k]: anchor, top = last_pl[k], hi[k]
            if s1[k] == 1: top = max(top, hi[k])
            else: anchor = np.nan
            if not 570 <= m[j] < 960: continue
            if not pos:
                if s1[k] == 1 and anchor == anchor and top > anchor and 600 <= m[j] < 840 and m[k] >= 570:
                    leg = top - anchor; f38, f62, f79 = top - .382 * leg, top - .618 * leg, top - .786 * leg
                    if f79 <= lo[k] <= f38 and c[k] > f62 and c[k] > o[k]: pos, e, px, peak, stop = True, j, o[j], o[j], f79
                continue
            if j == e: continue
            peak = max(peak, hi[k]) if k >= e else peak; pg = peak / px - 1
            if c[k] < stop or s1[k] == -1 or (pg >= .05 and c[k] / px - 1 <= .5 * pg): rows.append((tk, 'fib_pullback', True, d[e], d[j], o[j] / px - 1 - C2)); pos = False
        for h1t in (False, True):
            for (e0, x0, r0) in LIVE[h1t]:
                ie, ix = np.searchsorted(d, e0), np.searchsorted(d, x0); px = o[ie]; a = last_pl[ie - 1]
                day0 = (d >= np.datetime64(pd.Timestamp(e0).normalize())) & (np.arange(n) < ie); sh = hi[day0].max() if day0.any() else px
                tgt = a + 1.618 * (sh - a) if (a == a and sh > a) else np.nan
                if not tgt == tgt or tgt <= px: rows.append((tk, 'fib_tp_half', h1t, e0, x0, r0)); continue
                r1 = (tgt / px - 1 - C2) if (hi[ie:ix] >= tgt).any() else r0; rows.append((tk, 'fib_tp_half', h1t, e0, x0, .5 * r0 + .5 * r1))
        pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(fo)
    else:
        zz, atr1 = zigzag(r); conf_t = np.array([r.dt.values[z[0]] + np.timedelta64(60, 'm') for z in zz]) if zz else np.array([], dtype='datetime64[ns]')
        DS = {L: dow(H, L) for L in (2, 5, 12, 30)}; atr15 = compute_atr_series(hi, lo, c, 14)
        f1t = d[f1]; rows = []
        for h1t in (False, True):
            for (e0, x0, r0) in LIVE[h1t]:
                ie = np.searchsorted(d, e0); k = ie - 1
                if k < 40: continue
                kz = np.searchsorted(conf_t, e0, side='right'); Z = zz[max(0, kz - 4):kz]
                sw = [np.nan] * 10
                if len(Z) >= 4:
                    pts = [(z[2], z[1]) for z in Z]; aa = atr1[Z[-1][0]]
                    legs = [(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(3)]
                    sw = [legs[i][0] / aa for i in range(3)] + [legs[i][1] for i in range(3)] + [abs(legs[1][0] / legs[0][0]), abs(legs[2][0] / legs[1][0])]
                    sw += [(c[k] - pts[-1][0]) / aa, Z[-1][3]]
                lf = f1t[f1t < e0]; nsw = np.nan
                if len(lf) and s1[k] == 1: nsw = sum(1 for z in zz[:kz] if z[3] == 1 and r.dt.values[z[0]] >= lf[-1])
                path = (np.log(c[k - 31:k + 1] / c[k])) / (atr15[k] / c[k])
                rows.append((tk, h1t, e0, x0, r0, *sw, nsw, *(DS[L][k] for L in (2, 5, 12, 30)), *path))
        cols = ['tk', 'h1', 'entry', 'exit', 'ret'] + [f'sw{i}' for i in range(10)] + ['nsw', 'n2', 'n5', 'n12', 'n30'] + [f'p{i}' for i in range(32)]
        pd.DataFrame(rows, columns=cols).to_parquet(fo)
    print(tk, flush=True)
