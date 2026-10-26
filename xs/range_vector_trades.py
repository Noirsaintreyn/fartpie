"""Oct 26 2026: use the range x vector context inside the live engine (15m + 1h tracks, sticky 1h stop, LOCK5/0.5).
Variants written as 'var':
 live       the live engine
 stop_lod   also exit if a 15m close goes under the day's low as of entry (the low the flip 'confirmed'; 75-90% of those hold)
 tp13/tp17  take profit: trade >= +2%, today's RTH range >= 1.3 / 1.7 budgets (HAR, 1.7 sigma), and the high looks in
            (set >= 30 min ago, price >= 0.25 budget under it) -> out at the next open
 wtp13      same on the weekly range (weekly HAR budget, week high set on an earlier day or >= 30 min ago)
Per-trade entry context columns: used (share of today's budget used at entry), both (15m and 1h bull at entry), wk (1h flip came after
the week's low = week low likely in), age (bars since the day's low)
python3 xs/range_vector_trades.py BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
BARS, OUT, WT = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT, exist_ok=True)
def har(x): return 1.7 * (np.exp(-0.6186 + 0.2939 * x + 0.3661 * x.rolling(5).mean() + 0.1783 * x.rolling(22).mean()) / 1.665).shift(1)

def ctx(H):
    d = H.dt.dt.normalize(); rth = (H.m >= 570).values
    R = H[rth]; D = R.groupby(R.dt.dt.normalize()).agg(h=('high', 'max'), l=('low', 'min')); b = har(np.log(np.log(D.h / D.l).clip(lower=1e-5)))
    wk = D.index.to_period('W'); Wk = D.groupby(wk).agg(h=('h', 'max'), l=('l', 'min')); wb = har(np.log(np.log(Wk.h / Wk.l).clip(lower=1e-5)))
    lo_all = H.low.groupby(d).cummin().values                                                   # incl. premarket
    hr = H.high.where(rth); lr = H.low.where(rth)
    dhi, dlo = hr.groupby(d).cummax().values, lr.groupby(d).cummin().values
    w = H.dt.dt.to_period('W'); whi, wlo = hr.groupby(w).cummax().values, lr.groupby(w).cummin().values
    i = np.arange(len(H))
    thi = pd.Series(np.where(hr.values >= dhi, i, np.nan)).groupby(d.values).ffill().values; tlo = pd.Series(np.where(lr.values <= dlo, i, np.nan)).groupby(d.values).ffill().values
    twh = pd.Series(np.where(hr.values >= whi, i, np.nan)).groupby(w.values).ffill().values; twl = pd.Series(np.where(lr.values <= wlo, i, np.nan)).groupby(w.values).ffill().values
    return dict(b=b.reindex(d).values, wb=wb.reindex(w).values, lo_all=lo_all, dhi=dhi, dlo=dlo, thi=thi, tlo=tlo, whi=whi, wlo=wlo, twh=twh, twl=twl)

def run2(H, reg, s1, h1track, rule, X):
    o, c, hi, d, m = H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T = []; pos = False
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
    f1 = np.r_[False, (np.asarray(s1[1:]) == 1) & (np.asarray(s1[:-1]) != 1)]; lastf1 = pd.Series(np.where(f1, np.arange(len(H)), np.nan)).ffill().values
    for p in range(1, len(g)):
        j = g[p]
        if not pos:
            sig = (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1)
            if sig:
                pos, e, px, b1, peak, h1 = True, j, o[j], False, o[j], False; k = j - 1; slo = X['lo_all'][k]
                bb = X['b'][j]; used = (np.log(X['dhi'][k] / X['dlo'][k]) / bb) if (m[k] >= 570 and bb > 0) else np.nan
                info = (used, bool(KR[p] == 1 and KS[p] == 1), bool(lastf1[k] >= X['twl'][k]) if KS[p] == 1 else False, (k - X['tlo'][k]) if m[k] >= 570 else np.nan)
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or KS[p] == 1; now = c[j - 1] / px - 1 >= .02 and b1; h1 = h1 or now
        ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1)); pg = peak / px - 1
        if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex = True
        q = j - 1; gain = c[q] / px - 1; bb = X['b'][q]
        if rule == 'stop_lod' and c[q] < slo: ex = True
        if rule in ('tp13', 'tp17') and gain >= .02 and bb > 0 and m[q] >= 570:
            kk = 1.3 if rule == 'tp13' else 1.7
            if np.log(X['dhi'][q] / X['dlo'][q]) >= kk * bb and q - X['thi'][q] >= 2 and c[q] <= X['dhi'][q] * (1 - .25 * bb): ex = True
        if rule == 'wtp13' and gain >= .02 and X['wb'][q] > 0 and m[q] >= 570:
            wbb = X['wb'][q]
            if np.log(X['whi'][q] / X['wlo'][q]) >= 1.3 * wbb and q - X['twh'][q] >= 2 and c[q] <= X['whi'][q] * (1 - .25 * wbb): ex = True
        if ex: T.append((d[e], d[j], o[j] / px - 1 - C2) + info); pos = False
    return T

W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
SH, NS = int(os.getenv('SHARD', 0)), int(os.getenv('NSHARD', 1))
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); reg, s1 = states(H, vec); X = ctx(H); rows = []
    for rule in ('live', 'stop_lod', 'tp13', 'tp17', 'wtp13'):
        for h1t in (False, True): rows += [(tk, rule, h1t) + t for t in run2(H, reg, s1, h1t, rule, X)]
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret', 'used', 'both', 'wk', 'age']).to_parquet(fo); print(tk, len(rows), flush=True)
