"""Oct 26 2026: other trend detectors in the live engine (15m + 1h tracks, sticky 1h stop, LOCK5/0.5, entries at RTH opens), then the
live book (6 slots, ratio >= 1.5 S&P names, 8% breaker). Each model is a +1 / -1 state on 15m bars and on 1h bars, like the vector.
 vec          the live vector (SVM - Fractals port)
 donch20/55   close above / below the Donchian midpoint
 turtle20_10  long on a close above the prior 20-bar high, out on a close below the prior 10-bar low (classic breakout)
 supertrend   ATR(10) x 3 trailing band
 ema21_55     fast EMA above slow EMA
 kama         Kaufman adaptive moving average (10, 2, 30): bull while the close is above it and it is rising
 hma55        Hull moving average (55) slope
 tsmom        sign of the 130-bar return (about 5 sessions on 15m, ~20 on 1h)
 kalman       local-linear-trend Kalman filter: bull while the filtered slope is > 0
python3 xs/trend_models.py BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); A = sys.argv[1:]; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
BARS, OUT, WT = A; os.makedirs(OUT, exist_ok=True)

def turtle(df, n=20, x=10):
    h, l, c = df.high.values, df.low.values, df.close.values; hi = pd.Series(h).rolling(n).max().shift(1).values; lo = pd.Series(l).rolling(x).min().shift(1).values
    st = np.zeros(len(c), int); s = -1
    for i in range(len(c)):
        if s != 1 and c[i] > hi[i]: s = 1
        elif s == 1 and c[i] < lo[i]: s = -1
        st[i] = s
    return st
def kama(df, n=10, f=2, s=30):
    c = df.close.values; er = np.abs(c - np.r_[np.full(n, np.nan), c[:-n]]) / pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().values
    sc = (np.nan_to_num(er) * (2 / (f + 1) - 2 / (s + 1)) + 2 / (s + 1)) ** 2; k = np.empty(len(c)); k[0] = c[0]
    for i in range(1, len(c)): k[i] = k[i - 1] + sc[i] * (c[i] - k[i - 1])
    up = np.r_[False, k[1:] > k[:-1]]
    return np.where((c > k) & up, 1, -1)
def hma(df, n=55):
    c = pd.Series(df.close.values); w = lambda x, m: x.rolling(m).apply(lambda a: np.dot(a, np.arange(1, m + 1)) / (m * (m + 1) / 2), raw=True)
    h = w(2 * w(c, n // 2) - w(c, n), int(np.sqrt(n))).values
    return np.where(np.r_[False, h[1:] > h[:-1]], 1, -1)
def tsmom(df, n=130):
    c = df.close.values; return np.where(c > np.r_[np.full(n, np.nan), c[:-n]], 1, -1)
def kalman(df, q=1e-4, r=1.0):
    y = np.log(df.close.values); x = np.array([y[0], 0.0]); P = np.eye(2); F = np.array([[1, 1], [0, 1]]); Q = np.diag([q, q * .01]) * np.var(np.diff(y)) * 100; R = np.var(np.diff(y)) * r
    st = np.zeros(len(y), int)
    for i in range(len(y)):
        x = F @ x; P = F @ P @ F.T + Q; k = P[:, 0] / (P[0, 0] + R); x = x + k * (y[i] - x[0]); P = P - np.outer(k, P[0]); st[i] = 1 if x[1] > 0 else -1
    return st
M = {'vec': vec, 'donch20': lambda d: donch(d, 20), 'donch55': lambda d: donch(d, 55), 'turtle20_10': turtle, 'supertrend': supertrend,
     'ema21_55': emax, 'kama': kama, 'hma55': hma, 'tsmom': tsmom, 'kalman': kalman}
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
SH, NS = int(os.getenv('SHARD', 0)), int(os.getenv('NSHARD', 1))
for tk in CORE[SH::NS]:
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); rows = []
    for nm, f in M.items():
        reg, s1 = states(H, f)
        for h1t in (False, True): rows += [(tk, nm, h1t) + t for t in run(H, reg, s1, h1t)[0]]
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
