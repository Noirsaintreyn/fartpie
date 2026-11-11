"""LEVEL EVENTS: Rey's production detectors (backend.py via backtest_levels.METHODS) on QQQ DAILY and 4H(ish: 09:30-13:30 / 13:30-16:00) bars, point-in-time. Level set for trading day D uses bars through D-1 (lookback 120 bars).
First touch of each level on 15m bars 09:45-15:30 (prior bar closed fully on one side): from above = long at the level, from below = short. Stop s*ATR14(daily), target m*stop, out 15:55; stop and target in one bar = stop; cost 0.02/share.
Each event is also run on a PLACEBO level (level +/- 0.5 ATR, random sign). Saves events with features for the pairing study. usage: levels_events.py METHOD [METHOD..]"""
import sys, io, contextlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings('ignore')
METHS = sys.argv[1:]; sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location('bl', '/home/claude/fartpie/backtest_levels.py'); bl = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(bl)
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
x = pd.read_parquet('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/QQQ_15m_2018-01-01_2026-10-09.parquet'); dt = pd.to_datetime(x.datetime)
x['dt'] = (dt.dt.tz_localize('UTC') if dt.dt.tz is None else dt).dt.tz_convert('America/New_York').dt.tz_localize(None); x['day'] = x.dt.dt.normalize(); x['m'] = x.dt.dt.hour * 60 + x.dt.dt.minute
x = x[(x.m >= 570) & (x.m < 960)].sort_values('dt').reset_index(drop=True); x = x[x.groupby('day').day.transform('size') >= 20]
G = {d: g for d, g in x.groupby('day')}; DAYS = sorted(G)
dly = pd.DataFrame({'h': [G[d].high.max() for d in DAYS], 'l': [G[d].low.min() for d in DAYS], 'c': [G[d].close.iloc[-1] for d in DAYS]}, index=DAYS)
tr = np.maximum(dly.h - dly.l, np.maximum((dly.h - dly.c.shift(1)).abs(), (dly.l - dly.c.shift(1)).abs())); atr = tr.rolling(14).mean()
x['blk'] = np.where(x.m < 810, 0, 1); b4 = x.groupby(['day', 'blk']).agg(h=('high', 'max'), l=('low', 'min'), c=('close', 'last')).reset_index()
b4h, b4l, b4c = b4.h.values, b4.l.values, b4.c.values; b4day = b4.day.values
gex = pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); gex.index = gex.index.normalize(); g2 = (gex.call_gex - gex.put_gex)
TG = pd.read_pickle(SC + 'tg_features.pkl'); TG.index = pd.to_datetime(TG.index).normalize()
def hvn(di):
    lo = max(0, di - 5); bars = pd.concat([G[DAYS[k]] for k in range(lo, di)]); px0 = bars.close.mean(); bin_ = px0 * 0.0005
    n = int((bars.high.max() - bars.low.min()) / bin_) + 3; base = bars.low.min(); prof = np.zeros(n)
    for hh, ll, v in zip(bars.high.values, bars.low.values, bars.volume.values):
        a, b = int((ll - base) / bin_), int((hh - base) / bin_); prof[a:b + 1] += v / (b - a + 1)
    k = np.ones(5) / 5; sm = np.convolve(prof, k, 'same'); pk = [i for i in range(1, n - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]]; pk = sorted(pk, key=lambda i: -sm[i])[:3]
    return np.array([base + (i + .5) * bin_ for i in pk])
CFG = [(0.10, 1.0), (0.10, 1.5), (0.20, 1.5)]
def sim(side, L, hi, lo, cl, j, a):
    """side +1 long at L (touch from above) / -1 short; bars j.. ; returns R for each cfg"""
    out = []
    for s, m in CFG:
        risk = s * a; stop = L - side * risk; tgt = L + side * m * risk; res = None
        for q in range(j, len(hi)):
            hit_s = (lo[q] <= stop) if side > 0 else (hi[q] >= stop); hit_t = (hi[q] >= tgt) if side > 0 else (lo[q] <= tgt)
            if hit_s: res = -1.0; break
            if hit_t: res = float(m); break
        if res is None: res = side * (cl[-1] - L) / risk
        out.append(res - 0.02 / risk)
    return out
rng = np.random.default_rng(7); rows = []
for di in range(130, len(DAYS)):
    d = DAYS[di]; a = atr.iloc[di - 1]
    if not np.isfinite(a): continue
    g = G[d]; hi, lo, cl, mm = g.high.values, g.low.values, g.close.values, g.m.values; hv = hvn(di)
    # windows
    dh, dl, dc = dly.h.values[di - 120:di], dly.l.values[di - 120:di], dly.c.values[di - 120:di]
    k4 = np.searchsorted(b4day, np.datetime64(d)); sl = slice(max(0, k4 - 120), k4)
    feats = dict(gex=(g2.get(DAYS[di - 1], np.nan)), yz=TG.yz_pct.get(d, np.nan), har=TG.har_ratio.get(d, np.nan), vxn=TG.vxn_pct.get(d, np.nan))
    for mname in METHS:
        for tf, (H_, L_, C_) in (('1d', (dh, dl, dc)), ('4h', (b4h[sl], b4l[sl], b4c[sl]))):
            try:
                with contextlib.redirect_stdout(io.StringIO()): lv = bl.METHODS[mname](H_, L_, C_, tf)
            except Exception: continue
            for lvl in (lv or []):
                P0 = float(lvl['price'])
                for plc in (0, 1):
                    P = P0 if not plc else P0 + rng.choice([-.5, .5]) * a
                    for j in range(1, len(hi)):
                        if mm[j] < 585 or mm[j] > 930: continue
                        if lo[j] <= P <= hi[j] and (cl[j - 1] > P or cl[j - 1] < P):
                            side = 1 if cl[j - 1] > P else -1; R = sim(side, P, hi, lo, cl, j, a)
                            rows.append(dict(day=d, method=mname, tf=tf, plc=plc, side=side, atr=a, m=int(mm[j]), hvd=float(np.min(np.abs(hv - P))) / a, R1=R[0], R2=R[1], R3=R[2], **feats)); break
    if di % 200 == 0: print(di, len(DAYS), len(rows), flush=True)
pd.DataFrame(rows).to_pickle(SC + 'lvl_events_' + '_'.join(METHS) + '.pkl'); print('done', len(rows))
