"""Nov 10 2026: NOISE-AREA MOMENTUM x THE 15m QQQ VECTOR, ON THE CORRECTED ET CLOCK. DECLARED BEFORE RESULTS. (The earlier version, xs/noise_vec_stop.py, ran on the Central-time file and is void.)
Engines: noise-area momentum exactly as xs/new_models.py (k 1, decisions every 30 min 10:00-15:30, next-open fills, hard stop 40, cost 0.5, flat 15:55), with the vector regime (xs/signal_lab.vec on QQQ 15m, premarket + RTH, known at each 15m close):
  V0 base (band / VWAP exit)              V1 enter only when the vector agrees with the trade (long needs bull, short needs bear)
  V2 V1 and EXIT when the vector turns against (checked every 5m bar)      V3 no agreement needed, exit when the vector turns against
  V4 enter only AGAINST the vector (diagnostic: if V1 helps, V4 should be worse)
Report each as in new_models (pts/day both periods with t, long / short, corr with S6) and then the SYSTEM: S6 + w x engine for w = 0.25, 0.5, 1.0 (daily series, S6 at its live size x weight): pts/day, daily std, mean / std, worst month, months >= 50 pts by period.
PASS = pts/day > 0 with t >= 2 in both periods."""
import glob, sys
_n = open('xs/new_models.py').read(); exec(_n[:_n.index("# --- the current system's daily series")])
sys.argv = ['x', '2']; sys.path.insert(0, '/home/claude/fartpie/xs')
src = open('xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0]; exec(src)
H = pd.read_parquet(glob.glob('lse_clean/QQQ_15m_*.parquet')[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
mm = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(mm >= 240) & (mm < 960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True); REG = vec(H.rename(columns=str.lower))
CL = (H.dt + pd.Timedelta(minutes=15)).values
def regd(d):
    t = (pd.Timestamp(d) + pd.to_timedelta(GRID + 5, unit='m')).values; i = np.searchsorted(CL, t, 'right') - 1
    return np.where(i >= 0, REG[np.maximum(i, 0)], 0)
def noise2(d, mode, stop=40.0, start=600, end=930):
    a, _ = A[d]; o, h, l, c, v, vw = a; s = sg(d); op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op; rg = regd(d); rows = []; pos = 0; e = em = None; worst = 0.0; pend = None
    agree = mode in ('V1', 'V2'); vexit = mode in ('V2', 'V3'); anti = mode == 'V4'
    for j in range(NB):
        if np.isnan(c[j]): continue
        if pend is not None: pos, e, em, worst = pend, o[j], j, 0.0; pend = None
        if pos != 0:
            lo = (l[j] - e) if pos > 0 else (e - h[j]); worst = min(worst, lo)
            if lo <= -stop: rows.append((d, em, pos, -stop - COST, worst - COST)); pos = 0; continue
            if vexit and j < NB - 1 and not np.isnan(o[j + 1]) and rg[j] != pos: rows.append((d, em, pos, (o[j + 1] - e) * pos - COST, worst - COST)); pos = 0
        if j == NB - 1 or np.isnan(o[j + 1]):
            if j == NB - 1 and pos != 0: rows.append((d, em, pos, (c[j] - e) * pos - COST, worst - COST)); pos = 0
            continue
        t = GRID[j]
        if (t + 5) % 30 != 0 or t + 5 < start or t + 5 > end or not np.isfinite(s[j]): continue
        ub, lb = max(op, pc) * (1 + s[j]), min(op, pc) * (1 - s[j]); nx = o[j + 1]
        if not vexit:
            if pos == 1 and c[j] < max(ub, vw[j]): rows.append((d, em, 1, nx - e - COST, worst - COST)); pos = 0
            elif pos == -1 and c[j] > min(lb, vw[j]): rows.append((d, em, -1, e - nx - COST, worst - COST)); pos = 0
        if pos == 0:
            want = 1 if c[j] > ub else (-1 if c[j] < lb else 0)
            if want and (not agree or rg[j] == want) and (not anti or rg[j] == -want): pend = want
    return rows
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); UW = {'RA': 1.0, 'M1': .3, 'R1': .5, 'ON+': .4, 'ON-': .1, 'RULES': 1.0, 'V15a': .3}; base = pd.Series(0.0, index=DAYS)
for (t, u, y), n in zip(OUT['V2'], NAMES):
    t = t[t.sz > 0]; dd = pd.to_datetime(t.day).dt.normalize(); base = base.add((t.pts * t.sz * UW[n]).groupby(dd.values).sum().reindex(DAYS).fillna(0.0), fill_value=0)
def sysrow(s, w):
    x = base + w * s; out = []
    for lab, mm_ in (('14-21', x.index < '2022-01-01'), ('22-26', x.index >= '2022-01-01')):
        y = x[mm_]; mo = y.groupby(y.index.to_period('M')).sum(); out.append(f'{lab} {y.mean():+5.2f} std {y.std():5.1f} m/s {y.mean() / y.std():.3f} worst mo {mo.min():+5.0f} mo>=50 {np.mean(mo >= 50):3.0%}')
    return ' | '.join(out)
print(f'S6 alone: {sysrow(base * 0, 0)}\n')
for md in ('V0', 'V1', 'V2', 'V3', 'V4'):
    s, T = stats([r for d in DAYS for r in noise2(d, md)], f'{md}', base)
    for w in (.25, .5, 1.0): print(f'      S6 + {w:4.2f} x {md}: {sysrow(s, w)}')
