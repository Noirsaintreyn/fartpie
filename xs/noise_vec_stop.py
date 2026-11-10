"""Nov 10 2026: FLIPS + NOISE-AREA MOMENTUM, noise exits on the VECTOR, plain hard stop on the noise (Rey: 'for the flips + noise try it with the vector stop loss, regular stop just the noise;
if it shows anything we can improve leverage'). DECLARED BEFORE RESULTS. Real NQ 5m 2014 - Jan 2026 (noise replication of xs/nq_noise_onwide.py: sigma bands from the prior 14 sessions,
decisions at HH:00 / HH:30 from 10:00 to 15:30, next-5m-open fills, 1 pt cost, flat 15:55) + the 15m vector from QQQ premarket + RTH bars (xs/signal_lab.vec), regime known at each 15m close.
NOISE VARIANTS (all with a 40-pt hard stop, the 'regular stop'):
  N0  paper exit (close back through max(UB, VWAP) / min(LB, VWAP)) + hard stop
  N1  N0, entry only when the 15m vector agrees (long needs bull, short needs bear)
  N2  vector agrees at entry; EXIT when the vector turns against (every 15m close) instead of the band / VWAP stop; hard stop kept            <- Rey's idea
  N3  as N2 but no agreement needed at entry
FLIPS = the live Rule A engine (xs/flex50.py trade table, 1 contract, its own stop) and the full S6 stack (7 engines).
REPORT per variant: pts / day at 1 contract (all, 2014-21 / 2022-Jan26), losing days, worst day, 5th percentile day, year by year; correlation with Rule A and with S6; combined Rule A + noise and
S6 + noise (pts / day, worst day, 5th pct day, months >= 50 pts); and the leverage that fits the Lucid 50K day limit ($1,000 = 500 / worst-day-points MNQ) using the worst day and the 1st percentile day."""
import numpy as np, pandas as pd, importlib.util, pickle, glob, sys, warnings; warnings.filterwarnings('ignore')
sys.argv = ['x', '2']; sys.path.insert(0, '/home/claude/fartpie/xs')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
sp = importlib.util.spec_from_file_location('h', '/home/claude/fartpie/xs/nq_edge_hunt.py'); h = importlib.util.module_from_spec(sp); sp.loader.exec_module(h)
_nw = open('/home/claude/fartpie/xs/nq_noise_onwide.py').read(); exec(_nw[:_nw.index('def run(day')])           # DAYS (78-bar grid), K, MV, D
src = open('/home/claude/fartpie/xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0]; exec(src)
H = pd.read_parquet(glob.glob('/home/claude/fartpie/lse_clean/QQQ_15m_*.parquet')[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
mm = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(mm >= 240) & (mm < 960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True); REG = vec(H.rename(columns=str.lower))
CL = (H.dt + pd.Timedelta(minutes=15)).values; Hs = pd.Series(REG, index=H.dt + pd.Timedelta(minutes=15))                                      # regime known at the 15m close
def reg_day(day):
    t = pd.date_range(day + pd.Timedelta(hours=9, minutes=35), periods=78, freq='5min'); i = np.searchsorted(CL, t.values, 'right') - 1
    return np.where(i >= 0, REG[np.maximum(i, 0)], 0)
def run(day, prev, hist, mode, S=40):
    g = DAYS[day]
    if len(g) < 78 or len(hist) < 14: return None
    o, hi, lo, c = (g[x].values for x in 'ohlc'); op = o[0]; pc = prev if np.isfinite(prev) else op; sig = np.mean(hist, axis=0); rg = reg_day(day)
    vol = g.v.values.astype(float); vw = np.cumsum((hi + lo + c) / 3 * vol) / np.maximum(np.cumsum(vol), 1)
    agree = mode in ('N1', 'N2'); vexit = mode in ('N2', 'N3'); pos = 0; ent = 0.0; pnl = 0.0; start = 0
    for k in range(78):
        if pos and k >= start and S:
            st = ent - S * pos
            if (pos == 1 and lo[k] <= st) or (pos == -1 and hi[k] >= st): pnl += -S - 1; pos = 0
        if pos and vexit and k < 77 and rg[k] != pos: pnl += (o[k + 1] - ent) * pos - 1; pos = 0
        if k in K and k < 77:
            ub = max(op, pc) * (1 + sig[k]); lb = min(op, pc) * (1 - sig[k]); nx = o[k + 1]
            if not vexit:
                if pos == 1 and c[k] < max(ub, vw[k]): pnl += nx - ent - 1; pos = 0
                elif pos == -1 and c[k] > min(lb, vw[k]): pnl += ent - nx - 1; pos = 0
            if pos == 0:
                if c[k] > ub and (not agree or rg[k] == 1): pos, ent, start = 1, nx, k + 1
                elif c[k] < lb and (not agree or rg[k] == -1): pos, ent, start = -1, nx, k + 1
        if pos == 0 and vexit and k < 77 and mode == 'N3': pass
    if pos: pnl += (c[77] - ent) * pos - 1
    return pnl
idx = list(DAYS); MODES = ('N0', 'N1', 'N2', 'N3'); R = {m: {} for m in MODES}
for i, day in enumerate(idx):
    if i < 15: continue
    hist = [MV[x] for x in idx[i - 14:i] if MV[x] is not None]; prev = D.PC.get(day, np.nan)
    for m in MODES:
        r = run(day, prev, hist, m)
        if r is not None: R[m][day] = r
N = {m: pd.Series(R[m]) for m in MODES}
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); Dn = pd.read_pickle(f'{SP}/nq_days.pkl'); DAYS2 = Dn.index[(Dn.index >= '2018-02-01') & (Dn.index <= '2026-01-30')]
def eng(n):
    t = OUT['V2'][NAMES.index(n)][0]; t = t.copy(); t['day'] = pd.to_datetime(t.day); t = t[t.sz > 0]; return t.pts.groupby(t.day.values).sum().reindex(DAYS2).fillna(0.0)
RA = eng('RA'); S6 = sum(eng(n) for n in NAMES)
def st(s, lab):
    A = s.index < '2022-01-01'; mo = s.groupby(s.index.to_period('M')).sum()
    return f'{lab:<28} {s.mean():+6.2f} | {s[A].mean():+6.2f} | {s[~A].mean():+6.2f} | losing {np.mean(s < 0):4.0%} | worst {s.min():+6.0f} | p1 {s.quantile(.01):+6.0f} | p5 {s.quantile(.05):+6.0f} | mo>=50 {np.mean(mo >= 50):4.0%}'
print('=== pts / day at 1 contract: all | 2018-21 | 2022-Jan26 | losing days | worst day | 1st pct | 5th pct | months >= 50 pts (2018-Jan 2026 window) ===')
print('  ' + st(RA, 'Rule A (flips)')); print('  ' + st(S6, 'S6 (7 engines, V2)'))
NN = {m: N[m].reindex(DAYS2).fillna(0.0) for m in MODES}
for m in MODES: print('  ' + st(NN[m], f'{m} noise'))
print('\n=== correlation of daily points: noise variant vs Rule A | vs S6 ===')
for m in MODES: print(f'  {m}: {NN[m].corr(RA):+.2f} | {NN[m].corr(S6):+.2f}')
print('\n=== COMBINED at 1 contract each ===')
for base, bn in ((RA, 'Rule A'), (S6, 'S6')):
    print('  ' + st(base, bn + ' alone'))
    for m in MODES: print('  ' + st(base + NN[m], f'{bn} + {m}'))
print('\n=== LEVERAGE: MNQ that keep the worst day / the 1st-pct day inside the $1,000 daily loss limit (1 NQ = 10 MNQ, $2 a point per MNQ) ===')
for lab, s in (('Rule A + N2', RA + NN['N2']), ('S6', S6), ('S6 + N2', S6 + NN['N2']), ('Rule A', RA), ('N2 alone', NN['N2'])):
    w = abs(s.min()); p1 = abs(s.quantile(.01)); print(f'  {lab:<14} worst day {s.min():+6.0f} pts -> {int(500 // w):3} MNQ | 1st pct {s.quantile(.01):+6.0f} pts -> {int(500 // p1):3} MNQ | avg day at that size ${s.mean() * 2 * (500 // p1):,.0f}')
print('\nby year pts/day: ' + ' | '.join(f'{y}: RA {RA[RA.index.year == y].mean():+.1f} N2 {NN["N2"][NN["N2"].index.year == y].mean():+.1f} N0 {NN["N0"][NN["N0"].index.year == y].mean():+.1f}' for y in range(2018, 2027)))
