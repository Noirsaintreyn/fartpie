"""Nov 23 2026 report 406: STOCK ANALYSIS MODEL (separate from the flip screener). Rey: 'my stock analysis: company financials + macro regime + options data, then the vector is the cherry on top; high MEDIAN, high WIN RATE; use gex/vex/vanna/vega/charm/p-c and the models we found with edge'.
PRE-REGISTERED (written before any result): monthly decision at the last trading day t; entry = next-day OPEN; 10bp round trip; excess = vs SPY open-to-open over the same dates.
UNIVERSE: PIT S&P 500 members (monthly) among the 162 names with daily LSE candles (54 research + 108 pool); data 2018-07 .. 2026-09 entries. SPLIT: train = entries 2019-21; test = 2018-07..2018-12 + 2022-26.
SIGNALS (all at t, lagged data only): 
 F  fundamentals (SEC 10-K annual values, filed+3d, point in time): mean cross-sectional percentile of ROE (NI/equity), net margin (NI/rev), revenue growth (YoY), earnings yield (NI/(price*diluted shares)); needs >=3 of 4; direction +.
 O  options attention = the live gate's ratio (rolling-252 median of opt-vol/shares, shift 1); gate flag = ratio >= 1.5; direction +.
 R  name vol relief: Parkinson vol 5d mean <= its 20d mean (daily OHLC; report 404 logic on daily data); direction +.
 M  macro regime: VIX <= its 20d mean (report 402/403 logic); direction +.
 V  vector: daily vector (xs/vector_pine.run_state_machine on daily OHLC) regime == bull at t; entry only when bull; EXIT-E2 = first close in bear regime (next open) or 63 trading days.
ARMS (E1 = fixed 63-trading-day hold for all; E2 = hold while vector bull, V arms only): ALL | V | F5 (top quintile) | F5&V | G (ratio>=1.5) | G&V | F40&G (top 40% F and gate) | STACK = F40&G&R&V | STACK-noV | STACK&M.
METRICS per arm: n, WIN RATE (net>0), win vs SPY, MEDIAN net return, mean, median excess, month-clustered t of mean excess minus the same-month universe mean (arm-minus-universe), train | test.  PASS for a signal: same sign of (arm - universe) both halves and |pooled t| >= 2.
STAGE 2 (options greeks/flows on the 27-35 names with GEX/OI files; 21-day forward excess; monthly Spearman IC + tercile spread, train | test; declared directions): P1 put share of 5d option volume (low better); P2 opening-position imbalance 5d (net call opens minus put opens; high better); P3 put share of open interest (low better); P4 own net GEX z-score vs prior 126d (lower better); P5 IV minus 20d realised vol (higher better, VRP compensation).  Vanna/charm/vega from chains = stage 3."""
import sys, json, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); import vector_pine as vp
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; UP = '/mnt/user-data/uploads/fartpie/'; HZ = 63; COST = 1e-3
# ---- prices
D1 = pd.concat([pd.read_parquet(SP + 'lse_daily_54.parquet'), pd.read_parquet(SP + 'lse_daily_pool.parquet')]).drop_duplicates(['symbol', 'ts']); D1['day'] = pd.to_datetime(D1.ts).dt.normalize()
PXN = {}
for tk, g in D1.groupby('symbol'):
    g = g.sort_values('day').set_index('day'); g = g[~g.index.duplicated()]; g = g[(g.close > 0) & (g.high >= g.low)]
    if len(g) > 300: PXN[tk] = g
MD = pd.read_parquet(SP + 'macro_daily.parquet'); MD['day'] = pd.to_datetime(MD.ts).dt.normalize(); spy = MD[MD.symbol == 'SPY'].set_index('day').sort_index(); spy = spy[~spy.index.duplicated()]
vix = pd.read_parquet('data/yahoo/daily_v/VIX.parquet').close; vix.index = pd.to_datetime(vix.index).normalize(); Mflag = (vix <= vix.rolling(20).mean())
Mm = json.load(open('sp500_pit_membership.json')); fx = lambda t: 'META' if t == 'FB' else t; MEM = {k: {fx(t) for t in v} for k, v in Mm.items()}
# ---- ratio
_L = pd.read_csv(UP + 'xs/ext/optvol_ratio_live.csv', parse_dates=['date']); _L['r'] = 100 * _L.opt_contracts / _L.sh
_r = pd.concat([pd.read_csv(UP + 'xs/ext/optvol_ratio_sp.csv', parse_dates=['date']).rename(columns={'opt_ratio': 'r'})[['tk', 'date', 'r']], _L[['tk', 'date', 'r']]]).drop_duplicates(['tk', 'date']); RR = _r.pivot_table(index='date', columns='tk', values='r').sort_index()
RMED = RR.rolling(252, min_periods=126).median().shift(1)
# ---- SEC annual facts (point in time)
S = pd.read_parquet('xs/data/sec_facts.parquet')
for c in ('start', 'end', 'filed'): S[c] = pd.to_datetime(S[c], errors='coerce')
S = S[S.form.isin(['10-K', '10-K/A', '10-KT'])]; dur = (S.end - S.start).dt.days; S = S[(~S.kind.isin(['rev', 'ni', 'sh'])) | ((dur > 300) & (dur < 400))].sort_values('filed').drop_duplicates(['tk', 'kind', 'end'], keep='first'); S['tk'] = S.tk.map(fx)
SF = {k: g.sort_values('end') for k, g in S.groupby(['tk', 'kind'])}
def fund(tk, t):
    out = {}
    for kd in ('rev', 'ni', 'sh', 'eq'):
        g = SF.get((tk, kd))
        if g is None: return None
        g = g[g.filed + pd.Timedelta(days=3) <= t]
        if not len(g): return None
        c = g.iloc[-1]; p = g[(g.end < c.end - pd.Timedelta(days=330)) & (g.end > c.end - pd.Timedelta(days=400))]; out[kd] = (c.val, p.iloc[-1].val if len(p) else np.nan, c.end)
    return out
# ---- vector regimes (daily OHLC) + vol relief
REG = {}; PK = {}
for tk, g in PXN.items():
    regs = vp.run_state_machine(g[['high', 'low', 'close']].reset_index(drop=True))[0]; REG[tk] = pd.Series(regs, index=g.index)
    pv = np.log(g.high / g.low) ** 2 / (4 * np.log(2)); v5 = pv.rolling(5).mean() ** .5; v20 = pv.rolling(20).mean() ** .5; PK[tk] = (pv.rolling(5).mean() ** .5 <= (pv ** .5).rolling(20).mean())
cal = spy.index; me = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).last().values; me = [pd.Timestamp(d) for d in me if pd.Timestamp('2018-07-01') <= pd.Timestamp(d) <= pd.Timestamp('2026-08-31')]
print('names with prices', len(PXN), 'month-ends', len(me), flush=True)
rows = []
for t in me:
    mem = MEM.get(t.strftime('%Y-%m'), set()); cand = [tk for tk in PXN if tk in mem and t in PXN[tk].index]
    if len(cand) < 20: continue
    rec = []
    for tk in cand:
        g = PXN[tk]; pos = g.index.get_loc(t)
        if pos + HZ + 2 >= len(g): continue
        f = fund(tk, t); px = g.close.iloc[pos]; rt = RMED[tk].reindex([t], method='ffill').iloc[0] if tk in RMED.columns else np.nan
        r = dict(tk=tk, t=t, ratio=rt, V=int(REG[tk].iloc[pos] == 1), R=int(PK[tk].iloc[pos]), M=int(Mflag.get(t, False)))
        if f:
            sh = f['sh'][0]; ni = f['ni'][0]; rev = f['rev'][0]; eq = f['eq'][0]; r.update(roe=ni / eq if eq and eq > 0 else np.nan, mar=ni / rev if rev and rev > 0 else np.nan, gr=rev / f['rev'][1] - 1 if f['rev'][1] and f['rev'][1] > 0 else np.nan, ey=ni / (px * sh) if sh and sh > 0 else np.nan)
        # trades: E1 fixed 63d, E2 vector exit (open of next bar after first bear close, cap 63)
        e = pos + 1; ent = g.open.iloc[e]; x1 = g.open.iloc[e + HZ]; r['d_in'] = g.index[e]; r['d_x1'] = g.index[e + HZ]; r['g1'] = x1 / ent - 1
        j = None
        for k in range(e, e + HZ):
            if REG[tk].iloc[k] == -1: j = k + 1; break
        j = j if j is not None else e + HZ; r['g2'] = g.open.iloc[j] / ent - 1; r['d_x2'] = g.index[j]; rec.append(r)
    if len(rec) < 20: continue
    d = pd.DataFrame(rec)
    for c in ('roe', 'mar', 'gr', 'ey'): d[c + '_p'] = d[c].rank(pct=True)
    d['nf'] = d[['roe_p', 'mar_p', 'gr_p', 'ey_p']].notna().sum(axis=1); d['F'] = d[['roe_p', 'mar_p', 'gr_p', 'ey_p']].mean(axis=1).where(d.nf >= 3); d['Fp'] = d.F.rank(pct=True); rows.append(d)
T = pd.concat(rows, ignore_index=True); sp_o = spy.open
def sx(a, b): return sp_o.get(b, np.nan) / sp_o.get(a, np.nan) - 1
T['s1'] = [sx(a, b) for a, b in zip(T.d_in, T.d_x1)]; T['s2'] = [sx(a, b) for a, b in zip(T.d_in, T.d_x2)]
for k in ('1', '2'): T['n' + k] = T['g' + k] - COST; T['x' + k] = T['n' + k] - T['s' + k]
T['half'] = np.where((T.t >= '2019-01-01') & (T.t < '2022-01-01'), 'train', np.where((T.t < '2019-01-01') | (T.t >= '2022-01-01'), 'test', 'x')); T['G'] = (T.ratio >= 1.5).astype(int); T.to_parquet(SP + 'stock406.parquet')
print('trade rows', len(T), 'F available', T.F.notna().mean().round(2), 'ratio available', T.ratio.notna().mean().round(2), 'names/month', round(T.groupby('t').size().mean(), 1), flush=True)
# ---- arms
ARMS = {'ALL': T.tk.notna(), 'V (bull)': T.V == 1, 'F5 (top quintile)': T.Fp >= .8, 'F5 & V': (T.Fp >= .8) & (T.V == 1), 'G (ratio>=1.5)': T.G == 1, 'G & V': (T.G == 1) & (T.V == 1), 'F40 & G': (T.Fp >= .6) & (T.G == 1), 'STACK = F40&G&R&V': (T.Fp >= .6) & (T.G == 1) & (T.R == 1) & (T.V == 1), 'STACK without V': (T.Fp >= .6) & (T.G == 1) & (T.R == 1), 'STACK & M(VIX relief)': (T.Fp >= .6) & (T.G == 1) & (T.R == 1) & (T.V == 1) & (T.M == 1)}
def clt(d, col, base):  # month-clustered t of arm-minus-universe mean excess
    m = d.groupby('t')[col].mean() - base.groupby('t')[col].mean().reindex(d.groupby('t')[col].mean().index); m = m.dropna(); return m.mean() * 100, (m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 3 else np.nan)
def table(k, lab):
    print(f'\n{lab}'); print(f"{'arm':26s}{'n':>6s}{'win%':>7s}{'beatSPY%':>9s}{'med net%':>9s}{'mean net%':>10s}{'med exc%':>9s} | train: n win med | test: n win med | arm-uni excess pp (t) train | test")
    for nm, m in ARMS.items():
        if k == '2' and 'V' not in nm and 'STACK' not in nm: continue
        d = T[m & T['n' + k].notna() & (T.half != 'x')]; base = T[T['n' + k].notna() & (T.half != 'x') & ((T.V == 1) if k == '2' else True)]  # E2 baseline = all vector-bull names
        if len(d) < 30: print(f'{nm:26s} n {len(d)}'); continue
        s = f"{nm:26s}{len(d):6d}{(d['n'+k] > 0).mean()*100:7.1f}{(d['x'+k] > 0).mean()*100:9.1f}{d['n'+k].median()*100:9.2f}{d['n'+k].mean()*100:10.2f}{d['x'+k].median()*100:9.2f} |"
        for h in ('train', 'test'): dh = d[d.half == h]; s += f" {len(dh):5d} {(dh['n'+k] > 0).mean()*100:5.1f} {dh['n'+k].median()*100:6.2f} |"
        a = clt(d[d.half == 'train'], 'x' + k, base[base.half == 'train']); b = clt(d[d.half == 'test'], 'x' + k, base[base.half == 'test']); c = clt(d, 'x' + k, base)
        ok = np.sign(a[0]) == np.sign(b[0]) and abs(c[1]) >= 2; print(s + f' {a[0]:+5.2f} ({a[1]:+.1f}) | {b[0]:+5.2f} ({b[1]:+.1f}) | pooled t {c[1]:+.1f} {"PASS" if ok else ""}')
table('1', 'E1: FIXED 63-DAY HOLD (all arms)'); table('2', 'E2: HOLD WHILE DAILY VECTOR BULL (cap 63d), V arms')
# single-signal cross-section: does each analysis layer separate winners? (E1, medians)
print('\nSINGLE-SIGNAL SPREADS (E1 net median %, win %): top vs bottom half of each analysis layer')
for nm, col, thr in (('F (fundamentals)', 'Fp', .5), ('ratio percentile', None, None)):
    if col: x = T[T.Fp.notna() & (T.half != 'x')]; hi = x[x.Fp >= thr]; lo = x[x.Fp < thr]; print(f"  {nm:20s} top {hi.n1.median()*100:5.2f}% win {(hi.n1>0).mean()*100:4.1f}% | bottom {lo.n1.median()*100:5.2f}% win {(lo.n1>0).mean()*100:4.1f}%")
for nm, col in (('R vol relief', 'R'), ('M VIX relief', 'M'), ('V vector bull', 'V'), ('G gate', 'G')):
    x = T[T.half != 'x']; hi = x[x[col] == 1]; lo = x[x[col] == 0]; print(f"  {nm:20s} on  {hi.n1.median()*100:5.2f}% win {(hi.n1>0).mean()*100:4.1f}% (n{len(hi)}) | off {lo.n1.median()*100:5.2f}% win {(lo.n1>0).mean()*100:4.1f}% (n{len(lo)})")
