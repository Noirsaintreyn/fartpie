"""SMALL/MID-CAP RERUN (1h, ~730 days, RTH bars) + HUGE-UPSET OVERNIGHT TEST. DECLARED BEFORE RESULTS.
Universe: the 5 charted names (KGS LCID RIOT Z IREN) + 17 small/mid names from the earlier random pool (PLUG UPST CLOV TLRY BB SOFI FUBO OPEN ETSY DKNG ROKU CROX FIVE CHWY MTCH LYFT BBWI). Yahoo 1h RTH bars, ~Dec 2023-Nov 2026; halves 2023-24 vs 2025+ (no parameter is fitted, so it is a consistency split, not train/test).
PART A: own-history TP vs ADR TP vs none, vector exit (as report_vector_tp2.py). PART B: earnings agree/disagree windows (as before).
PART C (huge upsets / overnight): C1 position view: at each RTH close with the vector in a state s, day move in position direction measured in ADRs (close vs prior close / mean range%). Bucket by that. Report the position's overnight (close->next open) and next-day open->close return, vs a null where the state series is circularly shifted (500x, keeps persistence). Answers: after a huge move in our favour/against us, is holding overnight worse than average?
C2 move view (no vector): after a day with |move| >= m x ADR, continuation (sign of move x next return) overnight and next-day open->close, split into EARNINGS reaction days vs other days (mechanism: news jumps drift / no-news jumps reverse; Chan 2003, Savor 2012)."""

import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
rng = np.random.default_rng(9); KS = (None, .25, .5, .75, 1.0, 1.5)
def clean(H):
    m = H.index.hour * 60 + H.index.minute; return H[(m >= 240) & (m < 960)]
def local(tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_2018-01-01_*.parquet')[0]; H = pd.read_parquet(f, columns=['datetime', 'open', 'high', 'low', 'close'])
    H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.drop(columns='datetime').dropna().sort_index(); return clean(H[~H.index.duplicated()])
def yf15(tk):
    import yfinance as yf
    d = yf.download(tk, period='60d', interval='15m', prepost=True, progress=False, auto_adjust=False); d.columns = [c[0].lower() for c in d.columns]
    d.index = d.index.tz_convert('America/New_York').tz_localize(None); d = d[['open', 'high', 'low', 'close']].dropna(); return clean(d[~d.index.duplicated()])
def sparse(a, fn):
    T = [a]; k = 1
    while 2 * k <= len(a): p = T[-1]; T.append(fn(p[:-k], p[k:])); k *= 2
    return T
def rq(T, fn, l, r):                                                           # inclusive l..r arrays
    ln = r - l + 1; k = np.floor(np.log2(ln)).astype(int); out = np.empty(len(l))
    for kk in np.unique(k): m = k == kk; out[m] = fn(T[kk][l[m]], T[kk][r[m] - (1 << kk) + 1])
    return out
def prep(B):
    n = len(B); reg = run_state_machine(B.assign(datetime=B.index))[0]; day = B.index.normalize(); m = B.index.hour * 60 + B.index.minute; rth = (m >= 570) & (m < 960)
    d = pd.DataFrame({'h': B.high.where(rth).groupby(day).max(), 'l': B.low.where(rth).groupby(day).min(), 'c': B.close.where(rth).groupby(day).last()}).dropna(); rgp = (d.h - d.l) / d.c
    adr = rgp.rolling(20).mean().shift(); ratio = rgp.shift() / rgp.shift().rolling(20).mean().shift(); rk = ratio.expanding(min_periods=60).apply(lambda z: (z.iloc[:-1] < z.iloc[-1]).mean(), raw=False)
    P = dict(n=n, o=B.open.values, h=B.high.values, l=B.low.values, reg=reg, m=m, adr=adr.reindex(day).values, hv=(rk.reindex(day) >= 2 / 3).fillna(False).values, yr=B.index.year.values)
    P['Th'] = sparse(P['h'], np.maximum); P['Tl'] = sparse(P['l'], np.minimum); return P
def trades(P):
    reg, n = P['reg'], P['n']; ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]; T = []
    for a in range(len(ch) - 1):
        i, j = ch[a], ch[a + 1]
        if j + 1 >= n: continue
        if P['m'][i + 1] >= 480 and P['m'][i + 1] < 945 and not np.isnan(P['adr'][i + 1]): T.append((i + 1, j + 1, int(reg[i])))
    return np.array(T).reshape(-1, 3)
def ret(P, st, en, d, k):
    e = P['o'][st]; x = P['o'][en]; base = d * (x / e - 1)
    if k is None: return base
    a = P['adr'][st]; hi = rq(P['Th'], np.maximum, st, en - 1); lo = rq(P['Tl'], np.minimum, st, en - 1)
    tp_long = e * (1 + k * a); tp_short = e * (1 - k * a); hit = np.where(d > 0, hi >= tp_long, lo <= tp_short)
    return np.where(hit, k * a, base)
def run(PS, tag, eras, classes, nul=200):
    for cls in classes:
        for dirs in ('both', 'long'):
            print(f'\n#### {tag} | day class {cls} | directions {dirs}', flush=True)
            hdr = 'era        k     n  obs bp  null bp   p   win%   (obs - null)'; print(hdr)
            for en, (lo, hi) in eras.items():
                for k in KS:
                    N = 0; obs = 0; nl = np.zeros(nul); win = 0
                    for tk, P in PS.items():
                        T = trades(P)
                        if not len(T): continue
                        msk = (P['yr'][T[:, 0]] >= lo) & (P['yr'][T[:, 0]] <= hi)
                        if cls == 'HV-pre': msk &= P['hv'][T[:, 0]]
                        if dirs == 'long': msk &= T[:, 2] > 0
                        T = T[msk]
                        if not len(T): continue
                        r = ret(P, T[:, 0], T[:, 1], T[:, 2], k); L = T[:, 1] - T[:, 0]; N += len(T); obs += r.sum(); win += (r > 0).sum()
                        cand = np.where((P['m'] >= 480) & (P['m'] < 945) & ~np.isnan(P['adr']) & (P['yr'] >= lo) & (P['yr'] <= hi) & (P['hv'] if cls == 'HV-pre' else True))[0]
                        for z in range(nul):
                            st = rng.choice(cand, len(T)); en_ = np.minimum(st + L, P['n'] - 1); st = np.minimum(st, en_ - 1)
                            nl[z] += ret(P, st, en_, T[:, 2], k).sum()
                    if N < 20: print(f'{en:9s} {str(k):5s} {N:5d}  (too few)'); continue
                    o = obs / N * 1e4; nm = nl / N * 1e4; print(f'{en:9s} {str(k):5s} {N:5d} {o:+7.1f} {nm.mean():+8.1f} {(nm >= o).mean():5.2f} {win / N * 100:5.1f}   {o - nm.mean():+6.1f}', flush=True)
import yfinance as yf
QS = (50, 70, 85, 95, 100)
def prep2(B):
    P = prep(B); day = B.index.normalize(); m = B.index.hour * 60 + B.index.minute; rth = (m >= 570) & (m < 960)
    d = pd.DataFrame({'o': B.open.where(rth).groupby(day).first(), 'h': B.high.where(rth).groupby(day).max(), 'l': B.low.where(rth).groupby(day).min()}).dropna()
    fl = ((d.h - d.o) / d.o); fs = ((d.o - d.l) / d.o); P['q'] = {}
    for q in QS: P['q'][('L', q)] = fl.rolling(250, min_periods=120).quantile(q / 100).shift().reindex(day).values; P['q'][('S', q)] = fs.rolling(250, min_periods=120).quantile(q / 100).shift().reindex(day).values
    P['days'] = d.index; P['day'] = day; return P
def ret2(P, st, en, d, k):                           # k = ('q', q) | ('adr', x) | None
    e = P['o'][st]; x = P['o'][en]; base = d * (x / e - 1)
    if k is None: return base
    hi = rq(P['Th'], np.maximum, st, en - 1); lo = rq(P['Tl'], np.minimum, st, en - 1)
    if k[0] == 'adr': a = k[1] * P['adr'][st]
    else: a = np.where(d > 0, P['q'][('L', k[1])][st], P['q'][('S', k[1])][st])
    ok = ~np.isnan(a); a = np.nan_to_num(a, nan=9.0)
    hit = np.where(d > 0, hi >= e * (1 + a), lo <= e * (1 - a)) & ok
    return np.where(hit, a, base)
def keep_valid(P, T, k):
    if k is None or k[0] == 'adr': return np.ones(len(T), bool)
    return ~np.isnan(np.where(T[:, 2] > 0, P['q'][('L', k[1])][T[:, 0]], P['q'][('S', k[1])][T[:, 0]]))
def partA(PS, lo, hi, tag, nul=200):
    print(f'\n#### PART A | {tag} | all days, both directions', flush=True); print('TP rule             n   obs bp  null bp    p   win%  (obs-null)')
    for k, name in [(None, 'none'), (('adr', 1.0), '1.0 x ADR')] + [(('q', q), f'own p{q}' if q < 100 else 'own biggest day') for q in QS]:
        N = 0; obs = 0; nl = np.zeros(nul); win = 0
        for tk, P in PS.items():
            T = trades(P); T = T[(P['yr'][T[:, 0]] >= lo) & (P['yr'][T[:, 0]] <= hi)]; T = T[keep_valid(P, T, k)] if len(T) else T
            if not len(T): continue
            r = ret2(P, T[:, 0], T[:, 1], T[:, 2], k); L = T[:, 1] - T[:, 0]; N += len(T); obs += r.sum(); win += (r > 0).sum()
            cand = np.where((P['m'] >= 480) & (P['m'] < 945) & ~np.isnan(P['adr']) & (P['yr'] >= lo) & (P['yr'] <= hi))[0]
            if k is not None and k[0] == 'q': cand = cand[~np.isnan(P['q'][('L', k[1])][cand])]
            for z in range(nul):
                st = rng.choice(cand, len(T)); en_ = np.minimum(st + L, P['n'] - 1); st = np.minimum(st, en_ - 1); nl[z] += ret2(P, st, en_, T[:, 2], k).sum()
        if N < 20: print(f'{name:18s} {N:5d} (too few)'); continue
        o = obs / N * 1e4; nm = nl / N * 1e4; print(f'{name:18s} {N:5d} {o:+7.1f} {nm.mean():+8.1f} {(nm >= o).mean():6.2f} {win / N * 100:5.1f}  {o - nm.mean():+6.1f}', flush=True)
def earn_events(tk):
    d = yf.Ticker(tk).get_earnings_dates(limit=60).dropna(subset=['Surprise(%)']); out = []
    for ts, row in d.iterrows():
        dt = ts.tz_localize(None) if ts.tzinfo is None else ts.tz_convert('America/New_York').tz_localize(None); day = dt.normalize()
        react = day + pd.Timedelta(days=1) if dt.hour >= 16 else day; out.append((react, float(row['Surprise(%)'])))
    return out
def partB(PS, nul=300):
    print('\n#### PART B | earnings window (reaction day..+4 trading days), mega-caps, vector exit, no TP and TP=own p85', flush=True)
    rows = {}
    for tk, P in PS.items():
        ev = earn_events(tk); days = list(P['days']); dset = {d: i for i, d in enumerate(days)}; win = np.full(P['n'], 0.0)             # win = sign of surprise inside window else 0
        for react, sp in ev:
            j = np.searchsorted(np.array(days, dtype='datetime64[ns]'), np.datetime64(react))
            if j >= len(days): continue
            w = days[j:j + 5]; msk = np.isin(P['day'].values, np.array(w, dtype='datetime64[ns]')); win[msk] = np.sign(sp) if sp != 0 else 0
        P['ewin'] = win
    for kname, k in (('no TP', None), ('TP own p85', ('q', 85))):
        print(f'\n  -- {kname}'); print('  class     n   obs bp   null bp    p   win%')
        for cname in ('agree', 'disagree', 'none'):
            N = 0; obs = 0; nl = np.zeros(nul); wn = 0
            for tk, P in PS.items():
                T = trades(P); T = T[(P['yr'][T[:, 0]] >= 2019)]
                if not len(T): continue
                ew = P['ewin'][T[:, 0]]
                m = {'agree': (ew != 0) & (np.sign(ew) == T[:, 2]), 'disagree': (ew != 0) & (np.sign(ew) == -T[:, 2]), 'none': ew == 0}[cname]; T = T[m]; T = T[keep_valid(P, T, k)] if len(T) else T
                if not len(T): continue
                r = ret2(P, T[:, 0], T[:, 1], T[:, 2], k); L = T[:, 1] - T[:, 0]; N += len(T); obs += r.sum(); wn += (r > 0).sum()
                for z in range(nul):
                    st = np.empty(len(T), int)
                    for dd in (1, -1):                                              # random bar inside windows with the same surprise sign relation
                        sel = np.where(T[:, 2] == dd)[0]
                        if not len(sel): continue
                        want = (P['ewin'] == dd) if cname == 'agree' else (P['ewin'] == -dd) if cname == 'disagree' else (P['ewin'] == 0)
                        cand = np.where(want & (P['m'] >= 480) & (P['m'] < 945) & ~np.isnan(P['adr']) & (P['yr'] >= 2019))[0]
                        st[sel] = rng.choice(cand, len(sel))
                    en_ = np.minimum(st + L, P['n'] - 1); st = np.minimum(st, en_ - 1); nl[z] += ret2(P, st, en_, T[:, 2], k).sum()
            if N < 15: print(f'  {cname:9s} {N:4d} (too few)'); continue
            o = obs / N * 1e4; nm = nl / N * 1e4; print(f'  {cname:9s} {N:4d} {o:+8.1f} {nm.mean():+8.1f} {(nm >= o).mean():6.2f} {wn / N * 100:5.1f}', flush=True)
import yfinance as yf
NAMES = ['KGS','LCID','RIOT','Z','IREN','PLUG','UPST','CLOV','TLRY','BB','SOFI','FUBO','OPEN','ETSY','DKNG','ROKU','CROX','FIVE','CHWY','MTCH','LYFT','BBWI']
def h1(tk):
    d = yf.download(tk, period='730d', interval='1h', prepost=False, progress=False, auto_adjust=False); d.columns = [c[0].lower() for c in d.columns]
    d.index = d.index.tz_convert('America/New_York').tz_localize(None); d = d[['open','high','low','close']].dropna(); return d[~d.index.duplicated()]
PS = {}; BARS = {}
for tk in NAMES:
    try:
        B = h1(tk)
        if len(B) < 1500: print('skip', tk, len(B)); continue
        PS[tk] = prep2(B); BARS[tk] = B
    except Exception as e: print('skip', tk, str(e)[:80])
print('universe', len(PS), list(PS), flush=True)
partA(PS, 2023, 2024, 'SMALL/MID 1h half 1 (Dec23-24)', nul=150); partA(PS, 2025, 2026, 'SMALL/MID 1h half 2 (2025+)', nul=150)
partB(PS, nul=200)
# ---------------- PART C
def daily_table(tk):
    B = BARS[tk]; P = PS[tk]; day = B.index.normalize(); g = pd.DataFrame({'o': B.open.groupby(day).first(), 'c': B.close.groupby(day).last()}); idx = pd.Series(np.arange(len(B)), index=B.index).groupby(day).last()
    g['s'] = P['reg'][idx.values]; g['adr'] = pd.Series(P['adr'], index=B.index).groupby(day).first(); g['mv'] = g.c / g.c.shift() - 1
    g['gap'] = g.o.shift(-1) / g.c - 1; g['intra'] = g.c.shift(-1) / g.o.shift(-1) - 1; g['mva'] = g.mv / g.adr
    try:
        ev = earn_events(tk); g['earn'] = g.index.isin([r for r, _ in ev])
    except Exception: g['earn'] = False
    return g.dropna(subset=['mva', 'gap', 'intra']), P
TB = {tk: daily_table(tk) for tk in PS}
EDG = [-99, -3, -2, -1, 0, 1, 2, 3, 99]; LAB = ['<-3', '-3..-2', '-2..-1', '-1..0', '0..1', '1..2', '2..3', '>3']
def c1(longonly=False, nsh=500):
    print(f'\n#### C1 | position view | {"long-only (state=+1 only)" if longonly else "long & short"} | move in position direction (ADRs) | overnight = close->next open, next day = open->close, both in position direction (bp)')
    rows = {}; null_on = {l: [] for l in LAB}; null_nx = {l: [] for l in LAB}
    def collect(shiftmap):
        on = {l: [] for l in LAB}; nx = {l: [] for l in LAB}
        for tk, (g, P) in TB.items():
            s = g.s.values.astype(float)
            if shiftmap is not None: s = np.roll(s, shiftmap[tk])
            if longonly: s = np.where(s > 0, s, 0.)
            act = s != 0; pm = s * g.mva.values; b = pd.cut(pm, EDG, labels=LAB)
            for l in LAB:
                mk = act & (b == l)
                if mk.any(): on[l].append(s[mk] * g.gap.values[mk]); nx[l].append(s[mk] * g.intra.values[mk])
        return {l: (np.concatenate(on[l]) if on[l] else np.array([])) for l in LAB}, {l: (np.concatenate(nx[l]) if nx[l] else np.array([])) for l in LAB}
    on, nx = collect(None)
    for z in range(nsh):
        sm = {tk: int(rng.integers(20, len(TB[tk][0]) - 20)) for tk in TB}; a, b_ = collect(sm)
        for l in LAB: null_on[l].append(a[l].mean() * 1e4 if len(a[l]) else np.nan); null_nx[l].append(b_[l].mean() * 1e4 if len(b_[l]) else np.nan)
    print('bucket    n   overnight bp  (null)   p     | next-day o->c bp  (null)   p')
    for l in LAB:
        if len(on[l]) < 15: continue
        o = on[l].mean() * 1e4; n2 = nx[l].mean() * 1e4; a = np.array(null_on[l]); b_ = np.array(null_nx[l])
        print(f'{l:8s} {len(on[l]):5d}  {o:+8.1f}   ({np.nanmean(a):+6.1f}) {np.nanmean(a >= o):5.2f}  | {n2:+8.1f}   ({np.nanmean(b_):+6.1f}) {np.nanmean(b_ >= n2):5.2f}', flush=True)
c1(False); c1(True)
print('\n#### C2 | move view (no vector) | continuation = sign(day move) x next return (bp); t = mean/se')
rowsall = []
for tk, (g, P) in TB.items(): rowsall.append(g.assign(tk=tk))
G = pd.concat(rowsall); G['sg'] = np.sign(G.mv)
for m in (1.5, 2, 3):
    for nm, sub in (('earnings day', G[G.earn & (G.mva.abs() >= m)]), ('other days', G[~G.earn & (G.mva.abs() >= m)])):
        for side, ss in (('up', sub[sub.mv > 0]), ('down', sub[sub.mv < 0]), ('both', sub)):
            if len(ss) < 15: print(f'|move|>={m} ADR {nm:12s} {side:5s} n={len(ss)} (too few)'); continue
            co = (ss.sg * ss.gap).values * 1e4; cn = (ss.sg * ss.intra).values * 1e4
            print(f'|move|>={m} ADR {nm:12s} {side:5s} n={len(ss):4d} overnight {co.mean():+7.1f} (t {co.mean() / (co.std() / np.sqrt(len(co)) + 1e-9):+5.2f}) | next day o->c {cn.mean():+7.1f} (t {cn.mean() / (cn.std() / np.sqrt(len(cn)) + 1e-9):+5.2f})')
