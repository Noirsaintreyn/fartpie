"""AMD / PO3 and WYCKOFF (spring / upthrust) x VECTOR. Mechanical definitions fixed BEFORE results, no fitting. 8 mega-caps 15m 2019-26 (04:00-16:00 ET), train 2019-21 / test 2022+ (same rules both, nothing selected).
AMD (power of 3): Accumulation = premarket range 04:00-09:29 (>=8 bars). Manipulation = in the first 8 RTH bars (09:30-11:29) price takes out the premarket high (or low) -> sweep. Distribution = first later bar (until 12:00) that CLOSES back inside the premarket range = signal in the opposite direction of the sweep (swept high -> short, swept low -> long). One signal/day. Entry next bar open.
Wyckoff: range = prior 48 bars (extremes excluded current bar); only if range <= 8 x ATR14 (a real trading range, not a trend). Spring = bar low < range low and close back above it (volume >= 1.5x its 20-bar mean) -> long; Upthrust = high > range high and close back below (same volume rule) -> short. Signals 10:30-15:00 only, one per 16 bars. Entry next bar open.
Exit (primary): that day's last bar close. Secondary: vector's next opposite flip open.
Vector overlay: state at the signal bar close AGREES / DISAGREES with the signal direction. Trigger variant: take the vector's own flip (aligned with the signal) within 8 bars after the signal vs all other vector flips (EOD exit).
Null: random entries on the same ticker with the same minute-of-day and same direction, same exit rule (200 draws) = removes time-of-day drift. Classes ALL / HV-pre (prev day range top third of own history)."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_pine import run_state_machine
rng = np.random.default_rng(3)
def load(tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_2018-01-01_*.parquet')[0]; H = pd.read_parquet(f, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.drop(columns='datetime').dropna().sort_index(); H = H[~H.index.duplicated()]
    m = H.index.hour * 60 + H.index.minute; return H[(m >= 240) & (m < 960)]
def prep(tk):
    B = load(tk); n = len(B); reg = run_state_machine(B.assign(datetime=B.index))[0]; m = (B.index.hour * 60 + B.index.minute).values; day = B.index.normalize(); rth = (m >= 570) & (m < 960)
    h, l, c, o, v = B.high.values, B.low.values, B.close.values, B.open.values, B.volume.values
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]]))); atr = pd.Series(tr).rolling(14).mean().values
    d = pd.DataFrame({'h': B.high.where(rth).groupby(day).max(), 'l': B.low.where(rth).groupby(day).min()}).dropna(); rg = d.h - d.l
    pre = rg.shift() / rg.shift().rolling(20).mean().shift(); rk = pre.expanding(min_periods=60).apply(lambda z: (z.iloc[:-1] < z.iloc[-1]).mean(), raw=False); hv = (rk.reindex(day) >= 2 / 3).fillna(False).values
    # last bar index of each day
    dd = pd.Series(np.arange(n)).groupby(day.values).last(); eod = dd.reindex(day).values.astype(int)
    P = dict(n=n, o=o, h=h, l=l, c=c, v=v, reg=reg, m=m, day=day, hv=hv, yr=B.index.year.values, atr=atr, eod=eod, rth=rth)
    # --- AMD signals
    sig = {'AMD': [], 'WYC': []}; dayidx = pd.Series(np.arange(n)).groupby(day.values).apply(lambda s: s.values)
    for dt, ix in dayidx.items():
        mm = m[ix]; pm = ix[(mm >= 240) & (mm < 570)]; rb = ix[(mm >= 570)]
        if len(pm) < 8 or len(rb) < 14: continue
        PH, PL = h[pm].max(), l[pm].min(); swept = 0; fire = None
        for q, i in enumerate(rb[:16]):
            if q < 8:
                if swept == 0:
                    if h[i] > PH and l[i] >= PL: swept = 1
                    elif l[i] < PL and h[i] <= PH: swept = -1
            if swept == 1 and c[i] < PH and (q >= 1 or h[i] > PH): fire = (i, -1); break
            if swept == -1 and c[i] > PL and (q >= 1 or l[i] < PL): fire = (i, 1); break
            if swept != 0 and q >= 8 and False: break
        if fire and fire[0] + 1 < n and eod[fire[0] + 1] == eod[fire[0]]: sig['AMD'].append(fire)
    last = -99; W = 48; vm = pd.Series(v).rolling(20).mean().values
    for i in range(W + 20, n - 2):
        if not (630 <= m[i] <= 900) or i - last < 16 or np.isnan(atr[i]) or np.isnan(vm[i]) or eod[i + 1] != eod[i]: continue
        rh, rl = h[i - W:i].max(), l[i - W:i].min()
        if (rh - rl) > 8 * atr[i] or v[i] < 1.5 * vm[i]: continue
        if l[i] < rl and c[i] > rl: sig['WYC'].append((i, 1)); last = i
        elif h[i] > rh and c[i] < rh: sig['WYC'].append((i, -1)); last = i
    P['sig'] = {k: np.array(v_).reshape(-1, 2) for k, v_ in sig.items()}
    ch = [i for i in range(1, n) if reg[i] != reg[i - 1] and reg[i] != 0]; P['flips'] = np.array([(i, int(reg[i])) for i in ch]).reshape(-1, 2); P['chidx'] = np.array(ch)
    return P
def ret_eod(P, st, d): e = P['o'][st]; x = P['c'][P['eod'][st]]; return d * (x / e - 1)
def next_flip_exit(P, st, d):                                                    # exit at open after next opposite flip (state change away from d)
    ch = P['chidx']; out = np.full(len(st), np.nan)
    for q, (s, dd) in enumerate(zip(st, d)):
        j = ch[(ch >= s) & (P['reg'][ch] == -dd)]
        if len(j) and j[0] + 1 < P['n']: out[q] = dd * (P['o'][j[0] + 1] / P['o'][s] - 1)
    return out
def stats(PS, label, getsig, lo, hi, cls, nul=200, exit_='eod'):
    N = 0; obs = 0.0; nl = np.zeros(nul); win = 0
    for tk, P in PS.items():
        S = getsig(P)
        if not len(S): continue
        st = S[:, 0] + 1; d = S[:, 1]; ok = (P['yr'][st] >= lo) & (P['yr'][st] <= hi) & (P['hv'][st] if cls == 'HV-pre' else True); st, d = st[ok], d[ok]
        if not len(st): continue
        r = ret_eod(P, st, d) if exit_ == 'eod' else next_flip_exit(P, st, d); keep = ~np.isnan(r); r = r[keep]; st, d = st[keep], d[keep]
        if not len(r): continue
        N += len(r); obs += r.sum(); win += (r > 0).sum(); mins = P['m'][st]
        allc = np.where(P['rth'] & (P['yr'] >= lo) & (P['yr'] <= hi) & (P['hv'] if cls == 'HV-pre' else True))[0]; bym = {}
        for z in range(nul):
            cs = np.empty(len(st), int)
            for q in range(len(st)):
                mq = mins[q]
                if mq not in bym: bym[mq] = allc[P['m'][allc] == mq]
                cs[q] = rng.choice(bym[mq]) if len(bym[mq]) else st[q]
            nl[z] += (ret_eod(P, cs, d).sum() if exit_ == 'eod' else np.nansum(next_flip_exit(P, cs, d)))
    if N < 20: print(f'  {label:34s} n={N} (too few)'); return
    o = obs / N * 1e4; nm = nl / N * 1e4; print(f'  {label:34s} n={N:5d} obs {o:+7.1f} bp  null {nm.mean():+7.1f}  p={(nm >= o).mean():.2f}  win {win / N * 100:4.1f}%  (obs-null {o - nm.mean():+6.1f})', flush=True)
TK = ['AAPL', 'AMZN', 'AVGO', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']; PS = {tk: prep(tk) for tk in TK}
for tk, P in PS.items(): print(tk, 'AMD signals', len(P['sig']['AMD']), 'WYC', len(P['sig']['WYC']), flush=True)
def agree(kind, want):
    def g(P):
        S = P['sig'][kind]
        if not len(S): return S
        s = P['reg'][S[:, 0]]; return S[(s == S[:, 1]) if want else (s == -S[:, 1])]
    return g
def alls(kind): return lambda P: P['sig'][kind]
def trigger(kind, want):
    def g(P):
        S = P['sig'][kind]; F = P['flips']; out = []
        if not len(S): return np.zeros((0, 2), int)
        al = np.zeros(P['n'], int)
        for i, d in S: al[i + 1:i + 9] = d
        for i, d in F:
            if i + 1 >= P['n'] - 1 or P['eod'][i + 1] != P['eod'][min(i + 1, P['n'] - 1)] or not (480 <= P['m'][i + 1] < 945): continue
            if (al[i] == d) == want: out.append((i, d))
        return np.array(out).reshape(-1, 2)
    return g
for eraname, (lo, hi) in {'TRAIN 2019-21': (2019, 2021), 'TEST 2022+': (2022, 2026)}.items():
    for cls in ('ALL', 'HV-pre'):
        print(f'\n######## {eraname} | {cls} | exit = day close (null = same minute-of-day random entries, same direction)')
        for kind, nm in (('AMD', 'AMD/PO3 sweep-reclaim'), ('WYC', 'Wyckoff spring/upthrust')):
            stats(PS, f'{nm} alone', alls(kind), lo, hi, cls); stats(PS, f'{nm} + vector AGREES', agree(kind, True), lo, hi, cls); stats(PS, f'{nm} + vector DISAGREES', agree(kind, False), lo, hi, cls)
            stats(PS, f'vector flip within 8b AFTER {kind}', trigger(kind, True), lo, hi, cls); stats(PS, f'vector flip NOT preceded by {kind}', trigger(kind, False), lo, hi, cls)
        if cls == 'ALL':
            print('  -- secondary exit: vector next opposite flip')
            for kind, nm in (('AMD', 'AMD'), ('WYC', 'Wyckoff')): stats(PS, f'{nm} alone (vector exit)', alls(kind), lo, hi, cls, exit_='flip', nul=100); stats(PS, f'{nm} + vector agrees (vector exit)', agree(kind, True), lo, hi, cls, exit_='flip', nul=100)
