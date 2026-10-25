"""Signal lab (Oct 25 2026): is the edge in the vector, or in the universe + stops + slots around it?
Everything else is the live model: premarket+RTH 15m bars, 15m + 1h tracks, sticky 1h stop after +2%, LOCK5/0.5, entries/exits at
RTH opens, no buys after 14:00, S&P point-in-time, options ratio >= 1.5, 6 slots, ratio tie-break, 8% breaker (sim3).
 1 SIG   swap the vector for: Donchian midpoint (20, 55), Supertrend (10, 3), EMA cross (21/55)
 2 PAR   vector settings one at a time: pivot length, break ATR, level ATR, min ratchet move, confirm bars
 3 HOLD  re-enter after a profit-lock exit when price makes a new high in a still-bull trend; pyramid (add a slot once the trade
         moves to the sticky 1h stop); a daily-vector track
 4 ATT   attention measures beyond the ratio: option-volume surge, gap frequency (short interest / open interest: no history here)
 5 ENS   size by how many of the 15m / 1h / 4h / daily vectors agree at entry
Stage 1: python3 xs/signal_lab.py 1 [TK ...]   -> xs/ext/slab/<tk>.parquet (trades) + xs/ext/slab/att_<tk>.parquet (daily attention)
Stage 2: python3 xs/signal_lab.py 2"""
import sys, os, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
from vector_core import compute_atr_series, ema
import vector_core as VC
OUT = 'xs/ext/slab'; C2 = 2e-4
BASE = dict(PIVOT_LEN=VC.PIVOT_LEN, ATR_LEN=VC.ATR_LEN, BREAK_ATR=VC.BREAK_ATR, LEVEL_ATR=VC.LEVEL_ATR, MIN_MOVE_ATR=VC.MIN_MOVE_ATR,
            MIN_BARS_RATCH=VC.MIN_BARS_RATCH, CONFIRM_BARS=VC.CONFIRM_BARS, FAST_LEN=VC.FAST_LEN, SLOW_LEN=VC.SLOW_LEN)

def _piv(v, L, hi):
    n = len(v); out = np.full(n, np.nan)
    for i in range(2 * L, n):
        c = v[i - L]; left = v[i - 2 * L:i - L]; right = v[i - L + 1:i + 1]
        if (c > left.max() and c >= right.max()) if hi else (c < left.min() and c <= right.min()): out[i] = c
    return out

def vec(df, **kw):
    """vector_pine.run_state_machine with overridable settings -> regimes"""
    P = dict(BASE, **kw); L = P['PIVOT_LEN']
    high, low, close = df['high'].values, df['low'].values, df['close'].values; n = len(df)
    atr = compute_atr_series(high, low, close, P['ATR_LEN']); ef, es = ema(close, P['FAST_LEN']), ema(close, P['SLOW_LEN'])
    pl, ph = _piv(low, L, False), _piv(high, L, True); patr = np.r_[np.full(L, np.nan), atr[:-L]]
    LA, BA, MM, MB, CB = P['LEVEL_ATR'], P['BREAK_ATR'], P['MIN_MOVE_ATR'], P['MIN_BARS_RATCH'], P['CONFIRM_BARS']
    regime, level, bc, ac, lr = 0, np.nan, 0, 0, None; regs = np.zeros(n, int)
    for i in range(n):
        a = atr[i]
        if np.isnan(a) or np.isnan(ef[i]) or np.isnan(es[i]) and regime == 0: regs[i] = regime; continue
        cs = pl[i] - patr[i] * LA if not np.isnan(pl[i]) and not np.isnan(patr[i]) else np.nan
        cr = ph[i] + patr[i] * LA if not np.isnan(ph[i]) and not np.isnan(patr[i]) else np.nan
        if regime == 0: regime = 1 if ef[i] >= es[i] else -1; level = low[i] - a * LA if regime == 1 else high[i] + a * LA; lr = i
        sb = regime == 1 and close[i] < level - a * BA; rb = regime == -1 and close[i] > level + a * BA
        if regime == 1: bc, ac = (bc + 1 if sb else 0), 0
        elif regime == -1: ac, bc = (ac + 1 if rb else 0), 0
        if regime == 1 and bc >= CB:
            regime, bc, ac = -1, 0, 0; nr = cr if (not np.isnan(cr) and cr > close[i]) else high[i] + a * LA; level = max(nr, close[i] + a * LA); lr = i
        elif regime == -1 and ac >= CB:
            regime, bc, ac = 1, 0, 0; ns = cs if (not np.isnan(cs) and cs < close[i]) else low[i] - a * LA; level = min(ns, close[i] - a * LA); lr = i
        elif regime == 1 and not np.isnan(cs) and cs < close[i] and cs > level + a * MM and (lr is None or i - lr >= MB): level, lr = cs, i
        elif regime == -1 and not np.isnan(cr) and cr > close[i] and cr < level - a * MM and (lr is None or i - lr >= MB): level, lr = cr, i
        regs[i] = regime
    return regs

def donch(df, N):
    mid = (df.high.rolling(N).max() + df.low.rolling(N).min()) / 2
    return np.where(df.close.values > mid.values, 1, np.where(mid.isna().values, 0, -1))

def supertrend(df, n=10, m=3.0):
    h, l, c = df.high.values, df.low.values, df.close.values; atr = compute_atr_series(h, l, c, n); hl2 = (h + l) / 2
    up, dn = hl2 - m * atr, hl2 + m * atr; st = np.zeros(len(c), int); fu, fd = up.copy(), dn.copy(); d = 1
    for i in range(1, len(c)):
        if np.isnan(atr[i]): continue
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        if d == 1 and c[i] < fu[i]: d = -1
        elif d == -1 and c[i] > fd[i]: d = 1
        st[i] = d
    return st

def emax(df, f=21, s=55):
    return np.where(ema(df.close.values, f) >= ema(df.close.values, s), 1, -1)

SIGS = {'vec': lambda d: vec(d), 'donch20': lambda d: donch(d, 20), 'donch55': lambda d: donch(d, 55), 'supertrend': supertrend, 'ema21_55': emax}
PARS = {'piv5': dict(PIVOT_LEN=5), 'piv12': dict(PIVOT_LEN=12), 'brk0.2': dict(BREAK_ATR=.2), 'brk0.5': dict(BREAK_ATR=.5),
        'lvl0.05': dict(LEVEL_ATR=.05), 'lvl0.3': dict(LEVEL_ATR=.3), 'mm0.5': dict(MIN_MOVE_ATR=.5), 'mm1.2': dict(MIN_MOVE_ATR=1.2),
        'conf2': dict(CONFIRM_BARS=2), 'ema13_34': dict(FAST_LEN=13, SLOW_LEN=34)}

def load(f):
    H = pd.read_parquet(f); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= 240) & (m < 960) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; return H

def agg(H, freq, offset):
    x = H.set_index('dt')[['open', 'high', 'low', 'close']]
    g = x.groupby([x.index.normalize(), pd.Grouper(freq=freq, origin='start_day', offset=offset)])
    r = g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna()
    r.index = r.index.get_level_values(1); r = r.reset_index(); r.columns = ['dt', 'open', 'high', 'low', 'close']; return r

def known(H, r, st, length):
    s = pd.DataFrame({'known': r.dt + pd.Timedelta(length), 'r': st}).sort_values('known')
    return pd.merge_asof(pd.DataFrame({'t': H.dt + pd.Timedelta('15min')}), s, left_on='t', right_on='known', direction='backward').r.fillna(0).values

def states(H, smf):
    reg = np.asarray(smf(H[['dt', 'open', 'high', 'low', 'close']]))
    r = agg(H, '1h', '9h30min'); s1 = known(H, r, np.asarray(smf(r)), '1h')
    return reg, s1

def run(H, reg, s1, h1track, reentry=False, pyramid=False):
    """ah_sticky.run (sticky=True, LOCK5/0.5) + optional re-entry after a lock exit and pyramid add-on rows"""
    o, c, hi, d, m = H.open.values, H.close.values, H.high.values, H.dt.values, H.m.values
    g = [j for j in range(1, len(H)) if 570 <= m[j] < 960]; T, A = [], []; pos = False; rearm = None
    KR = [reg[j - 1] for j in g]; KS = [s1[j - 1] for j in g]
    for p in range(1, len(g)):
        j = g[p]; st_now = KS[p] if h1track else KR[p]
        if not pos:
            sig = (KS[p] == 1 and KS[p - 1] != 1) if h1track else (KR[p] == 1 and KR[p - 1] != 1)
            if reentry and rearm is not None:
                if st_now != 1: rearm = None
                elif c[j - 1] > rearm: sig = True; rearm = None
            if sig: pos, e, px, b1, peak, h1, addj = True, j, o[j], False, o[j], False, None
            continue
        if j == e: continue
        k = g[p - 1]
        if k >= e: peak = max(peak, hi[k])
        b1 = b1 or KS[p] == 1; now = c[j - 1] / px - 1 >= .02 and b1; h1 = h1 or now
        if h1 and addj is None: addj = j
        ex = KS[p] == -1 if h1track else ((KS[p] == -1) if h1 else (KR[p] == -1)); pg = peak / px - 1; lock = False
        if pg >= .05 and (c[j - 1] / px - 1) <= .5 * pg: ex, lock = True, not ex
        if ex:
            T.append((d[e], d[j], o[j] / px - 1 - C2))
            if pyramid and addj is not None and addj < j: A.append((d[addj], d[j], o[j] / o[addj] - 1 - C2))
            pos = False; rearm = peak if (lock and st_now == 1) else None
    return T, A

def daily_track(H, smf):
    R = H[H.m >= 570]; D = R.groupby(R.dt.dt.normalize()).agg(dt=('dt', 'first'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index(drop=True)
    s = np.asarray(smf(D)); T = []; pos = False
    for i in range(1, len(D) - 1):
        if not pos and s[i] == 1 and s[i - 1] != 1: pos, e = True, i + 1
        elif pos and s[i] == -1: T.append((D.dt.iloc[e], D.dt.iloc[i + 1], D.open.iloc[i + 1] / D.open.iloc[e] - 1 - C2)); pos = False
    return T, D, s

if sys.argv[1] == '1':
    os.makedirs(OUT, exist_ok=True)
    W = pd.read_parquet('xs/ext/wide_T.parquet'); core = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk)); att = sorted(set(W[W.sp].tk))
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_clean/*_15m_2018-01-01_*.parquet'))}
    exec(open('xs/wide_overlays.py').read().split("T = pd.concat([pd.read_parquet(f) for f in (glob")[0].split("if sys.argv[1] == '1':")[0].replace("fs = ", "_fs0 = "))
    exec(open('xs/wide_overlays.py').read().split("old = pd.read_csv")[1].split("T = pd.concat([pd.read_parquet(f) for f in (glob")[0].join(["old = pd.read_csv", ""]))
    for tk in (sys.argv[2:] or att):
        if tk not in fs: continue
        fa = f'{OUT}/att_{tk}.parquet'
        if not os.path.exists(fa):                                                # daily attention data for every S&P name in wide_T
            H = load(fs[tk]); R = H[H.m >= 570]; dd = R.dt.dt.normalize()
            sh = R.groupby(dd).volume.sum(); op = R.groupby(dd).open.first(); cl = R.groupby(dd).close.last()
            o_ = opts(tk); J = pd.concat([o_.rename('c'), sh.rename('sh')], axis=1).dropna(); J = J[J.sh > 0]
            rat = (J.c * 100 / J.sh).rolling(252, min_periods=60).median().shift(1)
            surge = (o_.rolling(5).mean() / o_.rolling(252, min_periods=60).median()).shift(1)
            gap = ((op / cl.shift(1) - 1).abs() > .02).astype(float).rolling(60).mean().shift(1)
            pd.DataFrame({'ratio': rat, 'surge': surge.reindex(rat.index), 'gapf': gap.reindex(rat.index)}).to_parquet(fa)
        if tk not in core: continue
        fo = f'{OUT}/{tk}.parquet'
        if os.path.exists(fo): continue
        H = load(fs[tk]); rows = []
        for nm, smf in list(SIGS.items()) + [(k, (lambda kw: (lambda d: vec(d, **kw)))(v)) for k, v in PARS.items()]:
            reg, s1 = states(H, smf)
            for h1t in (False, True):
                T, A = run(H, reg, s1, h1t, pyramid=nm == 'vec')
                rows += [(tk, nm, h1t, 'trade') + t for t in T]
                if nm == 'vec':
                    rows += [(tk, 'pyr_add', h1t, 'add') + t for t in A]
                    rows += [(tk, 'reentry', h1t, 'trade') + t for t in run(H, reg, s1, h1t, reentry=True)[0]]
            if nm == 'vec':                                                       # ensemble features: 1h / 4h / daily states at entry
                r4 = agg(H[H.m >= 570], '4h', '9h30min'); s4 = known(H, r4, vec(r4), '4h')
                DT, D, sd = daily_track(H, vec); rows += [(tk, 'daily', False, 'trade') + t for t in DT]
                sdk = pd.Series(sd, index=D.dt.dt.normalize()).shift(1)            # daily state known before the session
                E = pd.DataFrame({'dt': H.dt, 's1': s1, 's4': s4, 'sd': sdk.reindex(H.dt.dt.normalize()).values})
                E.to_parquet(f'{OUT}/ens_{tk}.parquet')
        pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'kind', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
    raise SystemExit

# ================================================================ stage 2
exec(open('xs/batch_oct25.py').read().split("PART = os.getenv('PART', 'ALL')")[0])     # sim3, show, head, live T
LIVE = T.copy()
M = json.load(open('sp500_pit_membership.json')); mem = {k: {'META' if t == 'FB' else t for t in v} for k, v in M.items()}; months = sorted(mem)
def member(tk, d):
    k = f'{d:%Y-%m}'; k = k if k in mem else (months[-1] if k > months[-1] else months[0]); return tk in mem[k]
ATT = {f.split('att_')[1][:-8]: pd.read_parquet(f) for f in glob.glob(f'{OUT}/att_*.parquet')}
S = pd.concat(pd.read_parquet(f) for f in glob.glob(f'{OUT}/[A-Z]*.parquet')); S['entry'] = pd.to_datetime(S.entry); S['exit'] = pd.to_datetime(S.exit)
S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)].reset_index(drop=True)
def look(tk, col, d):
    a = ATT.get(tk)
    return np.nan if a is None else a[col].asof(d)
def attach(D, cols):
    D = D.reset_index(drop=True)
    for c in cols: D[c] = np.nan
    for tk, ix in D.groupby('tk').groups.items():
        a = ATT.get(tk)
        if a is None: continue
        a = a[~a.index.duplicated()].sort_index(); d = D.loc[ix, 'entry'].dt.normalize()
        for c in cols: D.loc[ix, c] = a[c].reindex(d.values, method='ffill').values
    return D
S = attach(S, ['ratio']); mon = {}
_W = pd.read_parquet('xs/ext/wide_T.parquet'); SPM = _W.groupby([_W.tk, _W.entry.dt.strftime('%Y-%m')]).sp.first().to_dict()   # same membership as the live T
S['sp'] = [SPM.get((t, f'{e:%Y-%m}'), mon.setdefault((t, f'{e:%Y-%m}'), member(t, e))) for t, e in zip(S.tk, S.entry)]
S = S[S.sp & (S.ratio >= 1.5)]
def V(v): return S[S['var'] == v].copy()
base = V('vec'); print(f"reproduction check: lab vec trades {len(base)} vs live T {len(LIVE)} (avg {base.ret.mean():+.3%} vs {LIVE.ret.mean():+.3%})")
PART2 = os.getenv('PART', 'ALL')
if PART2 in ('ALL', 'SIG'):
    head('1 SIG  same model, different trend signal'); show('lab vector (reproduction)', base)
    for v in ('donch20', 'donch55', 'supertrend', 'ema21_55'): z = V(v); show(f'{v}  ({len(z)} trades, avg {z.ret.mean():+.2%})', z)
if PART2 in ('ALL', 'PAR'):
    head('2 PAR  vector settings, one at a time')
    for v in PARS: z = V(v); show(f'{v}  ({len(z)} trades, avg {z.ret.mean():+.2%})', z)
if PART2 in ('ALL', 'HOLD'):
    head('3 HOLD  re-entry, pyramiding, daily track'); show('lab vector', base)
    show('re-enter after a lock exit on a new high', V('reentry'))
    show('pyramid: add a slot when the 1h stop takes over', pd.concat([base, V('pyr_add')]))
    d = V('daily'); d['h1'] = True; show(f'+ daily-vector track ({len(d)} trades, avg {d.ret.mean():+.2%})', pd.concat([base, d]))
if PART2 in ('ALL', 'ATT'):
    head('4 ATT  attention beyond the options ratio')
    W = pd.read_parquet('xs/ext/wide_T.parquet'); W = W[W.sp & (W.ratio >= 1.0) & (W.entry >= '2019-01-01')].copy()
    W = attach(W.drop(columns=['surge', 'gapf'], errors='ignore'), ['surge', 'gapf'])
    core = W[W.ratio >= 1.5]; edge = W[(W.ratio < 1.5)]; show('live universe (ratio >= 1.5)', core)
    for q in (.8, .9):
        th = edge.surge.quantile(q); show(f'+ ratio 1.0-1.5 names with an option-volume surge (top {1 - q:.0%})', pd.concat([core, edge[edge.surge >= th]]))
        th = edge.gapf.quantile(q); show(f'+ ratio 1.0-1.5 names that gap often (top {1 - q:.0%})', pd.concat([core, edge[edge.gapf >= th]]))
    Z = core.copy(); Z['k'] = Z.groupby('entry').ratio.rank(pct=True) + Z.groupby('entry').surge.rank(pct=True).fillna(.5); show('tie-break: ratio rank + surge rank', Z, key='k')
    Z = core.copy(); Z['k'] = Z.groupby('entry').ratio.rank(pct=True) + Z.groupby('entry').gapf.rank(pct=True).fillna(.5); show('tie-break: ratio rank + gap-frequency rank', Z, key='k')
if PART2 in ('ALL', 'ENS'):
    head('5 ENS  size by how many of 15m / 1h / 4h / daily agree')
    EN = {f.split('ens_')[1][:-8]: pd.read_parquet(f).set_index('dt') for f in glob.glob(f'{OUT}/ens_*.parquet')}
    B = base.reset_index(drop=True); B['k'] = np.nan
    for tk, ix in B.groupby('tk').groups.items():
        x = EN.get(tk)
        if x is None: continue
        x = x[~x.index.duplicated()].sort_index(); k = 1 + (x.s1 == 1).astype(int) + (x.s4 == 1).astype(int) + (x.sd == 1).astype(int)
        B.loc[ix, 'k'] = k.reindex((B.loc[ix, 'entry'] - pd.Timedelta('1min')).values, method='ffill').values
    print(B.groupby('k').ret.agg(['mean', 'count']).assign(mean=lambda x: x['mean'] * 100).round(3).to_string())
    for nm, f in (('linear: k / 2.5 (0.5x .. 1.5x, average ~1x)', lambda k: k / 2.5), ('only 3-4 agree get 1.25x, else 0.75x', lambda k: np.where(k >= 3, 1.25, .75)), ('skip if fewer than 2 agree', None)):
        if f is None: show(nm, B[B.k >= 2]); continue
        B['w'] = f(B.k.fillna(2)); show(nm, B, size='w')
