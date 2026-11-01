"""Nov 1 2026: (1) breakout quality / liquidity sweeps, (2) the wave (base) before the flip, (3) Zarattini noise band - per trade, live engine trades.
Flip bar = the last 15m bar (1h bar for the 1h track) closed at or before the entry. Vector as live (04:00-16:00 bars).
 clv     close location in the flip bar (1 = closed at the high)          uwick   upper wick share of the bar
 beyond  how far the close got past the flip trigger, in ATR              relvol  flip-bar volume vs the median of the last 130 bars
 dur     bars in the bear phase before the flip (the base)                width   base high-low in ATR       durpct  dur vs this stock's past bases
 nb      move from today's open at the flip, in units of the average |move from open| at that time of day over 14 days (> 1 = outside the noise band)
 rev     the entry bar closed back below the trigger (failed break) -> test: exit at the next open"""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
import sys; sys.path.insert(0, '/home/claude/fartpie'); cwd = os.getcwd(); os.chdir('/home/claude/fartpie'); import live_screener as LS; os.chdir(cwd)
L = pd.read_parquet('/tmp/claude-0/ml/mtf_states.parquet').sort_values('entry').reset_index(drop=True)
def load(f):
    H = pd.read_parquet(f); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; return H[(m >= 240) & (m < 960) & (H.volume > 0) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
def feats(H, ln):
    V = LS.vec(H[['dt', 'open', 'high', 'low', 'close', 'volume']]); reg = V.reg.values; o, h, l, c, v = (H[k].values for k in ('open', 'high', 'low', 'close', 'volume'))
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[c[0], c[:-1]]), abs(l - np.r_[c[0], c[:-1]]))); atr = pd.Series(tr).ewm(alpha=1 / 14).mean().values
    trig = V.trig.values; rng = np.where(h > l, h - l, np.nan)
    start = np.zeros(len(H), int); s = 0
    for i in range(1, len(H)):
        if reg[i] != reg[i - 1]: s = i
        start[i] = s
    F = pd.DataFrame({'close_t': H.dt + ln, 'clv': (c - l) / rng, 'uwick': (h - np.maximum(o, c)) / rng,
                      'beyond': (c - np.r_[np.nan, trig[:-1]]) / atr, 'relvol': v / pd.Series(v).rolling(130, min_periods=40).median().shift(1).values})
    ps = np.r_[0, start[:-1]]                                                    # the phase that just ended at a flip bar = previous bar's phase
    idx = np.arange(len(H)); dur = idx - ps
    hh = pd.Series(h); ll = pd.Series(l); width = np.full(len(H), np.nan); durpct = np.full(len(H), np.nan); past = []
    for i in np.where((reg == 1) & (np.r_[0, reg[:-1]] == -1))[0]:            # bull flips
        a = ps[i]; width[i] = (h[a:i].max() - l[a:i].min()) / atr[i] if i > a else np.nan
        durpct[i] = (np.array(past) < dur[i]).mean() if len(past) >= 20 else np.nan; past.append(dur[i])
    F['dur'], F['width'], F['durpct'] = dur, width, durpct
    F['trig_prev'] = np.r_[np.nan, trig[:-1]]
    return F, V
rows = []
for tk, g in L.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = load(f[0]); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    F15, _ = feats(H, pd.Timedelta('15min')); H1 = LS.resample_1h(H); F1, _ = feats(H1, pd.Timedelta('1h'))
    R = H[(m >= 570)].copy(); R['d'] = R.dt.dt.normalize(); R['tod'] = R.dt.dt.hour * 60 + R.dt.dt.minute
    R['mv'] = (R.close / R.groupby('d').open.transform('first') - 1).abs()
    sig = R.pivot_table(index='d', columns='tod', values='mv').rolling(14, min_periods=10).mean().shift(1)   # prior 14 days, per time of day
    dop = R.groupby('d').open.first(); cl = H.set_index('dt').close; op = H.set_index('dt').open
    for e, h1, i0 in zip(g.entry, g.h1, g.index):
        Fx = F1 if h1 else F15; k = Fx.close_t.searchsorted(e, side='right') - 1
        if k < 1: continue
        r = Fx.iloc[k].to_dict(); r['i'] = i0
        tod = int(((e - pd.Timedelta('15min')).hour) * 60 + (e - pd.Timedelta('15min')).minute); d = e.normalize()
        try: r['nb'] = (cl[e - pd.Timedelta('15min')] / dop[d] - 1) / sig.loc[d, tod] if tod >= 570 else np.nan
        except KeyError: r['nb'] = np.nan
        try: eb_close = cl[e]; nxt = op[op.index > e].iloc[0]; nxt_t = op[op.index > e].index[0]
        except (KeyError, IndexError): eb_close = nxt = nxt_t = np.nan
        r['rev'] = bool(eb_close < r['trig_prev']) if r['trig_prev'] == r['trig_prev'] and eb_close == eb_close else False
        r['rev_exit'], r['rev_ret'] = nxt_t, (nxt / op.get(e, np.nan) - 1) if nxt == nxt else np.nan
        rows.append(r)
    print(tk, end=' ', flush=True)
F = pd.DataFrame(rows).set_index('i'); L = L.join(F.drop(columns=['close_t'])); L.to_parquet('/tmp/claude-0/ml/flipq.parquet')
TK = taken(L).merge(L, on=['tk', 'entry', 'exit', 'ret'], how='left'); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print(f'\n\nA) trades the live book takes ({len(TK)}): avg trade % by quintile of each feature (q1 = lowest)')
for c in ('clv', 'uwick', 'beyond', 'relvol', 'dur', 'width', 'durpct', 'nb'):
    q = pd.qcut(TK[c].rank(method='first'), 5, labels=[f'q{i}' for i in range(1, 6)])
    t = (TK.groupby([q, 'h']).ret.mean() * 100).unstack('h').round(2)
    print(f'  {c:<7} ' + '  '.join(f'{i}: {t.loc[i, "2019-21"]:+.2f}/{t.loc[i, "2022-26"]:+.2f}' for i in t.index) + f'   (cover {TK[c].notna().mean():.0%})')
t = TK.groupby(['rev', 'h']).ret.agg(['size', 'mean']).unstack('h'); print('  failed break (entry bar closed back under the trigger):'); print((t * [1, 1, 100, 100]).round(2).to_string())
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def sh(nm, Z): print(f'  {nm:<40}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
print('\nB) the book (yearly return / max DD)'); sh('live', L)
pr = lambda s: s.groupby(L.entry.dt.year).rank(pct=True)
for c in ('clv', 'beyond', 'relvol', 'dur', 'width', 'durpct', 'nb'):
    for sgn, nm in ((1, 'high'), (-1, 'low')):
        Z = L.copy(); Z['ratio'] = (pr(L.ratio) + sgn * pr(L[c]).fillna(.5)).values; sh(f'ratio + {c} ({nm} first)', Z)
Z = L.copy(); k = Z.rev & Z.rev_ret.notna(); Z.loc[k, 'exit'], Z.loc[k, 'ret'] = Z.rev_exit[k], Z.rev_ret[k]; sh(f'exit at the next open after a failed break ({k.mean():.0%} of trades)', Z)
sh('skip flips with nb < 1 (inside the noise band)', L[~(L.nb < 1)])
sh('skip flips with clv < 0.5 (closed in the lower half)', L[~(L.clv < .5)])
