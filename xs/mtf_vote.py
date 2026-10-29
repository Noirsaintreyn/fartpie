"""Oct 29 2026: the vector on every timeframe / session at entry, for the trades the live book takes and as the slot ranker.
States known at entry (last bar closed at or before the entry bar's open):
 v15r 15m RTH-only  v15x 15m incl. pre/after hours (04:00-20:00)  v5 5m (34 names)  v1h 1h  v4h 4h (2 RTH bars/day)  vD daily RTH
vote = number of bull states among v1h, v4h, vD, v15x (+ v5 where available)."""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
import sys; sys.path.insert(0, '/home/claude/fartpie'); cwd = os.getcwd(); os.chdir('/home/claude/fartpie'); import live_screener as LS; os.chdir(cwd)
L = prep(glob.glob('tm/*.parquet')); L = L[L['var'] == 'vec'].sort_values('entry').reset_index(drop=True)
def et(df): df = df.copy(); df['dt'] = pd.to_datetime(df.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); return df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
def agg(H, rule, off):
    x = H.set_index('dt')[['open', 'high', 'low', 'close', 'volume']]
    r = x.groupby([x.index.normalize(), pd.Grouper(freq=rule, origin='start_day', offset=off)]).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum')).dropna()
    r.index = r.index.get_level_values(1); return r.reset_index().rename(columns={'index': 'dt'})
def states(H, ln):                      # Series of vector state indexed by the bar's CLOSE time
    V = LS.vec(H[['dt', 'open', 'high', 'low', 'close', 'volume']]); return pd.Series(V.reg.values, index=V.dt + ln)
out = []
for tk, g in L.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    X = et(pd.read_parquet(f[0])); m = X.dt.dt.hour * 60 + X.dt.dt.minute
    R = X[(m >= 570) & (m < 960)].reset_index(drop=True); Xx = X[(m >= 240) & (m < 1200)].reset_index(drop=True)
    S = {'v15r': states(R, pd.Timedelta('15min')), 'v15x': states(Xx, pd.Timedelta('15min')),
         'v1h': states(agg(R, '1h', '9h30min'), pd.Timedelta('1h')), 'v4h': states(agg(R, '4h', '9h30min'), pd.Timedelta('4h'))}
    D = R.set_index('dt').resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
    S['vD'] = states(D, pd.Timedelta('16h'))
    f5 = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_5m_*.parquet')
    if f5:
        Y = pd.read_parquet(f5[0]); Y['dt'] = pd.to_datetime(Y.datetime).dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
        Y = Y.drop_duplicates('dt').sort_values('dt').reset_index(drop=True); my = Y.dt.dt.hour * 60 + Y.dt.dt.minute; S['v5'] = states(Y[(my >= 240) & (my < 1200)], pd.Timedelta('5min'))
    x = g[['entry']].copy()
    for k, s in S.items():
        s = s[~s.index.duplicated(keep='last')].sort_index(); k_ = s.index.searchsorted(x.entry.values, side='right') - 1
        x[k] = np.where(k_ >= 0, s.values[np.clip(k_, 0, None)], np.nan)
    out.append(x.drop(columns='entry')); print(tk, end=' ', flush=True)
L = L.join(pd.concat(out)); L.to_parquet('/tmp/claude-0/ml/mtf_states.parquet')
B = ['v1h', 'v4h', 'vD', 'v15x']; L['vote'] = (L[B] == 1).sum(axis=1) + (L['v5'] == 1).fillna(False).astype(int)
L['vote4'] = (L[B] == 1).sum(axis=1)
TK = taken(L); TK = TK.merge(L[['tk', 'entry'] + B + ['v5', 'v15r', 'vote', 'vote4']], on=['tk', 'entry'], how='left'); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print(f'\n\nA) trades the live book takes ({len(TK)}): avg trade % (n) by each state at entry')
for k in B + ['v5', 'v15r']:
    t = TK.groupby([k, 'h']).ret.agg(['size', 'mean']).unstack('h'); print(f'  {k}:  ' + '   '.join(f'{"bull" if i == 1 else "bear"}: ' + ' / '.join(f'{t.loc[i, ("mean", h)] * 100:+.2f}% ({int(t.loc[i, ("size", h)])})' for h in ('2019-21', '2022-26')) for i in t.index if i in (1, -1)))
print('\n  by vote (1h, 4h, daily, ext-hours 15m bull count; 0-4):'); print(TK.groupby(['vote4', 'h']).ret.agg(n='size', avg=lambda v: v.mean() * 100, win=lambda v: (v > 0).mean() * 100).round(2).unstack('h').to_string())
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def sh(nm, Z): print(f'  {nm:<38}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
print('\nB) the book with the vote as tie-break / filter (yearly return / max DD)')
sh('live (ratio)', L)
pr = lambda s: s.groupby(L.entry.dt.year).rank(pct=True)
for nm, key in (('vote4 first, then ratio', L.vote4 * 100 + pr(L.ratio)), ('vote first (incl 5m), then ratio', L.vote * 100 + pr(L.ratio)), ('ratio + vote4 (rank avg)', pr(L.ratio) + pr(L.vote4)),
                ('daily bull first, then ratio', (L.vD == 1) * 100 + pr(L.ratio)), ('4h bull first, then ratio', (L.v4h == 1) * 100 + pr(L.ratio)), ('ext-hours 15m bull first', (L.v15x == 1) * 100 + pr(L.ratio))):
    Z = L.copy(); Z['ratio'] = key.values; sh(nm, Z)
for k in (2, 3, 4):
    sh(f'filter: vote4 >= {k}', L[L.vote4 >= k])
sh('filter: skip daily bear', L[L.vD != -1])
