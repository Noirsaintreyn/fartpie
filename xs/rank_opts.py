"""Oct 28 2026: options-based tie-breaks beyond the ratio level (universe stays ratio >= 1.5). All known at the prior close.
 A (23 names with daily call/put volume, opt_summary 2018+; book re-run on just those names):
   surge   5d options/share volume vs its own 252d median (ratio CHANGE)     cshare  5d call share of contracts
 B (all names, opt_features IV 2019-02+):
   ivr     30d ATM IV vs its 252d median     div5  5d change in ATM IV     skew  put25 - call25 IV (lower = calls bid)     dskew 5d change in skew"""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('tm/*.parquet')); L = L[L['var'] == 'vec'].reset_index(drop=True); L['d'] = L.entry.dt.normalize()
EXT = '/mnt/user-data/uploads/fartpie/xs/ext'
def shares(tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    B = pd.read_parquet(f[0], columns=['datetime', 'volume']); t = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = t.dt.hour * 60 + t.dt.minute; k = (m >= 570) & (m < 960); return B.volume[k].groupby(t[k].dt.normalize()).sum()
FA = []
for f in glob.glob(f'{EXT}/opt_summary/*.csv'):
    tk = os.path.basename(f)[:-4]
    if tk not in set(L.tk): continue
    o = pd.read_csv(f, parse_dates=['date']).set_index('date').sort_index(); sh = shares(tk).reindex(o.index)
    c = o.call_vol + o.put_vol; r5 = c.rolling(5).sum() / sh.rolling(5).sum(); rd = c / sh
    X = pd.DataFrame({'surge': r5 / rd.rolling(252).median(), 'cshare': o.call_vol.rolling(5).sum() / c.rolling(5).sum()}).shift(1)   # prior close
    X['tk'] = tk; FA.append(X.reset_index().rename(columns={'date': 'd'}))
FA = pd.concat(FA)
Fo = pd.read_csv(f'{EXT}/opt_features.csv', parse_dates=['date']).rename(columns={'sym': 'tk'}).sort_values(['tk', 'date'])
g = Fo.groupby('tk'); iv = (Fo.iv_c50 + Fo.iv_p50) / 2; sk = Fo.iv_p25 - Fo.iv_c25
Fo['ivr'] = iv / iv.groupby(Fo.tk).transform(lambda s: s.rolling(252, min_periods=120).median()); Fo['div5'] = iv / iv.groupby(Fo.tk).shift(5) - 1
Fo['skew'] = sk; Fo['dskew'] = sk - sk.groupby(Fo.tk).shift(5)
for c in ('ivr', 'div5', 'skew', 'dskew'): Fo[c] = Fo.groupby('tk')[c].shift(1)
Fo = Fo.rename(columns={'date': 'd'})[['tk', 'd', 'ivr', 'div5', 'skew', 'dskew']]
def attach(Z, F, cols):
    Z = Z.sort_values('d'); Z['d'] = Z.d.astype('datetime64[ns]'); F = F.copy(); F['d'] = F.d.astype('datetime64[ns]'); F['tk'] = F.tk.astype(str); Z['tk'] = Z.tk.astype(str)
    return pd.merge_asof(Z, F.sort_values('d'), on='d', by='tk', direction='backward', tolerance=pd.Timedelta(days=7))
P = [(np.datetime64('2019-03-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
pr = lambda s, Z: s.groupby(Z.entry.dt.year).rank(pct=True)
def show(nm, Z, key):
    Y = Z.copy(); Y['ratio'] = key.fillna(key.median()).values
    print(f'  {nm:<30}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (sim(Y, a, b) for a, b in P)), flush=True)
def block(Z, feats):
    show('options ratio (live rank)', Z, Z.ratio)
    rng = np.random.default_rng(1); R = []
    for k in range(20):
        Y = Z.copy(); Y['ratio'] = rng.random(len(Y)); R.append([sim(Y, a, b)[0] for a, b in P])
    R = np.array(R); print(f'  {"random x20 median":<30}{np.median(R[:, 0]):>+8.1%}           {np.median(R[:, 1]):>+8.1%}    2022-26 range {R[:, 1].min():+.1%}..{R[:, 1].max():+.1%}')
    for f in feats:
        show(f'{f} high first', Z, Z[f]); show(f'{f} low first', Z, -Z[f])
        show(f'ratio + {f} (high)', Z, pr(Z.ratio, Z) + pr(Z[f], Z)); show(f'ratio + {f} (low)', Z, pr(Z.ratio, Z) - pr(Z[f], Z))
print('A) 23 names with daily call/put volume (book on these names only), 2019-03..2021 / 2022-26')
ZA = attach(L[L.tk.isin(FA.tk.unique())], FA, ['surge', 'cshare']).sort_values('entry').reset_index(drop=True)
print(f'  coverage surge {ZA.surge.notna().mean():.0%}'); block(ZA, ['surge', 'cshare'])
print('\nB) all names, IV features')
ZB = attach(L, Fo, ['ivr', 'div5', 'skew', 'dskew']).sort_values('entry').reset_index(drop=True)
print(f'  coverage ivr {ZB[ZB.entry >= "2019-03-01"].ivr.notna().mean():.0%}'); block(ZB, ['ivr', 'div5', 'skew', 'dskew'])
