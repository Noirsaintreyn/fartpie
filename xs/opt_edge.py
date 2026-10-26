"""What in the options ratio carries the edge? (Oct 26 2026). Live rules everywhere (ahx_clean trades, sticky, LOCK, 6 slots, 8% breaker).
Daily measures per name, all known at the prior close (shifted 1 day):
  r252  live: median over 252 days of option contracts x100 / LSE RTH shares        r63 / r21: same over 63 / 21 days
  rtrend  r21 / r252 (attention rising)        prem  median252 of option premium / stock dollar volume
  callr   median252 of CALL contracts x100 / shares        cshare  median63 of call share of option volume
  surge   5-day option volume / 252-day median        olev  log median252 of option contracts (size of the options market)
Stage 1 TK..: xs/ext/optedge/<tk>.parquet   Stage 2: deciles, top-N universes per measure, tie-breaks, combos."""
import sys, os, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
OUT = 'xs/ext/optedge'
if sys.argv[1] == '1':
    os.makedirs(OUT, exist_ok=True)
    old = pd.read_csv('xs/ext/optvol_ratio.csv', usecols=['tk', 'date', 'opt_contracts', 'opt_premium', 'call_c'], parse_dates=['date'])
    for tk in sys.argv[2:]:
        fo = f'{OUT}/{tk}.parquet'; fl = glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet'); ft = f'data/theta_vol/{tk}.csv'
        if os.path.exists(fo) or not fl or not os.path.exists(ft): continue
        try:
            p = [old[old.tk == tk][['date', 'opt_contracts', 'opt_premium', 'call_c']]]
            v = pd.read_csv(ft).dropna(subset=['date']); v['date'] = pd.to_datetime(v.date); p.append(v[['date', 'opt_contracts', 'opt_premium', 'call_c']])
            O = pd.concat(p).drop_duplicates('date', keep='last').set_index('date').sort_index()
            B = pd.read_parquet(fl[0], columns=['datetime', 'close', 'volume']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
            m = B.dt.dt.hour * 60 + B.dt.dt.minute; B = B[(m >= 570) & (m < 960)]; dd = B.dt.dt.normalize()
            sh = B.groupby(dd).volume.sum(); dv = (B.close * B.volume).groupby(dd).sum()
            J = pd.concat([O, sh.rename('sh'), dv.rename('dv')], axis=1).dropna(subset=['opt_contracts', 'sh']); J = J[J.sh > 0]
            r = J.opt_contracts * 100 / J.sh
            D = pd.DataFrame({'r252': r.rolling(252, min_periods=60).median(), 'r63': r.rolling(63, min_periods=30).median(), 'r21': r.rolling(21, min_periods=15).median(),
                              'prem': (J.opt_premium / J.dv).rolling(252, min_periods=60).median(), 'callr': (J.call_c * 100 / J.sh).rolling(252, min_periods=60).median(),
                              'cshare': (J.call_c / J.opt_contracts).rolling(63, min_periods=30).median(),
                              'surge': J.opt_contracts.rolling(5).mean() / J.opt_contracts.rolling(252, min_periods=60).median(),
                              'olev': np.log(J.opt_contracts.rolling(252, min_periods=60).median())})
            D['rtrend'] = D.r21 / D.r252
            D.shift(1).to_parquet(fo); print(tk, end=' ', flush=True)
        except Exception as e: print(f'\n{tk} ERR {str(e)[:60]}', flush=True)
    raise SystemExit
# ---------------------------------------------------------------- stage 2
exec(open('xs/batch_oct25.py').read().split("PART = os.getenv('PART', 'ALL')")[0])     # sim3, show, head, live T
M = json.load(open('sp500_pit_membership.json')); mem = {k: {'META' if t == 'FB' else t for t in v} for k, v in M.items()}; months = sorted(mem)
BAD = set(open('xs/ext/bad_data_names.txt').read().split())
MEAS = ['r252', 'r63', 'r21', 'rtrend', 'prem', 'callr', 'cshare', 'surge', 'olev']
A = []
for f in glob.glob(f'{OUT}/*.parquet'):
    tk = os.path.basename(f)[:-8]; fx = f'xs/ext/ahx_clean/{tk}.parquet'
    if tk in BAD or not os.path.exists(fx): continue
    x = pd.read_parquet(fx); x = x[x['var'] == 'base+sticky'].copy(); x['entry'] = pd.to_datetime(x.entry); x['exit'] = pd.to_datetime(x.exit)
    x = x[(x.entry >= '2019-01-01') & ((x.entry.dt.hour * 60 + x.entry.dt.minute) < 840)]
    if not len(x): continue
    D = pd.read_parquet(f); D = D[~D.index.duplicated()].sort_index(); d = x.entry.dt.normalize().values
    for c in MEAS: x[c] = D[c].reindex(d, method='ffill').values
    mm = {}; x['sp'] = [mm.setdefault(f'{e:%Y-%m}', tk in mem.get(f'{e:%Y-%m}', mem[months[-1]] if f'{e:%Y-%m}' > months[-1] else mem[months[0]])) for e in x.entry]
    A.append(x)
A = pd.concat(A, ignore_index=True); A = A[A.sp & A.r252.notna()].copy(); A['ratio'] = A.r252
print(f'{A.tk.nunique()} S&P names with options data, {len(A):,} trades')
L = A[A.r252 >= 1.5]; print(f'live universe rebuilt: {len(L):,} trades (live T {len(T):,})')
print('\nAVERAGE TRADE (%) and P(>= +3%) BY DECILE of each measure, all S&P trades (decile 10 = highest)')
for c in MEAS:
    q = pd.qcut(A[c].rank(method='first'), 10, labels=False) + 1; g = A.groupby(q).ret
    print(f'  {c:7s} avg ' + ' '.join(f'{v * 100:+.2f}' for v in g.mean()) + '   P3 ' + ' '.join(f'{v:.2f}' for v in g.apply(lambda r: (r >= .03).mean())))
head('UNIVERSE = each day the top N names by a measure (N = live median eligible count)')
PAN = {c: {} for c in MEAS}
for tk in A.tk.unique():
    D = pd.read_parquet(f'{OUT}/{tk}.parquet'); D = D[~D.index.duplicated()].sort_index()
    for c in MEAS: PAN[c][tk] = D[c]
PAN = {c: pd.DataFrame(v).sort_index() for c, v in PAN.items()}
NL = int((PAN['r252'] >= 1.5).sum(axis=1).loc['2019':].median())
print(f'top-N uses N = {NL} (median number of names with r252 >= 1.5 per day)')
DAY = {c: PAN[c].rank(axis=1, ascending=False) for c in MEAS}
for c in MEAS:
    R = DAY[c].stack(); key = pd.MultiIndex.from_arrays([A.entry.dt.normalize(), A.tk]); rk = R.reindex(key).values
    Z = A[rk <= NL].copy(); Z['ratio'] = Z[c]; show(f'top {NL} by {c} ({len(Z):,} trades)', Z)
head('LIVE UNIVERSE (r252 >= 1.5), different tie-break')
for c in MEAS:
    Z = L.copy(); Z['ratio'] = Z[c].fillna(Z[c].median()); show(f'tie-break by {c}', Z)
head('COMBOS on the live universe')
for c, q in (('cshare', .5), ('rtrend', .5), ('surge', .5), ('prem', .5)):
    th = L[c].quantile(q)
    show(f'live + {c} above its median', L[L[c] >= th]); show(f'live + {c} below its median', L[L[c] < th])
