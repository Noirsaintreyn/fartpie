import numpy as np, pandas as pd, glob, os
exec(open('/tmp/claude-0/ml/stopbook.py').read().split("P = [(np.datetime64")[0].split("S = pd.concat")[0])
exec("from collections import deque\n" + open('/tmp/claude-0/ml/stopbook.py').read().split("def sim(")[1].join(["def sim(", ""]).split("P = [(np.datetime64")[0]) if False else None
src = open('/tmp/claude-0/ml/stopbook.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
T = W[W.sp & (W.ratio >= 1.5)].copy(); T['w'] = 1.0
F = {}
for f in glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/*_15m_*.parquet'):
    tk = os.path.basename(f).split('_15m_')[0]
    if tk not in set(T.tk): continue
    B = pd.read_parquet(f, columns=['datetime', 'open', 'high', 'low', 'close']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = B.dt.dt.hour * 60 + B.dt.dt.minute; B = B[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True); day = B.dt.dt.normalize()
    D = B.groupby(day).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'))
    x = np.log(np.log(D.h / D.l).clip(lower=1e-5)); sig = (np.exp(-0.6186 + 0.2939 * x + 0.3661 * x.rolling(5).mean() + 0.1783 * x.rolling(22).mean()) / 1.665).shift(1)
    exp_ = (1.665 * sig)                                                          # expected day range (fraction), median-ish
    F[tk] = (B.dt.values, B.open.values, B.high.groupby(day).cummax().values, B.low.groupby(day).cummin().values, D.o.reindex(day).values, exp_.reindex(day).values)
def feat(tk, e):
    d, o, ch, cl, do, ex = F[tk]; k = np.searchsorted(d, e)
    if k >= len(d) or k == 0 or d[k - 1].astype('datetime64[D]') != d[k].astype('datetime64[D]'): return np.nan, np.nan   # first bar of the day: no session yet
    z = (o[k] / do[k] - 1) / ex[k]; used = np.log(ch[k - 1] / cl[k - 1]) / ex[k]; return z, used
FZ = [feat(t, e) for t, e in zip(T.tk, T.entry.values)]; T['z'] = [a for a, _ in FZ]; T['used'] = [b for _, b in FZ]
print(f'trades {len(T)}, with a session before entry {T.z.notna().mean():.0%}')
for c in ('z', 'used'):
    q = pd.qcut(T[c].rank(method='first'), 5, labels=False) + 1; g = T.groupby(q).agg(avg=('ret', 'mean'), lo=(c, 'min'), hi=(c, 'max'), n=('ret', 'size')); g['avg'] *= 100
    print(f'\n{c} quintiles (z = entry vs the open in expected-range units; used = share of expected range already traveled):'); print(g.round(3).to_string())
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<56}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
print(f'\n{"rule":<58}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', T)
for th in (.5, .75, 1.0):
    show(f'skip if entry is more than {th} expected ranges above the open', T[~(T.z > th)])
for th in (1.0, 1.3):
    show(f'skip if more than {th:.0%} of the expected range is used', T[~(T.used > th)])
z = T.copy(); z['w'] = np.where(z.z > .5, .5, 1.0); z['w'] = z.w / z.w.mean(); show('half size if entry > 0.5 ranges above the open (avg 1x)', z)
