"""Buys after 14:00 (live: none): all post-2pm flips, or only when the flip bar volume / day range used / day return passes a filter. Run on the Mac."""
import glob, json
exec(open('xs/batch_oct25.py').read().split("PART = os.getenv('PART', 'ALL')")[0])
M = json.load(open('sp500_pit_membership.json')); mem = {k: {'META' if t == 'FB' else t for t in v} for k, v in M.items()}; months = sorted(mem)
def member(tk, d):
    k = f'{d:%Y-%m}'; k = k if k in mem else (months[-1] if k > months[-1] else months[0]); return tk in mem[k]
X = []
for tk in sorted(set(T.tk)):
    f = f'xs/ext/ahx_clean/{tk}.parquet'; fa = f'xs/ext/slab/att_{tk}.parquet'
    if not (os.path.exists(f) and os.path.exists(fa)): continue
    x = pd.read_parquet(f); x = x[x['var'] == 'base+sticky'].copy(); x['entry'] = pd.to_datetime(x.entry); x['exit'] = pd.to_datetime(x.exit)
    x = x[(x.entry >= '2019-01-01') & ((x.entry.dt.hour * 60 + x.entry.dt.minute) >= 840)]
    a = pd.read_parquet(fa); a = a[~a.index.duplicated()].sort_index(); x['ratio'] = a.ratio.reindex(x.entry.dt.normalize().values, method='ffill').values
    x = x[x.ratio >= 1.5]; x = x[[member(t, e) for t, e in zip(x.tk, x.entry)]]
    if not len(x): continue
    B = bars(tk); d = B.dt.values; m = (B.dt.dt.hour * 60 + B.dt.dt.minute).values; v = B.volume.values if 'volume' in B else None
    if v is None:
        BB = pd.read_parquet(glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet')[0], columns=['datetime', 'volume']); BB['dt'] = pd.to_datetime(BB.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        v = BB.drop_duplicates('dt').set_index('dt').volume.reindex(B.dt.values).values
    vs = pd.Series(v); rv = (vs / vs.groupby(m).transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())).values
    day = B.dt.dt.normalize(); hi = B.high.groupby(day).cummax().values; lo = B.low.groupby(day).cummin().values
    D = B.groupby(day).agg(h=('high', 'max'), l=('low', 'min')); x_ = np.log(np.log(D.h / D.l).clip(lower=1e-5))
    sig = (np.exp(-0.6186 + 0.2939 * x_ + 0.3661 * x_.rolling(5).mean() + 0.1783 * x_.rolling(22).mean()) / 1.665).shift(1)
    budget = (1.7 * sig).reindex(day.values).values
    k = np.searchsorted(d, x.entry.values) - 1                                   # the flip bar = bar before the entry
    ok = k >= 0; k = np.clip(k, 0, len(d) - 1)
    x['rvol'] = np.where(ok, rv[k], np.nan); x['used'] = np.where(ok, np.log(hi[k] / lo[k]) / budget[k], np.nan)
    c = B.close.values; x['dayret'] = np.where(ok, c[k] / B.open.groupby(day).transform('first').values[k] - 1, np.nan)
    X.append(x)
X = pd.concat(X, ignore_index=True)
print(f'post-2pm flips in the live universe: {len(X)}, avg {X.ret.mean():+.3%}, P(>=3%) {(X.ret >= .03).mean():.3f}  (pre-2pm live avg {T.ret.mean():+.3%})')
for c in ('rvol', 'used', 'dayret'):
    q = pd.qcut(X[c].rank(method='first'), 3, labels=['low', 'mid', 'high']); g = X.groupby(q).ret.agg(['mean', 'count']); g['mean'] *= 100
    print(f'  post-2pm by {c} tercile:', ', '.join(f'{i} {r["mean"]:+.2f}% ({int(r["count"])})' for i, r in g.iterrows()))
head('BUYS AFTER 14:00 (live = none)')
show('allow all post-2pm flips', pd.concat([T, X], ignore_index=True))
for c, q, lab in (('rvol', .5, 'flip-bar volume above its median'), ('rvol', .67, 'flip-bar volume in the top third'), ('used', .5, 'day range used below its median'),
                  ('used', .33, 'day range used in the bottom third (room left)'), ('dayret', .5, 'day return above its median')):
    th = X[c].quantile(q); sel = X[X[c] >= th] if c in ('rvol', 'dayret') else X[X[c] <= th]
    show(f'post-2pm only if {lab} ({len(sel)})', pd.concat([T, sel], ignore_index=True))
for q in (.33, .5):
    th = X.rvol.quantile(q); sel = X[X.rvol <= th]; show(f'post-2pm only if flip-bar volume in the bottom {int(q*100)}% ({len(sel)})', pd.concat([T, sel], ignore_index=True))
sel = X[(X.rvol <= X.rvol.quantile(.5)) & (X.used <= X.used.quantile(.5))]; show(f'post-2pm only if low volume AND room left ({len(sel)})', pd.concat([T, sel], ignore_index=True))
