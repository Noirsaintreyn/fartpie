"""Oct 25 2026: better volatility for the low-in / high-in projection. Three daily vol forecasts (all known at the prior close):
 STD  : std of the last 20 daily close-to-close returns (current)
 HARRV: HAR on realized variance from 15m returns (Corsi 2009): RV_t+1 ~ RV_d + RV_w(5) + RV_m(22), fit on 2019-21, log form
 HARHL: HAR on the daily log high/low range (range-based, close to a high/low VECM's range forecast), same lags
Then the projection (high above a low that looks in, and the mirror) is re-measured in each model's sigma units:
better model = narrower bands in % of price at the same out-of-sample coverage (fit 2019-21, test 2022-26)."""
import glob, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
TKS = sys.argv[1:] or ['QQQ', 'SPY', 'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'META', 'AMZN', 'GOOGL']
def har_fit(y, X):
    b = np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)[0]; return b
rows = []; fits = []
for tk in TKS:
    f = glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet') or glob.glob(f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')
    if not f: continue
    B = pd.read_parquet(f[0], columns=['datetime', 'open', 'high', 'low', 'close']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    B['m'] = B.dt.dt.hour * 60 + B.dt.dt.minute; B = B[(B.m >= 570) & (B.m < 960)].drop_duplicates('dt').sort_values('dt'); B['d'] = B.dt.dt.normalize()
    D = B.groupby('d').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'))
    lr = np.log(B.close).diff(); lr[B.d != B.d.shift()] = np.nan                       # intraday 15m returns only
    rv = (lr ** 2).groupby(B.d).sum(); on = np.log(D.o / D.c.shift()) ** 2                # + overnight
    D['rv'] = rv + on.fillna(0); D['hl'] = np.log(D.h / D.l)
    D['std'] = np.log(D.c).diff().rolling(20).std()
    for col, nm in (('rv', 'harrv'), ('hl', 'harhl')):
        x = np.log(D[col].clip(lower=1e-8)); X = pd.concat([x, x.rolling(5).mean(), x.rolling(22).mean()], axis=1); y = x.shift(-1)
        tr = (D.index < '2022-01-01') & X.notna().all(1) & y.notna()
        b = har_fit(y[tr].values, X[tr].values); pred = b[0] + X.values @ b[1:]
        resid = (y[tr] - (b[0] + X[tr].values @ b[1:])).var()
        lvl = np.exp(pred + resid / 2)                                                    # forecast for the NEXT day
        D[nm] = np.sqrt(lvl) if col == 'rv' else lvl / 1.665                             # E[range] ~ 1.665 sigma (Parkinson)
        fits.append((tk, nm, *np.round(b, 3)))
    xh = np.log(D.hl.clip(lower=1e-8)); D['pool'] = np.exp(-0.6186 + 0.2939 * xh + 0.3661 * xh.rolling(5).mean() + 0.1783 * xh.rolling(22).mean()) / 1.665   # pooled, as live
    for c_ in ('std', 'harrv', 'harhl', 'pool'): D[c_] = D[c_].shift(1)                         # known at the prior close
    for d, g in B.groupby('d'):
        if d < pd.Timestamp('2019-01-01') or d not in D.index or len(g) < 24: continue
        sg = D.loc[d, ['std', 'harrv', 'harhl', 'pool']]
        if sg.isna().any(): continue
        H, L = g.high.max(), g.low.min()
        for cp in (630, 720, 840):
            k = g[g.m < cp]
            if len(k) < 2: continue
            lo, hi, last = k.low.min(), k.high.max(), k.close.iloc[-1]
            tl = k.m[k.low.idxmin()]; th = k.m[k.high.idxmax()]
            if cp - tl >= 30 and last >= lo * (1 + .5 * sg['std']):
                rows.append(dict(tk=tk, d=d, cp=cp, side='low in', holds=L >= lo, ext=H / lo - 1, **{f's_{k_}': v for k_, v in sg.items()}))
            if cp - th >= 30 and last <= hi * (1 - .5 * sg['std']):
                rows.append(dict(tk=tk, d=d, cp=cp, side='high in', holds=H <= hi, ext=1 - L / hi, **{f's_{k_}': v for k_, v in sg.items()}))
X = pd.DataFrame(rows); X['per'] = np.where(X.d.dt.year >= 2022, 'te', 'tr')
print(f'{len(X)} checkpoints\n')
print(f'{"side":<9}{"time":<7}{"model":<7}{"80% band: coverage 22-26":>26}{"avg width %":>13}{"95% band: coverage":>21}{"avg width %":>13}')
for side in ('low in', 'high in'):
    for cp in (630, 720, 840):
        a = X[(X.side == side) & (X.cp == cp)]
        for mdl in ('std', 'harrv', 'harhl'):
            u = a.ext / a[f's_{mdl}']; tr, te = a.per == 'tr', a.per == 'te'
            q80, q95 = u[tr].quantile(.8), u[tr].quantile(.95)
            print(f'{side:<9}{cp // 60}:{cp % 60:02d}  {mdl:<7}{(u[te] <= q80).mean():>26.1%}{(q80 * a[te][f"s_{mdl}"]).mean() * 100:>13.2f}{(u[te] <= q95).mean():>21.1%}{(q95 * a[te][f"s_{mdl}"]).mean() * 100:>13.2f}')
    print()
# does the vol model help predict whether the low/high HOLDS? (bigger forecast vs move so far)
X['used'] = X.ext                                                                        # placeholder
print('low/high holds rate by checkpoint (all models share it):', X.groupby(['side', 'cp']).holds.mean().round(3).to_dict())
print('\nHAR-range coefficients (const, day, week, month) by name:')
F = pd.DataFrame([f for f in fits if f[1] == 'harhl'], columns=['tk', 'm', 'c', 'bd', 'bw', 'bm']); print(F.to_string(index=False)); print('pooled mean:', F[['c', 'bd', 'bw', 'bm']].mean().round(4).to_dict())
print('\nquantiles of the move to the other extreme in HAR-range sigma units (all years):')
for side in ('low in', 'high in'):
    for cp in (630, 720, 840):
        a = X[(X.side == side) & (X.cp == cp)]; u = a.ext / a.s_pool; tr, te = a.per == 'tr', a.per == 'te'
        print(f'  pooled-live: 22-26 coverage of 19-21 80%/95% bands {(u[te] <= u[tr].quantile(.8)).mean():.1%} / {(u[te] <= u[tr].quantile(.95)).mean():.1%}', end='')
        print(f'  {side} {cp // 60}:{cp % 60:02d}  holds {a.holds.mean():.1%}  q50 {u.quantile(.5):.3f}  q80 {u.quantile(.8):.3f}  q95 {u.quantile(.95):.3f}')
