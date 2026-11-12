"""Nov 11 2026 (Rey): 5 mid/small-cap names with a BIG volatility forecast as of the Nov 10 close, then 15m and 1h vector charts (candles coloured by vector state only).
Universe: non-S&P options-active names in wide_T (+ mid-book names), market cap 0.3-25B, 20d avg dollar volume > $15M. Forecast: pooled HAR-RV (log daily variance from open/high/low/close incl. overnight gap, Yang-Zhang-style), next-5-day mean, fitted on data through Nov 10; compared with each name's own 1-year distribution. Implied vol: ATM IV of the nearest expiry >= 6 days (Yahoo, as of the fetch).
Charts: 15m vector on pre+RTH bars (04:00-16:00); 1h vector on RTH 15m bars resampled 09:30-anchored. Candle blue = bull state, pink = bear. Yahoo 15m history is 60 days."""
import sys, os, json, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from vector_pine import run_state_machine
ASOF = pd.Timestamp('2026-11-10')
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet')
U = sorted(set(W[~W.sp].tk.unique()) - {'QQQ', 'GLD', 'SLV', 'TLT', 'BRKB', 'BBBY', 'BBT', 'CRON'}) + ['KGS', 'MEDP', 'CART', 'MMI', 'WRLD']; U = sorted(set(U))
print('universe', len(U), flush=True)
D = yf.download(U, period='2y', interval='1d', auto_adjust=False, group_by='ticker', progress=False, threads=True)
rows = {}
for t in U:
    try: d = D[t].dropna(subset=['Open', 'High', 'Low', 'Close']); d = d[d.index <= ASOF + pd.Timedelta(days=0)]
    except Exception: continue
    if len(d) < 300: continue
    pc = d.Close.shift(); rv = np.log(d.Open / pc) ** 2 + .5 * np.log(d.High / d.Low) ** 2 - (2 * np.log(2) - 1) * np.log(d.Close / d.Open) ** 2     # Garman-Klass + overnight (Yang-Zhang family)
    rv = rv.clip(lower=1e-8); rows[t] = (d, rv)
print('names with data', len(rows), flush=True)
X = []; Y = []
for t, (d, rv) in rows.items():
    l = np.log(rv); f1 = l; f5 = np.log(rv.rolling(5).mean()); f22 = np.log(rv.rolling(22).mean()); tgt = np.log(rv[::-1].rolling(5).mean()[::-1].shift(-1))
    z = pd.concat([f1, f5, f22, tgt], axis=1).dropna(); X.append(z.iloc[:, :3].values); Y.append(z.iloc[:, 3].values)
Xa = np.vstack(X); Ya = np.concatenate(Y); A = np.c_[np.ones(len(Xa)), Xa]; beta = np.linalg.lstsq(A, Ya, rcond=None)[0]; s2 = np.var(Ya - A @ beta)
print('HAR pooled betas', beta.round(3), flush=True)
out = []
for t, (d, rv) in rows.items():
    l = np.log(rv); x = np.array([1, l.iloc[-1], np.log(rv.rolling(5).mean().iloc[-1]), np.log(rv.rolling(22).mean().iloc[-1])]); fc = np.exp(x @ beta + .5 * s2)
    r5 = rv.rolling(5).mean().dropna(); pct = (r5.iloc[-252:] < fc).mean(); yz20 = np.sqrt(rv.iloc[-20:].mean() * 252)
    dv = (d.Close * d.Volume).iloc[-20:].mean()
    out.append(dict(tk=t, last=d.Close.iloc[-1], har_vol=np.sqrt(fc * 252), yz20=yz20, har_pct=pct, ratio=np.sqrt(fc) / np.sqrt(r5.iloc[-252:].median()), dvol=dv))
O = pd.DataFrame(out).sort_values('har_pct', ascending=False)
O = O[(O.dvol > 15e6)].head(30).copy()
def mcap(t):
    try: return yf.Ticker(t).fast_info['market_cap']
    except Exception: return np.nan
O['mcap'] = [mcap(t) for t in O.tk]; O = O[(O.mcap > 3e8) & (O.mcap < 25e9)].copy()
print('shortlist after cap filter', len(O), flush=True)
def atm_iv(t):
    try:
        tk = yf.Ticker(t); ex = [e for e in tk.options if (pd.Timestamp(e) - pd.Timestamp('2026-11-11')).days >= 6]
        if not ex: return np.nan, None
        ch = tk.option_chain(ex[0]); px = tk.fast_info['last_price']; v = []
        for c in (ch.calls, ch.puts):
            c = c.dropna(subset=['impliedVolatility']); c = c[c.impliedVolatility > .01]
            if len(c): v.append(c.iloc[(c.strike - px).abs().argsort()[:1]].impliedVolatility.values[0])
        return (float(np.mean(v)) if v else np.nan), ex[0]
    except Exception: return np.nan, None
iv = [atm_iv(t) for t in O.tk]; O['iv'] = [a for a, _ in iv]; O['exp'] = [b for _, b in iv]
O['score'] = O.har_pct.rank(pct=True) + O.iv.rank(pct=True, na_option='bottom'); O = O.sort_values('score', ascending=False)
pd.set_option('display.width', 200); print(O[['tk', 'mcap', 'last', 'har_vol', 'yz20', 'har_pct', 'ratio', 'iv', 'exp']].assign(mcap=lambda x: (x.mcap / 1e9).round(1), har_vol=lambda x: (x.har_vol * 100).round(0), yz20=lambda x: (x.yz20 * 100).round(0), har_pct=lambda x: (x.har_pct * 100).round(0), iv=lambda x: (x.iv * 100).round(0)).round(2).to_string(index=False), flush=True)
PICK = list(O.tk.head(5)); O.head(5).to_csv('xs/vol_forecast_top5.csv', index=False); O.to_csv('xs/vol_forecast_shortlist.csv', index=False)
# ---------------- charts
BULL, BEAR, BG, FG = '#3b6cff', '#ff5d8f', '#0b0b0c', '#8a8d93'
def cs(ax, B, reg, title, sess_labels=True):
    n = len(B); x = np.arange(n); col = np.where(reg > 0, BULL, BEAR)
    for i in range(n):
        ax.plot([i, i], [B.low.iloc[i], B.high.iloc[i]], color=col[i], lw=.8, zorder=2)
        lo, hi = sorted([B.open.iloc[i], B.close.iloc[i]]); ax.add_patch(plt.Rectangle((i - .32, lo), .64, max(hi - lo, (B.high.max() - B.low.min()) * .0015), color=col[i], zorder=3))
    ax.set_facecolor(BG); ax.set_xlim(-1, n + 1); lo_, hi_ = B.low.min(), B.high.max(); ax.set_ylim(lo_ - (hi_ - lo_) * .04, hi_ + (hi_ - lo_) * .04)
    day = B.index.normalize(); b = [i for i in range(n) if i == 0 or day[i] != day[i - 1]]
    for i in b[1:]: ax.axvline(i - .5, color='#26282c', lw=.7, zorder=1)
    ax.set_xticks(b); ax.set_xticklabels([f'{day[i]:%b %d}' for i in b], color=FG, fontsize=8); ax.tick_params(axis='y', colors=FG, labelsize=8); ax.yaxis.tick_right()
    for s in ax.spines.values(): s.set_color('#26282c')
    ax.set_title(title, color='#d9dbe0', fontsize=10, loc='left')
    td = pd.Timestamp('2026-11-11'); idx = [i for i in range(n) if day[i] == td]
    if idx: ax.axvspan(idx[0] - .5, n, color='#ffffff', alpha=.035, zorder=0)
files = []
for tk in PICK:
    d15 = yf.download(tk, period='60d', interval='15m', prepost=True, progress=False, auto_adjust=False); d15.columns = [c[0].lower() for c in d15.columns]
    d15.index = d15.index.tz_convert('America/New_York').tz_localize(None); d15 = d15[['open', 'high', 'low', 'close']].dropna(); d15 = d15[~d15.index.duplicated()]
    m = d15.index.hour * 60 + d15.index.minute; d15 = d15[(m >= 240) & (m < 960)]
    reg15 = run_state_machine(d15.assign(datetime=d15.index))[0]
    rth = d15[(d15.index.hour * 60 + d15.index.minute) >= 570]
    h = rth.resample('60min', origin='start_day', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    reg1 = run_state_machine(h.assign(datetime=h.index))[0]
    cut15 = d15.index >= pd.Timestamp('2026-11-03'); cut1 = h.index >= pd.Timestamp('2026-10-14')
    r = O[O.tk == tk].iloc[0]
    fig, ax = plt.subplots(2, 1, figsize=(11, 8.4), facecolor=BG); fig.subplots_adjust(hspace=.30, left=.04, right=.93, top=.87, bottom=.06)
    fig.text(.04, .955, f'{tk}', color='#f1f2f4', fontsize=20, fontweight='bold'); fig.text(.04, .925, f'mkt cap ${r.mcap/1e9:.1f}B  ·  HAR forecast {r.har_vol*100:.0f}% ann. (top {100-r.har_pct*100:.0f}% of its year)  ·  20d YZ {r.yz20*100:.0f}%  ·  ATM implied {r.iv*100:.0f}%  ·  vector state as candle colour (blue bull / pink bear)', color=FG, fontsize=9)
    cs(ax[0], d15[cut15], reg15[cut15], f'15m vector (pre + RTH) - last 7 sessions;  shaded = Nov 11'); cs(ax[1], h[cut1], reg1[cut1], '1h vector (RTH) - last ~20 sessions')
    p = f'xs/vol_chart_{tk}.png'; fig.savefig(p, dpi=130, facecolor=BG); plt.close(fig); files.append(p)
print('FILES', files, flush=True)
