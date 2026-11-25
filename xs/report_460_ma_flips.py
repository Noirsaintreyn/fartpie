"""Report 460 (Nov 24 2026): does a moving-average trend filter (EMA / SMA, 15m, 1h and daily) improve the options-ratio gated flips, and which ones are worth drawing on the 15m chart.
Flips: xs/ext/wide_T.parquet (S&P member, options ratio >= 1.5) on the 54 live names, 15m and 1h tracks, 2019-01..2026-10.
Declared: for every MA (type EMA/SMA x length 20/50/100/200 x timeframe 15m/1h/daily = 24 per flip track), flips where the last completed 15m close is ABOVE the MA beat flips BELOW it [trend-aligned entries do better]. Direction is declared; the opposite sign is reported but cannot pass.
Information rule: the MA is the value on the last COMPLETED bar of its timeframe before the entry time; price = last completed 15m close. Test = month-paired difference in mean trade return (bp), |t|>=3.3 (Bonferroni over 48 tests) AND same sign in train 2019-21 and test 2018+2022+ to PASS; |t|>=2 shown as a lead.  Run: python3 xs/report_460_ma_flips.py"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
src = open('xs/ctx_rank.py').read(); i = src.index('SEC = dict('); SEC = eval(src[i + 6:src.index("\nX['sec']")].replace('\n', ' ').strip()); NAMES = sorted(SEC)
W = pd.read_parquet(UP + 'xs/ext/wide_T.parquet'); W = W[W.tk.isin(NAMES) & W.sp & (W.ratio >= 1.5)].copy().reset_index(drop=True); W['entry'] = pd.to_datetime(W.entry)
LENS = [20, 50, 100, 200]; rows = []
def bars(tk):
    d = pd.read_parquet(glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')[0]).rename(columns={'datetime': 'ts'}); d['dt'] = pd.to_datetime(d.ts, utc=True).dt.tz_convert(ET).dt.tz_localize(None); m = d.dt.dt.hour * 60 + d.dt.dt.minute
    return d[(m >= 570) & (m < 960)].sort_values('dt').reset_index(drop=True)
def mas(c, kind, L): return c.ewm(span=L, adjust=False).mean() if kind == 'EMA' else c.rolling(L).mean()
out = []
for tk in sorted(W.tk.unique()):
    B = bars(tk); F = W[W.tk == tk].sort_values('entry').copy(); B15 = pd.DataFrame({'ct': B.dt + pd.Timedelta('15min'), 'px': B.close}); F = pd.merge_asof(F, B15, left_on='entry', right_on='ct', direction='backward').drop(columns='ct')
    h = B.assign(k=(B.dt.dt.normalize() + pd.to_timedelta(((B.dt.dt.hour * 60 + B.dt.dt.minute - 570) // 60) * 60 + 570, unit='m'))); g = h.groupby('k'); H1 = pd.DataFrame({'close': g.close.last()}); H1['ct'] = H1.index + pd.Timedelta('60min'); H1.loc[H1.index.map(lambda t: t.hour * 60 + t.minute) == 930, 'ct'] = H1.index.normalize()[H1.index.map(lambda t: t.hour * 60 + t.minute) == 930] + pd.Timedelta('16h')
    dd = B.groupby(B.dt.dt.normalize()).close.last().to_frame('close'); dd['ct'] = dd.index + pd.Timedelta('16h')
    for tf, S in (('15m', pd.DataFrame({'close': B.close.values, 'ct': (B.dt + pd.Timedelta('15min')).values})), ('1h', H1.reset_index(drop=True)), ('daily', dd.reset_index(drop=True))):
        S = S.sort_values('ct').reset_index(drop=True)
        for kind in ('EMA', 'SMA'):
            for L in LENS:
                M = pd.DataFrame({'ct': S.ct, 'ma': mas(S.close, kind, L)}).dropna(); G = pd.merge_asof(F[['entry']], M, left_on='entry', right_on='ct', direction='backward')
                F[f'{kind}{L}_{tf}'] = (F.px.values > G.ma.values).astype(float) * np.where(G.ma.isna(), np.nan, 1.0)
    out.append(F)
F = pd.concat(out); y = F.entry.dt.year; F['mon'] = F.entry.dt.strftime('%Y-%m'); F['half'] = np.where(y.isin([2019, 2020, 2021]), 'train', 'test')
def paired(a, b):
    d = (a.groupby('mon').ret.mean() - b.groupby('mon').ret.mean()).dropna(); return d.mean() * 1e4, (d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan)
res = []
for h1 in (False, True):
    D = F[F.h1 == h1]; track = '1h' if h1 else '15m'; print(f'\n=== {track} flips n {len(D)} base {D.ret.mean() * 1e4:+.1f}bp, win {(D.ret > 0).mean():.0%} ===')
    for tf in ('15m', '1h', 'daily'):
        for kind in ('EMA', 'SMA'):
            for L in LENS:
                c = f'{kind}{L}_{tf}'; X = D[D[c].notna()]; a, b = X[X[c] == 1], X[X[c] == 0]
                if len(a) < 50 or len(b) < 50: continue
                d, t = paired(a, b); sg = []; txt = ''
                for hf in ('train', 'test'):
                    d2, t2 = paired(a[a.half == hf], b[b.half == hf]); sg.append(np.sign(d2)); txt += f' {hf} {d2:+.0f}/t{t2:.1f}'
                ok = t >= 3.3 and sg[0] == sg[1] == 1; res.append((track, c, d, t, ok, len(a), len(b)))
                print(f'{tf:5s} {kind} {L:3d}: above n {len(a):5d} {a.ret.mean() * 1e4:+6.1f}bp win {(a.ret > 0).mean():.0%} | below n {len(b):5d} {b.ret.mean() * 1e4:+6.1f}bp win {(b.ret > 0).mean():.0%} | diff {d:+6.1f}bp t {t:5.2f} |{txt}' + ('  PASS' if ok else '  lead' if t >= 2 and sg[0] == sg[1] == 1 else ''))
print('\nPASS:', [(r[0], r[1], round(r[2]), round(r[3], 2)) for r in res if r[4]]); print('leads (t>=2, same sign):', [(r[0], r[1], round(r[2]), round(r[3], 2)) for r in res if r[3] >= 2 and not r[4]])
