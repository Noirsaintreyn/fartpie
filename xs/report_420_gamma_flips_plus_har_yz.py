"""Nov 24 2026 report 420. Rey: 'see volatility on top of that (neg-gamma, high option-vol flips), HAR and YZ'.
DECLARED: base sets from report 419 (NEG&VOL3, LOW&VOL3; 18 names with gex).  Vol states known at the prior close: HARup = HAR next-day forecast (report 404 pooled fit on 2019-21) > its 22d mean RV (expansion), HARdn = forecast <= 22d mean (relief); YZup = Yang-Zhang(5d) > Yang-Zhang(20d) (daily OHLC, expansion), YZdn = <=.  Mechanism guess: negative gamma amplifies moves, so vol EXPANSION should help continuation (HARup/YZup); relief versions run as the opposite control.
8 comparisons (4 flags x 2 base sets) -> Bonferroni |t| >= 2.5, t = month-clustered mean difference of flag-on vs the rest of the same base set.  PRIMARY outcome (b) same-session 15:45 exit (intraday question); (a) vector exit secondary.  PASS: flag-on beats flag-off on win AND median in 2022 and 2023-26 (n>=30 cells) and |t| >= 2.5."""
import numpy as np, pandas as pd, os
_t = open('xs/report_404_name_vol_relief.py').read(); exec(_t[:_t.rindex("rows = []; TK0, s0")])
HARup = F > RVd.rolling(22).mean(); HARdn = ~HARup
D1 = pd.concat([pd.read_parquet(SP + 'lse_daily_54.parquet'), pd.read_parquet(SP + 'lse_daily_pool.parquet')]).drop_duplicates(['symbol', 'ts']); D1['day'] = pd.to_datetime(D1.ts).dt.normalize()
def yz(g, n):
    o, h, l, c = (np.log(g[k]) for k in ('open', 'high', 'low', 'close')); so = (o - c.shift(1)).rolling(n).var(); sc = (c - o).rolling(n).var(); rs = ((h - c) * (h - o) + (l - c) * (l - o)).rolling(n).mean(); k = .34 / (1.34 + (n + 1) / (n - 1)); return (so + k * sc + (1 - k) * rs).clip(lower=0) ** .5
YZ5 = {}; YZ20 = {}
for tk in NM:
    g = D1[D1.symbol == tk].sort_values('day').drop_duplicates('day').set_index('day')
    if len(g) < 100: continue
    YZ5[tk] = yz(g, 5); YZ20[tk] = yz(g, 20)
YZ5 = pd.DataFrame(YZ5).reindex(idx); YZ20 = pd.DataFrame(YZ20).reindex(idx); YZup = (YZ5 > YZ20).where(YZ5.notna() & YZ20.notna()); YZdn = ~YZup.astype(bool) & YZ5.notna() & YZ20.notna()
X_ = 'xs/ext/gex/'; G = {f[:-4]: pd.read_csv(X_ + f, parse_dates=['date']).set_index('date').sort_index().net_gex for f in os.listdir(X_)}
B = B0[B0.tk.isin(G)].copy().reset_index(drop=True); B['entry'] = pd.to_datetime(B.entry); B['d0'] = B.entry.dt.normalize() - pd.Timedelta(days=1)
pct = []; neg = []
PR = {tk: g.rolling(252, min_periods=126).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True).shift(1) for tk, g in G.items()}
for tk, d0 in zip(B.tk, B.d0): pct.append(PR[tk].reindex([d0], method='ffill').iloc[0]); neg.append(G[tk].shift(1).reindex([d0], method='ffill').iloc[0] < 0)
B['pct'] = pct; B['NEG'] = neg; B = B[B.pct.notna()].copy(); B['LOW'] = B.pct <= .25; B['VOL3'] = B.ratio >= B.ratio.quantile(2 / 3)
ed = pd.DatetimeIndex(B.entry).normalize(); a_ = np.minimum(idx.searchsorted(ed), nD - 1)
for nm, M_ in (('HARup', HARup), ('HARdn', HARdn), ('YZup', YZup), ('YZdn', YZdn)):
    S = M_.shift(1).astype(float); B[nm] = [S.values[a, ncol[t]] if t in ncol else np.nan for a, t in zip(a_, B.tk)]; B[nm] = B[nm] == 1
B['okv'] = np.isfinite([YZ5.shift(1).values[a, ncol[t]] if t in ncol else np.nan for a, t in zip(a_, B.tk)])
def sday(tk, e):
    try: return px_at(tk, pd.Timestamp(e).normalize() + pd.Timedelta(hours=15, minutes=45)) / px_at(tk, e) - 1 - 4e-4
    except Exception: return np.nan
B['a'] = B.ret - 4e-4; B['b'] = [sday(t, e) for t, e in zip(B.tk, B.entry)]; e_ = B.entry; B['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); B['mo'] = e_.dt.to_period('M')
print('flips', len(B), '| HARup share', B.HARup.mean().round(2), '| YZup share', B.YZup.mean().round(2))
for c, lab in (('b', '(b) PRIMARY same-session 15:45 exit'), ('a', '(a) vector exit')):
    D = B[B[c].notna()]
    for bname, bmask in (('ALL gex flips', D.NEG | ~D.NEG), ('NEG&VOL3', D.NEG & D.VOL3), ('LOW&VOL3', D.LOW & D.VOL3)):
        S = D[bmask]; print(f'\n=== {lab} | base {bname} n{len(S)} win {(S[c]>0).mean()*100:.1f}% med {S[c].median()*100:.2f}% mean {S[c].mean()*100:.2f}% ===  flag-on n | win% med% mean% | vs rest pp (t) | 22 win/med | 23-26 win/med')
        for nm in ('HARup', 'HARdn', 'YZup', 'YZdn'):
            on = S[S[nm]]; off = S[~S[nm]]
            if len(on) < 40: print(f'{nm:6s} n{len(on)}'); continue
            m = (on.groupby('mo')[c].mean() - off.groupby('mo')[c].mean()).dropna(); t = m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 3 else np.nan; s = f"{nm:6s}{len(on):5d} | {(on[c]>0).mean()*100:5.1f} {on[c].median()*100:5.2f} {on[c].mean()*100:5.2f} | {(on[c].mean()-off[c].mean())*100:+.2f} ({t:+.1f}) |"
            for era in ('22', '23-26'): x = on[on.era == era]; y = off[off.era == era]; s += f" {(x[c]>0).mean()*100:4.1f}/{x[c].median()*100:5.2f} (off {(y[c]>0).mean()*100:4.1f}/{y[c].median()*100:5.2f}) |" if len(x) >= 30 else ' n<30 |'
            print(s)
