"""Nov 23 2026 report 408. Rey: 'stock analysis picks the names, ENTER AT THE NEXT FLIP (15m or 1h)'.
DECLARED: eligible flips = the live gate (PIT S&P and ratio >= 1.5); analysis layers known the prior trading day: F40 = fundamentals percentile >= 0.6 (report 406 F at the latest month-end), R = name vol relief (Parkinson 5d <= 20d mean, daily OHLC), M = VIX <= 20d mean.
ARMS: BASE | F40 | R | M | F40&R | F40&R&M.  15m flips = the live book's 12,360 candidates (54 names); 1H flips = wide_T h1 rows (168-name pool, PIT S&P, ratio >= 1.5).  Trade level: n, win rate, median, mean (net 4bp), train 2019-21 | test 2018+2022-26 win and median, month-clustered t of arm-minus-base mean.
15m arms are also run through the live 6-slot book (true daily MTM; score() from report 402).  PASS: win rate and median up vs BASE in both halves with |pooled t| >= 2, or Calmar and MDD better in both halves."""
import numpy as np
_t = open('xs/report_402_five_ideas.py').read(); exec(_t[:_t.index("rows = []; TK0, s0")])
T6 = pd.read_parquet(SP + 'stock406.parquet')[['tk', 't', 'Fp']].dropna(); T6['t'] = T6.t.astype('datetime64[ns]'); T6 = T6.sort_values('t')
D1 = pd.concat([pd.read_parquet(SP + 'lse_daily_54.parquet'), pd.read_parquet(SP + 'lse_daily_pool.parquet')]).drop_duplicates(['symbol', 'ts']); D1['day'] = pd.to_datetime(D1.ts).dt.normalize(); PK = {}
for tk, g in D1.groupby('symbol'):
    g = g.sort_values('day').set_index('day'); g = g[~g.index.duplicated()]; pv = np.log(g.high / g.low) ** 2 / (4 * np.log(2)); PK[tk] = (pv.rolling(5).mean() ** .5 <= (pv ** .5).rolling(20).mean()).shift(1)
Mf = (vix <= vix.rolling(20).mean()).shift(1)
def feats(df):
    df = df.copy(); df['d0'] = pd.to_datetime(df.entry).dt.normalize() - pd.Timedelta(days=1); df['d0'] = df.d0.astype('datetime64[ns]'); df = df.sort_values('d0')
    df = pd.merge_asof(df, T6.rename(columns={'t': 'tm'}).sort_values('tm'), left_on='d0', right_on='tm', by='tk', direction='backward', tolerance=pd.Timedelta(days=45))
    df['R'] = [PK[t].reindex([d], method='ffill').iloc[0] if t in PK else np.nan for t, d in zip(df.tk, df.d0)]; df['M'] = Mf.reindex(df.d0, method='ffill').values
    df['F40'] = df.Fp >= .6; df['R'] = df.R.astype(float) == 1; df['M'] = df.M.astype(float) == 1; df['has'] = df.Fp.notna(); return df.sort_index()
ARM = {'BASE': lambda d: d.has, 'F40': lambda d: d.F40, 'R': lambda d: d.has & d.R, 'M': lambda d: d.has & d.M, 'F40&R': lambda d: d.F40 & d.R, 'F40&R&M': lambda d: d.F40 & d.R & d.M}
def trades(df, lab):
    df = df.copy(); df['n'] = df.ret - 4e-4; df['mo'] = pd.to_datetime(df.entry).dt.to_period('M'); df['half'] = np.where((pd.to_datetime(df.entry) >= '2019-01-01') & (pd.to_datetime(df.entry) < '2022-01-01'), 'train', np.where((pd.to_datetime(df.entry) < '2019-01-01') | (pd.to_datetime(df.entry) >= '2022-01-01'), 'test', 'x')); df = df[df.half != 'x']
    base = df[ARM['BASE'](df)]; print(f'\n{lab}: flips with fundamentals {len(base)} of {len(df)}'); print(f"{'arm':10s}{'n':>6s}{'win%':>7s}{'med%':>7s}{'mean%':>7s} | train win med | test win med | arm-base mean pp (t) pooled")
    for nm, f in ARM.items():
        d = df[f(df)]
        if len(d) < 40: print(nm, len(d)); continue
        m = d.groupby('mo').n.mean() - base.groupby('mo').n.mean().reindex(d.groupby('mo').n.mean().index); m = m.dropna(); t = m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 3 else np.nan
        s = f"{nm:10s}{len(d):6d}{(d.n>0).mean()*100:7.1f}{d.n.median()*100:7.2f}{d.n.mean()*100:7.2f} |"
        for h in ('train', 'test'): dh = d[d.half == h]; s += f" {(dh.n>0).mean()*100:5.1f} {dh.n.median()*100:5.2f} |"
        print(s + f" {m.mean()*100:+.2f} ({t:+.1f})")
# 15m: trades + book
F15 = feats(B0); trades(F15, '15m flips (live book candidates, 54 names)')
TK0, s0 = live_fn(B0); b = score('LIVE', TK0, s0); print('\n' + line(b) + f" | CAGR halves {b['ca'][0]:.1%} | {b['cb'][0]:.1%}")
for nm in ('F40', 'R', 'M', 'F40&R', 'F40&R&M'):
    Bf = B0[ARM[nm](F15).values].reset_index(drop=True); TKs, s = live_fn(Bf); d = score('book: ' + nm, TKs, s); print(line(d) + f" | kept {len(Bf)/len(B0):.0%} | CAGR halves {d['ca'][0]:.1%} | {d['cb'][0]:.1%}", flush=True)
# 1h flips
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[W.h1 & W.sp & (W.ratio >= 1.5) & (pd.to_datetime(W.entry) >= '2018-07-01') & (pd.to_datetime(W.exit) <= '2026-08-14')].reset_index(drop=True)
F1 = feats(W); trades(F1, '1H flips (wide_T h1, PIT S&P, ratio >= 1.5)')
