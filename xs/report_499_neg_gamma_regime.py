"""Report 499 (Nov 26 2026): Rey - 'test neg gamma as a range and volatility to know the move is likely; backtest how many times on high volatile days we respected neg gamma'.
FEASIBILITY: the historical gex files (xs/ext/gex/*.csv, 26 names, 2018-01..2025-12) hold only the daily NET GEX per name (no strike grid), so a zero-gamma LEVEL ('VT'), call/put walls and 'did price respect the level' CANNOT be backtested from them; that needs historical option chains (Theta pull on the Mac). What CAN be tested is neg gamma as a REGIME: does prior-close net_gex<0 change next-day range and trend/reversal behaviour, especially on high-vol days.
DECLARED BEFORE RESULTS (mechanism: dealers short gamma hedge WITH the move -> amplification; long gamma hedge AGAINST it -> pinning). Per name-day, RTH daily bars from the 15m cache, gex known at the prior close:
 H1 RANGE: log((H-L)/Cprev) larger when prior net_gex<0, controlling for log ATR14% (name's own prior vol) + name FE; sign +.
 H2 TREND: efficiency |C-O|/(H-L) larger when neg gamma; sign +.
 H3 MOMENTUM/REVERSAL: next-day return times the sign of the prior-day return (signed continuation) larger when neg gamma; sign +.
 H4 'RESPECT' proxy without a level: P(|C-O| < 0.5 x prior-day ATR) i.e. a quiet range day, lower when neg gamma (sign -).
Slices: all, high-vol (prior ATR14% top tercile within name), low-vol. Train 2018-2021 gex (declared per Rey's standing split: train 2019-21, test 2018 + 2022-25; 2018 -> test). Test statistic: coefficient on NEG, SE clustered by date. Pass = same sign train & test, pooled |t|>=2."""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
UP = '/mnt/user-data/uploads/fartpie/'; GX = 'xs/ext/gex/'; rows = []
for f in sorted(glob.glob(GX + '*.csv')):
    tk = os.path.basename(f)[:-4]; p = glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')
    if not p: continue
    H = pd.read_parquet(p[0]).rename(columns={'datetime': 'ts'}); H['dt'] = pd.to_datetime(H.ts, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].sort_values('dt')
    D = H.groupby(H.dt.dt.normalize()).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last')).dropna(); D = D[D.h > D.l]
    G = pd.read_csv(f, parse_dates=['date']).set_index('date').net_gex; D['g'] = G.reindex(D.index); D['gp'] = D.g.shift(1)   # gex dated d = prior close of d+1? use the value known BEFORE day t: gex of the previous trading day
    D['cp'] = D.c.shift(1); D['rng'] = np.log((D.h - D.l) / D.cp); tr = pd.concat([D.h - D.l, (D.h - D.cp).abs(), (D.l - D.cp).abs()], axis=1).max(axis=1); D['atr'] = tr.rolling(14).mean().shift(1) / D.cp
    D['eff'] = (D.c - D.o).abs() / (D.h - D.l); D['ret'] = D.c / D.cp - 1; D['pret'] = D.ret.shift(1); D['cont'] = D.ret * np.sign(D.pret); D['quiet'] = ((D.c - D.o).abs() / D.cp < .5 * D.atr).astype(float); D['tk'] = tk
    rows.append(D.dropna(subset=['gp', 'atr', 'rng', 'cont']).reset_index().rename(columns={'index': 'd', 'dt': 'd'}))
S = pd.concat(rows, ignore_index=True); S['d'] = pd.to_datetime(S.iloc[:, 0]); S['NEG'] = (S.gp < 0).astype(float); S['y'] = S.d.dt.year; S['tr'] = (S.y >= 2019) & (S.y <= 2021); S['latr'] = np.log(S.atr)
S['vt'] = S.groupby('tk').atr.transform(lambda x: pd.qcut(x, 3, labels=False, duplicates='drop'))
print(f'n {len(S)}, names {S.tk.nunique()}, days {S.d.nunique()}, neg-gamma share {S.NEG.mean():.0%}, by year: ' + ', '.join(f'{y} {g.NEG.mean():.0%}' for y, g in S.groupby('y')))
def eff(D, y):
    m = smf.ols(f'{y} ~ NEG + latr + C(tk)', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.d)[0]}); return m.params['NEG'], m.tvalues['NEG']
print(f"{'outcome':28s}| slice     | n tr/te    | NEG coef train | test | pooled (t) | means off -> on")
for y, lab in (('rng', 'H1 log range'), ('eff', 'H2 trend efficiency'), ('cont', 'H3 continuation (ret x sign prior)'), ('quiet', 'H4 quiet-day share')):
    for sl, F in (('all', S.vt >= 0), ('high-vol', S.vt == 2), ('low-vol', S.vt == 0)):
        D = S[F]; a, b, c = eff(D[D.tr], y), eff(D[~D.tr], y), eff(D, y); mo = D[D.NEG == 0][y].mean(); mn = D[D.NEG == 1][y].mean()
        print(f"{lab:28s}| {sl:9s} | {int(D.tr.sum()):5d}/{int((~D.tr).sum()):5d} | {a[0]:+.4f} | {b[0]:+.4f} | {c[0]:+.4f} (t {c[1]:+.1f}) | {mo:+.4f} -> {mn:+.4f}  {'PASS' if np.sign(a[0]) == np.sign(b[0]) and abs(c[1]) >= 2 else ''}")
H = S[S.vt == 2]; print(f"\nhigh-vol days: neg gamma {H.NEG.mean():.0%} of them; median range on neg {np.exp(H[H.NEG==1].rng.median())*100:.2f}% vs pos {np.exp(H[H.NEG==0].rng.median())*100:.2f}% (raw; ATR bucket not equal), share closing in the top/bottom 20% of the day's range: neg {((H[H.NEG==1].c-H[H.NEG==1].l)/(H[H.NEG==1].h-H[H.NEG==1].l)).pipe(lambda z:((z>.8)|(z<.2)).mean()):.0%} pos {((H[H.NEG==0].c-H[H.NEG==0].l)/(H[H.NEG==0].h-H[H.NEG==0].l)).pipe(lambda z:((z>.8)|(z<.2)).mean()):.0%}")
