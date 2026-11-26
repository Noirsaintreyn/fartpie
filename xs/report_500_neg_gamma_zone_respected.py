"""Report 500 (Nov 26 2026): Rey - 'does price SUPPORT negative gamma: if the neg-gamma zone is 100-110 and volatility says ~10 points, we expect price to trade 100-110 rather than 90-100'. Direct test of the zone with the ZERO-GAMMA LEVEL (zg) from data/theta_gex2 (12 names, 2018-2026, from raw Theta OI x EOD; known at the prior close).
DECLARED BEFORE RESULTS. Zone = the stretch between spot S (prior close) and the nearest zero-gamma crossing zg, only on days where net_gex<0 at spot (so positive gamma begins at zg). Up-zone: zg>S, down-zone: zg<S. D = |zg/S-1|; EM_h = atm_iv x sqrt(h/252) for h = 1 and 5 trading days (RTH daily bars from the 15m cache). From the prior close, over the next h days: Ez = excursion INTO the zone side (up zone: max high/S-1; down zone: 1-min low/S), Eo = excursion to the OPPOSITE side (same distance mirrored). 
Claim: price goes INTO the zone rather than the mirror side, and stays inside it. Metrics: (1) A = mean (Ez-Eo)/EM, tested vs 0 separately for up zones and down zones (drift cancels in the mirrored sign convention only if the zone effect is real: up zones have +drift, down zones -drift, so a drift artefact would show A>0 for up and A<0 for down; a zone effect shows A>0 for BOTH). (2) P_in = P(Ez >= D) (reaches zg) vs P_mir = P(Eo >= D) (reaches the mirror level) - respect = P_in > P_mir. (3) 'Contained' = reaches zg but does not pass it by more than 0.5 D: P(Ez >= D and Ez < 1.5 D) vs the mirror. Slices: all, high ATM IV (top tercile within name), Q = EM/D terciles (vol covers the zone), train 2019-21 vs test 2018+2022-26. Pass = A>0 in BOTH zone directions, same sign train & test, pooled |t|>=2 (date-clustered).
Null control: same stats on POSITIVE-gamma days (net_gex>0) with zg on the same side: zone claim should vanish."""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
UP = '/mnt/user-data/uploads/fartpie/'; rows = []
for f in sorted(glob.glob(UP + 'data/theta_gex2/*.csv')):
    tk = os.path.basename(f)[:-4]; p = glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')
    if not p: print('no price', tk); continue
    H = pd.read_parquet(p[0]).rename(columns={'datetime': 'ts'}); H['dt'] = pd.to_datetime(H.ts, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].sort_values('dt')
    D = H.groupby(H.dt.dt.normalize()).agg(h=('high', 'max'), l=('low', 'min'), c=('close', 'last')).dropna(); D.index.name = 'd'
    G = pd.read_csv(f, parse_dates=['date']).set_index('date'); G = G[~G.index.duplicated()]
    for h in (1, 5):
        hi = D.h[::-1].rolling(h, min_periods=h).max()[::-1].shift(-1); lo = D.l[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)   # window = next h days after day d
        X = pd.DataFrame({'c': D.c, 'hi': hi, 'lo': lo}).join(G[['net_gex', 'zg', 'atm_iv']], how='inner').dropna(); X['h'] = h; X['tk'] = tk; X['S'] = X.c; rows.append(X.reset_index())
S = pd.concat(rows, ignore_index=True); S = S[(S.zg > 0) & (S.atm_iv > .03) & (S.atm_iv < 3)]
S['D'] = (S.zg / S.S - 1).abs(); S = S[(S.D > .002) & (S.D < .15)]; S['up'] = S.zg > S.S
S['Ez'] = np.where(S.up, S.hi / S.S - 1, 1 - S.lo / S.S); S['Eo'] = np.where(S.up, 1 - S.lo / S.S, S.hi / S.S - 1); S['EM'] = S.atm_iv * np.sqrt(S.h / 252)
S['A'] = (S.Ez - S.Eo) / S.EM; S['Q'] = S.EM / S.D; S['pin'] = (S.Ez >= S.D).astype(float); S['pmir'] = (S.Eo >= S.D).astype(float)
S['cont'] = ((S.Ez >= S.D) & (S.Ez < 1.5 * S.D)).astype(float); S['contm'] = ((S.Eo >= S.D) & (S.Eo < 1.5 * S.D)).astype(float)
S['y'] = S.d.dt.year; S['tr'] = (S.y >= 2019) & (S.y <= 2021); S['ivt'] = S.groupby('tk').atm_iv.transform(lambda x: pd.qcut(x, 3, labels=False, duplicates='drop')); S['Qt'] = pd.qcut(S.Q, 3, labels=False)
def mt(D, col):
    if len(D) < 30: return (np.nan, np.nan)
    m = smf.ols(f'{col} ~ 1', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.d)[0]}); return (m.params['Intercept'], m.tvalues['Intercept'])
print(f'rows {len(S)} names {S.tk.nunique()}; neg-gamma zones up {int(((S.net_gex<0)&S.up&(S.h==1)).sum())} down {int(((S.net_gex<0)&~S.up&(S.h==1)).sum())}')
for lab, base in (('NEGATIVE gamma (net_gex<0)', S[S.net_gex < 0]), ('CONTROL positive gamma (net_gex>0)', S[S.net_gex > 0])):
    print(f'\n######## {lab}')
    for h in (1, 5):
        for dirn, dd in (('UP zone', True), ('DOWN zone', False)):
            for sl, F in (('all', None), ('high IV', 'ivt==2'), ('Q top (vol covers zone)', 'Qt==2'), ('Q low', 'Qt==0')):
                D = base[(base.h == h) & (base.up == dd)]; D = D.query(F) if F else D
                if len(D) < 30: continue
                a, b, c = mt(D[D.tr], 'A'), mt(D[~D.tr], 'A'), mt(D, 'A')
                print(f"h={h} {dirn:9s} {sl:24s} n {len(D):4d} | A train {a[0]:+.3f} test {b[0]:+.3f} pooled {c[0]:+.3f} (t {c[1]:+.1f}) | reach zg {D.pin.mean():.0%} vs mirror {D.pmir.mean():.0%} | contained {D.cont.mean():.0%} vs mirror {D.contm.mean():.0%}")
