"""Nov 10 2026: EDGE HUNT ON REAL NQ 5m (2014 - Jan 2026), ideas NOT already in the ~100 nq_* scripts. DECLARED BEFORE RESULTS.
Rey: 'dig into everything and see if you can find excess edge - macro, swing, game theory, anything'.
DATA   /mnt/user-data/uploads/Downloads/NQ_5Min.csv (MotiveWave, ET, European number format, contracts rolled by symbol), VIX / TNX / DXY dailies (Yahoo, from 2017).
DAY    RTH = 09:30-15:55 5m bars. O = 09:30 open, A = price at 12:00 (close of the 11:55 bar), C = 15:55 close. Prior-day H/L/C from the same contract only (roll days -> NaN).
OUTCOMES per day, in NQ points:  R_am = A - O (what Rule A trades live inside)  |  R_day = C - O  |  LR = ln(H-L of 09:30-12:00)  (size / stop width: volatility, not direction)
         R_after30 = A - price at 10:00 (does the move after the first half hour continue?)
FLAGS (family sizes: CAL 13, MAC 10, STR 11 = 34 tests x 3 outcomes; each outcome is judged separately)
  CAL  turn of month (last day + first 3), last 2 days of month, first day of month, quarter-end week, pre-holiday day, OPEX Friday, OPEX week, VIX-expiry Wednesday,
       Monday, Friday, FOMC day, day after FOMC, day-of-week = Tuesday..Thursday is the control
  MAC  prior-day VIX change > +10% | < -8%, VIX level tercile (hi/lo), VIX 5d change up, TNX 5d change up (>= +10 bp) / down (<= -10 bp), DXY 5d up / down (> 1%), NQ 5d return up / down
  STR  gap up / down > 0.5 x 20d avg range, overnight sweep of the prior high then back inside (fade), overnight sweep of the prior low then back inside, opens above / below the prior range,
       prior day closed in top / bottom 20% of its range, overnight range in its top third (20d), first-30-minute drive WITH the gap vs AGAINST it
PASS RULE  same sign in 2014-21 (A) and 2022-26 (B), |t| >= 2.0 in BOTH (two independent samples: chance of a fluke passing both ~ 0.05 x 0.05 / 2 per test, ~0.1 false passes expected in 34 tests x 3).
           (macro flags exist only from 2017, so A = 2017-21 for them)
Then the passing flags are applied to the 357 Rule A trades (live_data/nq_rule_a_trades.csv): average net points with / against, A vs B."""
import numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings('ignore')
ROOT = '/home/claude/fartpie'
def load():
    d = pd.read_csv('/mnt/user-data/uploads/Downloads/NQ_5Min.csv', sep=';', skiprows=2, header=None, names=['dt', 'sym', 'o', 'h', 'l', 'c', 'v'])
    for k in 'ohlc': d[k] = d[k].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    d['dt'] = pd.to_datetime(d.dt, format='%m/%d/%Y %I:%M %p'); return d.sort_values('dt').drop_duplicates('dt').set_index('dt')
def days(d):
    m = d.index.hour * 60 + d.index.minute; D = []; prev = None; rth = d[(m >= 570) & (m < 960)]; ovn = d
    for day, g in rth.groupby(rth.index.normalize()):
        if len(g) < 70: continue
        sym = g.sym.iloc[0]; o = g.o.iloc[0]; t = g.index.hour * 60 + g.index.minute
        gm = g[t < 720]; p10 = g.c[t == 595]; a = gm.c.iloc[-1]
        # overnight: previous calendar evening 18:00 -> 09:25 on the same contract
        w = d[(d.index >= day - pd.Timedelta(hours=6)) & (d.index < day + pd.Timedelta(hours=9, minutes=30))]; w = w[w.sym == sym]
        r = dict(day=day, sym=sym, O=o, A=a, C=g.c.iloc[-1], P10=p10.iloc[0] if len(p10) else np.nan, H=gm.h.max(), L=gm.l.min(), DH=g.h.max(), DL=g.l.min(), DC=g.c.iloc[-1],
                 OH=w.h.max() if len(w) else np.nan, OL=w.l.min() if len(w) else np.nan, n_o=len(w))
        D.append(r)
    D = pd.DataFrame(D).set_index('day')
    same = D.sym == D.sym.shift(1)
    for k, c in (('PC', 'DC'), ('PH', 'DH'), ('PL', 'DL')): D[k] = D[c].shift(1).where(same)
    D['R_am'] = D.A - D.O; D['R_day'] = D.C - D.O; D['LR'] = np.log((D.H - D.L).clip(lower=.25)); D['R_after30'] = D.A - D.P10
    D['R_first30'] = D.P10 - D.O; D['gap'] = D.O - D.PC; D['rng'] = D.DH - D.DL; D['rng20'] = D.rng.shift(1).rolling(20).mean()
    return D
def feats(D):
    F = pd.DataFrame(index=D.index); ix = D.index; dow = ix.dayofweek; mo = ix.to_period('M')
    pos = D.groupby(mo).cumcount(); rpos = D.groupby(mo).cumcount(ascending=False)
    F['turn_month'] = (rpos == 0) | (pos <= 2); F['last2'] = rpos <= 1; F['first_day'] = pos == 0
    F['qtr_end_wk'] = ix.month.isin([3, 6, 9, 12]) & (rpos <= 4)
    cal = pd.bdate_range(ix.min(), ix.max()); hol = cal.difference(ix); F['pre_hol'] = [(ix[i] + pd.offsets.BDay(1)) in hol for i in range(len(ix))]
    third = [x for x in ix if x.dayofweek == 4 and 15 <= x.day <= 21]; F['opex_fri'] = ix.isin(third)
    F['opex_wk'] = pd.Series(ix.isocalendar().week.values, index=ix).isin([pd.Timestamp(x).isocalendar()[1] for x in third]) & (pd.Series(ix.year, index=ix).values == pd.Series([pd.Timestamp(x).year for x in ix]).values)
    F['opex_wk'] = [(any((x - t).days in range(0, 5) and x.isocalendar()[1] == t.isocalendar()[1] and x.year == t.year for t in [pd.Timestamp(y) for y in third if abs((pd.Timestamp(y) - x).days) < 7])) for x in ix]
    F['vix_exp'] = [(x.dayofweek == 2 and 15 <= x.day <= 21) for x in ix]
    F['mon'] = dow == 0; F['fri'] = dow == 4
    FO = pd.to_datetime(open(f'{ROOT}/xs/nq_context_guards.py').read().split("FOMC = pd.to_datetime('''")[1].split("'''")[0].split())
    F['fomc'] = ix.isin(FO); nx = [ix[min(np.searchsorted(ix, f) + 1, len(ix) - 1)] for f in FO if f >= ix.min()]; F['fomc1'] = ix.isin(nx)
    V = {t: pd.read_parquet(f'{ROOT}/data/yahoo/daily_v/{t}.parquet').close for t in ('VIX', 'TNX', 'DX_Y_NYB')}
    vx = V['VIX']; vxp = vx.shift(1).reindex(ix, method='ffill')                      # prior-day values (known before the open)
    v_ch = (vx.pct_change().shift(1)).reindex(ix, method='ffill'); F['vix_up'] = v_ch > .10; F['vix_dn'] = v_ch < -.08
    q1, q2 = vxp[(ix < '2022')].quantile([1 / 3, 2 / 3]); F['vix_hi'] = vxp > q2; F['vix_lo'] = vxp < q1
    F['vix5_up'] = (vx.diff(5).shift(1)).reindex(ix, method='ffill') > 0
    tn = V['TNX'].diff(5).shift(1).reindex(ix, method='ffill'); F['tnx_up'] = tn >= .10; F['tnx_dn'] = tn <= -.10         # Yahoo ^TNX is the yield in percent -> .10 = 10 bp
    dx = V['DX_Y_NYB'].pct_change(5).shift(1).reindex(ix, method='ffill'); F['dxy_up'] = dx > .01; F['dxy_dn'] = dx < -.01
    r5 = (D.DC / D.DC.shift(5) - 1).shift(1); F['nq5_up'] = r5 > .02; F['nq5_dn'] = r5 < -.02
    g = D.gap / D.rng20; F['gap_up'] = g > .5; F['gap_dn'] = g < -.5
    inside_h = (D.OH > D.PH) & (D.O < D.PH); inside_l = (D.OL < D.PL) & (D.O > D.PL); F['swept_hi'] = inside_h & (D.PH.notna()); F['swept_lo'] = inside_l & (D.PL.notna())
    F['open_above'] = D.O > D.PH; F['open_below'] = D.O < D.PL
    clv = (D.DC - D.DL) / (D.DH - D.DL); clv = clv.shift(1).where(D.sym == D.sym.shift(1)); F['pclose_top'] = clv > .8; F['pclose_bot'] = clv < .2
    onr = (D.OH - D.OL); F['on_wide'] = onr > onr.shift(1).rolling(20).quantile(2 / 3)
    sg = np.sign(D.gap); drive = np.sign(D.R_first30); F['drive_with'] = (sg != 0) & (drive == sg); F['drive_against'] = (sg != 0) & (drive == -sg)
    return F.fillna(False).astype(bool)
def tstat(x):
    x = x.dropna(); return (x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 5 and x.std() > 0 else np.nan)
def diff_t(y, f):
    a, b = y[f].dropna(), y[~f].dropna()
    if len(a) < 15 or len(b) < 15: return np.nan, np.nan, len(a)
    d = a.mean() - b.mean(); se = np.sqrt(a.var() / len(a) + b.var() / len(b)); return d, d / se, len(a)
if __name__ == '__main__':
    d = load(); D = days(d); F = feats(D); print(f'days {len(D):,} {D.index.min():%Y-%m-%d} - {D.index.max():%Y-%m-%d} | O->12:00 avg {D.R_am.mean():+.1f} pts, O->close {D.R_day.mean():+.1f}', flush=True)
    A_ = D.index < '2022-01-01'; B_ = ~A_; start = {'vix_up': '2017', 'vix_dn': '2017', 'vix_hi': '2017', 'vix_lo': '2017', 'vix5_up': '2017', 'tnx_up': '2017', 'tnx_dn': '2017', 'dxy_up': '2017', 'dxy_dn': '2017'}
    out = []
    for y in ('R_am', 'R_day', 'R_after30', 'LR'):
        print(f'\n=== outcome {y}: flag | n A / B | diff (flag - not) A, B | t A, B | pass ===', flush=True); rows = []
        for c in F.columns:
            m0 = D.index >= pd.Timestamp(start.get(c, '2014') + '-01-01'); fa, fb = F[c] & m0, F[c]
            da, ta, na = diff_t(D[y][A_ & m0], F[c][A_ & m0]); db, tb, nb = diff_t(D[y][B_], F[c][B_])
            ok = np.isfinite(ta) and np.isfinite(tb) and np.sign(da) == np.sign(db) and abs(ta) >= 2 and abs(tb) >= 2
            rows.append((c, na, nb, da, db, ta, tb, ok))
        for c, na, nb, da, db, ta, tb, ok in sorted(rows, key=lambda r: -min(abs(r[5]) if np.isfinite(r[5]) else 0, abs(r[6]) if np.isfinite(r[6]) else 0)):
            print(f'  {c:<14} n {na:5} / {nb:5} | {da:+8.2f} {db:+8.2f} | t {ta:+5.2f} {tb:+5.2f} | {"PASS" if ok else ""}', flush=True)
            if ok: out.append((y, c))
    print('\nPASSING (outcome, flag):', out, flush=True)
    # ---- the same flags on Rule A's own trades (net points per 1 NQ, after cost) ----
    X = pd.read_csv(f'{ROOT}/live_data/nq_rule_a_trades.csv', parse_dates=['day', 'entry']).set_index('day'); X = X[X.index.isin(F.index)]
    X['A'] = X.index < '2022-01-01'; Fx = F.loc[X.index]; sg = np.sign(D.loc[X.index, 'gap']); X['with_gap'] = (sg == X.dir); X['against_gap'] = (sg == -X.dir)
    X['long'] = X.dir == 1; X['short'] = X.dir == -1
    X['to_swept_hi'] = Fx['swept_hi'].values & (X.dir.values == -1); X['to_swept_lo'] = Fx['swept_lo'].values & (X.dir.values == 1)
    print(f'\n=== RULE A trades ({len(X)}; avg net pts {X.net_pts.mean():+.1f}; A {X[X.A].net_pts.mean():+.1f} n {X.A.sum()}, B {X[~X.A].net_pts.mean():+.1f} n {(~X.A).sum()}) | flag: n A/B | avg pts flag vs not, A | B | t A, B ===', flush=True)
    cols = list(F.columns) + ['with_gap', 'against_gap', 'long', 'short', 'to_swept_hi', 'to_swept_lo']; rows = []
    for c in cols:
        f = Fx[c].values if c in F.columns else X[c].values; f = pd.Series(f, index=X.index)
        da, ta, na = diff_t(X.net_pts[X.A], f[X.A]); db, tb, nb = diff_t(X.net_pts[~X.A], f[~X.A]); rows.append((c, na, nb, da, db, ta, tb))
    for c, na, nb, da, db, ta, tb in sorted(rows, key=lambda r: -min(abs(r[5]) if np.isfinite(r[5]) else 0, abs(r[6]) if np.isfinite(r[6]) else 0)):
        print(f'  {c:<14} n {na:4} / {nb:4} | {da:+7.1f} {db:+7.1f} | t {ta:+5.2f} {tb:+5.2f} | {"PASS" if np.isfinite(ta) and np.isfinite(tb) and np.sign(da) == np.sign(db) and abs(ta) >= 2 and abs(tb) >= 2 else ""}', flush=True)
    F.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/nq_flags.pkl'); D.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/nq_days.pkl')
