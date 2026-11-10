"""Nov 9 2026: NAKED LONG CALLS FOR ACCURACY (what a $700 cash account can trade) - strike, expiry and exit choices, plus the trade-quality filter. DECLARED BEFORE RESULTS.
Flips: the screener's bull flips in the 23 names with ThetaData history (one per name per day). Real EOD quotes: bought at the ASK, sold at the BID. 1 premium unit per trade.
Structures: strike ITM (nearest at or below spot - 0.5 sigma x sqrt(days)) or ATM (nearest at or above spot) | expiry ~30 days (21-45) or ~60 days (45-75)
Exits (daily closing marks at the bid): VEC at the screener's exit (or expiry) | TP50 take +50% / stop -50%, else the screener exit | TP30 +30% / -40%, else the screener exit
Filters: all flips (2019-21 / 2022-26) and, 2021-26 only, the trade-quality model's TOP DECILE (walk-forward, out of sample; xs/trade_quality.py).
Puts: the screener produces bull flips only - long puts need bear-flip signals built from the per-stock bars (Mac); not testable here.
Reported: win rate, average and median return on premium, months positive and worst month (1 unit per trade), by year."""
import numpy as np, pandas as pd, warnings, io, contextlib; warnings.filterwarnings('ignore')
_o = open('/home/claude/fartpie/xs/opt_flips.py').read(); exec(_o[:_o.index('def entry_all(')])
_q = open('/home/claude/fartpie/xs/trade_quality.py').read(); exec(_q[:_q.index("\nif __name__ == '__main__':")])
with contextlib.redirect_stdout(io.StringIO()): OS = walk(X[['tk', 'day', 'yr', 'ret'] + FEAT].copy(), 'stock')
OS['q'] = OS.groupby('yr').p_lr.transform(lambda s_: s_.rank(pct=True)); SC = OS.sort_values('q').drop_duplicates(['tk', 'day'], keep='last').set_index(['tk', 'day']).q
EXP = {'M30': (21, 45, 30), 'M60': (45, 75, 60)}
DI = lambda x: np.datetime64(x, 'D').astype(np.int64)
TF = pd.read_parquet(f'{U}/xs/ext/wide_T.parquet'); TF = TF[TF.sp & (TF.ratio >= 1.5)].copy(); TF['d0'] = TF.entry.dt.normalize(); TF['d1'] = TF.exit.dt.normalize(); TF = TF.sort_values(['tk', 'entry']).drop_duplicates(['tk', 'd0'])
def trade(bk, d0, d1, tenor, mny, exit_):
    if d0 not in bk.rng: return None
    S0, sg = bk.S[d0], bk.sig[d0]
    if not (np.isfinite(S0) and np.isfinite(sg)): return None
    a, b = bk.rng[d0]; lo, hi, tg = EXP[tenor]; dte = bk.dte[a:b]; ok_ = (dte >= lo) & (dte <= hi)
    if not ok_.any(): return None
    dsel = dte[ok_][np.argmin(np.abs(dte[ok_] - tg))]; ix = a + np.where((dte == dsel) & (bk.r[a:b] == 1) & (bk.ask[a:b] > 0))[0]
    if not len(ix): return None
    e = bk.e[ix[0]]; w = sg * np.sqrt(max(1, round(dsel * 252 / 365))); K = bk.K[ix]
    c = ix[K <= S0 * (1 - .5 * w)] if mny == 'ITM' else ix[K >= S0]
    if not len(c): return None
    j = c[np.argmax(bk.K[c])] if mny == 'ITM' else c[np.argmin(bk.K[c])]
    bi, ak = bk.bid[j], bk.ask[j]
    if bi <= 0 or ak - bi > .3 * (ak + bi) / 2: return None
    Kj = bk.K[j]; prem = ak; end = min(d1, e)
    tp, sl = {'VEC': (None, None), 'TP50': (.5, -.5), 'TP30': (.3, -.4)}[exit_]
    if tp is not None:
        for d in bk.days[(bk.days > d0) & (bk.days < end)]:
            v = bk.leg(d, e, Kj, 1, +1, 'TOUCH'); r_ = v / prem - 1
            if r_ >= tp or r_ <= sl: return r_
    return bk.leg(end, e, Kj, 1, +1, 'TOUCH') / prem - 1
if __name__ == '__main__':
    F = TF[TF.tk.isin(B)].copy(); F = F[F.d0 >= '2019-01-01']; rows = []
    for r_ in F.itertuples():
        bk = B[r_.tk]; d0, d1 = DI(r_.d0), DI(r_.d1); q = SC.get((r_.tk, r_.d0), np.nan)
        for tenor in EXP:
            for mny in ('ITM', 'ATM'):
                for ex in ('VEC', 'TP50', 'TP30'):
                    v = trade(bk, d0, d1, tenor, mny, ex)
                    if v is not None: rows.append((r_.tk, r_.d0, tenor, mny, ex, v, q))
    Z = pd.DataFrame(rows, columns=['tk', 'd0', 'ten', 'mny', 'ex', 'ret', 'q']); Z['yr'] = Z.d0.dt.year; Z['mo'] = Z.d0.dt.to_period('M')
    def line(z):
        mo = z.groupby('mo').ret.sum()
        return f'win {np.mean(z.ret > 0):3.0%}, avg {z.ret.mean():+6.1%}, med {z.ret.median():+6.1%} (n {len(z):4d}), months+ {np.mean(mo > 0):3.0%}, worst mo {mo.min():+5.1f}u'
    print('=== naked long calls on ALL flips | 2019-21 ; 2022-26 ===', flush=True)
    for (ten, mny, ex), g_ in Z.groupby(['ten', 'mny', 'ex']):
        print(f'  {ten} {mny} {ex:<5} ' + line(g_[g_.yr <= 2021]) + ' ; ' + line(g_[g_.yr >= 2022]), flush=True)
    print('\n=== TOP-DECILE quality flips (out of sample, 2021-26) vs the other flips, same years ===', flush=True)
    W = Z[(Z.yr >= 2021) & Z.q.notna()]
    for (ten, mny, ex), g_ in W.groupby(['ten', 'mny', 'ex']):
        t_ = g_[g_.q >= .9]; print(f'  {ten} {mny} {ex:<5} top: ' + line(t_) + ' | rest: ' + line(g_[g_.q < .9]) + ' | top by year: ' + ' '.join(f'{y % 100}:{v:+.0%}' for y, v in t_.groupby('yr').ret.mean().items()), flush=True)
# --- follow-up (EXPLORATORY, added after seeing that the screener's tight exits sink the options): option exits INDEPENDENT of the screener -
#     take profit / stop on the option itself, else out after 10 trading days (or expiry); same entries, real fills
def trade2(bk, d0, tenor, mny, tp, sl, hold=10):
    if d0 not in bk.rng: return None
    k0 = np.searchsorted(bk.days, d0); d1 = bk.days[min(len(bk.days) - 1, k0 + hold)]
    return trade_tp(bk, d0, d1, tenor, mny, tp, sl)
def trade_tp(bk, d0, d1, tenor, mny, tp, sl):
    S0, sg = bk.S[d0], bk.sig[d0]
    if not (np.isfinite(S0) and np.isfinite(sg)): return None
    a, b = bk.rng[d0]; lo, hi, tg = EXP[tenor]; dte = bk.dte[a:b]; ok_ = (dte >= lo) & (dte <= hi)
    if not ok_.any(): return None
    dsel = dte[ok_][np.argmin(np.abs(dte[ok_] - tg))]; ix = a + np.where((dte == dsel) & (bk.r[a:b] == 1) & (bk.ask[a:b] > 0))[0]
    if not len(ix): return None
    e = bk.e[ix[0]]; w = sg * np.sqrt(max(1, round(dsel * 252 / 365))); K = bk.K[ix]
    c = ix[K <= S0 * (1 - .5 * w)] if mny == 'ITM' else ix[K >= S0]
    if not len(c): return None
    j = c[np.argmax(bk.K[c])] if mny == 'ITM' else c[np.argmin(bk.K[c])]; bi, ak = bk.bid[j], bk.ask[j]
    if bi <= 0 or ak - bi > .3 * (ak + bi) / 2: return None
    Kj = bk.K[j]; prem = ak; end = min(d1, e)
    for d in bk.days[(bk.days > d0) & (bk.days < end)]:
        r_ = bk.leg(d, e, Kj, 1, +1, 'TOUCH') / prem - 1
        if r_ >= tp or r_ <= sl: return r_
    return bk.leg(end, e, Kj, 1, +1, 'TOUCH') / prem - 1
if __name__ == '__main__':
    rows = []
    for r_ in F.itertuples():
        bk = B[r_.tk]; d0 = DI(r_.d0); q = SC.get((r_.tk, r_.d0), np.nan)
        for tenor in EXP:
            for mny in ('ITM', 'ATM'):
                for nm, tp, sl in (('TP30/SL40', .3, -.4), ('TP50/SL50', .5, -.5), ('TP100/SL50', 1.0, -.5)):
                    v = trade2(bk, d0, tenor, mny, tp, sl)
                    if v is not None: rows.append((r_.tk, r_.d0, tenor, mny, nm, v, q))
    Z2 = pd.DataFrame(rows, columns=['tk', 'd0', 'ten', 'mny', 'ex', 'ret', 'q']); Z2['yr'] = Z2.d0.dt.year; Z2['mo'] = Z2.d0.dt.to_period('M')
    print('\n=== EXPLORATORY: option-only exits (TP / SL on the option, else out after 10 trading days) | all flips 2019-21 ; 2022-26 | top-decile quality flips 2021-26 ===', flush=True)
    for (ten, mny, ex), g_ in Z2.groupby(['ten', 'mny', 'ex']):
        w_ = g_[(g_.yr >= 2021) & (g_.q >= .9)]
        print(f'  {ten} {mny} {ex:<10} ' + line(g_[g_.yr <= 2021]) + ' ; ' + line(g_[g_.yr >= 2022]) + ' | top decile: ' + line(w_), flush=True)
