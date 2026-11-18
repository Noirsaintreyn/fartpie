"""Nov 18 2026 report 332. LARGE-CAP SCREENER (54 names, 12,360 candidate vector flips, slot book of 303/324): gate / rank the flips with (1) VALUE (sales yield, book/market; the one small-cap factor that survived 326) and (2) the BANK-PAPER INPUTS of 331 used as flip-time context.  Declared BEFORE running.
THEORY 1 (value): a flip says price turned up; value says the price is anchored below the cash the business earns.  Flips that start from cheap prices have room to re-rate; flips in expensive names are momentum-on-momentum and give back faster.  Why it should work: the value premium (anchoring + limited arbitrage).  Why it would NOT here: the 54 names are mega-cap growth (value there is a style bet that lost 2019-21 and in the AI run; sales yield is mostly a sector label: tech/semi are 'expensive').  Value is therefore also tested WITHIN sector via ranks inside the 54 (sector label not removed -- stated limit).
 VAL = mean of cross-sectional percentile ranks (among the 54) of sales/market-cap and book/market-cap at the PRIOR month-end, SEC annual values usable filed+3d, shares aligned to the split-adjusted price (split table).
 Variants (slot book, 6 slots, 2bp, 8% breaker; live ratio order vs): drop bottom-third VAL | take only top-half VAL | rank boost -K*(VAL top third) K=2 | boost AND 3-signal ranker.
THEORY 2 (bank inputs as flip context, from 331, each with a stated path): breadth washout (share of the 753 small caps above 200d in the bottom tercile; participation exhausted -> bigger rebounds), broadening (RSP/SPY 3m up), oil falling, CTA flow (NQ dpos > 0 from report 312, lagged), vol-control crowding (SPY 20d vol in bottom tercile of trailing year), copper/gold up.  Each: flip return ON minus OFF, train 2019-21 | test 2018+2022-26, month-clustered t.  Admit only with the same sign in both halves and |t|>=2 in one.
Direction of every test fixed above (ON should be better, except crowding where ON should be worse)."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from sec_panel import one
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
B = pd.read_parquet(SP + 'b302.parquet'); B = B[B.exit < END].copy(); B['h1'] = False; B['entry'] = pd.to_datetime(B.entry)
# ---- value panel for the 54
A = pd.DataFrame([r for f in glob.glob('data/fund/sec/*.json') for r in one(f)]); A['avail'] = A.filed + pd.Timedelta(days=3); A = A.sort_values('avail')
spl = pd.read_parquet(SP + 'splits.parquet'); spl['date'] = pd.to_datetime(spl.date)
PX = {os.path.basename(f)[:-8]: pd.read_parquet(f).close for f in glob.glob('data/yahoo/daily/*.parquet')}; PX = pd.DataFrame(PX).sort_index(); PX.index = pd.to_datetime(PX.index).tz_localize(None) if PX.index.tz else PX.index
def fac(tk, d):  # split factor for shares reported at 'd' -> current split-adjusted basis
    s = spl[(spl.tk == tk) & (spl.date > d)]; return float(s.ratio.prod()) if len(s) else 1.0
A['shs_cur'] = [ (sh if sh == sh else w) * fac(tk, d) if (sh == sh or w == w) else np.nan for tk, d, sh, w in zip(A.ticker, A.filed, A.get('shs'), A.get('shs_w'))]
ME = PX.resample('ME').last().index; rk = {}
for t in ME:
    px = PX.loc[:t].iloc[-1]; a = A[(A.avail <= t) & (A.end >= t - pd.Timedelta(days=550))].groupby('ticker').tail(1).set_index('ticker'); mc = (px.reindex(a.index) * a.shs_cur)
    d = pd.DataFrame({'sy': a.rev / mc, 'bm': a['eq'] / mc}).where(mc > 0)
    rk[t] = d.rank(pct=True).mean(axis=1)
RK = pd.DataFrame(rk).T; print(f'value panel: {RK.shape[1]} names with ranks, {RK.notna().sum(axis=1).median():.0f} median names/month')
B['mprev'] = B.entry.dt.to_period('M').dt.to_timestamp('M') - pd.offsets.MonthEnd(1); B['VAL'] = [RK.at[m, tk] if (m in RK.index and tk in RK.columns) else np.nan for m, tk in zip(B.mprev, B.tk)]
B['VALhi'] = B.VAL >= 2 / 3; B['VALlo'] = B.VAL <= 1 / 3; B['trn'] = (B.entry >= '2019-01-01') & (B.entry < '2022-01-01'); B['tst'] = ~B.trn
print(f'flips with a value rank: {B.VAL.notna().mean():.0%}')
def tt(s, m):
    g = s.groupby(m.loc[s.index]).mean(); return f'{s.mean() * 100:+5.2f}% (t{g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 5 else np.nan:+4.1f},n{len(s):4d})'
B['m'] = B.entry.dt.to_period('M')
print('\n(1) flip return % by value bucket (train | test)')
for lab, f in (('top third (cheap)', B.VALhi), ('middle', ~B.VALhi & ~B.VALlo & B.VAL.notna()), ('bottom third (rich)', B.VALlo)):
    a, b = B[f & B.trn], B[f & B.tst]; print(f'  {lab:20s} {tt(a.ret, a.m)} | {tt(b.ret, b.m)}      ret/vol20: {(a.ret / a.vol20).mean():+.2f} | {(b.ret / b.vol20).mean():+.2f}')
print('  spearman(VAL, ret) train | test:', round(B[B.trn][['VAL', 'ret']].corr(method='spearman').iloc[0, 1], 3), '|', round(B[B.tst][['VAL', 'ret']].corr(method='spearman').iloc[0, 1], 3))
# ---- slot sim (303 machinery)
def sim(Bs, key):
    global T0
    T0 = Bs.assign(_k=key).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(6); R = np.zeros(len(D)); Wt = np.zeros(len(D))
    for tk, e, x, r, ra in tk_:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / 6; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w; Wt[a:b + 1] += w
    s = pd.Series(R * np.where(Wt > 1, 1 / Wt, 1.0), index=D); o = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; c = e.iloc[-1] ** (1 / y) - 1; o.append(f'{c:6.1%} / DD {(e / e.cummax() - 1).min():6.1%} / Calmar {c / -(e / e.cummax() - 1).min():4.2f}')
    return ' | '.join(o)
B['TREND'] = B.trend.astype(bool); B['HI52'] = B.h52b.astype(bool); B['SIZE'] = B.ldvp >= 2 / 3; ns3 = B[['TREND', 'HI52', 'SIZE']].sum(axis=1).values
brank = B.brank.values + 1e-6 * B.ratio.rank(ascending=False).values; Bv = B[B.VAL.notna()].copy(); bv = Bv.brank.values + 1e-6 * Bv.ratio.rank(ascending=False).values
print('\n(2) slot book  [2019-22 | 2023-26]  CAGR / maxDD / Calmar   (only flips with a value rank; live order = ratio rank)')
print(' live, all flips                       ', sim(B, brank)); print(' live, flips with a rank only          ', sim(Bv, bv))
print(' drop bottom-third VAL                 ', sim(Bv[~Bv.VALlo], bv[~Bv.VALlo.values])); print(' only top-half VAL                     ', sim(Bv[Bv.VAL >= .5], bv[(Bv.VAL >= .5).values]))
print(' rank boost K=2 (VAL top third)        ', sim(Bv, bv - 2 * Bv.VALhi.values))
ns3v = Bv[['TREND', 'HI52', 'SIZE']].sum(axis=1).values; print(' 3-signal ranker K=2 (reference)       ', sim(Bv, bv - 2 * ns3v)); print(' ranker + VAL boost K=2                ', sim(Bv, bv - 2 * ns3v - 2 * Bv.VALhi.values))
# ---- bank inputs as flip context
import yfinance as yf
MP = pd.read_parquet(SP + 'macro_px.parquet'); XX = yf.download(['RSP', 'USO', 'HG=F', 'GC=F'], start='2015-01-01', end='2026-11-17', auto_adjust=True, progress=False)['Close']; MP = MP.join(XX); MM = MP.resample('ME').last()
Csm = pd.read_parquet(SP + 'sm_close.parquet').dropna(how='all', axis=1); above = (Csm > Csm.rolling(200).mean()).where(Csm.notna()).mean(axis=1).resample('ME').last()
vol = np.log(MP.SPY.dropna()).diff().rolling(20).std().mul(np.sqrt(252)); vpct = vol.rolling(252).apply(lambda s: (s <= s.iloc[-1]).mean(), raw=False).resample('ME').last()
tr_ = lambda s: s[(s.index >= '2019-01-01') & (s.index < '2022-01-01')]; qa = tr_(above).quantile(1 / 3); qv = tr_(vpct).quantile(1 / 3)
d312 = pd.read_parquet(SP + 'd312.parquet'); d312.index = pd.to_datetime(d312.index).normalize(); cta = (d312.NQ_dpos.round(3) > 0)
ST = {'breadth washout': above <= qa, 'broadening (RSP/SPY up)': ((MM.RSP / MM.SPY) / (MM.RSP / MM.SPY).shift(3) - 1) > 0, 'oil falling': (MM.USO / MM.USO.shift(3) - 1) < 0,
      'vol-control crowded (expect WORSE)': vpct <= qv, 'copper/gold up': ((MM['HG=F'] / MM['GC=F']) / (MM['HG=F'] / MM['GC=F']).shift(3) - 1) > 0}
print('\n(3) bank inputs as FLIP CONTEXT: mean flip return % ON vs OFF (train | test), ON-OFF diff')
for k, s in ST.items():
    on = B.mprev.map(s).astype(float); row = []
    for nm, m in (('train', B.trn), ('test ', B.tst)):
        a, b = B[m & (on == 1)], B[m & (on == 0)]
        if len(a) < 20 or len(b) < 20: row.append(f'{nm} n/a'); continue
        d = a.ret.mean() - b.ret.mean(); ga, gb = a.ret.groupby(a.m).mean(), b.ret.groupby(b.m).mean(); se = np.sqrt(ga.var() / len(ga) + gb.var() / len(gb)); row.append(f'{nm} ON {a.ret.mean() * 100:+.2f} ({len(a)}) OFF {b.ret.mean() * 100:+.2f} ({len(b)}) diff {d * 100:+.2f} (t{d / se:+.1f})')
    print(f'  {k:36s} ' + ' | '.join(row))
c_on = B.day.map(cta) if 'day' in B else None
if c_on is not None:
    on = pd.to_datetime(B.entry).dt.normalize().map(cta).astype(float); row = []
    for nm, m in (('train', B.trn), ('test ', B.tst)):
        a, b = B[m & (on == 1)], B[m & (on == 0)]; d = a.ret.mean() - b.ret.mean(); ga, gb = a.ret.groupby(a.m).mean(), b.ret.groupby(b.m).mean(); se = np.sqrt(ga.var() / len(ga) + gb.var() / len(gb)); row.append(f'{nm} ON {a.ret.mean() * 100:+.2f} ({len(a)}) OFF {b.ret.mean() * 100:+.2f} ({len(b)}) diff {d * 100:+.2f} (t{d / se:+.1f})')
    print(f'  {"CTA buying (NQ dpos>0, lagged)":36s} ' + ' | '.join(row))
