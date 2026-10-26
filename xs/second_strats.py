"""Oct 26 2026: candidate second strategies (daily, close-to-close, 5 bp per side), to run next to the vector book.
 REV5    short-term reversal (Jegadeesh 1990, Lehmann 1990): each close buy the 10% worst 5-day performers of the point-in-time S&P 500,
         hold 5 days (5 overlapping sleeves, 1/5 each)
 REV5r   same inside the options-ratio >= 1.5 names only
 REV1    1-day reversal: buy the 10% worst 1-day performers at the close, sell next close
 REV5v   REV5 only when the equal-weight S&P's 20-day vol is above its 1-year median (Nagel 2012: reversal = paid liquidity provision)
Options signals on the 38 names with daily option summaries (weekly, Friday close to Friday close; long top third / short bottom third):
 SKEW    Xing, Zhang, Zhao 2010: flattest put skew long, steepest short
 PCR     Pan, Poteshman 2006 (volume version): lowest put/call volume ratio long, highest short
 IVCH    Cremers, Weinbaum 2010-style: largest 1-week rise in ATM IV short, largest fall long
writes OUT/daily.parquet (one column per strategy, daily returns)
python3 xs/second_strats.py EXT_DIR PIT_JSON OUT_DIR"""
import sys, json, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
EXT, PIT, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True); TC = 5e-4
P = pd.read_parquet(f'{EXT}/close_px_sp.parquet').pivot_table(index='date', columns='ticker', values='close').sort_index()
P = P[P.index >= '2017-06-01']; R1 = P.pct_change(fill_method=None)
M = json.load(open(PIT)); mem = pd.DataFrame(False, index=P.index, columns=P.columns)
for ym, tks in M.items():
    k = (P.index.strftime('%Y-%m') == ym); c = [t for t in tks if t in P.columns]; mem.loc[k, c] = True
first = min(M); mem.loc[P.index.strftime('%Y-%m') < first, [t for t in M[first] if t in P.columns]] = True
W = pd.read_parquet(f'{EXT}/wide_T.parquet'); W['d'] = W.entry.dt.normalize()
rat = W.groupby(['d', 'tk']).ratio.first().unstack().reindex(P.index).ffill(limit=30).reindex(columns=P.columns)
def rev(look, hold, univ, gate=None):
    past = P / P.shift(look) - 1; past = past.where(univ & P.notna() & P.shift(look).notna())
    q = past.rank(axis=1, pct=True); pick = (q <= .1).astype(float); pick = pick.div(pick.sum(axis=1), axis=0).fillna(0)
    if gate is not None: pick = pick.mul(gate.astype(float), axis=0)
    ret = pd.Series(0.0, index=P.index); 
    for k in range(hold):                                 # sleeve formed k+1 days ago, held hold days
        w = pick.shift(k + 1); ret += (w * R1).sum(axis=1) / hold
    turn = (pick - pick.shift(1)).abs().sum(axis=1) / hold
    return ret - turn * TC
ew = R1.where(mem).mean(axis=1); vol20 = ew.rolling(20).std(); gate = vol20 > vol20.rolling(252, min_periods=120).median()
S = {'REV5': rev(5, 5, mem), 'REV5r': rev(5, 5, mem & (rat >= 1.5)), 'REV1': rev(1, 1, mem), 'REV5v': rev(5, 5, mem, gate), 'EW_SP': ew}
# options
O = {}
for f in glob.glob(f'{EXT}/opt_summary/*.csv'):
    tk = os.path.basename(f)[:-4]; d = pd.read_csv(f, parse_dates=['date']).set_index('date'); O[tk] = d
def opanel(col): return pd.DataFrame({t: d[col] for t, d in O.items()}).sort_index()
skew, iv = opanel('skew'), opanel('atm_iv'); pcr = opanel('put_vol') / opanel('call_vol')
spot = opanel('spot'); r1o = spot.pct_change(fill_method=None)
fri = spot.index[spot.index.dayofweek == 4]
def weekly_ls(sig, hi_long=True):
    s = sig.reindex(fri); q = s.rank(axis=1, pct=True)
    lo, hi = (q <= 1 / 3).astype(float), (q > 2 / 3).astype(float)
    L, Sh = (hi, lo) if hi_long else (lo, hi)
    L = L.div(L.sum(axis=1), axis=0); Sh = Sh.div(Sh.sum(axis=1), axis=0)
    L, Sh = L.reindex(spot.index).ffill().shift(1), Sh.reindex(spot.index).ffill().shift(1)
    long_ = (L * r1o).sum(axis=1); ls = long_ - (Sh * r1o).sum(axis=1); return long_, ls
for nm, sig, hl in (('SKEW', skew, True), ('PCR', pcr, False), ('IVCH', iv - iv.shift(5), False)):
    lo_, ls_ = weekly_ls(sig, hl); S[nm + '_long'] = lo_; S[nm + '_ls'] = ls_
S['EW_opt38'] = r1o.mean(axis=1)
pd.DataFrame(S).to_parquet(f'{OUT}/daily.parquet'); print('ok')
