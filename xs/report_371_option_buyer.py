"""Nov 22 2026 report 371. Rey: 'I trade options off the flip, being wrong is a decent hit (often we don't break through or hold); find something, maybe an options-data signal that lines up'.
RESEARCH (literature read before declaring): (1) Baltussen-Da-Lammers-Martens 'Hedging demand and market intraday momentum': intraday momentum is strong on negative net-gamma days and insignificant on positive days; (2) Barbon-Buraschi 'Gamma Fragility': negative dealer gamma -> intraday momentum, positive gamma -> intraday reversal, stronger in illiquid underlyings; (3) Quantpedia/Bondarenko line: long option portfolios profit only ~28% of the time because implied vol exceeds realized (volatility risk premium ~3%, positive ~85% of the time); buyers only get paid in extreme vol spikes.
MEASURE (real option P&L, EOD quotes in theta_compact): at the flip-day close buy the call nearest the money with 14-45 DTE (closest to 21) at the ASK (skip if spread > 25% of mid); exit at the BID on the earlier of the vector exit day and 5 trading days later (same-day exits skipped).  Return = bid_exit / ask_entry - 1.  Names with ThetaData EOD quotes AND a GEX file (net_gex, atm_iv).
HYPOTHESES (declared; features at the prior close; train 2019-21 | test 2022-26; terciles on train; week-clustered t; direction declared):
 G1 NET GEX: more NEGATIVE dealer net gamma (own-name net_gex / its trailing-126d std, lower) -> BETTER call P&L (flip continues; momentum regime).
 V1 IV/RV: lower atm_iv / realized vol (20d close-to-close, annualised) -> BETTER (cheap options relative to what the stock actually does).
 V2 IV LEVEL: lower atm_iv -> BETTER (less premium to overcome).
 Also report: the unconditional call P&L, the share of calls that profit, and P&L of a 'filter' that takes only (negative-gamma tercile AND cheap-IV/RV tercile).
Pass: same sign both halves, |t|>=2 in at least one; only 3 hypotheses so ~0.15 false passes expected."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet'); b['day'] = pd.to_datetime(b.day); b['exit'] = pd.to_datetime(b.exit); b['entry'] = pd.to_datetime(b.entry)
names = [os.path.basename(f)[:-4] for f in glob.glob(U + 'xs/ext/gex/*.csv')]; names = [n for n in names if os.path.exists(U + f'data/theta_compact/{n}.parquet') and n in set(b.tk)]
print('names', len(names), flush=True); rows = []
for tk in names:
    g = pd.read_csv(U + f'xs/ext/gex/{tk}.csv', parse_dates=['date']).set_index('date').sort_index(); g['gz'] = (g.net_gex - g.net_gex.rolling(126, min_periods=60).mean()) / g.net_gex.rolling(126, min_periods=60).std()
    lc = np.log(g.lse_close); rv = lc.diff().rolling(20).std() * np.sqrt(252); g['ivrv'] = g.atm_iv / rv; g = g.shift(1)  # prior-day values
    t = pd.read_parquet(U + f'data/theta_compact/{tk}.parquet'); t = t[(t.r == 1) & (t.K / t.S > .9) & (t.K / t.S < 1.1)]; t['d'] = pd.to_datetime(t.d); days = np.sort(t.d.unique()); byd = {d: x for d, x in t.groupby('d')}
    for _, r in b[b.tk == tk].iterrows():
        D = r.day
        if D not in byd: continue
        x = byd[D]; x = x[(x.dte >= 14) & (x.dte <= 45) & (x.ask > 0) & (x.bid > 0)]
        if not len(x): continue
        S = x.S.median(); e = x.loc[(x.dte - 21).abs().idxmin(), 'e']; x = x[x.e == e]; k = x.loc[(x.K - S).abs().idxmin()]
        if (k.ask - k.bid) / ((k.ask + k.bid) / 2) > .25: continue
        i0 = np.searchsorted(days, np.datetime64(D)); X = min(r.exit.normalize(), pd.Timestamp(days[min(i0 + 5, len(days) - 1)]))
        if X <= D: continue
        y = byd.get(X); y = y[(y.e == e) & (y.K == k.K)] if y is not None else None
        if y is None or not len(y): continue
        rows.append(dict(tk=tk, day=D, ret_u=r.ret, call=y.bid.iloc[0] / k.ask - 1, gz=g.gz.get(D, np.nan), ivrv=g.ivrv.get(D, np.nan), iv=g.atm_iv.get(D, np.nan), spread=(k.ask - k.bid) / ((k.ask + k.bid) / 2), dte=k.dte))
    print(tk, len(rows), end=' | ', flush=True)
A = pd.DataFrame(rows); A['wk'] = A.day.dt.to_period('W').astype(str); A['half'] = np.where(A.day < '2022-01-01', 'train', 'test'); A.to_parquet(SP + 'A371.parquet')
print('\nflips with a priced call: %d | mean call P&L %+.1f%% | median %+.1f%% | profitable %.0f%% | underlying mean ret at same flips %+.2f%%' % (len(A), A.call.mean() * 100, A.call.median() * 100, (A.call > 0).mean() * 100, A.ret_u.mean() * 100))
for h in ('train', 'test'):
    S = A[A.half == h]; print('  %s: n %d mean %+.1f%% median %+.1f%% profitable %.0f%% | avg spread paid %.1f%%' % (h, len(S), S.call.mean() * 100, S.call.median() * 100, (S.call > 0).mean() * 100, S.spread.mean() * 100))
tr = A[A.half == 'train']
def test(lab, col, lowgood):
    lo, hi = tr[col].quantile([1 / 3, 2 / 3]); out = []
    for h in ('train', 'test'):
        D_ = A[(A.half == h) & A[col].notna()]; lowS = D_[D_[col] <= lo]; hiS = D_[D_[col] >= hi]; T, R_ = (lowS, hiS) if lowgood else (hiS, lowS)
        Z = pd.concat([T.assign(x=1.0), R_.assign(x=0.0)]); m = sm.OLS(Z.call.values, sm.add_constant(Z.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Z.wk)[0]}); out.append((m.params[1] * 100, m.tvalues[1], T.call.mean() * 100, R_.call.mean() * 100, (T.call > 0).mean() * 100, (R_.call > 0).mean() * 100, len(T)))
    ok = np.sign(out[0][0]) == np.sign(out[1][0]) == 1 and max(abs(out[0][1]), abs(out[1][1])) >= 2
    print('%-34s TRAIN good-third %+.1f%% vs other-end %+.1f%% (diff %+.1f pts, t %+.1f, profitable %.0f/%.0f%%, n %d) | TEST %+.1f%% vs %+.1f%% (diff %+.1f, t %+.1f, %.0f/%.0f%%, n %d)  %s' % (lab, out[0][2], out[0][3], out[0][0], out[0][1], out[0][4], out[0][5], out[0][6], out[1][2], out[1][3], out[1][0], out[1][1], out[1][4], out[1][5], out[1][6], 'PASS' if ok else ''), flush=True)
test('G1 low (negative) net GEX z', 'gz', True); test('V1 low IV/RV', 'ivrv', True); test('V2 low ATM IV', 'iv', True)
print('(call P&L shown in %; the diff is good-end minus the opposite end)')
both = A.dropna(subset=['gz', 'ivrv']); lg, hg = tr.gz.quantile(1 / 3), tr.ivrv.quantile(1 / 3)
for h in ('train', 'test'):
    S = both[both.half == h]; f = S[(S.gz <= lg) & (S.ivrv <= hg)]; print('combined filter %-5s: n %d of %d, mean call P&L %+.1f%% vs all %+.1f%%, profitable %.0f%% vs %.0f%%' % (h, len(f), len(S), f.call.mean() * 100, S.call.mean() * 100, (f.call > 0).mean() * 100, (S.call > 0).mean() * 100))
