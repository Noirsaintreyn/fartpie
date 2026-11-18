"""Nov 18 2026 report 338. FORMAL STRESS TEST 2/3: the large-cap options-ratio vector screener (live book: point-in-time S&P, ratio >= 1.5, 15m pivot-ratchet bull flips, 6 slots, 8% breaker, 54 research names, Jan 2019 - Sep 2026; the slot simulator of 303/324/325).  Declared before running.
 S1 Sharpe + PSR + DEFLATED Sharpe (N trials in {50, 200, 1000, 5000}; skill-less trial SR sd 0.5 annualised, assumption).
 S2 stationary block bootstrap of the daily book (5,000): P(SR<0), CAGR 5-95%, 5% worst max drawdown (%).
 S3 RANDOM-ENTRY PLACEBO: each taken flip replaced by the SAME ticker held the SAME number of 15m bars from a random start (2,000 draws of the book's mean trade return): does the flip beat random timing in the same names?  (Controls for drift/beta of mega-cap growth.)
 S4 COST stress: extra bp per trade on top of the 2+2 bp already in the simulator (0 / 5 / 10 / 20 / 40).
 S5 LATE-ENTRY stress (the check that cost the profile Rey was shown 36% of its edge): entry delayed by 1 / 2 / 4 fifteen-minute bars (price = that bar's open, exit unchanged); live bot enters next bar open.
 S6 NEIGHBOURHOOD: slots {4, 5, 6, 8, 10} x breaker {0, 5%, 8%, 12%}: share profitable (CAGR>0), min / median CAGR and Sharpe.
 S7 CONCENTRATION: remove the best 5% / 10% of taken trades.  Yearly table.
Not run: look-ahead replacement and exchange-feed match (need the live signal code).  The slot sim spreads each trade evenly over its holding days (checked against the real daily series in 300)."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from stress_lib import *
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; B0 = pd.read_parquet(SP + 'b302.parquet'); B0 = B0[B0.exit < END].copy(); B0['entry'] = pd.to_datetime(B0.entry); B0['exit'] = pd.to_datetime(B0.exit); B0['h1'] = False
def sim(Bs, N=6, brk=.08, extra=0.0, ret=None):
    global T0
    X = Bs.copy(); X['ret'] = (X.ret if ret is None else ret); key = X.brank.values + 1e-6 * X.ratio.rank(ascending=False).values; T0 = X.assign(_k=key).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(N, brk); R = np.zeros(len(D))
    for tk, e, x, r, ra in tk_:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / N; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4 - extra)) ** (1 / n) - w
    return pd.Series(R, index=D), tk_
def cagr(s): e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; return e.iloc[-1] ** (1 / y) - 1
def mdd(s): e = (1 + s).cumprod(); return float((e / e.cummax() - 1).min())
s0, TK = sim(B0); s = s0[s0.index >= '2019-01-01']; x = s.values; print(f'book: {len(TK)} trades taken, CAGR {cagr(s):.1%}, max DD {mdd(s):.1%}, Sharpe {sharpe(x):.2f}, invested days {np.mean(x != 0):.0%}')
print(f'\nS1 PSR(0) {psr(x):.3f}'); [print(f'   DSR N={n:5d}: {dsr(x, n)[0]:.3f} (expected max skill-less SR {dsr(x, n)[1]:.2f})') for n in (50, 200, 1000, 5000)]
Bi = stationary_bootstrap(x, 3000, 20); cg = np.array([(np.prod(1 + x[i]) ** (252 / len(x)) - 1) for i in Bi]); dd = np.array([float((np.cumprod(1 + x[i]) / np.maximum.accumulate(np.cumprod(1 + x[i])) - 1).min()) for i in Bi]); bs = np.array([sharpe(x[i]) for i in Bi])
print(f'\nS2 block bootstrap: P(SR<0) {np.mean(bs < 0):.4f} | SR 5-95% {np.percentile(bs, 5):.2f}-{np.percentile(bs, 95):.2f} | CAGR 5-95% {np.percentile(cg, 5):.1%}-{np.percentile(cg, 95):.1%} | max DD actual {mdd(s):.1%}, median {np.median(dd):.1%}, worst-5% {np.percentile(dd, 5):.1%}')
# ---- bars for S3 / S5
O = {}
for tk in B0.tk.unique():
    f = sorted(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))
    if not f: continue
    H = pd.read_parquet(f[-1], columns=['datetime', 'open']); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); O[tk] = H.open.sort_index()
print(f'\nbars available for {len(O)} of {B0.tk.nunique()} tickers')
rng = np.random.default_rng(5); act = []; hold = []; tks = []
for tk, e, xx, r, ra in TK:
    if tk not in O: continue
    o = O[tk]; i = o.index.searchsorted(e); j = o.index.searchsorted(xx)
    if i >= len(o) or j >= len(o) or j <= i: continue
    act.append(r); hold.append(j - i); tks.append(tk)
act = np.array(act); pl = np.zeros(2000)
for d in range(2000):
    rs = []
    for tk, h in zip(tks, hold):
        o = o_ = O[tk].values; i = rng.integers(0, len(o_) - h - 1); rs.append(o_[i + h] / o_[i] - 1 - .0002)
    pl[d] = np.mean(rs)
print(f'S3 random-entry placebo (same tickers, same hold in bars): actual mean trade {act.mean():+.3%} (n{len(act)}) | placebo mean {pl.mean():+.3%} sd {pl.std():.3%} | p(placebo >= actual) = {np.mean(pl >= act.mean()):.4f}; win rate actual {np.mean(act > 0):.0%}')
print('\nS4 cost stress (extra bp per trade round trip on top of the simulator\'s 2+2):')
for bp in (0, 5, 10, 20, 40):
    z, _ = sim(B0, extra=bp * 1e-4); z = z[z.index >= '2019-01-01']; print(f'   +{bp:2d} bp: CAGR {cagr(z):.1%}, max DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}')
print('\nS5 late entry (delay in 15m bars; price = that bar open):')
for k in (0, 1, 2, 4):
    ret = B0.ret.values.copy(); n_ok = 0
    if k:
        for tk in O:
            m = (B0.tk == tk).values; o = O[tk]; ie = o.index.searchsorted(B0.entry[m].values); ix = o.index.searchsorted(B0.exit[m].values); idl = np.minimum(ie + k, len(o) - 1); ok = (idl < ix) & (ie < len(o)) & (ix < len(o))
            ov = o.values; new = ov[np.minimum(ix, len(o) - 1)] / ov[idl] - 1 - .0002; old = B0.ret.values[m]; ret[np.where(m)[0]] = np.where(ok, new, old); n_ok += ok.sum()
    z, _ = sim(B0, ret=ret); z = z[z.index >= '2019-01-01']; print(f'   delay {k} bars ({15 * k:2d} min): CAGR {cagr(z):.1%}, max DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}, mean candidate-trade ret {np.mean(ret):+.3%}')
print('\nS6 neighbourhood (slots x breaker):'); res = []
for N in (4, 5, 6, 8, 10):
    for br in (0, .05, .08, .12):
        z, _ = sim(B0, N=N, brk=br); z = z[z.index >= '2019-01-01']; res.append((N, br, cagr(z), sharpe(z.values), mdd(z)))
R = pd.DataFrame(res, columns=['N', 'brk', 'cagr', 'sr', 'dd']); print(R.pivot(index='N', columns='brk', values='cagr').map(lambda v: f"{v:.1%}").to_string()); print(f'   profitable {np.mean(R.cagr > 0):.0%} of {len(R)} | CAGR min {R.cagr.min():.1%} median {R.cagr.median():.1%} | Sharpe min {R.sr.min():.2f} median {R.sr.median():.2f}')
srt = sorted(TK, key=lambda t: -t[3]); print('\nS7 concentration: drop the best trades')
for q in (.05, .10):
    k = int(len(TK) * q); drop = {(t[0], t[1]) for t in srt[:k]}; Bd = B0[[(a, b) not in drop for a, b in zip(B0.tk, B0.entry)]]; z, _ = sim(Bd); z = z[z.index >= '2019-01-01']; print(f'   without the best {q:.0%} ({k} trades): CAGR {cagr(z):.1%}, max DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}')
yr = (1 + s).groupby(s.index.year).prod() - 1; print('   by year:', {int(k): f'{v:.1%}' for k, v in yr.items()}); print(f'   halves: 2019-22 CAGR {cagr(s[s.index < "2023-01-01"]):.1%} Sharpe {sharpe(s[s.index < "2023-01-01"].values):.2f} | 2023-26 CAGR {cagr(s[s.index >= "2023-01-01"]):.1%} Sharpe {sharpe(s[s.index >= "2023-01-01"].values):.2f}')
