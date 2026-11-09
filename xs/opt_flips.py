"""Nov 9 2026: THE SCREENER IN OPTIONS, REAL PRICES. DECLARED BEFORE RESULTS.
The screener only uses option VOLUME (the ratio). Its options sister model (xs/options_model.py, xs/put_sell_prob.py) was a price PROXY: capped calls failed,
but the short-put proxy had an edge - after a flip the stock ends above entry - 0.5 sigma in 5 days 81.6% vs 76.8% of random days (2019-21), 77.8% vs 72.6% (2022-26).
This tests it with REAL ThetaData end-of-day bid / ask (data/theta_compact, 35 S&P names + SPY / QQQ / IWM, 2018 - Oct 2026).
Trades: screener flips (point-in-time S&P, options ratio >= 1.5, both tracks), one per name per day, in names with option data. Sold / bought at THAT day's close quotes.
Spot = put-call-parity spot (same units as strikes); sigma = 20-day stdev of daily log changes (splits removed). td = trading days to expiry.
Structures (expiry: WEEKLY = 5-10 days to expiry, closest to 7; MONTHLY = 21-45 days, closest to 30):
  PCS k   put credit spread: short the highest put at or below S (1 - k sigma sqrt(td)), long the highest put at or below S (1 - (k + 1) sigma sqrt(td)); k = 0, 0.5, 1
          risk = width - credit (the long put is the stop). Return on risk.
  CSP k   cash-secured short put at the same short strike; return on the strike (cash secured).
  CALL    long call, nearest strike at or above S; return on premium.   CDS  call debit spread, long that call, short the call at or above S (1 + sigma sqrt(td)).
Exits: EXP = hold to expiry (intrinsic at expiry spot) | VEC = close at the screener's own exit (that day's close quotes) or expiry, whichever first.
Fills: TOUCH = sell at the bid, buy at the ask (worst case) | MID = at the mid. A short leg needs bid >= 0.05 and spread <= 50% of mid, else no trade.
Baseline: the same structures on every 5th trading day in the same names (VEC uses a holding time drawn from the flips' own holding times).
Report 2019-21 / 2022-26: trades, win rate, average return, flips vs baseline."""
import numpy as np, pandas as pd, glob, os, warnings; warnings.filterwarnings('ignore')
U = '/mnt/user-data/uploads/fartpie'
T = pd.read_parquet(f'{U}/xs/ext/wide_T.parquet'); T = T[T.sp & (T.ratio >= 1.5)].copy()
T['d0'] = T.entry.dt.normalize(); T['d1'] = T.exit.dt.normalize(); T = T.sort_values(['tk', 'entry']).drop_duplicates(['tk', 'd0'])
D = lambda x: np.datetime64(x, 'D').astype(np.int64)
class Book:
    def __init__(self, X):
        X = X.sort_values(['d', 'e', 'r', 'K']).reset_index(drop=True)
        self.d = X.d.values.astype('datetime64[D]').astype(np.int64); self.e = X.e.values.astype('datetime64[D]').astype(np.int64)
        self.K, self.r, self.bid, self.ask, self.dte = X.K.values.astype(float), X.r.values, X.bid.values.astype(float), X.ask.values.astype(float), X.dte.values
        self.days, st = np.unique(self.d, return_index=True); self.rng = dict(zip(self.days, zip(st, np.r_[st[1:], len(self.d)])))
        S = X.groupby('d').S.first(); self.S = dict(zip(self.days, S.values.astype(float)))
        r = np.log(S).diff(); r[r.abs() > .3] = np.nan; self.sig = dict(zip(self.days, r.rolling(20, min_periods=15).std().values))
    def spot_at(self, d):
        k = np.searchsorted(self.days, d, 'right') - 1; return self.S[self.days[k]] if k >= 0 else np.nan
    def leg(self, d, e, K, r, side, fill):
        if d >= e or d not in self.rng: return max(0.0, (self.spot_at(min(d, e)) - K) * r)
        a, b = self.rng[d]; w = np.where((self.e[a:b] == e) & (self.r[a:b] == r) & (self.K[a:b] == K))[0]
        if not len(w) or self.ask[a + w[0]] <= 0: return max(0.0, (self.S[d] - K) * r)
        bi, ak = self.bid[a + w[0]], self.ask[a + w[0]]; return (bi + ak) / 2 if fill == 'MID' else (bi if side > 0 else ak)
B = {os.path.basename(f)[:-8]: Book(pd.read_parquet(f)) for f in sorted(glob.glob(f'{U}/data/theta_compact/*.parquet'))}
WIN = {'W': (5, 10, 7), 'M': (21, 45, 30)}
def entry_all(bk, d0, d1s):
    """all structures for one entry day; d1s = {'EXP': None, 'VEC': exit day}"""
    out = []
    if d0 not in bk.rng: return out
    S0, sg = bk.S[d0], bk.sig[d0]
    if not (np.isfinite(S0) and np.isfinite(sg)): return out
    a, b = bk.rng[d0]
    for ten, (lo, hi, tg) in WIN.items():
        dte = bk.dte[a:b]; ok_ = (dte >= lo) & (dte <= hi)
        if not ok_.any(): continue
        dsel = dte[ok_][np.argmin(np.abs(dte[ok_] - tg))]; ix = a + np.where(dte == dsel)[0]; e = bk.e[ix[0]]; td = max(1, int(round(dsel * 252 / 365))); w = sg * np.sqrt(td)
        K, r, bi, ak = bk.K[ix], bk.r[ix], bk.bid[ix], bk.ask[ix]
        def px(j, side, fill): return (bi[j] + ak[j]) / 2 if fill == 'MID' else (ak[j] if side > 0 else bi[j])
        puts = np.where((r == -1) & (bi >= .05) & (ak > bi))[0]; puts_any = np.where((r == -1) & (ak > 0))[0]; calls = np.where((r == 1) & (ak > 0))[0]
        for st, k in (('PCS', 0.0), ('PCS', 0.5), ('PCS', 1.0), ('CSP', 0.5), ('CSP', 1.0), ('CALL', 0), ('CDS', 0)):
            if st in ('PCS', 'CSP'):
                s = puts[K[puts] <= S0 * (1 - k * w)]
                if not len(s): continue
                s = s[np.argmax(K[s])]
                if ak[s] - bi[s] > .5 * (ak[s] + bi[s]) / 2: continue
                l = None
                if st == 'PCS':
                    lc = puts_any[(K[puts_any] <= S0 * (1 - (k + 1) * w)) & (K[puts_any] < K[s])]
                    if not len(lc): continue
                    l = lc[np.argmax(K[lc])]
                nm = f'{st} {k:g}'
            else:
                cc = calls[K[calls] >= S0]
                if not len(cc): continue
                s = cc[np.argmin(K[cc])]
                if bi[s] <= 0 or ak[s] - bi[s] > .5 * (ak[s] + bi[s]) / 2: continue
                l = None
                if st == 'CDS':
                    lc = calls[(K[calls] >= S0 * (1 + w)) & (K[calls] > K[s]) & (bi[calls] > 0)]
                    if not len(lc): continue
                    l = lc[np.argmin(K[lc])]
                nm = st
            for ex, d1 in d1s.items():
                if st in ('CALL', 'CDS') and ex == 'EXP' and ten == 'M': continue
                dx = e if d1 is None else min(d1, e)
                for fill in ('TOUCH', 'MID'):
                    if st == 'CSP':
                        pnl = px(s, -1, fill) - bk.leg(dx, e, K[s], -1, -1, fill); ret = pnl / K[s]
                    elif st == 'PCS':
                        net = px(s, -1, fill) - px(l, +1, fill); risk = (K[s] - K[l]) - net
                        if net <= 0 or risk <= 0: continue
                        pnl = net - bk.leg(dx, e, K[s], -1, -1, fill) + bk.leg(dx, e, K[l], -1, +1, fill); ret = pnl / risk
                    elif st == 'CALL':
                        prem = px(s, +1, fill); pnl = bk.leg(dx, e, K[s], 1, +1, fill) - prem; ret = pnl / prem
                    else:
                        net = px(s, +1, fill) - px(l, -1, fill)
                        if net <= 0: continue
                        pnl = bk.leg(dx, e, K[s], 1, +1, fill) - bk.leg(dx, e, K[l], 1, -1, fill) - net; ret = pnl / net
                    out.append((ten, nm, ex, fill, ret, pnl > 0))
    return out
if __name__ == '__main__':
    F = T[T.tk.isin(B)].copy(); F = F[F.d0 >= '2019-01-01']
    hold = np.array([max(0, np.searchsorted(B[r_.tk].days, D(r_.d1)) - np.searchsorted(B[r_.tk].days, D(r_.d0))) for r_ in F.itertuples()]); rng = np.random.default_rng(0)
    base = []
    for tk in sorted(set(F.tk)):
        dd_ = B[tk].days[B[tk].days >= D('2019-01-01')]
        for j in range(0, len(dd_) - 1, 5): base.append((tk, dd_[j], dd_[min(j + int(rng.choice(hold)), len(dd_) - 1)]))
    print(f'flips with option data: {len(F)} ({F.tk.nunique()} names), median hold {np.median(hold):.0f} trading days, mean {hold.mean():.1f}; baseline days {len(base)}', flush=True)
    R = []
    for lab, rows in (('flip', [(r_.tk, D(r_.d0), D(r_.d1)) for r_ in F.itertuples()]), ('base', base)):
        for tk, d0, d1 in rows:
            for x in entry_all(B[tk], d0, {'EXP': None, 'VEC': d1}): R.append((lab, tk, d0) + x)
    R = pd.DataFrame(R, columns=['set', 'tk', 'd0', 'ten', 'st', 'ex', 'fill', 'ret', 'win']); R['d0'] = pd.to_datetime(R.d0.values.astype('datetime64[D]'))
    R['per'] = np.where(R.d0.dt.year >= 2022, '22-26', '19-21'); R.to_parquet('/tmp/claude-0/opt_flips_R.parquet')
    print('\naverage return per trade (PCS on risk, CSP on cash, calls on premium) | win rate | n  -  flips vs baseline', flush=True)
    for (ten, st, ex, fill), g in R.groupby(['ten', 'st', 'ex', 'fill'], sort=False):
        out = []
        for per in ('19-21', '22-26'):
            for s in ('flip', 'base'):
                t = g[(g.per == per) & (g.set == s)]; out.append(f'{t.ret.mean() * 100:+7.2f}% {t.win.mean():3.0%} {len(t):5d}')
        print(f'  {ten} {st:<7} {ex} {fill:<5} | 19-21 flip {out[0]}  base {out[1]} | 22-26 flip {out[2]}  base {out[3]}', flush=True)
