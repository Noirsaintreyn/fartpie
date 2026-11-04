"""Nov 4 2026: ICT order blocks (OB) and rejection blocks (RB), 5m / 15m / 30m / 1h / 4h / daily, stocks and futures. DECLARED BEFORE RESULTS.
DATA  stocks: 5m bars (lse_ohlcv_cache, UTC -> ET), RTH 09:30-16:00 only, 2018-2026, single stocks with 5m files (ETFs left out); 15m..4h built from 5m (09:30-anchored), daily = RTH day.
      futures: MNQ 1-minute last prices (Dec 2024 - Nov 2025, 24h) -> 5m..4h bars (wicks from 1-min prints, so slightly understated);
               Yahoo NQ=F ES=F GC=F CL=F 1h (last ~730 days) -> 1h and 4h; Yahoo daily since 2010 -> D.
RULES (per bar series; ATR = Wilder 14 of that series; swing = 2-bar fractal, known 2 bars later)
  bullish OB: a down candle i; within the next 3 bars a close above its high that is >= 1.5 ATR above its low AND above the last confirmed swing high (displacement + break of structure).
              zone = [low_i, high_i], live from the confirming bar.  bearish OB: mirror.
  bullish RB: a confirmed swing low whose lower wick (body bottom - low) is >= 50% of the candle's range; zone = [low, body bottom].  bearish RB: mirror at swing highs.
  retest: the first bar after the zone goes live that trades back into it (bullish: low <= zone top), within 100 bars; a close beyond the far edge before that cancels the zone;
          a bar that opens beyond the far edge is skipped. Entry = zone near edge (or the open, if it opened inside the zone).
OUTCOMES from the entry, using the ATR at formation:  SYM = +1 ATR target before -1 ATR stop within 50 bars (same bar: counted as the stop) -> win rate;
          ICT = stop 0.1 ATR beyond the far edge, target 2R -> mean R (timeouts marked to the 50th bar);  D10 = move after 10 bars, in ATR, in the block's direction.
PLACEBOS  OB: every 5th ordinary candle of the same colour with NO displacement/BOS, same zone/retest/outcome rules.   RB: swing lows/highs with a SMALL wick (< 50% of range).
          RANDOM: entries at random bars (1 in 20), both directions, SYM and D10 only.
PASS (per timeframe x asset class x block type): beats RANDOM entries on SYM win rate AND a MATCHED RANDOM 2R trade (same direction, same risk in ATR, entry at a random
          bar's close) on ICT mean R, in BOTH halves, each z >= 2. (A first run compared with the placebo zones only; their tighter zones lose more to the conservative
          same-bar rule, so that comparison flattered the blocks; it is still printed.)"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from numba import njit
ROOT = '/mnt/user-data/uploads/fartpie'
@njit(cache=True)
def wilder(h, l, c, n):
    a = np.full(len(c), np.nan); tr = np.empty(len(c)); tr[0] = h[0] - l[0]
    for i in range(1, len(c)): tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    if len(c) <= n: return a
    s = tr[:n].mean(); a[n - 1] = s
    for i in range(n, len(c)): s = s + (tr[i] - s) / n; a[i] = s
    return a
@njit(cache=True)
def outcome(o, h, l, c, a, f, zb, zt, d, maxwait, hold):
    """first retest of zone [zb, zt] after bar f, direction d (+1 bull / -1 bear). returns (touch index, sym win 1/0, ict R, d10) or touch = -1"""
    n = len(c); at = a[f]
    if not (at > 0): return -1, 0.0, 0.0, 0.0, 0.0
    for t in range(f + 1, min(n, f + 1 + maxwait)):
        if d == 1:
            if o[t] < zb: return -1, 0.0, 0.0, 0.0, 0.0
            if l[t] <= zt:
                e = zt if o[t] >= zt else o[t]; st = e - at; tg = e + at; s2 = zb - .1 * at; r = e - s2; t2 = e + 2 * r
                sym = -1.0; ict = 9.0
                if l[t] <= st: sym = 0.0
                if l[t] <= s2: ict = -1.0
                for k in range(t + 1, min(n, t + 1 + hold)):
                    if sym < 0:
                        if l[k] <= st: sym = 0.0
                        elif h[k] >= tg: sym = 1.0
                    if ict == 9.0:
                        if l[k] <= s2: ict = -1.0
                        elif h[k] >= t2: ict = 2.0
                    if sym >= 0 and ict != 9.0: break
                last = c[min(n - 1, t + hold)]
                if sym < 0: sym = 1.0 if last > e else 0.0
                if ict == 9.0: ict = (last - e) / r if r > 0 else 0.0
                d10 = (c[min(n - 1, t + 10)] - e) / at
                return t, sym, ict, d10, r / at
            if c[t] < zb: return -1, 0.0, 0.0, 0.0, 0.0
        else:
            if o[t] > zt: return -1, 0.0, 0.0, 0.0, 0.0
            if h[t] >= zb:
                e = zb if o[t] <= zb else o[t]; st = e + at; tg = e - at; s2 = zt + .1 * at; r = s2 - e; t2 = e - 2 * r
                sym = -1.0; ict = 9.0
                if h[t] >= st: sym = 0.0
                if h[t] >= s2: ict = -1.0
                for k in range(t + 1, min(n, t + 1 + hold)):
                    if sym < 0:
                        if h[k] >= st: sym = 0.0
                        elif l[k] <= tg: sym = 1.0
                    if ict == 9.0:
                        if h[k] >= s2: ict = -1.0
                        elif l[k] <= t2: ict = 2.0
                    if sym >= 0 and ict != 9.0: break
                last = c[min(n - 1, t + hold)]
                if sym < 0: sym = 1.0 if last < e else 0.0
                if ict == 9.0: ict = (e - last) / r if r > 0 else 0.0
                d10 = (e - c[min(n - 1, t + 10)]) / at
                return t, sym, ict, d10, r / at
            if c[t] > zt: return -1, 0.0, 0.0, 0.0, 0.0
    return -1, 0.0, 0.0, 0.0, 0.0
@njit(cache=True)
def scan(o, h, l, c, a):
    """events: rows (kind, dir, formed, touch, sym, ict, d10); kind 0 OB, 1 OB-placebo, 2 RB, 3 RB-placebo, 4 random"""
    n = len(c); out = []
    sh = np.zeros(n, np.bool_); sl = np.zeros(n, np.bool_)
    for i in range(2, n - 2):
        if h[i] > max(h[i - 1], h[i - 2]) and h[i] >= max(h[i + 1], h[i + 2]): sh[i] = True
        if l[i] < min(l[i - 1], l[i - 2]) and l[i] <= min(l[i + 1], l[i + 2]): sl[i] = True
    lsh = np.full(n, np.nan); lsl = np.full(n, np.nan); ch = np.nan; cl = np.nan
    for j in range(n):                                   # last swing CONFIRMED by bar j (pivot at j-2)
        if j >= 2 and sh[j - 2]: ch = h[j - 2]
        if j >= 2 and sl[j - 2]: cl = l[j - 2]
        lsh[j] = ch; lsl[j] = cl
    pc = 0
    for i in range(20, n - 3):
        if not (a[i] > 0): continue
        if c[i] < o[i]:                                  # bullish OB candidate (down candle)
            f = -1
            for k in range(1, 4):
                j = i + k
                if c[j] > h[i] and c[j] - l[i] >= 1.5 * a[i] and lsh[j] == lsh[j] and c[j] > lsh[j]: f = j; break
            if f > 0:
                t, s, r, d, rk = outcome(o, h, l, c, a, f, l[i], h[i], 1, 100, 50)
                if t > 0: out.append((0.0, 1.0, float(f), float(t), s, r, d, rk))
            else:
                pc += 1
                if pc % 5 == 0:
                    t, s, r, d, rk = outcome(o, h, l, c, a, i + 1, l[i], h[i], 1, 100, 50)
                    if t > 0: out.append((1.0, 1.0, float(i + 1), float(t), s, r, d, rk))
        elif c[i] > o[i]:                                # bearish OB candidate (up candle)
            f = -1
            for k in range(1, 4):
                j = i + k
                if c[j] < l[i] and h[i] - c[j] >= 1.5 * a[i] and lsl[j] == lsl[j] and c[j] < lsl[j]: f = j; break
            if f > 0:
                t, s, r, d, rk = outcome(o, h, l, c, a, f, l[i], h[i], -1, 100, 50)
                if t > 0: out.append((0.0, -1.0, float(f), float(t), s, r, d, rk))
            else:
                pc += 1
                if pc % 5 == 0:
                    t, s, r, d, rk = outcome(o, h, l, c, a, i + 1, l[i], h[i], -1, 100, 50)
                    if t > 0: out.append((1.0, -1.0, float(i + 1), float(t), s, r, d, rk))
        rg = h[i] - l[i]
        if rg > 0 and i + 2 < n:
            if sl[i]:
                bb = min(o[i], c[i]); big = (bb - l[i]) >= .5 * rg
                if bb > l[i]:
                    t, s, r, d, rk = outcome(o, h, l, c, a, i + 2, l[i], bb, 1, 100, 50)
                    if t > 0: out.append((2.0 if big else 3.0, 1.0, float(i + 2), float(t), s, r, d, rk))
            if sh[i]:
                bt = max(o[i], c[i]); big = (h[i] - bt) >= .5 * rg
                if h[i] > bt:
                    t, s, r, d, rk = outcome(o, h, l, c, a, i + 2, bt, h[i], -1, 100, 50)
                    if t > 0: out.append((2.0 if big else 3.0, -1.0, float(i + 2), float(t), s, r, d, rk))
        if i % 20 == 0 and i + 51 < n:                   # random entries at the close, both directions, SYM + D10
            for dd in (1.0, -1.0):
                e = c[i]; at = a[i]; sym = -1.0
                for k in range(i + 1, i + 51):
                    up = h[k] >= e + at; dn = l[k] <= e - at
                    if dd > 0:
                        if dn: sym = 0.0; break
                        if up: sym = 1.0; break
                    else:
                        if up: sym = 0.0; break
                        if dn: sym = 1.0; break
                if sym < 0: sym = 1.0 if (c[i + 50] - e) * dd > 0 else 0.0
                out.append((4.0, dd, float(i), float(i), sym, 0.0, (c[i + 10] - e) * dd / at, 0.0))
    return out
@njit(cache=True)
def matched(o, h, l, c, a, risk, dirs, hold):
    """for each event: a pseudo-random bar, entry at its close, same direction and same risk (in ATR), stop -1R / target +2R, same conservative rules -> R"""
    n = len(c); out = np.full(len(risk), np.nan)
    for k in range(len(risk)):
        u = 20 + (k * 104729 + 7) % (n - hold - 25); at = a[u]
        if not (at > 0) or not (risk[k] > 0): continue
        e = c[u]; r = risk[k] * at; d = dirs[k]; st = e - d * r; tg = e + 2 * d * r; res = 9.0
        for j in range(u + 1, u + 1 + hold):
            if (d > 0 and l[j] <= st) or (d < 0 and h[j] >= st): res = -1.0; break
            if (d > 0 and h[j] >= tg) or (d < 0 and l[j] <= tg): res = 2.0; break
        if res == 9.0: res = (c[u + hold] - e) * d / r
        out[k] = res
    return out
# ---------------------------------------------------------------- data
def rs(D, rule, off):
    x = D.set_index('dt')[['open', 'high', 'low', 'close']]
    g = x.groupby([x.index.normalize(), pd.Grouper(freq=rule, origin='start_day', offset=off)]) if off is not None else x.resample(rule)
    r = g.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if off is not None: r.index = r.index.get_level_values(1)
    r = r.reset_index(); r.columns = ['dt', 'open', 'high', 'low', 'close']; return r
def stocks():
    out = {}
    for f in sorted(glob.glob(f'{ROOT}/lse_ohlcv_cache/*_5m_*.parquet')):
        tk = os.path.basename(f).split('_')[0]
        if tk in ('SPY', 'QQQ', 'SMH', 'XLE', 'XLF') or tk in out: continue
        H = pd.read_parquet(f); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt')[['dt', 'open', 'high', 'low', 'close']]
        S = {'5m': H.reset_index(drop=True)}
        for tf, rule in (('15m', '15min'), ('30m', '30min'), ('1h', '1h'), ('4h', '4h')): S[tf] = rs(H, rule, '9h30min')
        d = H.groupby(H.dt.dt.normalize()).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index().rename(columns={'dt': 'dt'})
        d.columns = ['dt', 'open', 'high', 'low', 'close']; S['D'] = d; out[tk] = S
    return out
def futures():
    out = {}
    M = pd.read_parquet(f'{ROOT}/mnq_orderflow_1min.parquet'); M['dt'] = pd.to_datetime(M.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    M = M[['dt', 'last_price']].dropna().drop_duplicates('dt').sort_values('dt'); M['open'] = M['high'] = M['low'] = M['close'] = M.last_price
    S = {}
    for tf, rule in (('5m', '5min'), ('15m', '15min'), ('30m', '30min'), ('1h', '1h'), ('4h', '4h')): S[tf] = rs(M, rule, None)
    out['MNQ (1-min prints)'] = S
    import yfinance as yf
    for sym in ('NQ=F', 'ES=F', 'GC=F', 'CL=F'):
        S = {}
        try:
            Y = yf.download(sym, period='730d', interval='1h', progress=False, auto_adjust=False)
            if len(Y):
                Y.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in Y.columns]; Y = Y.reset_index(); Y['dt'] = pd.to_datetime(Y.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
                Y = Y[['dt', 'open', 'high', 'low', 'close']].dropna(); S['1h'] = Y.reset_index(drop=True); S['4h'] = rs(Y, '4h', None)
            Dd = yf.download(sym, start='2010-01-01', interval='1d', progress=False, auto_adjust=False)
            Dd.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in Dd.columns]; Dd = Dd.reset_index(); Dd['dt'] = pd.to_datetime(Dd.iloc[:, 0])
            S['D'] = Dd[['dt', 'open', 'high', 'low', 'close']].dropna().reset_index(drop=True)
        except Exception as ex: print('  !', sym, str(ex)[:60])
        out[sym] = S
    return out
if __name__ == '__main__':
    KIND = {0: 'OB', 1: 'OB placebo', 2: 'RB', 3: 'RB placebo', 4: 'random'}
    ev = []
    for cls_, loader in (('stocks', stocks), ('futures', futures)):
        U = loader(); print(cls_, 'instruments', len(U), list(U)[:6], flush=True)
        for tk, S in U.items():
            for tf, B in S.items():
                if len(B) < 300: continue
                o, h, l, c = (B[k].values.astype(np.float64) for k in ('open', 'high', 'low', 'close')); a = wilder(h, l, c, 14)
                r = scan(o, h, l, c, a)
                if not r: continue
                R = pd.DataFrame(r, columns=['kind', 'dir', 'f', 't', 'sym', 'ict', 'd10', 'risk']); R['ts'] = B.dt.values[R.t.astype(int)]
                mk = R.kind.isin([0.0, 2.0]).values; R['mrand'] = np.nan
                if mk.any(): R.loc[mk, 'mrand'] = matched(o, h, l, c, a, R.risk.values[mk].astype(np.float64), R.dir.values[mk].astype(np.float64), 50)
                mid = B.dt.iloc[len(B) // 2] if cls_ == 'futures' else pd.Timestamp('2022-01-01')
                R['half'] = np.where(R.ts < mid, 'H1', 'H2'); R['cls'] = cls_; R['tf'] = tf; R['tk'] = tk; ev.append(R)
            print(' ', tk, flush=True)
    E = pd.concat(ev, ignore_index=True); E['kind'] = E.kind.map(KIND); E.to_parquet('/tmp/claude-0/ml/ob_rb_events.parquet')
    print(f'\nevents {len(E):,}', flush=True)
    TFS = ['5m', '15m', '30m', '1h', '4h', 'D']
    def z(p1, n1, p2, n2):
        p = (p1 * n1 + p2 * n2) / (n1 + n2); se = np.sqrt(max(p * (1 - p), 1e-9) * (1 / n1 + 1 / n2)); return (p1 - p2) / se
    def zm(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 2 and len(b) > 2 else np.nan
    for cls_ in ('stocks', 'futures'):
        print(f'\n==== {cls_.upper()}  (cells: n | SYM win% vs placebo | ICT mean R vs placebo | D10 ATR vs placebo; per half H1 / H2)', flush=True)
        for tf in TFS:
            Z = E[(E.cls == cls_) & (E.tf == tf)]
            if not len(Z): continue
            rnd = Z[Z.kind == 'random']; print(f'  [{tf}] random entries: SYM win {rnd.sym.mean():.1%} (n {len(rnd):,}), D10 {rnd.d10.mean():+.3f} ATR', flush=True)
            for blk, pl in (('OB', 'OB placebo'), ('RB', 'RB placebo')):
                for dr, dn in ((1, 'bull'), (-1, 'bear')):
                    s = f'    {blk} {dn:<4}'; ok = True
                    for hf in ('H1', 'H2'):
                        A = Z[(Z.kind == blk) & (Z.dir == dr) & (Z.half == hf)]; P = Z[(Z.kind == pl) & (Z.dir == dr) & (Z.half == hf)]
                        if len(A) < 30 or len(P) < 30: s += f' | {hf} n{len(A)} (too few)'; ok = False; continue
                        Rr = rnd[(rnd.dir == dr) & (rnd.half == hf)]; zs = z(A.sym.mean(), len(A), Rr.sym.mean(), len(Rr)); mr = A.mrand.dropna(); zi = zm(A.ict, mr)
                        ok = ok and zs >= 2 and zi >= 2
                        s += f' | {hf} n{len(A):>6,} SYM {A.sym.mean():.1%} vs rnd {Rr.sym.mean():.1%} (z{zs:+.1f}) ICT {A.ict.mean():+.3f}R vs matched rnd {mr.mean():+.3f}R (z{zi:+.1f}) [placebo zone {P.ict.mean():+.3f}R] risk {A.risk.median():.2f}ATR'
                    print(s + ('   PASS' if ok else ''), flush=True)
