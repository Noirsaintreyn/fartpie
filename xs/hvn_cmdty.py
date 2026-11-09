"""Nov 9 2026: HVN MEAN REVERSION ON GOLD (GC) AND CRUDE (CL) vs NQ, SAME RULES (Rey: try more mean-reverting assets). DECLARED BEFORE RESULTS.
Data: Yahoo continuous futures NQ=F GC=F CL=F (the only intraday CL / GC reachable here): 1h bars Jun 2024 - Nov 2026, 5m bars only the last ~60 days. Same resolution
for all three, so the comparison is like for like. Continuous contracts: roll days add noise (real and placebo alike).
Main sessions (ET): NQ 09:30-16:00 | GC 08:20-13:30 | CL 09:00-14:30. 1h bars: NQ 10-16, GC 08-13, CL 09-14 (clock-aligned bars inside the session).
0) IS IT MORE MEAN-REVERTING? within-session lag-1 autocorrelation of bar returns and the variance ratio VR = var(k-bar return) / (k x var(1-bar)) (< 1 = mean-reverting),
   5m (k = 6) and 1h (k = 3).
1) HVN FADE, as xs/hvn_mr.py: levels = prior main-session volume profile (bars' volume spread over their range, bin = median ADR / 60), POC + the 2 biggest other
   peaks >= 0.05 ADR apart; FIRST touch with the bar before fully on one side; limit fill at the level; stop inside the touch bar = stop; stop + target in one bar = stop.
   5m: stop 0.05 ADR, targets 0.05 / 0.10 ADR. 1h: stop 0.10 ADR, targets 0.10 / 0.20 ADR (coarser bars). Out at the session close. Cost: NQ 1 pt, GC $0.20, CL $0.02.
   ADR = 20-day mean of the full-day range (from 1h bars). Results in R (multiples of the stop), so the three can be compared.
   Pairings: THOD line (anchor = session open, sigma = 1h Garman-Klass x Hurst over the last 60 1h bars, lines +-1, 2, 3 sigma, within 0.25 sigma), EXH (range used >= 0.8 x
   expected range, k fit on the first half). No gold / oil options data here, so no gamma regime for GC / CL.
   PLACEBO: level moved 0.03-0.12 ADR to a random side, 5 draws. 1h split into halves by date."""
import os, numpy as np, pandas as pd, yfinance as yf
CACHE = '/home/claude/fartpie/data/yahoo'; os.makedirs(CACHE, exist_ok=True)
SYM = {'NQ': ('NQ=F', 570, 960, range(10, 16), 1.0), 'GC': ('GC=F', 500, 810, range(8, 13), .20), 'CL': ('CL=F', 540, 870, range(9, 14), .02)}
def get(sym, iv, per):
    f = f'{CACHE}/{sym.replace("=", "")}_{iv}.parquet'
    if os.path.exists(f): return pd.read_parquet(f)
    Y = yf.download(sym, period=per, interval=iv, progress=False, auto_adjust=False)
    if isinstance(Y.columns, pd.MultiIndex): Y.columns = Y.columns.get_level_values(0)
    Y = Y.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']].dropna(); Y.index = pd.DatetimeIndex(Y.index).tz_convert('America/New_York').tz_localize(None)
    Y.to_parquet(f); return Y
def bot_sig(B, t):
    k = B.index.searchsorted(t, 'left'); b = B.iloc[max(0, k - 102):k]
    if len(b) < 40: return np.nan
    r = b.iloc[-60:]; var = np.maximum(.5 * np.log(r.high / r.low) ** 2 - (2 * np.log(2) - 1) * np.log(r.close / r.open) ** 2, 1e-12); s = np.sqrt(var.mean()) * r.close.values[-1]
    cl = b.close.values; lags = range(2, 20); tau = [np.std(cl[g:] - cl[:-g]) for g in lags]; H_ = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    mult = 1.0 + .3 * (H_ - .5) / .5 if H_ > .6 else (1.0 - .2 * (.5 - H_) / .5 if H_ < .4 else 1.0); return s * mult if abs(mult - 1) * 2 > .1 else s
def sessions(B, a0, a1, hrs, iv):
    mm = B.index.hour * 60 + B.index.minute
    if iv == '1h': msk = np.isin(B.index.hour, list(hrs)) & (B.index.minute == 0)
    else: msk = (mm >= a0) & (mm < a1)
    S = B[msk & (B.index.dayofweek < 5)].copy(); S['day'] = S.index.normalize(); S['m'] = S.index.hour * 60 + S.index.minute
    return {d: g for d, g in S.groupby('day') if len(g) >= (4 if iv == '1h' else 40)}
def hvns(g, adr, bn):
    lo0 = np.floor(g.low.min() / bn); nb = int(np.floor(g.high.max() / bn) - lo0) + 1; vp = np.zeros(nb)
    for lo_, hi_, v_ in zip(g.low.values, g.high.values, g.volume.values):
        a, b = int(np.floor(lo_ / bn) - lo0), int(np.floor(hi_ / bn) - lo0); vp[a:b + 1] += v_ / (b - a + 1)
    sm = np.convolve(vp, np.ones(3) / 3, 'same'); px = (lo0 + np.arange(nb) + .5) * bn; w = max(3, int(nb / 20))
    pk = sorted([k for k in range(nb) if sm[k] > 0 and sm[k] == sm[max(0, k - w):k + w + 1].max()], key=lambda k: -sm[k]); out = []
    for k in pk:
        if all(abs(px[k] - q) >= .05 * adr for q in out): out.append(px[k])
        if len(out) == 3: break
    return out
def fade(hi, lo, cl, j, L, side, sd, td, cost):
    S, T = L - side * sd, L + side * td
    if (lo[j] <= S) if side > 0 else (hi[j] >= S): return (-sd - cost) / sd
    for q in range(j + 1, len(hi)):
        st = (lo[q] <= S) if side > 0 else (hi[q] >= S); tg = (hi[q] >= T) if side > 0 else (lo[q] <= T)
        if st: return (-sd - cost) / sd
        if tg: return (td - cost) / sd
    return ((cl[-1] - L) * side - cost) / sd
def run(asset, iv, rng):
    sym, a0, a1, hrs, cost = SYM[asset]; B1 = get(sym, '1h', '730d'); B = B1 if iv == '1h' else get(sym, '5m', '60d')
    D1 = B1.resample('1D').agg({'high': 'max', 'low': 'min'}).dropna(); ADR = (D1.high - D1.low).rolling(20).mean().shift(1)
    SS = sessions(B, a0, a1, hrs, iv); days = sorted(SS); bn = float(np.nanmedian(ADR)) / 60
    SG = {d: bot_sig(B1, SS[d].index[0]) for d in days}
    rngd = {d: SS[d].high.max() - SS[d].low.min() for d in days}; half = days[len(days) // 2]
    kk = np.nanmedian([rngd[d] / SG[d] for d in days if d < half and np.isfinite(SG[d]) and SG[d] > 0])
    sd_, tds = (.05, (.05, .10)) if iv == '5m' else (.10, (.10, .20)); rows = []
    for d0, d in zip(days[:-1], days[1:]):
        adr = ADR.get(d, np.nan)
        if (d - d0).days > 4 or not np.isfinite(adr): continue
        g = SS[d]; hi, lo, cl, op = g.high.values, g.low.values, g.close.values, g.open.values; mm = g.m.values; sg = SG[d]; R = sg * kk; o0 = op[0]
        for plc in (0, 1, 1, 1, 1, 1):
            levs = hvns(SS[d0], adr, bn)
            if plc: levs = [L + rng.choice([-1, 1]) * rng.uniform(.03, .12) * adr for L in levs]
            for L in levs:
                for p in range(1, len(hi)):
                    if not (lo[p] <= L <= hi[p]): continue
                    if iv == '5m' and (mm[p] < a0 + 15 or mm[p] > a1 - 30): break
                    side = 1 if lo[p - 1] > L else (-1 if hi[p - 1] < L else 0)
                    if side == 0: break
                    thod = np.isfinite(sg) and min(abs(L - (o0 - side * k * sg)) for k in (1, 2, 3)) <= .25 * sg
                    exh = np.isfinite(R) and ((hi[:p].max() - L) >= .8 * R if side > 0 else (L - lo[:p].min()) >= .8 * R)
                    rows.append(dict(asset=asset, iv=iv, day=d, plc=plc, side=side, thod=bool(thod), exh=bool(exh), h2=d >= half,
                                     r11=fade(hi, lo, cl, p, L, side, sd_ * adr, tds[0] * adr, cost), r21=fade(hi, lo, cl, p, L, side, sd_ * adr, tds[1] * adr, cost))); break
    return pd.DataFrame(rows), len(days)
def mr_stats(asset, iv):
    sym, a0, a1, hrs, cost = SYM[asset]; B = get(sym, iv, '730d' if iv == '1h' else '60d'); SS = sessions(B, a0, a1, hrs, iv); k = 6 if iv == '5m' else 3; ac, v1, vk = [], [], []
    for g in SS.values():
        r = np.diff(np.log(g.close.values))
        if len(r) < k: continue
        ac.append(np.c_[r[:-1], r[1:]]); v1.extend(r); rk = np.log(g.close.values[k::k]) - np.log(g.close.values[:-k:k])[:len(g.close.values[k::k])]; vk.extend(rk)
    A_ = np.vstack(ac); return np.corrcoef(A_[:, 0], A_[:, 1])[0, 1], np.var(vk) / (k * np.var(v1))
if __name__ == '__main__':
    print('0) within main session: lag-1 autocorrelation of bar returns, variance ratio (< 1 = mean-reverting)', flush=True)
    for iv in ('5m', '1h'):
        print(f'  {iv}: ' + '   '.join(f'{a} ac1 {mr_stats(a, iv)[0]:+.3f} VR {mr_stats(a, iv)[1]:.2f}' for a in SYM), flush=True)
    rng = np.random.default_rng(5)
    for iv in ('1h', '5m'):
        print(f'\n1) HVN fade, {iv} bars | trades a month, HOLD (target first), R per trade (1:1 / 2:1) | placebo R (1:1 / 2:1)' + (' | 1st half ; 2nd half' if iv == '1h' else ''), flush=True)
        for a in SYM:
            E, nd = run(a, iv, rng); mo = nd / 21
            for nm, f in (('all HVN touches', lambda z: z.index == z.index), ('THOD line', lambda z: z.thod), ('EXH (range used)', lambda z: z.exh), ('THOD & EXH', lambda z: z.thod & z.exh)):
                X = E[f(E)]; parts = [('', X)] if iv == '5m' else [('1st ', X[~X.h2]), ('2nd ', X[X.h2])]; cells = []
                for lab, Z in parts:
                    r_, p_ = Z[Z.plc == 0], Z[Z.plc == 1]; m_ = mo if iv == '5m' else mo / 2
                    cells.append(f'{lab}{len(r_) / m_:4.1f}/mo {np.mean(r_.r11 > 0):3.0%} {r_.r11.mean():+.2f} / {r_.r21.mean():+.2f} R (plc {p_.r11.mean():+.2f} / {p_.r21.mean():+.2f})')
                print(f'  {a} {nm:<18} ' + ' ; '.join(cells), flush=True)
