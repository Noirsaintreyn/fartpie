"""Nov 22 2026 report 350. Rey: if the opening-range breakout works, shouldn't entries AFTER the first 15/30 minutes be higher probability?  Declared before running.
Two readings, both tested on the 12,320 large-cap long flips (15m, entry = next bar open):
 T) TIMING: flips by entry time bucket: 09:30-09:45 (inside the 15m range window), 09:45-10:00, 10:00-10:30, 10:30-12:00, 12:00-14:00, 14:00-16:00; per-trade return, win rate.  THEORY: flips in the first bars are overnight/gap carry-over (state machine flips on the open bar), noisy; once the opening range exists the flip has a reference.
 O) ORB STATE at entry (last COMPLETED bar close vs the session's opening range; OR15 = first 15m bar, OR30 = first two bars; only flips entering after the range is complete): ABOVE the range high (breakout confirmed), INSIDE, BELOW the range low.  THEORY (Rey): ORB works because a break of the opening range persists; a long flip with price already above the range should beat one inside/below.
Train 2019-21 | test 2018+2022-26 per-trade; then the 6-slot book skipping flips that are not above the OR (for the post-OR flips) and, separately, skipping first-bar flips.  Why it fails: the flip itself already is a breakout above the pivot ratchet, so ORB state is mostly redundant; and report 342-343 found no gradient in prior-bar features."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from stress_lib import *
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; B0 = pd.read_parquet(SP + 'b302.parquet'); B0 = B0[B0.exit < END].copy(); B0['entry'] = pd.to_datetime(B0.entry); B0['exit'] = pd.to_datetime(B0.exit); B0['h1'] = False; B0 = B0.reset_index(drop=True)
def sim(Bs, N=6, brk=.08):
    global T0
    key = Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values; T0 = Bs.assign(_k=key).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(N, brk); R = np.zeros(len(D))
    for tk, e, x, r, ra in tk_:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / N; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w
    return pd.Series(R, index=D), len(tk_)
def cagr(s): e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; return e.iloc[-1] ** (1 / y) - 1
def mdd(s): e = (1 + s).cumprod(); return float((e / e.cummax() - 1).min())
n = int(B0.shape[0]); S15 = np.full(n, np.nan); S30 = np.full(n, np.nan)
for tk in B0.tk.unique():
    f = sorted(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')); H = pd.read_parquet(f[-1]); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index(); mm = H.index.hour * 60 + H.index.minute; H = H[(mm >= 570) & (mm < 960)]
    day = H.index.normalize(); first = pd.Series(np.arange(len(H)), index=H.index).groupby(day).transform('min').values; hi, lo, cl = H.high.values, H.low.values, H.close.values; idx = H.index.values
    m = np.where((B0.tk == tk).values)[0]; ie = np.searchsorted(idx, B0.entry.values[m])
    for j, i in zip(m, ie):
        if i >= len(H) or i < 1: continue
        f0 = first[i]; k = i - f0   # bars of the session completed before the entry bar
        for nb, arr in ((1, S15), (2, S30)):
            if k >= nb:
                rh, rl = hi[f0:f0 + nb].max(), lo[f0:f0 + nb].min(); c = cl[i - 1]; arr[j] = 1 if c > rh else (-1 if c < rl else 0)
B0['s15'] = S15; B0['s30'] = S30; mn = (B0.entry.dt.hour * 60 + B0.entry.dt.minute).values; tr = ((B0.entry >= '2019-01-01') & (B0.entry < '2022-01-01')).values; te = ~tr
def f(m): s = B0.ret.values[m]; return '   n/a' if m.sum() < 25 else f'{s.mean():+.3%} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, n{len(s)}, win {(s > 0).mean():.0%})'
print('T) TIMING: entry-time buckets (share of all flips)')
for lab, a, b in (('09:30-09:45', 570, 585), ('09:45-10:00', 585, 600), ('10:00-10:30', 600, 630), ('10:30-12:00', 630, 720), ('12:00-14:00', 720, 840), ('14:00-16:00', 840, 960)):
    m = (mn >= a) & (mn < b); print(f'   {lab} ({m.mean():4.0%}): train {f(tr & m)} | test {f(te & m)}')
for col, lab in (('s15', 'OR15 (first 15m bar)'), ('s30', 'OR30 (first 30m)')):
    print(f'\nO) ORB state, {lab}: flips entering after the range is complete: {B0[col].notna().mean():.0%}')
    for v, nm in ((1, 'ABOVE range high'), (0, 'INSIDE'), (-1, 'BELOW range low')): m = (B0[col] == v).values; print(f'   {nm:17s} ({m.mean():4.0%}): train {f(tr & m)} | test {f(te & m)}')
s0, n0 = sim(B0); s0 = s0[s0.index >= '2019-01-01']; print(f'\nBOOK baseline: {n0} trades, CAGR {cagr(s0):.1%}, DD {mdd(s0):.1%}, Sharpe {sharpe(s0.values):.2f}')
for lab, keep in (('skip first-bar (09:30) flips', ~((mn >= 570) & (mn < 585))), ('skip flips in first 30 min', ~((mn >= 570) & (mn < 600))), ('post-OR15 only if ABOVE', ~(B0.s15.isin([0, -1]).values)), ('post-OR30 only if ABOVE', ~(B0.s30.isin([0, -1]).values)), ('ONLY flips above OR15', (B0.s15 == 1).values)):
    z, nt = sim(B0[keep]); z = z[z.index >= '2019-01-01']; print(f'BOOK {lab:28s}: {nt} trades ({keep.sum()} candidates), CAGR {cagr(z):.1%}, DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}')
