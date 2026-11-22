"""Nov 22 2026 report 353. 'Stocks in play' 5-minute ORB (Zarattino/Pagani/Aziz: top relative-volume names, opening-range break in the direction of the first candle, 10% ATR stop, exit at the close) on OUR 54 large caps with the 5m bars available (partial: check count).  Declared before running.
Per name-day: first 5m candle (09:30-09:35): direction = candle colour (long if close > open, short if close < open; dojis skipped).  Entry = stop order at the candle's high (long) / low (short) on the first later bar that trades through it; stop = entry -/+ 0.10 x ATR14 (daily); no target; exit at the 15:55 bar's close; cost 2 bp + 1 bp slippage round trip; R = P&L / (0.10 x ATR).  Relative volume rv0 = first-candle volume / mean of the same name's previous 14 first-candle volumes.  'In play' = top K by rv0 each day among available names (K = 3, 5) vs all name-days vs bottom third.  Long and short reported separately.  Train 2019-21 | test 2018 + 2022-26.
THEORY: unusual early volume = news/imbalance; the first candle's direction persists for the rest of the day (heavy-tail winners, like the NQ ORB's rvol terciles in report 336).  Why it fails: 54 mega-caps are rarely 'in play' (little news-driven dispersion) vs the 7,000-stock paper universe; the 10% ATR stop is hit constantly (win rate ~15-30%) so costs dominate; the paper's numbers are backtested net of its own cost model."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; fs = sorted(glob.glob(SP + 'lse5m/*.parquet')); print(f'5m files: {len(fs)}'); CANDLE = len(sys.argv) > 1 and sys.argv[1] == 'candle'; print('stop = opposite end of the first candle (R = candle range)' if CANDLE else 'stop = 0.10 x ATR14'); rows = []
for fn in fs:
    tk = fn.split('/')[-1][:-8]; H = pd.read_parquet(fn); tcol = 'ts' if 'ts' in H.columns else 'datetime'; H.index = pd.to_datetime(H[tcol], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index(); mm = H.index.hour * 60 + H.index.minute; H = H[(mm >= 570) & (mm < 960)]
    if len(H) < 1000: continue
    day = H.index.normalize(); dd = H.groupby(day).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last')); atr = (dd.h - dd.l).rolling(14).mean().shift(1); first = H.groupby(day).head(1); fv = pd.Series(first.volume.values, index=first.index.normalize()); rv = fv / fv.rolling(14).mean().shift(1)
    for d, x in H.groupby(day):
        if len(x) < 70 or x.index[0].hour * 60 + x.index[0].minute != 570 or pd.isna(atr.get(d)) or pd.isna(rv.get(d)): continue
        o0, h0, l0, c0 = x.open.iloc[0], x.high.iloc[0], x.low.iloc[0], x.close.iloc[0]
        if c0 == o0: continue
        dr = 1 if c0 > o0 else -1; ent = h0 if dr == 1 else l0; rest = x.iloc[1:]; rd = (h0 - l0) if CANDLE else .10 * atr[d]; stop = (l0 if dr == 1 else h0) if CANDLE else ent - dr * .10 * atr[d]; px = None
        for tt, b in rest.iterrows():
            if px is None:
                if (dr == 1 and b.high >= ent) or (dr == -1 and b.low <= ent):
                    px = max(b.open, ent) if dr == 1 else min(b.open, ent)
                    if (dr == 1 and b.low <= stop) or (dr == -1 and b.high >= stop): rows.append((tk, d, dr, rv[d], (min(stop, px) - px) if dr == 1 else (px - max(stop, px)), rd, px)); px = 'done'; break
            else:
                if (dr == 1 and b.low <= stop) or (dr == -1 and b.high >= stop): xp = min(stop, b.open) if dr == 1 else max(stop, b.open); rows.append((tk, d, dr, rv[d], (xp - px) * dr, rd, px)); px = 'done'; break
        if px is not None and px != 'done': rows.append((tk, d, dr, rv[d], (rest.close.iloc[-1] - px) * dr, rd, px))
T = pd.DataFrame(rows, columns=['tk', 'd', 'dir', 'rv', 'pnl', 'atr', 'px']); T['net'] = T.pnl - .0003 * T.px; T['R'] = T.net / T.atr; T['d'] = pd.to_datetime(T.d)
print(f'{len(T)} trades, {T.tk.nunique()} names, {T.d.min().date()} -> {T.d.max().date()}')
T['rk'] = T.groupby('d').rv.rank(ascending=False, method='first'); T['n'] = T.groupby('d').rv.transform('size'); tr = (T.d >= '2019-01-01') & (T.d < '2022-01-01'); te = ~tr & (T.d >= '2018-01-01')
def f(m): s = T.R[m]; return '   n/a' if len(s) < 25 else f'{s.mean():+.3f}R (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, n{len(s)}, win {(s > 0).mean():.0%}, pct {T.net[m].div(T.px[m]).mean():+.3%})'
for lab, m in (('ALL name-days', T.d > '2000'), ('top-3 rv0 each day', T.rk <= 3), ('top-5 rv0 each day', T.rk <= 5), ('bottom third by rv0 rank', T.rk > T.n * 2 / 3)):
    for sd, ms in (('both', T.dir != 0), ('long', T.dir == 1), ('short', T.dir == -1)): print(f'{lab:26s} {sd:5s} train {f(tr & m & ms)} | test {f(te & m & ms)}')
