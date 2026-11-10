"""Nov 9 2026: LONG / SHORT ON THE HIGH-OPTIONS-RATIO NAMES - BIGGER MOVES, FEWER CHOP-OUTS? (Rey). DECLARED BEFORE RESULTS.
1) FULL HISTORY, bull flips the screener takes candidates for (wide_T, 2019 - Sep 2026): by options-ratio fifth - average return, average SIZE of the move |ret|,
   win rate, and CHOP-OUTS = trades out the same day at a loss. Does a higher ratio mean bigger moves and fewer chop-outs?
2) ROUGH LONG / SHORT (no bear flips in the backtest data; Yahoo hourly bars Nov 2024 - Nov 2026 only): the exact Pine vector port (xs/vector_pine.py) on
   09:30-aligned 1h bars for the 54 names (like the screener's 1h track). Long at the next bar's open after a bull flip, short after a bear flip, out at the next
   open after the opposite flip; 2 bp a side; overnight holds allowed. Split: the names' median options ratio top half vs bottom half; shorts also only when SPY was
   below its 50-day average at the prior close (weak tape). Average return, win rate, |move|, chop-outs (out within 2 bars at a loss). Hint only - 2 years, bull market."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); from vector_pine import run_state_machine
U = '/mnt/user-data/uploads/fartpie'
T = pd.read_parquet(f'{U}/xs/ext/wide_T.parquet'); T = T[T.sp & (T.ratio >= 1.5)].copy(); T['per'] = np.where(T.entry < '2023-01-01', 'a', 'b')
T['same'] = T.exit.dt.normalize() == T.entry.dt.normalize()
if __name__ == '__main__':
    cuts = T[T.per == 'a'].ratio.quantile([.2, .4, .6, .8]).values; T['f'] = np.digitize(T.ratio, cuts)
    print(f'=== 1) bull flips by options-ratio fifth (cuts {", ".join(f"{c:.2f}" for c in cuts)}) | avg, |move|, win, chop-outs (same-day loss) | 2019-22 ; 2023-26 ===', flush=True)
    for k in range(5):
        cells = []
        for p_ in 'ab':
            z = T[(T.f == k) & (T.per == p_)]; cells.append(f'{z.ret.mean():+.2%}, |{z.ret.abs().mean():.2%}|, win {np.mean(z.ret > 0):.0%}, chop {np.mean(z.same & (z.ret < 0)):.0%} (n {len(z)})')
        print(f'  fifth {k + 1} {"(lowest)" if k == 0 else "(highest)" if k == 4 else "":<9} ' + ' ; '.join(cells), flush=True)
    med = T.groupby('tk').ratio.median(); hi_names = set(med[med >= med.median()].index)
    SPY = pd.read_parquet('/home/claude/fartpie/data/yahoo/daily_v/SPY.parquet').close; weak = (SPY < SPY.rolling(50).mean()).shift(1)
    rows = []
    for tk in sorted(med.index):
        f = f'/home/claude/fartpie/data/yahoo/h1/{tk}.parquet'
        if not os.path.exists(f): continue
        H = pd.read_parquet(f); mm = H.index.hour * 60 + H.index.minute; H = H[(mm >= 570) & (mm < 960)].copy()
        if len(H) < 300: continue
        reg = run_state_machine(H.reset_index().rename(columns={'index': 'datetime', 'Datetime': 'datetime'}))[0]; o = H.open.values; t = H.index
        pos = 0; ent = None
        for i in range(1, len(H) - 1):
            if reg[i] != reg[i - 1] and reg[i] != 0:
                if pos != 0:
                    r = (o[i + 1] / ent[1] - 1) * pos - 4e-4; rows.append((tk, t[ent[0]], t[i + 1], pos, r, i + 1 - ent[0]))
                pos = int(reg[i]); ent = (i + 1, o[i + 1])
    Z = pd.DataFrame(rows, columns=['tk', 'e', 'x', 'side', 'ret', 'bars']); Z = Z[Z.e >= '2024-12-15']
    Z['grp'] = np.where(Z.tk.isin(hi_names), 'high-ratio half', 'low-ratio half'); Z['weak'] = Z.e.dt.normalize().map(weak).fillna(False).astype(bool)
    Z['chop'] = (Z.bars <= 2) & (Z.ret < 0)
    print(f'\n=== 2) ROUGH long / short on 1h vector flips, {Z.tk.nunique()} names, {Z.e.min().date()} - {Z.x.max().date()} | avg, win, |move|, chop-outs (<= 2 bars at a loss), trades ===', flush=True)
    def ln(z): return f'avg {z.ret.mean():+.2%}, win {np.mean(z.ret > 0):.0%}, |{z.ret.abs().mean():.2%}|, chop {z.chop.mean():.0%} (n {len(z)})'
    for grp in ('high-ratio half', 'low-ratio half'):
        for side, nm in ((1, 'LONG '), (-1, 'SHORT')):
            z = Z[(Z.grp == grp) & (Z.side == side)]; print(f'  {grp:<16} {nm} all: {ln(z)} | SPY weak only: {ln(z[z.weak])}', flush=True)
    print(f'  (SPY below its 50-day average on {Z.drop_duplicates("e").weak.mean():.0%} of entry days in this window)', flush=True)
