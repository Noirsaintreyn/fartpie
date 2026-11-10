"""Nov 9 2026: POSITIVE-GAMMA FLIPS COMBINED WITH EVERYTHING ELSE, AND GAMMA FLIPS (GEX sign changes) (Rey). DECLARED BEFORE RESULTS.
18 screener names with per-stock GEX, 2019 - Dec 2025; every flag from the PRIOR close.
Flags: POS (net GEX > 0) alone and with: high / low Yang-Zhang third, HAR above / at-or-below the stock's normal, ATM IV up / down over 5 days,
prior-day relative volume top third, within 5% of the 52-week high, 20-day strength vs SPY > 0, options ratio top third.
GAMMA FLIPS: GFLIP+ net GEX turned positive within the last 5 sessions | GFLIP- turned negative within the last 5 sessions.
1) per trade vs the rest of the 18 names, 2019-22 / 2023-25. CHOSEN: the flag (n >= 150 in 2019-22) with the biggest 2019-22 edge over the rest; JUDGED on 2023-25.
2) the chosen flag: gets the slot first (all 54 names) | swap: replaces the weakest LOSING holding (hourly prices, Dec 2024 - Dec 2025, xs/scr_gamma_swap.py)."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_rank_lit.py').read(); exec(_s[:_s.index("\nif __name__ == '__main__':")])
GX = []
for f in glob.glob('/mnt/user-data/uploads/fartpie/xs/ext/gex/*.csv'):
    g = pd.read_csv(f, parse_dates=['date']).sort_values('date'); g['tk'] = f.split('/')[-1][:-4]; s_ = np.sign(g.net_gex)
    g['ivup'] = g.atm_iv / g.atm_iv.shift(5) - 1; g['gfp'] = (s_ > 0) & (s_.shift(1).rolling(5).min() < 0); g['gfn'] = (s_ < 0) & (s_.shift(1).rolling(5).max() > 0)
    g['nxt'] = g.date.shift(-1); GX.append(g[['tk', 'nxt', 'net_gex', 'ivup', 'gfp', 'gfn']].dropna(subset=['nxt']))
GX = pd.concat(GX).set_index(['tk', 'nxt']); GXT = set(GX.index.get_level_values(0))
X = base2.join(GX, on=['tk', 'day']); X['per'] = np.where(X.day < '2023-01-01', 'a', 'b'); A_ = X[X.per == 'a']
yq1, yq2 = A_.yz.quantile([1 / 3, 2 / 3]); rvq = A_.rvol.quantile(2 / 3); raq = A_.ratio.quantile(2 / 3); P_ = X.net_gex > 0
FL = {'POS': P_, 'POS + high YZ': P_ & (X.yz >= yq2), 'POS + low YZ': P_ & (X.yz < yq1), 'POS + HAR above normal': P_ & (X.harrel > 1), 'POS + HAR at/below normal': P_ & (X.harrel <= 1),
      'POS + IV rising': P_ & (X.ivup > 0), 'POS + IV falling': P_ & (X.ivup <= 0), 'POS + high rel volume': P_ & (X.rvol >= rvq), 'POS + near 52w high': P_ & (X.hi52 >= .95),
      'POS + strong vs SPY': P_ & (X.rs20 > 0), 'POS + top-third ratio': P_ & (X.ratio >= raq), 'GFLIP+ (turned positive <= 5d)': X.gfp.fillna(False).astype(bool),
      'GFLIP- (turned negative <= 5d)': X.gfn.fillna(False).astype(bool)}
G = X.tk.isin(GXT) & X.net_gex.notna() & (X.day < '2026-01-01')
if __name__ == '__main__':
    print(f'{G.sum():,} signals in the 18 names with GEX\n\n=== 1) per trade: flagged avg return, win (n) vs the rest | 2019-22 ; 2023-25 ===', flush=True); ED = {}
    for nm, f in FL.items():
        cells = []; ed = []
        for p_ in 'ab':
            a, b = X[G & f & (X.per == p_)], X[G & ~f & (X.per == p_)]; ed.append(a.ret.mean() - b.ret.mean())
            cells.append(f'{a.ret.mean():+.2%} win {np.mean(a.ret > 0):.0%} (n {len(a)}) vs {b.ret.mean():+.2%}')
        if (G & f & (X.per == 'a')).sum() >= 150: ED[nm] = ed
        print(f'  {nm:<32} ' + ' ; '.join(cells) + f'  | edge {ed[0]:+.2%} / {ed[1]:+.2%}', flush=True)
    best = max(ED, key=lambda k_: ED[k_][0]); print(f'\n  -> chosen on 2019-22: {best} (edge {ED[best][0]:+.2%}) | 2023-25 edge {ED[best][1]:+.2%} ({"holds" if ED[best][1] > 0 else "FAILS"})', flush=True)
    def st3(s):
        out = []
        for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2026-01-01')):
            z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25
            out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
        return ' | '.join(out)
    print('\n=== 2) slots, all 54 names | CAGR / max DD 2019-22 | 2023-25 ===', flush=True)
    for lab, k in (('live (ratio)', -X.ratio), (f'{best} first, then ratio', -FL[best].fillna(False).astype(int) * 100 - X.ratio)):
        T0 = X.assign(_k=k.values).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); print(f'  {lab:<44} {st3(s_[s_.index < "2026-01-01"])}', flush=True)
    pd.DataFrame({'flag': FL[best].fillna(False).astype(bool)}).to_parquet('/tmp/claude-0/scr_gamma_flag.parquet'); print(f'  (flag "{best}" saved for the swap test)', flush=True)
