"""Nov 9 2026: 'HIGH-PROBABILITY' SCREENER FLIPS - NEGATIVE GAMMA + YANG-ZHANG / HAR / RISING IV (Rey). DECLARED BEFORE RESULTS.
Per-stock options data (xs/ext/gex/*.csv: net GEX, ATM IV, call / put volume) exists for 18 of the 54 screener names (42% of signals), 2018 - Dec 2025.
No open interest by strike here, so no true vanna / vega exposure: rising ATM IV (5-day) stands in for the vega side. All flags from the PRIOR close.
  NEG       stock's net GEX < 0 (dealers short gamma - moves extend)
  NEG+YZ    NEG and Yang-Zhang 20d in its top third (2019-22 cut)
  NEG+HAR   NEG and HAR forecast above the stock's own normal (HAR vs trailing-year median > 1)
  NEG+IVUP  NEG and ATM IV up over the last 5 days
  POS       net GEX > 0 (for contrast)
1) Points per trade (avg return, win rate, n) for each flag vs the rest, same 18 names, 2019-22 / 2023-25.
2) Slots (all 54 names, 6 slots, stand-in path): high-probability flips get the slot first, then the ratio; 2019-22 / 2023-25.
(Replacement only for high-probability flips: xs/scr_hp_swap.py, hourly prices, Dec 2024 - Dec 2025.)"""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_rank_vol.py').read(); exec(_s[:_s.index("\nif __name__ == '__main__':")])
GX = []
for f in glob.glob('/mnt/user-data/uploads/fartpie/xs/ext/gex/*.csv'):
    tk = f.split('/')[-1][:-4]; g = pd.read_csv(f, parse_dates=['date']).sort_values('date'); g['tk'] = tk
    g['ivup'] = g.atm_iv / g.atm_iv.shift(5) - 1; g['nxt'] = g.date.shift(-1); GX.append(g[['tk', 'nxt', 'net_gex', 'atm_iv', 'ivup']].dropna(subset=['nxt']))
GX = pd.concat(GX).set_index(['tk', 'nxt']); GXT = set(GX.index.get_level_values(0))
X = base_T0.join(GX, on=['tk', 'day']); X['per'] = np.where(X.day < '2023-01-01', 'a', 'b'); yq = X[X.per == 'a'].yz.quantile(2 / 3)
X['NEG'] = X.net_gex < 0; X['POS'] = X.net_gex > 0; X['NEG+YZ'] = X.NEG & (X.yz >= yq); X['NEG+HAR'] = X.NEG & (X.harrel > 1); X['NEG+IVUP'] = X.NEG & (X.ivup > 0)
if __name__ == '__main__':
    G = X[X.tk.isin(GXT) & X.net_gex.notna() & (X.day < '2026-01-01')]
    print(f'18-name universe: {len(G):,} signals with GEX ({G.NEG.mean():.0%} in negative gamma)\n\n=== 1) per trade: avg return, win rate (n) | 2019-22 ; 2023-25 ===', flush=True)
    for f in ('NEG', 'NEG+YZ', 'NEG+HAR', 'NEG+IVUP', 'POS'):
        cells = []
        for p_ in 'ab':
            a, b = G[(G.per == p_) & G[f]], G[(G.per == p_) & ~G[f]]
            cells.append(f'{a.ret.mean():+.2%} win {np.mean(a.ret > 0):.0%} (n {len(a)}) vs rest {b.ret.mean():+.2%} win {np.mean(b.ret > 0):.0%}')
        print(f'  {f:<9} ' + ' ; '.join(cells), flush=True)
    def st3(s):
        out = []
        for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2026-01-01')):
            z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25
            out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
        return ' | '.join(out)
    print('\n=== 2) slots, all 54 names: high-probability flips first, then ratio | CAGR / max DD 2019-22 | 2023-25 ===', flush=True)
    for lab in ('live (ratio only)', 'NEG', 'NEG+YZ', 'NEG+HAR', 'NEG+IVUP'):
        k = -X.ratio if lab.startswith('live') else (-X[lab].fillna(False).astype(int) * 100 - X.ratio)
        T0 = X.assign(_k=k.values).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); s_ = s_[s_.index < '2026-01-01']
        print(f'  {lab:<20} {st3(s_)}', flush=True)
# --- follow-up (EXPLORATORY, decided after seeing that positive-gamma flips did better in both periods): positive gamma first
if __name__ == '__main__':
    for lab, k in (('POS first, then ratio', -X.POS.fillna(False).astype(int) * 100 - X.ratio), ('ratio, POS breaks near-ties (rank-sum)', X.groupby('entry').ratio.rank(ascending=False) - X.POS.fillna(False).astype(int) * 1.5)):
        T0 = X.assign(_k=k.values).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); s_ = s_[s_.index < '2026-01-01']
        print(f'  {lab:<38} {st3(s_)}', flush=True)
