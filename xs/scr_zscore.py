"""Nov 9 2026: WHERE IS THE STOCK WHEN IT FLIPS? OVERSOLD vs EXTENDED, LOW vs HIGH OF ITS RANGE (Rey). DECLARED BEFORE RESULTS.
Screener bull flips (54 names, 2019 - Sep 2026). All from the PRIOR daily close (Yahoo, adjusted):
  Z20     (close - 20-day mean) / 20-day stdev of closes        (negative = oversold, positive = extended)
  RNG20   position in the 20-day high-low range, 0 = at the low, 1 = at the high
  RNG60   the same over 60 days
1) avg trade return and win rate by FIFTH of each (cuts from 2019-22), 2019-22 / 2023-26.
2) slots (6, stand-in path): when flips arrive together, prefer the most OVERSOLD (lowest Z20) | the most EXTENDED (highest Z20) | lowest / highest RNG20,
   alone and as rank-sum with the options ratio. CHOSEN on 2019-22, JUDGED on 2023-26."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_rank_lit.py').read(); exec(_s[:_s.index("\nif __name__ == '__main__':")])
Zf = []
for tk in TKS:
    Y = dv(tk); c = Y.close; z = (c - c.rolling(20).mean()) / c.rolling(20).std()
    r20 = (c - Y.low.rolling(20).min()) / (Y.high.rolling(20).max() - Y.low.rolling(20).min()); r60 = (c - Y.low.rolling(60).min()) / (Y.high.rolling(60).max() - Y.low.rolling(60).min())
    f = pd.DataFrame({'tk': tk, 'z20': z, 'rng20': r20, 'rng60': r60}, index=Y.index).reset_index().rename(columns={'index': 'date', 'Date': 'date'})
    f['nxt'] = f.date.shift(-1); Zf.append(f.dropna(subset=['nxt']))
Zf = pd.concat(Zf).set_index(['tk', 'nxt'])[['z20', 'rng20', 'rng60']]
X = base2.join(Zf, on=['tk', 'day']); X['per'] = np.where(X.day < '2023-01-01', 'a', 'b')
if __name__ == '__main__':
    print(f'features found: z20 {X.z20.notna().mean():.0%}\n\n=== 1) avg trade return (win rate, n) by fifth | 2019-22 ; 2023-26 ===', flush=True)
    for f in ('z20', 'rng20', 'rng60'):
        cuts = X[X.per == 'a'][f].quantile([.2, .4, .6, .8]).values; b_ = np.digitize(X[f], cuts); b_ = np.where(X[f].isna(), -1, b_)
        print(f'  {f} (cuts {", ".join(f"{c:.2f}" for c in cuts)}):', flush=True)
        for k in range(5):
            g_ = X[b_ == k]; cells = [f'{g_[g_.per == p_].ret.mean():+.2%} ({np.mean(g_[g_.per == p_].ret > 0):.0%}, n {(g_.per == p_).sum()})' for p_ in 'ab']
            print(f'    fifth {k + 1} {"(lowest)" if k == 0 else "(highest)" if k == 4 else "":<9} ' + ' ; '.join(cells), flush=True)
    def st5(s):
        out = []
        for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
            z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
        return ' | '.join(out)
    print('\n=== 2) pick order when flips arrive together | CAGR / max DD 2019-22 (CHOOSE) | 2023-26 (JUDGE) ===', flush=True); RES = {}
    g = X.groupby('entry'); rr = g.ratio.rank(ascending=False)
    for lab, key in (('live: highest ratio', -X.ratio), ('most oversold (lowest Z20)', g.z20.rank(ascending=True, na_option='bottom')),
                     ('most extended (highest Z20)', g.z20.rank(ascending=False, na_option='bottom')), ('lowest in 20d range', g.rng20.rank(ascending=True, na_option='bottom')),
                     ('highest in 20d range', g.rng20.rank(ascending=False, na_option='bottom')), ('ratio + most oversold', rr + g.z20.rank(ascending=True, na_option='bottom')),
                     ('ratio + most extended', rr + g.z20.rank(ascending=False, na_option='bottom')), ('ratio + lowest in range', rr + g.rng20.rank(ascending=True, na_option='bottom')),
                     ('ratio + highest in range', rr + g.rng20.rank(ascending=False, na_option='bottom'))):
        T0 = X.assign(_k=np.asarray(key)).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); RES[lab] = s_
        print(f'  {lab:<28} {st5(s_)}', flush=True)
    best = max([k for k in RES if not k.startswith('live')], key=lambda k: (1 + RES[k][RES[k].index < '2023-01-01']).prod())
    print(f'  -> chosen on 2019-22: {best}', flush=True)
