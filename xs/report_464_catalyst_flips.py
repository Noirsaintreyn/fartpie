"""Report 464 (Nov 24 2026): flips with a REASON to move. Rey: not fundamentals - specific things that lead to money: business deals and the macro that moves that type of company.
DECLARED BEFORE RESULTS. Sample: structure_small daily bull vector flips 2019-2026Q1, entry px < $10 (primary) and < $20 (robustness). Target: flip return r >= +10% (primary), >= +30%, plus mean/median/win. Eras: 2019-21 | 2022 | 2023-26Q1.
Company catalysts (SEC 8-K item flags from data in scratchpad/ext/8k_wide.csv; known by the flip day: filed before the flip day, or the same day with acceptance before 20:00Z = before the close), window = prior 7 calendar days:
 DEAL  item 1.01 (material definitive agreement) or 2.01 (completed acquisition/disposition)   [declared: better]
 PR    item 7.01/8.01 (Reg FD / other events: press-release news) and NOT 2.02                   [better]
 EARN  item 2.02                                                                               [better: an event]
 NEG   items 3.02 (unregistered equity sale = dilution), 1.02 (agreement terminated), 1.03 (bankruptcy), 2.05 (restructuring)   [worse; control]
 INS   open-market insider buy (Form 4 code P) in the prior 30 days                            [better]
Sector macro tailwind (SIC 2-digit group -> its own driver, 20-day change known the day before): oil/gas (13, 29): WTI up; metals/mining (10, 12, 14, 33): dollar DOWN; banks/finance (60-64): curve (10y-2y) steepening; health/biotech (28, 80, 38): XLV beats SPY; tech/comm/software (35, 36, 37, 48, 73): XLK beats SPY; utilities (49): 10y yield down.  TAIL = driver favorable (mapped names only)    [better]
Reason-to-move gate: GATE = DEAL | INS | TAIL (any positive reason).  PASS (per flag, declared): flag beats no-flag on P(r>=10%) in EVERY era AND month-paired t >= 2.2 on the mean (7 flags, Bonferroni); NEG must be worse."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d).astype('datetime64[ns]'); O = O[O.px < 20].copy().reset_index(drop=True)
O['era3'] = np.select([O.d.dt.year <= 2021, O.d.dt.year == 2022], ['2019-21', '2022'], '2023-26'); O['mon'] = O.d.dt.strftime('%Y-%m')
K = pd.read_csv(SP + 'ext/8k_wide.csv'); K = K.dropna(subset=['items']).copy(); K['filed'] = pd.to_datetime(K.filed).astype('datetime64[ns]'); K['acc'] = pd.to_datetime(K.accept, utc=True, errors='coerce')
K['known'] = np.where(K.acc.dt.hour < 20, K.filed, K.filed + pd.Timedelta(days=1)); K['known'] = pd.to_datetime(K.known).astype('datetime64[ns]')
it = K['items'].astype(str).str.split(',')
K['DEAL'] = it.apply(lambda l: '1.01' in l or '2.01' in l); K['EARN'] = it.apply(lambda l: '2.02' in l); K['PR'] = it.apply(lambda l: ('7.01' in l or '8.01' in l) and '2.02' not in l); K['NEG'] = it.apply(lambda l: any(x in l for x in ('3.02', '1.02', '1.03', '2.05')))
def flag(col):
    k = K[K[col]][['ticker', 'known']].drop_duplicates(); out = np.zeros(len(O), bool); g = {t: x.known.values for t, x in k.groupby('ticker')}
    for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
        a = g.get(tk)
        if a is not None: out[i] = ((a <= d) & (a > d - np.timedelta64(7, 'D'))).any()
    return out
for c in ('DEAL', 'PR', 'EARN', 'NEG'): O[c] = flag(c)
I = pd.read_pickle(SP + 'ext/insider_P_hist.pkl'); I = I[I.TRANS_CODE == 'P']; g = {t: x.fd.values.astype('datetime64[ns]') for t, x in I.groupby('ISSUERTRADINGSYMBOL')}
O['INS'] = [((g[t] < d) & (g[t] >= d - np.timedelta64(30, 'D'))).any() if t in g else False for t, d in zip(O.tk.values, O.d.values)]
fr = pd.read_parquet(SP + 'fred.parquet'); fr.index = pd.to_datetime(fr.index).astype('datetime64[ns]'); fr = fr.ffill(); mp = pd.read_parquet(SP + 'macro_px.parquet'); mp.index = pd.to_datetime(mp.index).astype('datetime64[ns]'); mp = mp.ffill()
rel = lambda e: (mp[e].pct_change(20) - mp.SPY.pct_change(20))
Dm = pd.DataFrame({'oil': fr.DCOILWTICO.pct_change(20) > 0, 'usd_dn': fr.DTWEXBGS.pct_change(20) < 0, 'steep': (fr.T10Y2Y - fr.T10Y2Y.shift(20)) > 0, 'xlv': rel('XLV') > 0, 'xlk': rel('XLK') > 0, 'y_dn': (fr.DGS10 - fr.DGS10.shift(20)) < 0}).shift(1)
Dm = Dm.reindex(Dm.index.union(O.d.unique())).ffill().reindex(O.d.values)
sec = pd.to_numeric(O.sec, errors='coerce'); grp = np.select([sec.isin([13, 29]), sec.isin([10, 12, 14, 33]), sec.between(60, 64), sec.isin([28, 80, 38]), sec.isin([35, 36, 37, 48, 73]), sec == 49], ['oil', 'usd_dn', 'steep', 'xlv', 'xlk', 'y_dn'], '')
tail = np.full(len(O), np.nan)
for i, gname in enumerate(grp):
    if gname: tail[i] = float(bool(Dm[gname].values[i]))
O['TAIL'] = tail; O['GATE'] = O.DEAL | O.INS | (O.TAIL == 1)
def mp_(a, b):
    x = (a.groupby('mon').r.mean() - b.groupby('mon').r.mean()).dropna(); return x.mean() * 100, (x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan)
def rep(S, label):
    print(f'\n##### {label}: flips {len(S)} | base P(r>=10%) {(S.r >= .1).mean():.1%} P(r>=30%) {(S.r >= .3).mean():.1%} mean {S.r.mean() * 100:+.1f}% median {S.r.median() * 100:+.1f}% win {(S.r > 0).mean():.0%}')
    print(f"{'flag':6s} {'n':>5s} {'share':>6s} | {'P>=10%':>7s} vs no | {'P>=30%':>7s} vs no | {'mean':>7s} vs no | {'median':>7s} | {'win':>4s} | month-paired mean diff (pp) t | P(>=10%) flag vs no by era 2019-21 / 2022 / 2023-26")
    for f in ('DEAL', 'PR', 'EARN', 'NEG', 'INS', 'TAIL', 'GATE'):
        T = S[S.TAIL.notna()] if f == 'TAIL' else S; a = T[T[f] == 1] if f == 'TAIL' else T[T[f]]; b = T[T[f] == 0] if f == 'TAIL' else T[~T[f]]
        if len(a) < 20: print(f'{f:6s} n {len(a)} (too few)'); continue
        d, t = mp_(a, b); era = ' / '.join(f"{(a[a.era3 == e].r >= .1).mean():.0%} v {(b[b.era3 == e].r >= .1).mean():.0%} (n{(a.era3 == e).sum()})" for e in ('2019-21', '2022', '2023-26'))
        print(f"{f:6s} {len(a):5d} {len(a) / len(T):6.0%} | {(a.r >= .1).mean():7.1%} vs {(b.r >= .1).mean():4.1%} | {(a.r >= .3).mean():7.1%} vs {(b.r >= .3).mean():4.1%} | {a.r.mean() * 100:+6.1f}% vs {b.r.mean() * 100:+5.1f}% | {a.r.median() * 100:+6.1f}% | {(a.r > 0).mean():3.0%} | {d:+6.1f} t {t:5.2f} | {era}")
rep(O[O.px < 10], 'px < $10'); rep(O, 'px < $20 (robustness)')
