"""Nov 8 2026: TRADE REVIEW of the prop system (S1: Rule A, 75 / 50, flat 12:00, <= 2 a day, POC-to-VAH skip, Yang-Zhang + VWAP ladder) - what limits payouts besides
good vs bad entries. Goal: 2 payouts a month per account = 10 qualifying days a month (Lucid 50K: +$150 days; 150K: +$250) and enough profit for two payouts.
1 FREQUENCY  trades a month, days with a trade, QUALIFYING DAYS a month at 50K / 150K ladder size, by year
2 EXITS      target / stop / noon: share, average, by year
3 PATHS      max favorable (MFE) / adverse (MAE) excursion: how far stopped trades went our way first; how deep winners went first; noon exits
4 TIME       minutes to target / stop; entry time buckets; weekday; first vs second trade of the day
5 STREAKS    longest losing runs and how often 2 full stops in a row (= a 50K at 10 micros is gone)
6 WHAT-IFS   per trade, 1 contract: breakeven stop after +20 / +30 / +40; half off at +20 / +30 (rest as is) - win rate and points per trade
NQ points per contract after 1 pt cost; 2018 - Oct 2026 unless noted."""
_e = open('xs/nq_account_ev.py').read(); exec(_e[:_e.rindex("if __name__ == '__main__':")])
rows = []
for idx, (net, qe, p, how) in full.items():
    if skip2.loc[idx] <= 0: continue
    day = pd.Timestamp(XA.at[idx, 'day']); k = dpos.get(day)
    yzh = bool(YZF.iloc[k]) if k is not None else False; vx = bool(X.at[idx, 'VWAPX']); size = (1.0, .5, .3)[min(int(yzh) + int(vx), 2)]
    pa = np.array(p); rows.append(dict(idx=idx, day=day, j=int(XA.at[idx, 'j']), net=net - 1.0, how=how, mfe=pa.max() - 1.0, mae=pa.min() - 1.0, bars=len(pa) / 2 * 5, size=size, path=pa))
T = pd.DataFrame(rows).sort_values(['day', 'j']).reset_index(drop=True); T['yr'] = T.day.dt.year; T['me'] = m5[T.j.values] - 5; T['wd'] = T.day.dt.day_name().str[:3]
T['nth'] = T.groupby('day').cumcount() + 1
ALLD = pd.DatetimeIndex(D_[:NS]); ALLD = ALLD[ALLD >= T.day.min()]
def whatif(r, kind, X_):
    p = r.path - 1.0
    if kind == 'be':
        hit = np.where(p >= X_)[0]
        if len(hit) == 0: return r.net
        after = np.where(p[hit[0]:] <= 0)[0]
        return -1.0 if len(after) else r.net
    if kind == 'half': return .5 * X_ + .5 * r.net if r.mfe >= X_ else r.net
if __name__ == '__main__':
    mo = len(ALLD) / 21
    print('1 FREQUENCY (S1 trades)', flush=True)
    for nm, sc, thr in (('50K', 1.0, 150), ('150K', 2.25, 250)):
        T['usd'] = T.net * 10 * T['size'] * sc * 2; dP = T.groupby('day').usd.sum().reindex(ALLD).fillna(0.0)
        print(f'  {nm}: {len(T) / mo:.1f} trades a month, {(dP != 0).sum() / mo:.1f} days with a trade a month ({(dP != 0).mean():.0%} of days), QUALIFYING days (>= ${thr}) {(dP >= thr).sum() / mo:.1f} a month '
              f'| by year ' + ' '.join(f'{y} {(dP[dP.index.year == y] >= thr).sum() / (dP.index.year == y).sum() * 21:.1f}' for y in range(2018, 2027)), flush=True)
    print(f'  ladder sizes: 10 micros {np.mean(T["size"] == 1):.0%}, 5 {np.mean(T["size"] == .5):.0%}, 3 {np.mean(T["size"] == .3):.0%} -> a qualifying day at 50K needs +7.5 / +15 / +25 points', flush=True)
    print('\n2 EXITS', flush=True)
    for h, g in T.groupby('how'):
        print(f'  {h:<7} {len(g) / len(T):4.0%} avg {g.net.mean():+6.1f} | by year ' + ' '.join(f'{y} {np.mean(T[T.yr == y].how == h):3.0%}' for y in range(2018, 2027)), flush=True)
    print('\n3 PATHS', flush=True)
    st = T[T.how == 'stop']; tg = T[T.how == 'target']; nn = T[T.how == 'noon']
    print(f'  stopped trades ({len(st)}): went our way first by >= +10 {np.mean(st.mfe >= 10):.0%}, +20 {np.mean(st.mfe >= 20):.0%}, +30 {np.mean(st.mfe >= 30):.0%}, +40 {np.mean(st.mfe >= 40):.0%}; median MFE {st.mfe.median():+.0f}', flush=True)
    print(f'  target trades ({len(tg)}): went against first by <= -15 {np.mean(tg.mae <= -15):.0%}, -25 {np.mean(tg.mae <= -25):.0%}, -35 {np.mean(tg.mae <= -35):.0%}, -45 {np.mean(tg.mae <= -45):.0%}; median MAE {tg.mae.median():+.0f}', flush=True)
    print(f'  noon exits ({len(nn)}): avg {nn.net.mean():+.1f}, win {np.mean(nn.net > 0):.0%}, between -10 and +10: {np.mean(nn.net.abs() <= 10):.0%}; median MFE {nn.mfe.median():+.0f}, reached +30 {np.mean(nn.mfe >= 30):.0%}, +50 {np.mean(nn.mfe >= 50):.0%}', flush=True)
    print('\n4 TIME', flush=True)
    print(f'  minutes in trade: target median {tg.bars.median():.0f}, stop median {st.bars.median():.0f}, noon median {nn.bars.median():.0f}', flush=True)
    for lab, m in (('09:45-10:00', T.me < 600), ('10:00-10:30', (T.me >= 600) & (T.me < 630)), ('10:30-11:00', (T.me >= 630) & (T.me < 660)), ('11:00-11:30', T.me >= 660)):
        g = T[m]; print(f'  entry {lab}: n{len(g):3d} win {np.mean(g.net > 0):3.0%} avg {g.net.mean():+6.1f} target {np.mean(g.how == "target"):3.0%} stop {np.mean(g.how == "stop"):3.0%}', flush=True)
    for w in ('Mon', 'Tue', 'Wed', 'Thu', 'Fri'):
        g = T[T.wd == w]; print(f'  {w}: n{len(g):3d} win {np.mean(g.net > 0):3.0%} avg {g.net.mean():+6.1f}', flush=True)
    for k_ in (1, 2):
        g = T[T.nth == k_]; print(f'  trade #{k_} of the day: n{len(g):3d} win {np.mean(g.net > 0):3.0%} avg {g.net.mean():+6.1f}', flush=True)
    print('\n5 STREAKS', flush=True)
    los = (T.net <= 0).values; runs = []; c = 0
    for v in los: c = c + 1 if v else (runs.append(c) or 0)
    runs.append(c); runs = np.array([r for r in runs if r > 0]); full_stops = (T.how == 'stop').values
    two = np.sum(full_stops[1:] & full_stops[:-1])
    print(f'  losing runs: longest {runs.max()}, runs of 3+ {np.sum(runs >= 3)}, of 4+ {np.sum(runs >= 4)} | back-to-back full stops {two} times in {len(T)} trades ({two / (len(ALLD) / 252):.1f} a year)', flush=True)
    print('\n6 WHAT-IFS per trade (1 contract): win rate, avg | 2018-21 avg / 2022-26 avg', flush=True)
    base = T.net.values; yy = T.yr.values
    print(f'  as is                     win {np.mean(base > 0):3.0%} avg {base.mean():+6.2f} | {base[yy < 2022].mean():+6.2f} / {base[yy >= 2022].mean():+6.2f}', flush=True)
    for kind, xs_ in (('be', (20, 30, 40)), ('half', (20, 30))):
        for x_ in xs_:
            v = np.array([whatif(r, kind, x_) for r in T.itertuples()])
            lab = f'breakeven after +{x_}' if kind == 'be' else f'half off at +{x_}'
            print(f'  {lab:<25} win {np.mean(v > 0):3.0%} avg {v.mean():+6.2f} | {v[yy < 2022].mean():+6.2f} / {v[yy >= 2022].mean():+6.2f}', flush=True)
    T.drop(columns=['path']).to_parquet('/tmp/claude-0/ml/nq_trade_review.parquet')
