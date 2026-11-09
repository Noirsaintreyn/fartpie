"""Nov 8 2026: ONE-CLASS SVM 'PEAK-TRADE' DETECTOR - learn the shape of the context around the best trades, not a win/loss boundary. DECLARED BEFORE RESULTS.
Why: trained classifiers failed because 'good flip' labels are noisy (win rate barely moves); a one-class model only describes where the big winners live.
Trades: noise momentum on real NQ futures (all days, both directions, stop 0.15 ADR), 2014 - Jan 2026. Context at entry (all known then): direction; prior-close Greeks
percentiles (GEX, DEX, ATM IV, skew, put/call volume and OI), vanna and charm signs, vanna flow x direction, IV 5-day change sign; Yang-Zhang 1 / 5 / 20-day percentiles;
gap and the open's place vs the prior value area (trade direction); overnight range / ADR; entry minute.
Walk-forward by year (train on all earlier years, test the next, 2017 - 2025): two RBF one-class SVMs (nu 0.15) - one on the top 20% of training trades by points (peaks),
one on the bottom 20% (worst) - CONTRAST score = peak score - worst score. Take trades whose contrast is in the top 30% / 15% of the training trades' scores.
Compared with the hand-built gate (negative gamma + DEX not high). Points per trade and a day at 1 contract, by year."""
_y = open('/home/claude/fartpie/xs/dte_yz.py').read(); exec(_y[:_y.rindex("if __name__ == '__main__':")])
_gr = open('/home/claude/fartpie/xs/dte_greeks.py').read(); exec(_gr[_gr.index('def m1_dir'):_gr.index('def r1_dir')])
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
def feats(T):
    rows = []
    for r_ in T.itertuples():
        d = pd.Timestamp(r_.day); f = DAYF.get(d); k = di_.get(d)
        if f is None or k is None or k == 0: rows.append(None); continue
        val, poc, vah = VA[days[k - 1]]; o0 = DY.o[d]; pc = DY.c[days[k - 1]]; adr = DY.adr[d]; dr = r_.dir
        loc = (1 if o0 > vah else (-1 if o0 < val else 0)) * dr; ovr = NSO['NR'].get(d, np.nan) / adr if np.isfinite(adr) else np.nan
        rows.append([dr, f.get('gex2p'), f.get('dexp'), f.get('ivp'), f.get('skewp'), f.get('pcvp'), f.get('pcop'), np.sign(f.get('net_vanna')), np.sign(f.get('net_charm')),
                     (f.get('vflow') or 0) * dr, np.sign(f.get('iv5')), DY.yz1p.get(d), DY.yz5p.get(d), DY.yz20p.get(d), (o0 - pc) / adr * dr, loc, ovr, r_.m if hasattr(r_, 'm') else np.nan])
    return rows
if __name__ == '__main__':
    T = m1_dir(); T = T[T.day.isin(DY.index)].reset_index(drop=True)
    tm = []
    for r_ in T.itertuples():
        ix = gi[pd.Timestamp(r_.day)]; w = np.where(np.isclose(o[ix], r_.e))[0]; tm.append(m[ix[w[0]]] if len(w) else np.nan)
    T['m'] = tm; FX = feats(T); keep = [i for i, f_ in enumerate(FX) if f_ is not None and np.isfinite(np.array(f_, dtype=float)).all()]
    T = T.loc[keep].reset_index(drop=True); Z = np.array([FX[i] for i in keep], dtype=float); yr = T.day.dt.year.values
    print(f'noise trades with full context: {len(T)} ({yr.min()} - {yr.max()}), avg {T.pts.mean():+.2f}', flush=True)
    SC = np.full(len(T), np.nan); THR = {}
    for Y in range(2017, 2026):
        tr, te = yr < Y, (yr == Y) if Y < 2025 else (yr >= 2025)
        sc_ = StandardScaler().fit(Z[tr]); Ztr = sc_.transform(Z[tr]); p = T.pts.values[tr]; hi, lo = np.quantile(p, [.8, .2])
        top = OneClassSVM(nu=.15, gamma='scale').fit(Ztr[p >= hi]); bot = OneClassSVM(nu=.15, gamma='scale').fit(Ztr[p <= lo])
        s_tr = top.decision_function(Ztr) - bot.decision_function(Ztr); s_te = top.decision_function(sc_.transform(Z[te])) - bot.decision_function(sc_.transform(Z[te]))
        SC[te] = s_te; THR[Y] = (np.quantile(s_tr, .7), np.quantile(s_tr, .85))
        print(f'  {Y}: train {tr.sum()}, test {te.sum()} | test avg: top30% {T.pts.values[te][s_te >= THR[Y][0]].mean():+6.2f} (n{(s_te >= THR[Y][0]).sum()}) | '
              f'top15% {T.pts.values[te][s_te >= THR[Y][1]].mean():+6.2f} | rest {T.pts.values[te][s_te < THR[Y][0]].mean():+6.2f}', flush=True)
    te_all = yr >= 2017; t30 = np.array([THR[y if y < 2025 else 2025][0] if y >= 2017 else np.inf for y in yr]); t15 = np.array([THR[y if y < 2025 else 2025][1] if y >= 2017 else np.inf for y in yr])
    gate = np.array([bool(DY.gex.get(pd.Timestamp(d), np.nan) < 0) and not (fget(pd.Timestamp(d), 'dexp') >= 2 / 3) for d in T.day])
    AD = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2017-01-01') and ok(d)]); AD18 = AD[AD >= '2018-02-01']
    print(f'\n  selection (2017 - Jan 2026 unless noted)        trades  win   pts/trade  pts/day  2017-21 / 2022-25 per trade', flush=True)
    for lab, msk, ad in (('every noise trade', te_all, AD), ('OCSVM contrast top 30%', te_all & (SC >= t30), AD), ('OCSVM contrast top 15%', te_all & (SC >= t15), AD),
                         ('hand gate: neg gamma + DEX ok (2018+)', te_all & gate, AD18), ('OCSVM top 30% AND hand gate (2018+)', te_all & gate & (SC >= t30), AD18)):
        t = T[msk]; P = t.groupby('day').pts.sum().reindex(ad).fillna(0.0); a = t.pts[t.day.dt.year <= 2021].mean(); b = t.pts[t.day.dt.year >= 2022].mean()
        print(f'  {lab:<44} {len(t):6d} {np.mean(t.pts > 0):4.0%} {t.pts.mean():+9.2f} {P.mean():+8.2f}  {a:+6.2f} / {b:+6.2f}', flush=True)
    T.assign(sc=SC).to_parquet('/tmp/claude-0/ml/dte_ocsvm_noise.parquet')
