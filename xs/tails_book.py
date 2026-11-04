"""Nov 4 2026: book test of 'prefer likely-BIG trades' (follows xs/tails.py). DECLARED BEFORE RESULTS. 40 matched runs vs live, 6 slots, 2019-21 / 2022-26.
  V1/V2  K = ratio x (1 + w x ATR% quintile/4), w = .25 / 1.0       (magnitude tilt: higher-vol trades had higher avg return in both halves)
  V3     skip the lowest ATR% quintile (within year)
  M1/M2  K = ratio x (1 + w x P(BIG)/base), P(BIG) cross-fit: model trained on the OTHER half (no trade scored by a model that saw it), w = .25 / 1.0
  M3     K = ratio x (1 + .5 x (P(BIG) - P(BAD)) / base)   (asymmetry, cross-fit)
PASS = beats live ret/DD in >= 70% of runs in both halves."""
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
from sklearn.ensemble import HistGradientBoostingClassifier
T = pd.read_parquet('/tmp/claude-0/ml/tails_T.parquet')
num = ['gap', 'runup', 'z20', 'prior', 'margin', 'tod', 'open_bar', 'qqq', 'whip', 'tkrec', 'rv', 'rvrel', 'dnext', 'dlast', 'spy200', 'vts', 'vixp', 'spy20', 'ratio', 'h1f',
       'egap', 'esurp', 'rel60', 'rel20', 'hi52', 'dratio', 'atr_pct', 'squeeze', 'bbw', 'd_hi20', 'd_hi80', 'base', 'trend50', 'e10e20', 'egap_open', 'rvol', 'dd_days']   # crowd left out: it counts later same-day flips (look-ahead)
Xf = T[num].astype(float); P = {}
for tgt in ('BIG', 'BAD'):
    p = np.full(len(T), np.nan)
    for a, b in (('19-21', '22-26'), ('22-26', '19-21')):
        tr, te = (T.per == a).values, (T.per == b).values
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=300, min_samples_leaf=80, l2_regularization=1.0, random_state=0).fit(Xf[tr], T[tgt][tr]); p[te] = m.predict_proba(Xf[te])[:, 1]
    P[tgt] = p
q = T.groupby(T.entry.dt.year).atr_pct.transform(lambda z: pd.qcut(z.rank(method='first'), 5, labels=False)).fillna(2).values
B0 = ev(T); line('L live', B0)
def K(nm, k): Y = T.copy(); Y['ratio'] = k; line(nm, ev(Y), B0)
K('V1 ATR% tilt w.25', T.ratio * (1 + .25 * q / 4)); K('V2 ATR% tilt w1.0', T.ratio * (1 + 1.0 * q / 4))
Y = T[q > 0].copy(); line('V3 skip lowest ATR% quintile (drops 20%)', ev(Y), B0)
bb = np.nanmean(P['BIG']); K('M1 P(BIG) tilt w.25', T.ratio * (1 + .25 * P['BIG'] / bb)); K('M2 P(BIG) tilt w1.0', T.ratio * (1 + 1.0 * P['BIG'] / bb))
K('M3 P(BIG)-P(BAD) tilt', T.ratio * (1 + .5 * (P['BIG'] - P['BAD']) / bb))
