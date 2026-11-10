"""Nov 9 2026: CAN WE TELL A GOOD FLIP FROM A BAD ONE BEFORE ENTRY? ACCURACY FIRST, ONE TRADE A DAY (Rey: for options, accuracy and risk > return). DECLARED BEFORE RESULTS.
Every screener bull flip 2019 - Sep 2026 (54 names; options targets on the 23 names with ThetaData). Features known before entry: options ratio and its rank in the bar,
Yang-Zhang, HAR, HAR vs normal, prior-day relative volume, gap, 52-week high, 20-day strength vs SPY, prior-day return, idiosyncratic vol, put / call volume,
realized - implied vol, own gamma sign, SPY 20-day trend, entry time, 1h vs 15m track, flips that bar, weekday.
Targets (win = > 0):  STOCK  the screener trade itself | PCS  monthly put credit spread, short strike 0.5 sigma below spot, held to expiry (real fills, bid / ask)
                      | CALL  weekly call closed at the screener exit (real fills)
Models: logistic regression (standardised, median-filled) and shallow gradient boosting, WALK-FORWARD: trained on all years before Y, tested on Y, Y = 2021..2026.
Judged out of sample only: AUC; win rate and average return of the model's TOP-1 flip per day vs a RANDOM flip per day vs all flips; top decile vs bottom decile."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
_s = open('/home/claude/fartpie/xs/scr_gamma_combo.py').read(); exec(_s[:_s.index("\nif __name__ == '__main__':")])
X = X.copy(); X['tm'] = X.entry.dt.hour * 60 + X.entry.dt.minute; X['n'] = X.groupby('entry').tk.transform('size'); X['rk'] = X.groupby('entry').ratio.rank(ascending=False)
X['h1f'] = X.h1.astype(float); X['wd'] = X.day.dt.weekday; X['gam'] = np.sign(X.net_gex); X['yr'] = X.day.dt.year
spy20 = SPYd.close.pct_change(20).shift(1); X['spy20'] = X.day.map(spy20)
FEAT = ['ratio', 'rk', 'yz', 'har', 'harrel', 'rvol', 'gap', 'hi52', 'rs20', 'rev1', 'ivol', 'pcr', 'vrp', 'gam', 'spy20', 'tm', 'h1f', 'n', 'wd']
R = pd.read_parquet('/tmp/claude-0/opt_flips_R.parquet'); R['d0'] = pd.to_datetime(R.d0)
first = X.sort_values('entry').drop_duplicates(['tk', 'day']).set_index(['tk', 'day'])[FEAT + ['yr']]
def opt_target(ten, st, ex):
    z = R[(R.set == 'flip') & (R.ten == ten) & (R.st == st) & (R.ex == ex) & (R.fill == 'TOUCH')].copy(); z['day'] = z.d0.astype('datetime64[ns]')
    z = z.set_index(['tk', 'day'])[['ret']].join(first, how='inner').reset_index(); return z
def walk(Z, label):
    Z = Z.dropna(subset=['ret']).reset_index(drop=True); Z['y'] = (Z.ret > 0).astype(int); Z['p_lr'] = np.nan; Z['p_gb'] = np.nan
    for Y in range(2021, 2027):
        tr, te = Z.yr < Y, Z.yr == Y
        if te.sum() == 0 or tr.sum() < 300: continue
        med = Z.loc[tr, FEAT].median(); A, B_ = Z.loc[tr, FEAT].fillna(med), Z.loc[te, FEAT].fillna(med); mu, sd = A.mean(), A.std().replace(0, 1)
        lr = LogisticRegression(C=.1, max_iter=2000).fit((A - mu) / sd, Z.loc[tr, 'y']); Z.loc[te, 'p_lr'] = lr.predict_proba((B_ - mu) / sd)[:, 1]
        gb = HistGradientBoostingClassifier(max_depth=2, max_iter=150, learning_rate=.05, min_samples_leaf=300, random_state=0).fit(Z.loc[tr, FEAT], Z.loc[tr, 'y'])
        Z.loc[te, 'p_gb'] = gb.predict_proba(Z.loc[te, FEAT])[:, 1]
    O = Z[Z.p_lr.notna()].copy(); rng = np.random.default_rng(0); O['rnd'] = rng.random(len(O))
    print(f'\n=== {label}: out-of-sample 2021-2026, {len(O):,} trades, base win rate {O.y.mean():.0%}, avg {O.ret.mean():+.2%} ===', flush=True)
    print('  AUC by year (logistic / boosting): ' + ' | '.join(f'{y}: {roc_auc_score(g_.y, g_.p_lr):.2f} / {roc_auc_score(g_.y, g_.p_gb):.2f}' for y, g_ in O.groupby('yr') if g_.y.nunique() > 1)
          + f' | pooled {roc_auc_score(O.y, O.p_lr):.3f} / {roc_auc_score(O.y, O.p_gb):.3f}', flush=True)
    for col, nm in (('p_lr', 'logistic'), ('p_gb', 'boosting')):
        q = O.groupby('yr')[col].transform(lambda s_: s_.rank(pct=True)); top, bot = O[q >= .9], O[q <= .1]
        t1 = O.loc[O.groupby('day')[col].idxmax()]; r1 = O.loc[O.groupby('day').rnd.idxmax()]
        print(f'  {nm:<9} top decile: win {top.y.mean():.0%}, avg {top.ret.mean():+.2%} | bottom decile: win {bot.y.mean():.0%}, avg {bot.ret.mean():+.2%} | '
              f'TOP-1 a day: win {t1.y.mean():.0%}, avg {t1.ret.mean():+.2%} (n {len(t1)}) vs RANDOM-1 a day: win {r1.y.mean():.0%}, avg {r1.ret.mean():+.2%}', flush=True)
        print('             top-1 a day by year: ' + ' | '.join(f'{y}: win {g_.y.mean():.0%} avg {g_.ret.mean():+.1%}' for y, g_ in t1.groupby('yr')), flush=True)
    return O
if __name__ == '__main__':
    S = X[['tk', 'day', 'yr', 'ret'] + FEAT].copy(); OS = walk(S, 'STOCK: the screener trade')
    # follow-up: use the STOCK model's out-of-sample score to pick OPTIONS trades on the same flips (23 names); top decile = about 1 flip a day
    OS['q'] = OS.groupby('yr').p_lr.transform(lambda s_: s_.rank(pct=True)); sc = OS.sort_values('q').drop_duplicates(['tk', 'day'], keep='last').set_index(['tk', 'day']).q
    print(f'\n=== options on flips picked by the STOCK model (logistic, out of sample): top decile = {OS[OS.q >= .9].shape[0] / OS.day.nunique():.1f} flips a day ===', flush=True)
    for ten, st, ex, nm in (('W', 'CALL', 'VEC', 'weekly call, screener exit'), ('W', 'CALL', 'EXP', 'weekly call, to expiry'), ('M', 'PCS 0.5', 'EXP', 'monthly put spread 0.5 sigma, to expiry'),
                            ('M', 'CSP 0.5', 'EXP', 'monthly cash-secured put 0.5 sigma, to expiry')):
        z = R[(R.set == 'flip') & (R.ten == ten) & (R.st == st) & (R.ex == ex) & (R.fill == 'TOUCH')].copy(); z['day'] = z.d0.astype('datetime64[ns]'); z = z[z.day.dt.year >= 2021]
        z['q'] = [sc.get((t, d), np.nan) for t, d in zip(z.tk, z.day)]; z = z.dropna(subset=['q']); t_, b_ = z[z.q >= .9], z[z.q < .9]
        print(f'  {nm:<42} top decile: win {np.mean(t_.ret > 0):.0%}, avg {t_.ret.mean():+.2%}, median {t_.ret.median():+.1%} (n {len(t_)}) | rest: win {np.mean(b_.ret > 0):.0%}, avg {b_.ret.mean():+.2%} | '
              + 'top decile by year: ' + ' '.join(f'{y % 100}:{g_.ret.mean():+.0%}' for y, g_ in t_.groupby(t_.day.dt.year)), flush=True)
    walk(opt_target('M', 'PCS 0.5', 'EXP'), 'PCS: monthly put credit spread 0.5 sigma OTM, to expiry, real fills')
    walk(opt_target('W', 'CALL', 'VEC'), 'CALL: weekly call closed at the screener exit, real fills')
# --- follow-up 2: the stock-model top decile -> monthly put credit spread book (1 unit of risk per spread), out of sample 2021-2026
if __name__ == '__main__':
    z = R[(R.set == 'flip') & (R.ten == 'M') & (R.st == 'PCS 0.5') & (R.ex == 'EXP') & (R.fill == 'TOUCH')].copy(); z['day'] = z.d0.astype('datetime64[ns]'); z = z[z.day.dt.year >= 2021]
    z['q'] = [sc.get((t, d), np.nan) for t, d in zip(z.tk, z.day)]; z = z.dropna(subset=['q']); t_ = z[z.q >= .9].copy(); t_['mo'] = t_.day.dt.to_period('M')
    mo = t_.groupby('mo').ret.sum(); top5 = t_.ret.sort_values().iloc[:5]
    print(f'\n=== put-spread book on the stock-model top decile: {len(t_) / t_.mo.nunique():.1f} spreads in active months | month avg {mo.mean():+.2f} units of risk, months positive {np.mean(mo > 0):.0%}, '
          f'worst month {mo.min():+.2f}, best {mo.max():+.2f} | worst 5 trades {", ".join(f"{x:+.0%}" for x in top5)} | full losses (<= -90%) {np.mean(t_.ret <= -.9):.0%} of trades', flush=True)
