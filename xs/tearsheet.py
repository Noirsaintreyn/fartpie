"""
Phase 0 / step 3 - cross-sectional tearsheet.

For each sleeve, at weekly rebalance dates (last trading day of each week),
in-universe names only:
  score -> percentile rank within (date, sector)  [sector-neutral]
  rank IC vs y_h (residual label, entry next open)       h in 5/20/60
  Newey-West t-stat of the weekly IC series (lags = h/5 overlap)
  quintile long-short on y_h, turnover, net of costs (COST_BPS one-way per trade)
  deciles of y20, IC by year, first/second-half split
Pre-registered pass rule (decided before looking):
  NW t(IC_20) >= 3  AND  net LS20 > 0 in BOTH halves.

Usage:  python3 xs/tearsheet.py              # all sleeves
        python3 xs/tearsheet.py rmom12_1 on60
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sleeves import SLEEVES  # noqa: E402

X = os.path.dirname(os.path.abspath(__file__))
from cfg import D, REP  # noqa: E402
HORIZONS = (5, 20, 60)
COST_BPS = 5.0   # one-way, S&P 500 names: half-spread + slippage + commissions
PASS_T = 3.0


def nw_tstat(x, lags):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 20:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(s / n)


def rebalance_dates(dates):
    d = pd.Series(pd.to_datetime(sorted(set(dates))))
    return set(d.groupby(d.dt.to_period('W')).max())


def prep(labels, score_wide, rb):
    s = score_wide.stack(future_stack=True).rename('score').reset_index()
    s.columns = ['date', 'ticker', 'score']
    df = labels[labels.date.isin(rb)].merge(s, on=['date', 'ticker'], how='inner')
    df = df.dropna(subset=['score'])
    df = df[df.groupby('date')['ticker'].transform('size') >= 30]
    df['s'] = df.groupby(['date', 'sector'])['score'].rank(pct=True) - 0.5
    df['q'] = df.groupby('date')['s'].transform(
        lambda v: pd.qcut(v.rank(method='first'), 5, labels=False))
    df['dec'] = df.groupby('date')['s'].transform(
        lambda v: pd.qcut(v.rank(method='first'), 10, labels=False))
    return df


def ic_series(df, h):
    y = f'y{h}'
    g = df.dropna(subset=[y]).groupby('date')
    return g.apply(lambda d: d['s'].rank().corr(d[y].rank()) if len(d) > 50 else np.nan,
                   include_groups=False)


def ls_stats(df, h):
    y = f'y{h}'
    d = df.dropna(subset=[y])
    top = d[d.q == 4].groupby('date')[y].mean()
    bot = d[d.q == 0].groupby('date')[y].mean()
    ls = (top - bot).dropna()
    # turnover: each cohort rebalances every n weeks
    n = max(1, h // 5)
    dates = sorted(d.date.unique())
    sets = {dt: (set(d[(d.date == dt) & (d.q == 4)].ticker), set(d[(d.date == dt) & (d.q == 0)].ticker))
            for dt in dates}
    tos = []
    for i in range(n, len(dates)):
        a, b = sets[dates[i - n]], sets[dates[i]]
        t = [1 - len(a[k] & b[k]) / max(1, len(b[k])) for k in (0, 1)]
        tos.append(np.mean(t))
    to = float(np.mean(tos)) if tos else np.nan
    cost = 2 * 2 * to * COST_BPS / 1e4          # 2 legs x (sell+buy) x turnover
    net = ls - cost
    sh = []
    for off in range(n):
        sub = net.iloc[off::n]
        if len(sub) > 10 and sub.std() > 0:
            sh.append(sub.mean() / sub.std() * np.sqrt(252 / h))
    return ls, net, to, cost, float(np.mean(sh)) if sh else np.nan


def run(name, labels, rb):
    score = SLEEVES[name]()
    df = prep(labels, score, rb)
    lines = [f'==== {name} ====', f'obs {len(df):,}  rebalances {df.date.nunique()}  '
             f'names/date {len(df) / max(1, df.date.nunique()):.0f}']
    row = {'sleeve': name}
    for h in HORIZONS:
        ic = ic_series(df, h)
        t = nw_tstat(ic, max(1, h // 5))
        ls, net, to, cost, sh = ls_stats(df, h)
        ann = net.mean() * 252 / h
        lines.append(f'h={h:>2}  IC {ic.mean():+.4f}  IC_IR {ic.mean() / ic.std():+.3f}  '
                     f'NW-t {t:+.2f}  hit {(ic.dropna() > 0).mean():.0%}   LS gross {ls.mean() * 1e4:+.1f}bp  '
                     f'net {net.mean() * 1e4:+.1f}bp/{h}d  ann {ann:+.2%}  Sharpe {sh:+.2f}  '
                     f'turnover {to:.0%}')
        row.update({f'IC{h}': ic.mean(), f't{h}': t, f'net{h}_ann': ann, f'sharpe{h}': sh,
                    f'to{h}': to})
        if h == 20:
            ic20, net20 = ic, net
    dec = df.dropna(subset=['y20']).groupby('dec')['y20'].mean() * 1e4
    mono = pd.Series(dec.values).rank().corr(pd.Series(range(len(dec)), dtype=float))
    lines.append('y20 by decile (bp): ' + ' '.join(f'{v:+.0f}' for v in dec.values) +
                 f'   monotonicity {mono:+.2f}')
    yr = ic20.groupby(ic20.index.year).mean()
    lines.append('IC20 by year: ' + '  '.join(f'{y}:{v:+.3f}' for y, v in yr.items()))
    mid = sorted(ic20.index)[len(ic20) // 2]
    h1, h2 = net20[net20.index < mid], net20[net20.index >= mid]
    i1, i2 = ic20[ic20.index < mid], ic20[ic20.index >= mid]
    lines.append(f'halves: IC20 {i1.mean():+.4f} | {i2.mean():+.4f}   '
                 f'net LS20 {h1.mean() * 1e4:+.1f}bp | {h2.mean() * 1e4:+.1f}bp')
    passed = (row['t20'] >= PASS_T) and h1.mean() > 0 and h2.mean() > 0
    # sign-flipped pass (sleeve works in reverse) reported separately, not auto-passed
    flip = (row['t20'] <= -PASS_T) and h1.mean() < 0 and h2.mean() < 0
    lines.append(f'PASS: {"YES" if passed else "no"}' + ('   (significant in REVERSE)' if flip else ''))
    row.update({'mono': mono, 'net20_h1': h1.mean(), 'net20_h2': h2.mean(), 'pass': passed,
                'reverse': flip})
    return '\n'.join(lines), row, df[['date', 'ticker', 's']].rename(columns={'s': name})


def main():
    os.makedirs(REP, exist_ok=True)
    names = sys.argv[1:] or list(SLEEVES)
    labels = pd.read_parquet(os.path.join(D, 'labels.parquet'))
    rb = rebalance_dates(labels.date)
    texts, rows, scores = [], [], None
    for n in names:
        txt, row, sc = run(n, labels, rb)
        print(txt + '\n', flush=True)
        texts.append(txt)
        rows.append(row)
        scores = sc if scores is None else scores.merge(sc, on=['date', 'ticker'], how='outer')
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(REP, 'summary.csv'), index=False)
    rk = scores.groupby('date')[names].rank()
    rk['date'] = scores['date']
    corr = rk.groupby('date')[names].corr().groupby(level=1).mean()
    corr = corr.loc[names, names]
    ctxt = 'Average cross-sectional rank correlation between sleeves:\n' + corr.round(2).to_string()
    print(ctxt)
    with open(os.path.join(REP, 'tearsheet.txt'), 'w') as f:
        f.write(f'cost {COST_BPS}bp one-way | pass rule NW t(IC20)>={PASS_T} and net LS20>0 both halves\n\n')
        f.write('\n\n'.join(texts) + '\n\n' + ctxt + '\n')


if __name__ == '__main__':
    main()
