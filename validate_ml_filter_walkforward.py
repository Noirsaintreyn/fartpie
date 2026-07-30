"""
Validate the ML filter (backtest_ml_filter.py) with a proper rolling
walk-forward instead of trusting one static 70/30 split. Reuses the
already-collected features/labels (backtest_ml_filter_events.csv) - only
the train/test splitting and model fitting are redone, so this is cheap.

For each (instrument, timeframe) file, independently: split its own
timeline into K sequential folds, then for each fold k >= 2, train on
everything before fold k (expanding window) and evaluate on fold k. This
retrains and re-tests the filter at K-1 different points spread across each
file's 12-year history, instead of one arbitrary split point, and pools all
of those out-of-sample predictions into one combined holdout set. If the
filter's lift is a real, consistent effect, it should show up robustly
across most/all folds, not just the one we happened to look at before.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

FEATURE_COLS = ['vwap_distance_norm', 'vol_forecast_pct_of_price', 'atr_distance']


def zscore_test(p1, n1, p2, n2):
    x1, x2 = p1 * n1, p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0 or n1 == 0 or n2 == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    return z, 2 * (1 - norm.cdf(abs(z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='backtest_ml_filter_events.csv')
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--random-accuracy', type=float, default=0.4227)
    ap.add_argument('--random-n', type=int, default=29257)
    args = ap.parse_args()

    events = pd.read_csv(args.events)
    oos_rows = []
    fold_summary_rows = []

    for (inst, tf), g in events.groupby(['instrument', 'timeframe']):
        g = g.sort_values('t').reset_index(drop=True)
        fold_edges = np.quantile(g['t'], np.linspace(0, 1, args.k_folds + 1))
        fold_ids = np.digitize(g['t'], fold_edges[1:-1], right=True)
        g = g.assign(fold=fold_ids)

        for k in range(1, args.k_folds):
            train = g[g['fold'] < k]
            test = g[g['fold'] == k]
            if len(train) < 200 or len(test) < 50:
                continue
            X_train = pd.get_dummies(train[FEATURE_COLS + ['level_type']], columns=['level_type'])
            X_test = pd.get_dummies(test[FEATURE_COLS + ['level_type']], columns=['level_type'])
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, train['bounced'])
            test = test.copy()
            test['pred_proba'] = model.predict_proba(X_test)[:, 1]
            test['instrument'], test['timeframe'], test['fold'] = inst, tf, k

            # per-fold quick diagnostic: does the filter beat unconditional
            # *within this one fold*, tested against random (small-N per
            # fold, so this is a consistency check, not a claim of
            # significance on its own - the pooled test below is the real one)
            for lt, lg in test.groupby('level_type'):
                if len(lg) < 20:
                    continue
                median_score = lg['pred_proba'].median()
                filt = lg[lg['pred_proba'] >= median_score]
                fold_summary_rows.append({
                    'instrument': inst, 'timeframe': tf, 'fold': k, 'level_type': lt,
                    'n': len(lg), 'unconditional_acc': lg['bounced'].mean(),
                    'filtered_acc': filt['bounced'].mean(), 'n_filtered': len(filt),
                })

            oos_rows.append(test)

    oos = pd.concat(oos_rows, ignore_index=True)
    fold_summary = pd.DataFrame(fold_summary_rows)
    oos.to_csv('backtest_ml_filter_walkforward_oos.csv', index=False)
    fold_summary.to_csv('backtest_ml_filter_walkforward_folds.csv', index=False)

    print(f"Total pooled out-of-sample events (across all folds, all files): {len(oos)}")
    print(f"Folds run per file: {args.k_folds - 1}\n")

    print("=== Per-fold consistency check (lift = filtered_acc - unconditional_acc) ===")
    pd.set_option('display.width', 160)
    print(fold_summary.pivot_table(index=['instrument', 'timeframe', 'fold'], columns='level_type',
                                    values=['unconditional_acc', 'filtered_acc']).round(3).to_string())

    print("\n=== POOLED walk-forward result (the real test - many folds, many time periods) ===")
    p_rand, n_rand = args.random_accuracy, args.random_n
    n_comparisons = oos['level_type'].nunique()
    alpha = 0.05 / n_comparisons

    for level_type, g in oos.groupby('level_type'):
        n_total = len(g)
        unconditional_acc = g['bounced'].mean()
        median_score = g['pred_proba'].median()
        filtered = g[g['pred_proba'] >= median_score]
        filtered_acc = filtered['bounced'].mean()
        n_filtered = len(filtered)

        # what fraction of folds showed POSITIVE lift (sign consistency check)
        lt_folds = fold_summary[fold_summary['level_type'] == level_type]
        pct_positive_folds = (lt_folds['filtered_acc'] > lt_folds['unconditional_acc']).mean()

        z_uncond, p_uncond = zscore_test(unconditional_acc, n_total, p_rand, n_rand)
        z_filt, p_filt = zscore_test(filtered_acc, n_filtered, p_rand, n_rand)

        print(f"\n{level_type} (pooled walk-forward holdout n={n_total}):")
        print(f"  Unconditional accuracy: {unconditional_acc:.4f}  (z={z_uncond:.2f}, p={p_uncond:.5f} vs random)")
        print(f"  ML-filtered accuracy:   {filtered_acc:.4f}  n={n_filtered}  (z={z_filt:.2f}, p={p_filt:.5f} vs random, "
              f"Bonferroni threshold={alpha:.5f})")
        print(f"  Filter lift: {filtered_acc - unconditional_acc:+.4f}")
        print(f"  Fraction of individual folds where filter beat unconditional: {pct_positive_folds:.1%} "
              f"({int(len(lt_folds))} folds tested)")


if __name__ == '__main__':
    main()
