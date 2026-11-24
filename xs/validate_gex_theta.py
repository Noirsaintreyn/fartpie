"""Compare data/theta_gex/<TK>.csv (rebuilt from raw ThetaData) with the existing xs/ext/gex/<TK>.csv: sign agreement, rank correlation, scale ratio, by year. Pass bar (declared): sign agreement >= 85% and spearman >= 0.8 on QQQ before any other ticker is trusted."""
import sys, pandas as pd, numpy as np
for tk in sys.argv[1:]:
    a = pd.read_csv(f'xs/ext/gex/{tk}.csv', parse_dates=['date']).set_index('date').net_gex; b = pd.read_csv(f'data/theta_gex/{tk}.csv', parse_dates=['date']).set_index('date').net_gex; j = pd.concat([a.rename('old'), b.rename('new')], axis=1).dropna()
    print(tk, 'days', len(j), 'sign agree', round(float((np.sign(j.old) == np.sign(j.new)).mean()), 3), 'spearman', round(float(j.old.rank().corr(j.new.rank())), 3), 'median ratio new/old', round(float((j.new / j.old).median()), 3), 'NEG share old/new', round(float((j.old < 0).mean()), 2), round(float((j.new < 0).mean()), 2))
    print(j.groupby(j.index.year).apply(lambda x: round(float((np.sign(x.old) == np.sign(x.new)).mean()), 2)).to_dict())
