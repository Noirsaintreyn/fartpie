"""515f: Rey: the earnings-window lift fails only in 2022 (market dump), so test the mechanism: earnings is a binary-event vol/attention shock; in a down-market regime the surprise skews negative so doubles fall. Declared: lift(E14) within ATRP should be >0 when the index is in an up regime (idx_reg>0 / idx_dd shallow) in EVERY era, and <=0 when down. Pooled across eras with month-clustered t."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/report_515d_earnings_window.py').read().split("print('names with 2.02")[0])
print('idx_reg values', P.idx_reg.value_counts(dropna=False).to_dict()); P['up'] = (P.idx_reg > 0).astype(int); P['dd'] = P.idx_dd
print('share of days index up-regime by year:', P.groupby('yr').up.mean().round(2).to_dict())
P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
for lab, m in (('INDEX UP regime', P.up == 1), ('INDEX DOWN regime', P.up == 0), ('all', P.up.isin([0, 1]))):
    for c in ('E14',):
        row = []
        for e in ('2018', '2019-21', '2022', '2023-26'):
            mk = m & (P.era4 == e); 
            try: a, b, d, t = lift(c, mk); row.append(f'{e}: {d:+.3f} (t {t:.1f}, P1 {a:.3f} P0 {b:.3f}, n1 {int(((P[c] == 1) & mk).sum())})')
            except Exception: row.append(f'{e}: n/a')
        try: a, b, d, t = lift(c, m); row.append(f'POOLED: {d:+.3f} (t {t:.1f})')
        except Exception: pass
        print(f'{lab:18s} ' + ' | '.join(row))
a, b, d, t = lift('E14', P.yr != 2022); print('POOLED excluding 2022 (all regimes):', f'{d:+.3f} t {t:.1f}')
a, b, d, t = lift('E14', P.yr >= 0); print('POOLED all years:', f'{d:+.3f} t {t:.1f}')
# 2022 by half
for lab, mk in (('2022 H1', (P.yr == 2022) & (P.d.dt.month <= 6)), ('2022 H2', (P.yr == 2022) & (P.d.dt.month > 6))):
    try: a, b, d, t = lift('E14', mk); print(lab, f'{d:+.3f} t {t:.1f} P1 {a:.3f} P0 {b:.3f}')
    except Exception as e: print(lab, 'n/a')
# size of reaction: forward return to earnings day for flagged days in 2022 vs else (does the event skew negative in dumps?)
