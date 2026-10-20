"""
Mid/small-cap panel (run with XS_U=mid for the rest of the pipeline).

In-universe on trading day d when ALL hold (every test uses data known before d):
  - DoltHub has an options print for the symbol in (d-10 days, d]      -> point-in-time, optionable
  - not an S&P 500 member that month (sp500_pit_membership.json)
  - operating company: SEC submissions with a SIC code, not funds/trusts (SIC 6221, 6722, 6726, 6792)
  - prior close >= $5 and 20d median dollar volume (lagged 1 day) >= $5M
Sector: SIC mapped to GICS-style buckets.  Fundamentals: xs/raw/ext_fund -> data_mid/fund_q, fund_a.

Usage: XS_U=mid python3 xs/panel_build_mid.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

X = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(X)
os.environ.setdefault('XS_U', 'mid')
sys.path.insert(0, X)
from cfg import D, VH, RAW  # noqa: E402

START, END = '2019-03-01', '2026-09-18'
MIN_PX, MIN_DV = 5.0, 5e6
FUND_EXCL = {'6221', '6722', '6726', '6792', '6798x'}


def sic_sector(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return 'Unknown'
    if 1300 <= s <= 1399 or 2900 <= s <= 2999: return 'Energy'
    if 1000 <= s <= 1499 or 2400 <= s <= 2699 or 2800 <= s <= 2829 or 2850 <= s <= 2899 or 3000 <= s <= 3399: return 'Materials'
    if 2830 <= s <= 2836 or 3840 <= s <= 3851 or 8000 <= s <= 8099: return 'Health Care'
    if 2000 <= s <= 2199 or 2840 <= s <= 2844 or 5400 <= s <= 5499: return 'Consumer Staples'
    if 3570 <= s <= 3579 or 3600 <= s <= 3699 and not 3630 <= s <= 3639 or 3800 <= s <= 3899 or 7370 <= s <= 7379: return 'Information Technology'
    if 2700 <= s <= 2799 or 4800 <= s <= 4899 or 7800 <= s <= 7999: return 'Communication Services'
    if 4900 <= s <= 4999: return 'Utilities'
    if 6500 <= s <= 6599 or s == 6798: return 'Real Estate'
    if 6000 <= s <= 6799: return 'Financials'
    if 2200 <= s <= 2399 or 3630 <= s <= 3639 or 3710 <= s <= 3716 or 3900 <= s <= 3999 or 5200 <= s <= 5999 or 7000 <= s <= 7299: return 'Consumer Discretionary'
    return 'Industrials'


def main():
    subs = {}
    for f in glob.glob(os.path.join(RAW, 'ext_fund', '*.sub.json')):
        j = json.load(open(f))
        subs[os.path.basename(f)[:-9]] = j
    ok = {s for s, j in subs.items() if j.get('sic') and str(j['sic']) not in FUND_EXCL}
    print(f'SEC submissions {len(subs)}  operating companies {len(ok)}')

    vh = pd.read_csv(VH, usecols=['date', 'act_symbol'])
    vh['date'] = pd.to_datetime(vh['date'])
    mem = json.load(open(os.path.join(ROOT, 'sp500_pit_membership.json')))
    sp_month = {(m, t) for m, v in mem.items() for t in v}

    frames, nopx = [], []
    for s in sorted(ok):
        p = os.path.join(RAW, 'ext_prices', f'{s}.parquet')
        if not os.path.exists(p):
            nopx.append(s)
            continue
        df = pd.read_parquet(p)
        df['date'] = pd.to_datetime(df['datetime']).dt.normalize()
        df = df[(df.close > 0) & (df.open > 0)].drop_duplicates('date').sort_values('date')
        f = df.adjclose / df.close
        dv = (df.close * df.volume).rolling(20, min_periods=10).median().shift(1)
        out = pd.DataFrame({'date': df.date, 'ticker': s, 'open': df.open * f, 'close': df.adjclose,
                            'vol': df.volume.astype(float), 'px_raw_prev': df.close.shift(1), 'dv20': dv})
        out['ret'] = out.close.pct_change()
        out['ret_on'] = out.open / out.close.shift(1) - 1
        out['ret_id'] = out.close / out.open - 1
        # DoltHub presence (point-in-time optionable)
        pr = vh.loc[vh.act_symbol == s, 'date'].sort_values().values
        k = np.searchsorted(pr, out.date.values, 'right') - 1
        last = np.where(k >= 0, pr[np.clip(k, 0, None)], np.datetime64('NaT'))
        fresh = (out.date.values - last) <= np.timedelta64(10, 'D')
        month = out.date.dt.strftime('%Y-%m')
        not_sp = ~np.array([(m, s) in sp_month or (m, s.replace('-', '.')) in sp_month for m in month])
        out['in_univ'] = fresh & not_sp & (out.px_raw_prev >= MIN_PX) & (out.dv20 >= MIN_DV)
        out['sector'] = sic_sector(subs[s]['sic'])
        out['src'] = 'yahoo'
        frames.append(out[(out.date >= pd.Timestamp(START) - pd.Timedelta(days=400)) & (out.date <= END)])
    panel = pd.concat(frames, ignore_index=True)
    # same corrupt-series guard as the S&P build (spike-and-reverse inside the universe window)
    r = panel.groupby('ticker').ret
    spike = (panel.ret.abs() > 0.35) & (np.sign(panel.ret) != np.sign(r.shift(-1))) & (r.shift(-1).abs() > 0.20) & panel.in_univ
    bad = spike.groupby(panel.ticker).sum()
    bad = set(bad[bad >= 2].index)
    panel = panel[~panel.ticker.isin(bad)]
    panel = panel.drop(columns=['px_raw_prev', 'dv20']).sort_values(['date', 'ticker']).reset_index(drop=True)
    panel.to_parquet(os.path.join(D, 'panel.parquet'), index=False)

    # fundamentals -> data_mid
    q = [pd.read_parquet(f) for f in glob.glob(os.path.join(RAW, 'ext_fund', '*.q.parquet'))]
    a = [pd.read_parquet(f) for f in glob.glob(os.path.join(RAW, 'ext_fund', '*.a.parquet'))]
    pd.concat(q, ignore_index=True).to_parquet(os.path.join(D, 'fund_q.parquet'), index=False)
    pd.concat(a, ignore_index=True).to_parquet(os.path.join(D, 'fund_a.parquet'), index=False)

    u = panel[panel.in_univ & (panel.date >= START)]
    per = u.groupby('date').size()
    pres = vh[vh.date >= START].groupby(vh.date.dt.to_period('M')).act_symbol.nunique()
    print(f'panel rows {len(panel):,}  tickers {panel.ticker.nunique()}  in-universe names/day: '
          f'median {per.median():.0f} (min {per.min()}, max {per.max()})')
    print(f'operating cos without Yahoo prices (likely delisted -> survivorship gap): {len(nopx)}  '
          f'corrupt series dropped: {len(bad)}')
    print('sector mix (in-universe rows):', (u.sector.value_counts(normalize=True) * 100).round(1).to_dict())


if __name__ == '__main__':
    main()
