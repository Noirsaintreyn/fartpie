"""
Insider-trading signals from SEC Insider Transactions Data Sets (Form 4).

Only open-market trades: code P (purchase, acquired) and S (sale, disposed).
Grants, option exercises, gifts, tax withholding etc. are ignored.
A trade is usable from the first session AFTER its Form 4 FILING_DATE.

Per ticker, per trading day t (all trailing windows end at t):
  buyers63   distinct insiders with an open-market purchase, last 63 sessions
  offbuy63   same, officers only (CEO/CFO/other officers)
  buy_usd63  $ purchased, last 63 sessions
  npr126     (buy tx - sell tx) / (buy tx + sell tx), last 126 sessions (NaN if none)
  sell_10b5  share of sales flagged 10b5-1 (2023+ only), informational

Output: xs/data/insider_daily.parquet (long: date, ticker, features)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

X = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, X)
from cfg import D  # noqa: E402
RAW = os.path.join(X, 'raw', 'insider_345')


def load():
    subs, trans, owners = [], [], []
    for q in sorted(glob.glob(os.path.join(RAW, '*'))):
        rd = lambda f, cols: pd.read_csv(os.path.join(q, f), sep='\t', usecols=cols, dtype=str,
                                         quoting=3, on_bad_lines='skip')
        subs.append(rd('SUBMISSION.tsv', ['ACCESSION_NUMBER', 'FILING_DATE', 'DOCUMENT_TYPE',
                                          'ISSUERCIK']))
        trans.append(rd('NONDERIV_TRANS.tsv', ['ACCESSION_NUMBER', 'TRANS_CODE', 'TRANS_SHARES',
                                              'TRANS_PRICEPERSHARE', 'TRANS_ACQUIRED_DISP_CD']))
        owners.append(rd('REPORTINGOWNER.tsv', ['ACCESSION_NUMBER', 'RPTOWNERCIK',
                                               'RPTOWNER_RELATIONSHIP', 'RPTOWNER_TITLE']))
    s = pd.concat(subs).drop_duplicates('ACCESSION_NUMBER')
    t = pd.concat(trans)
    o = pd.concat(owners).drop_duplicates('ACCESSION_NUMBER')   # first reporting owner
    s = s[s.DOCUMENT_TYPE.isin(['4', '4/A'])]
    t = t[((t.TRANS_CODE == 'P') & (t.TRANS_ACQUIRED_DISP_CD == 'A')) |
          ((t.TRANS_CODE == 'S') & (t.TRANS_ACQUIRED_DISP_CD == 'D'))]
    df = t.merge(s, on='ACCESSION_NUMBER').merge(o, on='ACCESSION_NUMBER', how='left')
    df['filed'] = pd.to_datetime(df.FILING_DATE, format='%d-%b-%Y', errors='coerce')
    df['usd'] = pd.to_numeric(df.TRANS_SHARES, errors='coerce') * pd.to_numeric(
        df.TRANS_PRICEPERSHARE, errors='coerce')
    rel = df.RPTOWNER_RELATIONSHIP.fillna('')
    df['officer'] = rel.str.contains('Officer')
    df['cik'] = df.ISSUERCIK.str.zfill(10)
    return df.dropna(subset=['filed'])


def main():
    df = load()
    fq = pd.read_parquet(os.path.join(D, 'fund_q.parquet'), columns=['file_ticker', 'cik']).drop_duplicates()
    import sys
    sys.path.insert(0, X)
    from events import ticker_map
    dates = pd.read_parquet(os.path.join(D, 'resid.parquet'))['res'].index
    tmap = ticker_map(fq.file_ticker.unique(), pd.read_parquet(
        os.path.join(D, 'panel.parquet'), columns=['ticker']).ticker.unique())
    cik_of = dict(zip(fq.file_ticker, fq.cik.astype(str).str.zfill(10)))
    df = df[df.filed >= dates[0] - pd.Timedelta(days=200)]
    k = np.searchsorted(dates.values, (df.filed + pd.Timedelta(days=1)).values)
    df = df[k < len(dates)]
    df['ai'] = k[k < len(dates)]                      # index of first usable session
    print(f'open-market Form 4 trades: {len(df):,}  buys {(df.TRANS_CODE == "P").sum():,}')

    rows = []
    n = len(dates)
    for t, f in tmap.items():
        g = df[df.cik == cik_of.get(f)]
        if g.empty:
            continue
        b, s_ = g[g.TRANS_CODE == 'P'], g[g.TRANS_CODE == 'S']
        nb = np.bincount(b.ai, minlength=n).astype(float)
        ns = np.bincount(s_.ai, minlength=n).astype(float)
        usd = np.bincount(b.ai, weights=b.usd.fillna(0), minlength=n)
        nb126 = pd.Series(nb).rolling(126, min_periods=1).sum().values
        ns126 = pd.Series(ns).rolling(126, min_periods=1).sum().values
        tot = nb126 + ns126
        npr = np.where(tot > 0, (nb126 - ns126) / np.where(tot > 0, tot, 1), np.nan)
        usd63 = pd.Series(usd).rolling(63, min_periods=1).sum().values
        # distinct buyers in trailing 63 sessions
        buyers = np.zeros(n)
        offb = np.zeros(n)
        if len(b):
            ev = b[['ai', 'RPTOWNERCIK', 'officer']].drop_duplicates(['ai', 'RPTOWNERCIK'])
            for i in np.unique(np.concatenate([ev.ai.values + j for j in range(63)])):
                if i >= n:
                    continue
                w = ev[(ev.ai <= i) & (ev.ai > i - 63)]
                buyers[i] = w.RPTOWNERCIK.nunique()
                offb[i] = w[w.officer].RPTOWNERCIK.nunique()
        rows.append(pd.DataFrame({'date': dates, 'ticker': t, 'buyers63': buyers, 'offbuy63': offb,
                                  'buy_usd63': usd63, 'npr126': npr}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(os.path.join(D, 'insider_daily.parquet'), index=False)
    last = out[out.date >= '2019-01-01']
    print(f'tickers {out.ticker.nunique()}  share of ticker-days with >=1 buyer {(last.buyers63 > 0).mean():.1%}  '
          f'>=2 buyers {(last.buyers63 >= 2).mean():.2%}  npr coverage {last.npr126.notna().mean():.1%}')


if __name__ == '__main__':
    main()
