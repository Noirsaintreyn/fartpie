"""
Earnings events + fundamental signals, point-in-time.

Announcement (reaction) day: the trading day with the highest volume ratio
(volume / median of prior 60 sessions) in (period_end+7d, filed]. The 10-Q/10-K
filed date is a hard upper bound, so no future info. If no day has ratio >= 1.5,
fall back to the first trading day >= filed. Validated against exact 8-K Item 2.02
timestamps for the names in stock-screener/data/raw/sec_submissions.

Per event (known at CLOSE of reaction day rd):
  sue      (EPS - EPS_4q_ago) / sd(prior 8 such diffs, min 4)      seasonal random walk
  sue_rev  same for revenue
  ear      residual return rd-1..rd (beta+sector adjusted)
Annual (known the session after the 10-K filed date, held 400 days):
  accruals = (NI - CFO) / avg(assets)        asset_growth = assets / assets_prev - 1

Outputs: xs/data/events.parquet, xs/data/annual_sig.parquet
"""
import json
import os
import sys

import numpy as np
import pandas as pd

X = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, X)
from panel_build import ALIAS  # noqa: E402

from cfg import D, U  # noqa: E402
SUBS = os.path.join(os.path.dirname(os.path.dirname(X)), 'stock-screener', 'data', 'raw', 'sec_submissions')


def ticker_map(file_tickers, panel_tickers):
    ft = set(file_tickers)
    m = {}
    for t in panel_tickers:
        for cand in (ALIAS.get(t, t), t):
            if cand in ft:
                m[t] = cand
                break
    return m


RAW_SUBS = os.path.join(X, 'raw', 'sec_submissions')


def exact_release_days(dates):
    """cik -> sorted DatetimeIndex of reaction days for 8-K Item 2.02 (earnings releases).
    Accepted before 16:00 ET -> reaction that session; at/after 16:00 -> next session."""
    out = {}
    if U != 'sp':
        ext = os.path.join(X, 'raw', 'ext_fund')
        for fn in (f for f in os.listdir(ext) if f.endswith('.sub.json')) if os.path.isdir(ext) else []:
            j = json.load(open(os.path.join(ext, fn)))
            days = []
            for a in j.get('releases', []):
                ts = pd.Timestamp(a)
                ts = (ts.tz_localize('UTC') if ts.tzinfo is None else ts).tz_convert('America/New_York')
                day = ts.normalize().tz_localize(None)
                k = dates.searchsorted(day)
                if k < len(dates) and dates[k] == day and ts.hour >= 16:
                    k += 1
                if k < len(dates):
                    days.append(dates[k])
            out[str(j['cik']).zfill(10)] = pd.DatetimeIndex(sorted(set(days)))
        return out
    if not os.path.isdir(RAW_SUBS):
        return out
    for fn in os.listdir(RAW_SUBS):
        try:
            d = json.load(open(os.path.join(RAW_SUBS, fn)))
        except Exception:
            continue
        blocks = [d['filings']['recent']] + d.get('older', [])
        days = []
        for r in blocks:
            for i, fm in enumerate(r.get('form', [])):
                if fm not in ('8-K', '8-K/A') or '2.02' not in (r['items'][i] or ''):
                    continue
                ts = pd.Timestamp(r['acceptanceDateTime'][i])
                ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts
                ts = ts.tz_convert('America/New_York')
                day = ts.normalize().tz_localize(None)
                k = dates.searchsorted(day)
                if k < len(dates) and dates[k] == day and ts.hour >= 16:
                    k += 1
                if k < len(dates):
                    days.append(dates[k])
        out[fn[:-5].zfill(10)] = pd.DatetimeIndex(sorted(set(days)))
    return out


def std_surprise(s, n=8, min_n=4):
    d = s - s.shift(4)
    sd = d.shift(1).rolling(n, min_periods=min_n).std()
    return d / sd.replace(0, np.nan)


def main():
    p = pd.read_parquet(os.path.join(D, 'panel.parquet'), columns=['date', 'ticker', 'vol'])
    V = p.pivot(index='date', columns='ticker', values='vol').sort_index()
    dates = V.index
    base = V.rolling(60, min_periods=30).median().shift(1)
    ratio = V / base
    res = pd.read_parquet(os.path.join(D, 'resid.parquet'))['res']

    fq = pd.read_parquet(os.path.join(D, 'fund_q.parquet'))
    fa = pd.read_parquet(os.path.join(D, 'fund_a.parquet'))
    tmap = ticker_map(fq.file_ticker.unique(), V.columns)
    inv = {}
    for t, f in tmap.items():
        inv.setdefault(f, []).append(t)

    exact = exact_release_days(dates)
    print(f'exact 8-K release dates for {len(exact)} companies')
    ev = []
    for f, g in fq.groupby('file_ticker'):
        if f not in inv:
            continue
        g = g.sort_values('end').drop_duplicates('end').reset_index(drop=True)
        g = g[g['filed'].notna()]
        # quarters must be ~consecutive for the lag-4 comparison
        g['sue'] = std_surprise(g['eps'])
        g['sue_rev'] = std_surprise(g['rev'])
        for t in inv[f]:
            r = ratio[t]
            for _, e in g.iterrows():
                if e['filed'] < dates[0] or e['filed'] > dates[-1]:
                    continue
                win = r[(r.index > e['end'] + pd.Timedelta(days=7)) & (r.index <= e['filed'])]
                fb = dates[dates.searchsorted(e['filed'])] if dates.searchsorted(e['filed']) < len(dates) else None
                if len(win.dropna()) and win.max() >= 1.5:
                    rd, how = win.idxmax(), 'volume'
                else:
                    rd, how = fb, 'filed'
                if rd is None:
                    continue
                i = dates.get_loc(rd)
                ear = res[t].iloc[max(0, i - 1):i + 1].sum(min_count=1)
                k = dates.searchsorted(e['filed'])
                avail = dates[k] if k < len(dates) else None   # strictly PIT: 10-Q/10-K on file
                # exact: first 8-K 2.02 reaction day after period end, on/before the filing
                rd_x = ear_x = None
                xd = exact.get(str(e['cik']).zfill(10))
                if xd is not None and len(xd):
                    c = xd[(xd > e['end']) & (xd <= (avail if avail is not None else e['filed']))]
                    if len(c):
                        rd_x = c[0]
                        j = dates.get_loc(rd_x)
                        ear_x = res[t].iloc[max(0, j - 1):j + 1].sum(min_count=1)
                ev.append({'ticker': t, 'end': e['end'], 'filed': e['filed'], 'rd': rd, 'how': how,
                           'avail': avail, 'rd_x': rd_x, 'ear_x': ear_x,
                           'vol_ratio': win.max() if len(win.dropna()) else np.nan,
                           'eps': e['eps'], 'sue': e['sue'], 'sue_rev': e['sue_rev'], 'ear': ear})
    ev = pd.DataFrame(ev).sort_values(['ticker', 'rd'])
    ev.to_parquet(os.path.join(D, 'events.parquet'), index=False)
    print(f"events {len(ev):,}  tickers {ev.ticker.nunique()}  dated by volume {(ev.how == 'volume').mean():.1%}  "
          f"sue coverage {ev.sue.notna().mean():.1%}  ear coverage {ev.ear.notna().mean():.1%}")
    print(f"exact-dated events: {ev.rd_x.notna().mean():.1%}   "
          f"exact vs volume-inferred day match: {(ev.rd_x == ev.rd)[ev.rd_x.notna()].mean():.1%}")
    print(f"lag filed - rd (days): median {(ev.filed - ev.rd).dt.days.median()}  "
          f"p90 {(ev.filed - ev.rd).dt.days.quantile(.9)}")

    # ---- validation vs exact 8-K Item 2.02 dates ----
    if os.path.isdir(SUBS):
        hits, n = 0, 0
        errs = []
        for fn in os.listdir(SUBS):
            t = fn[:-5]
            if t not in ev.ticker.values:
                continue
            r = json.load(open(os.path.join(SUBS, fn)))['data']['filings']['recent']
            true = []
            for i, fm in enumerate(r['form']):
                if fm == '8-K' and '2.02' in r['items'][i]:
                    ts = pd.Timestamp(r['acceptanceDateTime'][i]).tz_convert('America/New_York')
                    day = ts.normalize().tz_localize(None)
                    k = dates.searchsorted(day)
                    if k >= len(dates):
                        continue
                    if dates[k] == day and ts.hour >= 16:
                        k += 1
                    if k < len(dates):
                        true.append(dates[k])
            true = pd.DatetimeIndex(sorted(set(true)))
            for rd in ev.loc[ev.ticker == t, 'rd']:
                if len(true) == 0 or rd < true.min():
                    continue
                j = true.searchsorted(rd)
                cands = [true[x] for x in (j - 1, j) if 0 <= x < len(true)]
                gap = min(abs(dates.get_loc(rd) - dates.get_loc(c)) for c in cands if c in dates)
                n += 1
                hits += gap == 0
                errs.append(gap)
        errs = np.array(errs)
        print(f"8-K validation: {n} events, exact {hits / max(1, n):.1%}, within 1 day {(errs <= 1).mean():.1%}, "
              f"within 3 {(errs <= 3).mean():.1%}")

    # ---- annual signals ----
    rows = []
    for f, g in fa.groupby('file_ticker'):
        for t in inv.get(f, []):
            for _, a in g.iterrows():
                k = dates.searchsorted(a['filed'] + pd.Timedelta(days=1))
                if k >= len(dates):
                    continue
                pair = [v for v in (a['assets'], a['assets_prev']) if pd.notna(v)]
                avg_at = float(np.mean(pair)) if pair else np.nan
                rows.append({'ticker': t, 'avail': dates[k],
                             'accruals': (a['ni'] - a['cfo']) / avg_at if avg_at > 0 else np.nan,
                             'asset_growth': a['assets'] / a['assets_prev'] - 1
                             if a['assets_prev'] and a['assets_prev'] > 0 else np.nan})
    an = pd.DataFrame(rows).sort_values(['ticker', 'avail'])
    an.to_parquet(os.path.join(D, 'annual_sig.parquet'), index=False)
    print(f"annual signal rows {len(an):,}  accruals {an.accruals.notna().mean():.1%}  "
          f"asset_growth {an.asset_growth.notna().mean():.1%}")


if __name__ == '__main__':
    main()
