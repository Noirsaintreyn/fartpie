"""
One-time batch build of real GICS-level sub-industry classification
(e.g. "Semiconductors", not just the 11 broad "Information Technology"
sectors in sp500_sectors.json), via yfinance's .info - free, no key.

Resumable: writes sp500_subindustries.json incrementally as it goes,
skips tickers already resolved on a rerun, so an interrupted run loses
no progress. Paced with a small delay per request since .info hits
Yahoo's quote-summary endpoint per ticker (no batching available for
this field, unlike price history).

Usage: .venv311/bin/python build_sp500_subindustries.py
"""
import json
import time

import yfinance as yf

OUT_PATH = 'sp500_subindustries.json'
REQUEST_DELAY = 0.15


def main():
    with open('sp500_pit_all_tickers.json') as f:
        tickers = json.load(f)
    tickers_yf = [t.replace('.', '-') for t in tickers]

    try:
        with open(OUT_PATH) as f:
            result = json.load(f)
    except FileNotFoundError:
        result = {}

    todo = [t for t in tickers_yf if t not in result]
    print(f"{len(result)} already resolved, {len(todo)} to fetch\n")

    failed = []
    for i, t in enumerate(todo):
        try:
            info = yf.Ticker(t).info
            industry = info.get('industry') or info.get('industryDisp')
            sector = info.get('sector') or info.get('sectorDisp')
            if industry:
                result[t] = {'industry': industry, 'sector': sector}
            else:
                failed.append(t)
        except Exception:
            failed.append(t)
        time.sleep(REQUEST_DELAY)

        if (i + 1) % 25 == 0:
            with open(OUT_PATH, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  {i+1}/{len(todo)} processed, {len(result)} resolved so far...", flush=True)

    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*80}\nDone: {len(result)}/{len(tickers_yf)} tickers resolved")
    if failed:
        print(f"Failed/no industry ({len(failed)}): {failed[:30]}{'...' if len(failed) > 30 else ''}")

    from collections import Counter
    industries = Counter(v['industry'] for v in result.values())
    print(f"\n{len(industries)} distinct sub-industries. Largest groups:")
    for name, count in industries.most_common(15):
        print(f"  {count:>3}  {name}")


if __name__ == '__main__':
    main()
