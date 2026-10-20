"""Compact extracts from local SEC caches (stdlib only).
xbrl_facts.csv : ticker,tag,start,end,val,filed,form   (selected share-count / payout tags)
8k_items.csv   : cik,ticker,filed,accept,items           (8-K / 8-K/A)"""
import json, os, csv, glob, sys
PART=int(sys.argv[1]); NP=int(sys.argv[2])
X = os.path.dirname(os.path.abspath(__file__))
SEC = os.path.join(X, '..', 'sec_edgar_cache')
TAGS = {'us-gaap': ['PaymentsForRepurchaseOfCommonStock', 'CommonStockSharesOutstanding',
                    'WeightedAverageNumberOfDilutedSharesOutstanding', 'CommonStockDividendsPerShareDeclared',
                    'CommonStockDividendsPerShareCashPaid', 'PaymentsOfDividendsCommonStock', 'PaymentsOfDividends',
                    'StockIssuedDuringPeriodValueShareBasedCompensation', 'ShareBasedCompensation'],
        'dei': ['EntityCommonStockSharesOutstanding']}
with open(os.path.join(X, 'ext', f'xbrl_facts_{PART}.csv'), 'w', newline='') as g:
    w = csv.writer(g); w.writerow(['ticker', 'tag', 'start', 'end', 'val', 'filed', 'form'])
    for fn in sorted(glob.glob(os.path.join(SEC, '*.json')))[PART::NP]:
        try: j = json.load(open(fn))
        except Exception as e: print('bad', fn, e); continue
        t = j.get('ticker') or os.path.basename(fn).split('_')[0]
        facts = (j.get('facts') or {}).get('facts', {})
        for ns, tags in TAGS.items():
            for tag in tags:
                for unit, arr in facts.get(ns, {}).get(tag, {}).get('units', {}).items():
                    for r in arr:
                        if r.get('form', '').startswith(('10-', '20-', '40-')) or ns == 'dei':
                            w.writerow([t, tag, r.get('start', ''), r['end'], r['val'], r['filed'], r.get('form', '')])
print('xbrl done')
if PART: sys.exit()
with open(os.path.join(X, 'ext', '8k_items.csv'), 'w', newline='') as g:
    w = csv.writer(g); w.writerow(['cik', 'ticker', 'filed', 'accept', 'items'])
    for fn in sorted(glob.glob(os.path.join(X, 'raw', 'sec_submissions', '*.json'))):
        j = json.load(open(fn)); r = j['filings']['recent']; tk = (j.get('tickers') or [''])[0]
        for i, f in enumerate(r['form']):
            if f in ('8-K', '8-K/A'):
                w.writerow([j['cik'], tk, r['filingDate'][i], r['acceptanceDateTime'][i], r['items'][i]])
print('8k done')
