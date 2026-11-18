"""Point-in-time ANNUAL fundamentals panel from SEC companyfacts (small-cap universe).  Output: $SP/sec_annual.parquet
One row per (ticker, fiscal period end) with the ORIGINAL 10-K values (earliest filing of that period = no restatement look-ahead) and the filing date `filed`.
A factor built from a row is tradable from the first trading day AFTER `filed`.  Flow items require a ~1-year duration; stocks are point values."""
import json, glob, os, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
TAGS = {  # name: (kind, [tags in priority order])
    'rev': ('f', ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax']),
    'cogs': ('f', ['CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold']),
    'gp': ('f', ['GrossProfit']), 'opinc': ('f', ['OperatingIncomeLoss']), 'ni': ('f', ['NetIncomeLoss', 'ProfitLoss']),
    'cfo': ('f', ['NetCashProvidedByUsedInOperatingActivities']), 'capex': ('f', ['PaymentsToAcquirePropertyPlantAndEquipment']),
    'da': ('f', ['DepreciationDepletionAndAmortization', 'DepreciationAndAmortization', 'DepreciationAmortizationAndAccretionNet']),
    'sga': ('f', ['SellingGeneralAndAdministrativeExpense']), 'rnd': ('f', ['ResearchAndDevelopmentExpense']), 'intexp': ('f', ['InterestExpense']),
    'sbc': ('f', ['ShareBasedCompensation']), 'shs_w': ('f', ['WeightedAverageNumberOfSharesOutstandingBasic']),
    'assets': ('s', ['Assets']), 'liab': ('s', ['Liabilities']), 'ca': ('s', ['AssetsCurrent']), 'cl': ('s', ['LiabilitiesCurrent']),
    'eq': ('s', ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest']),
    'ltd': ('s', ['LongTermDebtNoncurrent', 'LongTermDebt']), 'cash': ('s', ['CashAndCashEquivalentsAtCarryingValue']),
    'ar': ('s', ['AccountsReceivableNetCurrent']), 'inv': ('s', ['InventoryNet']), 'ppe': ('s', ['PropertyPlantAndEquipmentNet']),
    'shs': ('s', ['CommonStockSharesOutstanding']),
}
def one(f):
    tk = os.path.basename(f)[:-5]; j = json.load(open(f)); g = j['facts'].get('us-gaap', {}); out = {}
    for name, (kind, tags) in TAGS.items():
        for tg in tags:
            if tg not in g: continue
            u = g[tg]['units']; u = u.get('USD') or u.get('shares') or next(iter(u.values()))
            df = pd.DataFrame(u)
            if df.empty or 'form' not in df: continue
            df = df[df.form.isin(['10-K', '10-K/A', '10-KT'])].copy()
            df['end'] = pd.to_datetime(df.end); df['filed'] = pd.to_datetime(df.filed)
            if kind == 'f':
                df['start'] = pd.to_datetime(df.start); dur = (df.end - df.start).dt.days; df = df[(dur >= 340) & (dur <= 380)]
            if df.empty: continue
            df = df.sort_values('filed').drop_duplicates('end', keep='first')[['end', 'filed', 'val']]
            for e, fl, v in df.itertuples(index=False):
                r = out.setdefault(e, {}); r.setdefault(name, v); r['filed'] = max(r.get('filed', fl), fl)   # latest of the items' first filings = when ALL are public
            break
    return [dict(ticker=tk, end=e, **r) for e, r in out.items()]
if __name__ == '__main__':
    R = []
    for f in sorted(glob.glob('data/fund/sec_small/*.json')): R += one(f)
    P = pd.DataFrame(R); P['filed'] = pd.to_datetime(P.filed); P = P.dropna(subset=['filed']).sort_values(['ticker', 'end'])
    P.to_parquet(SP + 'sec_annual.parquet'); print(P.shape, P.ticker.nunique()); print(P.notna().mean().round(2).to_string())
