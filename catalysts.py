"""Nov 25 2026: catalyst block for flip alerts (Rey: 'add catalysts and cut if it's bad'). catalyst_block(tk, asof=None) -> dict(text, bad, good, tags).
Sources (free): SEC submissions API (8-K item codes, S-3/424B offering filings, Form 4 count), Yahoo (earnings dates + 2-day reaction, last headlines).
BAD (report 481, flagged for a cut/warning): 8-K 3.01 listing deficiency, 1.02 agreement terminated, 1.03 bankruptcy, 2.05/2.06 restructuring or impairment, 4.02 non-reliance, or an earnings reaction <= -5% in the last 8 days. NOT bad: 3.02 equity sale (those flips did BETTER in report 481: +15%/+23% mean, n 17/12).
GOOD tags: earnings reaction >= +5%, 8-K 1.01/2.01 deal, new 13D."""
import json, os, re, time, requests, pandas as pd
H = {'User-Agent': 'research rey@example.com'}; _HERE = os.path.dirname(os.path.abspath(__file__)); _CIK = None
NAMES = {'1.01': 'agreement', '1.02': 'agreement terminated', '1.03': 'bankruptcy', '2.01': 'acquisition/disposal', '2.02': 'earnings', '2.03': 'debt', '2.05': 'restructuring', '2.06': 'impairment', '3.01': 'listing notice', '3.02': 'equity sale', '4.02': 'non-reliance', '5.02': 'mgmt change', '5.07': 'vote', '7.01': 'Reg FD', '8.01': 'other event'}
BADIT = {'3.01', '1.02', '1.03', '2.05', '2.06', '4.02'}
def _cik(tk):
    global _CIK
    if _CIK is None:
        p = os.path.join(_HERE, 'data/fund/company_tickers.json')
        try: j = json.load(open(p)) if os.path.exists(p) else requests.get('https://www.sec.gov/files/company_tickers.json', headers=H, timeout=25).json()
        except Exception: j = {}
        _CIK = {v['ticker']: v['cik_str'] for v in j.values()}
    return _CIK.get(tk)
def catalyst_block(tk, asof=None, days=10):
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now().normalize(); lines, tags, bad, good = [], [], [], []
    try:
        c = _cik(tk); s = requests.get(f'https://data.sec.gov/submissions/CIK{c:010d}.json', headers=H, timeout=25).json()['filings']['recent']
        df = pd.DataFrame({k: s[k] for k in ('filingDate', 'form', 'items')}); df['filingDate'] = pd.to_datetime(df.filingDate); w = df[(df.filingDate >= asof - pd.Timedelta(days=days)) & (df.filingDate <= asof + pd.Timedelta(days=1))]
        for r in w[w.form == '8-K'].itertuples():
            its = [i for i in str(r.items).split(',') if i and i != '9.01']; txt = ', '.join(NAMES.get(i, i) for i in its)
            if txt: lines.append(f'8-K {r.filingDate:%m-%d}: {txt}')
            if BADIT & set(its): bad.append('8-K ' + ', '.join(NAMES[i] for i in its if i in BADIT))
            if {'1.01', '2.01'} & set(its): good.append('deal 8-K')
        off = w[w.form.isin(['S-3', 'S-1', '424B5', '424B3'])]
        if len(off): lines.append('offering filing: ' + ', '.join(sorted(set(off.form))))
        f4 = w[w.form == '4']
        if len(f4): lines.append(f'{len(f4)} Form 4 filing(s) (insider buy vs sell: check the filing)')
        if (w.form.isin(['SC 13D', 'SC 13D/A', 'SCHEDULE 13D', 'SCHEDULE 13D/A'])).any(): good.append('13D')
    except Exception as e: lines.append(f'SEC lookup failed ({str(e)[:40]})')
    try:
        import yfinance as yf
        T = yf.Ticker(tk); ed = T.get_earnings_dates(limit=6)
        if ed is not None and len(ed):
            ed.index = ed.index.tz_localize(None).normalize(); near = ed[(ed.index >= asof - pd.Timedelta(days=days)) & (ed.index <= asof + pd.Timedelta(days=14))]
            for d_ in near.index:
                h = T.history(start=d_ - pd.Timedelta(days=6), end=d_ + pd.Timedelta(days=6))
                if len(h) and d_ <= asof:
                    h.index = h.index.tz_localize(None).normalize(); a = h[h.index < d_]; b = h[h.index > d_]
                    if len(a) and len(b): rx = b.Close.iloc[0] / a.Close.iloc[-1] - 1; lines.append(f'earnings {d_:%m-%d}: 2-day reaction {rx:+.0%}'); (bad if rx <= -.05 else good if rx >= .05 else []).append(f'earnings {rx:+.0%}')
                else: lines.append(f'earnings {d_:%m-%d} (upcoming)')
        for x in (T.news or [])[:3]:
            cc = x.get('content', x); lines.append('news: ' + str(cc.get('title') or x.get('title'))[:100])
    except Exception: pass
    return dict(text='\n'.join(lines) or 'no catalysts found', bad=bad, good=good, flag='BAD CATALYST: ' + '; '.join(bad) if bad else ('catalyst: ' + '; '.join(good) if good else 'no clear catalyst'))

def _script(t):
    """bold script ('cursive') letters for the sign-off; Discord has no cursive font, these are Unicode letters"""
    o = ''
    for ch in t:
        if ch.isupper(): o += chr(0x1D4D0 + ord(ch) - 65)
        elif ch.islower(): o += chr(0x1D4EA + ord(ch) - 97)
        else: o += ch
    return o
SIGN = '*~' + _script('Valiant') + '*'
def insiders(tk, days=30, asof=None, maxf=8):
    """open-market insider trades from the Form 4 XML (code P = buy, S = sale) over the last `days`; -> dict(buys, sells, buy_usd, sell_usd, who)"""
    import xml.etree.ElementTree as ET
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now().normalize(); out = dict(buys=0, sells=0, buy_usd=0., sell_usd=0., who=[])
    c = _cik(tk)
    if not c: return out
    s = requests.get(f'https://data.sec.gov/submissions/CIK{c:010d}.json', headers=H, timeout=25).json()['filings']['recent']
    df = pd.DataFrame({k: s[k] for k in ('filingDate', 'form', 'accessionNumber', 'primaryDocument')}); df['filingDate'] = pd.to_datetime(df.filingDate)
    df = df[(df.form == '4') & (df.filingDate >= asof - pd.Timedelta(days=days)) & (df.filingDate <= asof + pd.Timedelta(days=1))].head(maxf)
    for r in df.itertuples():
        try:
            x = requests.get(f"https://www.sec.gov/Archives/edgar/data/{c}/{r.accessionNumber.replace('-', '')}/{r.primaryDocument.split('/')[-1]}", headers=H, timeout=20).text
            R = ET.fromstring(x); who = (R.findtext('.//reportingOwner/reportingOwnerRelationship/officerTitle') or ('Director' if R.findtext('.//reportingOwner/reportingOwnerRelationship/isDirector') in ('1', 'true') else 'Owner'))
            for t in R.findall('.//nonDerivativeTransaction'):
                code = t.findtext('transactionCoding/transactionCode'); sh = float(t.findtext('transactionAmounts/transactionShares/value') or 0); px = float(t.findtext('transactionAmounts/transactionPricePerShare/value') or 0)
                if code == 'P': out['buys'] += 1; out['buy_usd'] += sh * px; out['who'].append(f'{who} bought')
                elif code == 'S': out['sells'] += 1; out['sell_usd'] += sh * px
        except Exception: continue
    return out
def _usd(v): return f'${v/1e6:.1f}M' if v >= 1e6 else f'${v/1e3:.0f}k'
def report(tk, asof=None, extra=None):
    """quick catalyst / news / insider report for a flip alert, ends with the Valiant sign-off. Never raises: a failed source just drops its line."""
    cb = catalyst_block(tk, asof); L = [f'**Catalyst report \u00b7 {tk}**']
    for ln in cb['text'].split('\n'):
        if ln and not ln.startswith('news:') and 'Form 4' not in ln and ln != 'no catalysts found': L.append('\u2022 ' + ln)
    try:
        i = insiders(tk, asof=asof)
        if i['buys'] or i['sells']: L.append('\u2022 insiders 30d (open market): ' + ' \u00b7 '.join(x for x in (f"{i['buys']} buy(s) {_usd(i['buy_usd'])}" if i['buys'] else '', f"{i['sells']} sale(s) {_usd(i['sell_usd'])}" if i['sells'] else '') if x) + (f" ({', '.join(sorted(set(i['who'])))})" if i['who'] else ''))
        else: L.append('\u2022 insiders 30d: no open-market buys or sales')
        if i['buys'] and not i['sells']: cb['good'].append('insider buying')
    except Exception: pass
    nw = [ln[6:] for ln in cb['text'].split('\n') if ln.startswith('news:')]
    if nw: L += ['\u2022 news: ' + n for n in nw[:3]]
    for e in (extra or []): L.append('\u2022 ' + e)
    L.append(('\u26a0\ufe0f ' if cb['bad'] else '\u2022 ') + cb['flag'] + ('' if not cb['bad'] else ' (warning: size down or skip)')); L.append(SIGN)
    return '\n'.join(L)
