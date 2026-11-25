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
_PAIRS = None; _GR = {}
def _pairs():
    global _PAIRS
    if _PAIRS is None:
        p = os.path.join(_HERE, 'insider_P_pairs.csv'); _PAIRS = {}
        if os.path.exists(p):
            for r in pd.read_csv(p, dtype=str).itertuples(): _PAIRS.setdefault(r.tk, {}).setdefault(r.cik, []).append(pd.Timestamp(r.d))
    return _PAIRS
def buys(tk, start, end, maxf=10):
    """Form 4 open-market trades filed in [start, end]: list of dict(side 'P'/'S', usd, cik, name, who (role/title), date). Parsed from the Form 4 XML."""
    import xml.etree.ElementTree as ET
    c = _cik(tk); out = []
    if not c: return out
    s = requests.get(f'https://data.sec.gov/submissions/CIK{c:010d}.json', headers=H, timeout=25).json()['filings']['recent']
    df = pd.DataFrame({k: s[k] for k in ('filingDate', 'form', 'accessionNumber', 'primaryDocument')}); df['filingDate'] = pd.to_datetime(df.filingDate)
    df = df[(df.form == '4') & (df.filingDate >= pd.Timestamp(start)) & (df.filingDate <= pd.Timestamp(end))].head(maxf)
    for r in df.itertuples():
        try:
            x = requests.get(f"https://www.sec.gov/Archives/edgar/data/{c}/{r.accessionNumber.replace('-', '')}/{r.primaryDocument.split('/')[-1]}", headers=H, timeout=20).text; R = ET.fromstring(x)
            who = (R.findtext('.//reportingOwner/reportingOwnerRelationship/officerTitle') or ('Director' if R.findtext('.//reportingOwner/reportingOwnerRelationship/isDirector') in ('1', 'true') else ('10% owner' if R.findtext('.//reportingOwner/reportingOwnerRelationship/isTenPercentOwner') in ('1', 'true') else 'Owner')))
            cik = (R.findtext('.//reportingOwner/reportingOwnerId/rptOwnerCik') or '').lstrip('0'); nm = R.findtext('.//reportingOwner/reportingOwnerId/rptOwnerName') or ''
            for t in R.findall('.//nonDerivativeTransaction'):
                code = t.findtext('transactionCoding/transactionCode')
                if code in ('P', 'S'): out.append(dict(side=code, usd=float(t.findtext('transactionAmounts/transactionShares/value') or 0) * float(t.findtext('transactionAmounts/transactionPricePerShare/value') or 0), cik=cik, name=nm, who=who, date=r.filingDate))
        except Exception: continue
    return out
def insiders(tk, days=30, asof=None, maxf=8):
    """open-market insider trades over the last `days` (code P = buy, S = sale) -> dict(buys, sells, buy_usd, sell_usd, who)"""
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now().normalize(); out = dict(buys=0, sells=0, buy_usd=0., sell_usd=0., who=[])
    for b in buys(tk, asof - pd.Timedelta(days=days), asof + pd.Timedelta(days=1), maxf):
        if b['side'] == 'P': out['buys'] += 1; out['buy_usd'] += b['usd']; out['who'].append(f"{b['who']} bought")
        else: out['sells'] += 1; out['sell_usd'] += b['usd']
    return out
def grade(tk, asof=None):
    """Nov 25 2026 (Rey: 'for unusual buyers maybe grade A'): GRADE A = an insider open-market buy in the last 30 days by someone with NO purchase at this issuer in the prior 24 months (report 489 FIRST flag: lead-level, not a filter). History = insider_P_pairs.csv (SEC insider datasets, through Mar 2026) plus a live Form 4 scan of the gap since. -> dict(a: bool, buyers: [dict(who, name, usd)], any_buy: bool)"""
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now().normalize(); key = (tk, str(asof.date()))
    if key in _GR: return _GR[key]
    res = dict(a=False, buyers=[], any_buy=False)
    try:
        w0 = asof - pd.Timedelta(days=30); cur = [b for b in buys(tk, w0, asof + pd.Timedelta(days=1), 10) if b['side'] == 'P']
        if cur:
            res['any_buy'] = True; hist = _pairs().get(tk, {}); gap0 = pd.Timestamp('2026-04-01'); gap = [b for b in buys(tk, gap0, w0, 40) if b['side'] == 'P'] if w0 > gap0 else []
            for b in cur:
                prior = [d for d in hist.get(b['cik'], []) if asof - pd.Timedelta(days=730) <= d < w0] + [g['date'] for g in gap if g['cik'] == b['cik'] and g['date'] >= asof - pd.Timedelta(days=730)]
                if not prior: res['buyers'].append(dict(who=b['who'], name=b['name'], usd=b['usd']))
            res['a'] = bool(res['buyers'])
    except Exception: pass
    _GR[key] = res; return res
def _usd(v): return f'${v/1e6:.1f}M' if v >= 1e6 else f'${v/1e3:.0f}k'
def report(tk, asof=None, extra=None):
    """quick catalyst / news / insider report for a flip alert, ends with the Valiant sign-off. Never raises: a failed source just drops its line."""
    cb = catalyst_block(tk, asof); L = [f'**Catalyst report \u00b7 {tk}**']
    for ln in cb['text'].split('\n'):
        if ln and not ln.startswith('news:') and 'Form 4' not in ln and ln != 'no catalysts found': L.append('\u2022 ' + ln)
    try:
        gr = grade(tk, asof)
        if gr['a']: L.insert(1, '\u2605 GRADE A: unusual insider buyer - ' + '; '.join(f"{b['who']} {b['name'].title()} bought {_usd(b['usd'])} (first purchase here in 24+ months)" for b in gr['buyers'][:2]))
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

_NM = {}
def name(tk):
    """company name from the SEC submissions file"""
    if tk in _NM: return _NM[tk]
    try: c = _cik(tk); _NM[tk] = requests.get(f'https://data.sec.gov/submissions/CIK{c:010d}.json', headers=H, timeout=25).json().get('name', '')
    except Exception: _NM[tk] = ''
    return _NM[tk]

# ---------------------------------------------------------------- setup scorecard (Nov 25 2026): every factor gets a tone and the stat of how it did before
_FS = None; _SUB = {}
def _fstats():
    global _FS
    if _FS is None:
        p = os.path.join(_HERE, 'daily_factor_stats.json'); _FS = json.load(open(p)) if os.path.exists(p) else {}
    return _FS
def _subs(tk):
    if tk in _SUB: return _SUB[tk]
    c = _cik(tk); df = pd.DataFrame(columns=['filingDate', 'form', 'items', 'accessionNumber', 'primaryDocument'])
    if c:
        s = requests.get(f'https://data.sec.gov/submissions/CIK{c:010d}.json', headers=H, timeout=25).json()['filings']['recent']; df = pd.DataFrame({k: s[k] for k in ('filingDate', 'form', 'items', 'accessionNumber', 'primaryDocument')}); df['filingDate'] = pd.to_datetime(df.filingDate)
    _SUB[tk] = df; return df
def earn_timing(tk, asof, win=30):
    """'early' / 'late' / 'normal' / None (unknown): the latest 8-K item 2.02 in the last `win` days vs the same company's 2.02 about a year earlier (aligned +365d, +-40d)"""
    try:
        df = _subs(tk); e = df[(df.form == '8-K') & df['items'].astype(str).str.contains('2.02')].filingDate.sort_values(); asof = pd.Timestamp(asof); cur = e[(e <= asof) & (e > asof - pd.Timedelta(days=win))]
        if not len(cur): return 'normal'
        cur = cur.iloc[-1]; tgt = cur - pd.Timedelta(days=365); prv = e[((e - tgt).abs() <= pd.Timedelta(days=40)) & (e < cur - pd.Timedelta(days=200))]
        if not len(prv): return None
        sh = (cur - prv.iloc[(prv - tgt).abs().argmin()]).days - 365; return 'early' if sh <= -5 else ('late' if sh >= 5 else 'normal')
    except Exception: return None
def negbad(tk, asof):
    try:
        df = _subs(tk); asof = pd.Timestamp(asof); w = df[(df.form == '8-K') & (df.filingDate <= asof + pd.Timedelta(days=1)) & (df.filingDate > asof - pd.Timedelta(days=7))]; hit = [i for it in w['items'].astype(str) for i in it.split(',') if i in BADIT]
        return NAMES.get(hit[0], hit[0]) if hit else None
    except Exception: return None
def filing_change(tk, asof):
    """cosine similarity of the latest 10-K/10-Q (last 90d) vs the same form a year earlier (report 491); None if unavailable"""
    import collections
    try:
        df = _subs(tk); df = df[df.form.isin(['10-K', '10-Q'])]; asof = pd.Timestamp(asof); cur = df[(df.filingDate <= asof) & (df.filingDate > asof - pd.Timedelta(days=90))]
        if not len(cur): return None
        cur = cur.sort_values('filingDate').iloc[-1]; prv = df[(df.form == cur.form) & ((df.filingDate - (cur.filingDate - pd.Timedelta(days=365))).abs() <= pd.Timedelta(days=45))]
        if not len(prv): return None
        prv = prv.iloc[(prv.filingDate - (cur.filingDate - pd.Timedelta(days=365))).abs().argmin()]; c = _cik(tk)
        def wc(r): t = requests.get(f"https://www.sec.gov/Archives/edgar/data/{c}/{r.accessionNumber.replace('-', '')}/{r.primaryDocument}", headers=H, timeout=60).text.lower(); return collections.Counter(re.findall(r'[a-z]{3,}', re.sub(r'<[^>]+>', ' ', t)))
        a, b = wc(cur), wc(prv); k = set(a) & set(b); return sum(a[w] * b[w] for w in k) / ((sum(v * v for v in a.values()) ** .5) * (sum(v * v for v in b.values()) ** .5))
    except Exception: return None
def _st(x, base=''):
    return f"{x['hits']}/{x['n']} reach +10% ({x['hits'] / x['n']:.0%})" if x and x.get('n') else 'no history'
def scorecard(tk, asof, atrp, gr=None):
    """-> dict(grade, pts, cohort (all years), cohort_test, rows=[(factor, today, tone '+'/'0'/'-', history stat)]). Points (report 492): +1 ATRP top third of the group, +1 unusual insider buy, +1 stock record >= 60% over >= 5 flips, -1 earnings filed early, -2 bad 8-K. Filing-text change is shown with its measured result but carries NO points (report 491/493: no edge; the paper's direction failed here)."""
    FS = _fstats(); fa = FS.get('factors', {}); rows = []; pts = 0; asof = pd.Timestamp(asof)
    hi = atrp is not None and atrp == atrp and atrp >= FS.get('atrp_cut', 9); pts += hi
    rows.append(('Volatility (ATR %)', f"{atrp:.1%}" + (' · top third of group' if hi else ' · below the top third'), '+' if hi else '0', _st(fa.get('HI', {}).get('on' if hi else 'off'))))
    g = gr if gr is not None else grade(tk, asof); un = bool(g.get('a')); pts += un
    rows.append(('Insider buying', ('UNUSUAL buyer: ' + '; '.join(f"{b['who']} {_usd(b['usd'])}" for b in g['buyers'][:1])) if un else ('buys, but not unusual' if g.get('any_buy') else 'none in 30 days'), '+' if un else '0', _st(fa.get('FIRST', {}).get('on' if un else 'off')) + ('' if un else ' (baseline)')))
    s = FS.get('stock', {}).get(tk); sk, sn = (s if s else (0, 0)); rec = sn >= 5 and sk / sn >= .6; pts += rec
    rows.append(('This stock’s record', f"{sk} of {sn} earlier flips reached +10%" if sn else 'no earlier flips on file', '+' if rec else ('-' if sn >= 5 and sk / sn < .4 else '0'), (f"{sk}/{sn} on this stock" if sn else '') + (' · ' + _st(fa.get('REC', {}).get('on' if rec else 'off')) + (' (record ≥60%)' if rec else ' (others)'))))
    et = earn_timing(tk, asof); early = et == 'early'; pts -= early
    rows.append(('Earnings timing', {'early': 'filed EARLIER than last year', 'late': 'filed later than last year', 'normal': 'on the usual schedule / none in 30 days', None: 'unknown (no prior-year date)'}[et], '-' if early else '0', _st(fa.get('EARLY30', {}).get('on')) + ' when early' if early else (_st(fa.get('LATE30', {}).get('on')) + ' when late (no edge)' if et == 'late' else '')))
    nb = negbad(tk, asof); pts -= 2 * bool(nb)
    rows.append(('Bad 8-K (7 days)', nb or 'none', '-' if nb else '0', (_st(fa.get('NEGBAD', {}).get('on')) + ' when present') if nb else ''))
    sim = filing_change(tk, asof); ft = FS.get('filing', {}); ch = sim is not None and sim <= ft.get('cut', 0)
    rows.append(('Filing text vs last year', 'unavailable' if sim is None else (f"CHANGED (similarity {sim:.3f})" if ch else f"similar ({sim:.3f})"), '0', ('changed: ' + _st(ft.get('changed')) + ' vs same: ' + _st(ft.get('same')) + ' · no edge, not graded') if ft else ''))
    grd = 'A' if pts >= 2 else ('B' if pts == 1 else ('C' if pts == 0 else 'D'))
    return dict(grade=grd, pts=int(pts), cohort=FS.get('grades', {}).get(grd), cohort_test=FS.get('grades_test', {}).get(grd), rows=rows)
