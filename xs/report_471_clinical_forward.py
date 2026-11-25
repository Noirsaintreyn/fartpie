"""Report 471 (Nov 24 2026): dated forward clinical catalyst. Hypothesis: a small-cap biotech/pharma whose registered Phase 2/3 trial reaches its PRIMARY COMPLETION date within the next ~5 weeks has a dated reason to move (readout / data / financing around it), so a vector flip just before it reaches +10% (REACH) more often than a flip in a similar pipeline company with nothing due.
DECLARED BEFORE RESULTS. Universe: bull vector flips (structure_small) px<20/<10/<5, SIC 283x/2800s/384x/8731/80xx names (pharma, biotech, devices, health). Base group = top 30% base5 score (train-cut). Sponsor->ticker: company name from SEC facts entityName, normalised (drop inc/corp/ltd/plc/holdings/therapeutics suffixes), matched to ClinicalTrials.gov lead sponsor name by prefix; only tickers with >=1 matched Phase 2/3 interventional trial enter (flag vs no-flag compare like with like); match rate reported.
Flag FWD35 = a matched Phase 2/3 trial with primary completion date in (flip, flip+35d] (dates with only YYYY-MM are put mid-month). FWD35_P3 = Phase 3 only. PLACEBOS: BACK35 = completion date in (flip-35d, flip] (already happened), FAR = (flip+35d, flip+70d]. LEAK CAVEAT: the API returns the CURRENT record, not the date as known at the flip (history endpoint is blocked), so completed trials carry their ACTUAL date; BACK35 and FAR show how much of any lift is hindsight. PASS (declared) = FWD35 beats no-flag on REACH in every era with n>=8 and pooled z>=2 AND BACK35 does not show the same lift."""
import numpy as np, pandas as pd, json, os, re, time, requests, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; CACHE = SP + 'ext/ct_cache/'; os.makedirs(CACHE, exist_ok=True)
O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d).astype('datetime64[ns]'); O = O[O.px < 20].copy().reset_index(drop=True)
O['era3'] = np.select([O.d.dt.year <= 2021, O.d.dt.year == 2022], ['2019-21', '2022'], '2023-26'); O['tr'] = (O.d.dt.year <= 2021).values
sec = pd.to_numeric(O.sec, errors='coerce'); SC = pd.read_csv(SP + 'ext/sic.csv'); O = O.merge(SC[['tk', 'sic']], on='tk', how='left')
HEALTH = ((O.sic >= 2830) & (O.sic <= 2836)) | (O.sic == 8731) | ((O.sic >= 3840) & (O.sic <= 3859)) | ((O.sic >= 8000) & (O.sic <= 8099)) | O.sec.astype(str).isin(['28', '38', '80'])
O = O[HEALTH].reset_index(drop=True); tks = sorted(O.tk.unique()); print('health flips', len(O), 'tickers', len(tks))
SUF = r'\b(inc|incorporated|corp|corporation|ltd|limited|plc|co|company|holdings|holding|llc|lp|sa|nv|ag|the|group)\b'
def norm(s): return re.sub(r'\s+', ' ', re.sub(SUF, ' ', re.sub(r'[^a-z0-9 ]', ' ', str(s).lower()))).strip()
def name(tk):
    f = f'data/fund/sec_small/{tk}.json'
    return json.load(open(f)).get('entityName') if os.path.exists(f) else None
H = {'User-Agent': 'research rey@example.com'}
def studies(core):
    f = CACHE + re.sub(r'\W+', '_', core) + '.pkl'
    if os.path.exists(f): return pd.read_pickle(f)
    rows, tok = [], None
    for _ in range(12):
        p = {'query.spons': core, 'pageSize': 1000, 'fields': 'NCTId,LeadSponsorName,PrimaryCompletionDate,PrimaryCompletionDateType,Phase,StudyType', 'filter.advanced': 'AREA[Phase](PHASE2 OR PHASE3)'}
        if tok: p['pageToken'] = tok
        r = None
        for a in range(3):
            try:
                r = requests.get('https://clinicaltrials.gov/api/v2/studies', params=p, headers=H, timeout=60)
                if r.status_code == 200: break
            except Exception: pass
            time.sleep(3 * (a + 1))
        if r is None or r.status_code != 200: break
        j = r.json()
        for s in j.get('studies', []):
            ps = s['protocolSection']; rows.append((ps['identificationModule']['nctId'], ps['sponsorCollaboratorsModule']['leadSponsor']['name'], ps.get('statusModule', {}).get('primaryCompletionDateStruct', {}).get('date'), ','.join(ps.get('designModule', {}).get('phases', []))))
        tok = j.get('nextPageToken')
        if not tok: break
    D = pd.DataFrame(rows, columns=['nct', 'sponsor', 'pcd', 'phase']); pd.to_pickle(D, f); time.sleep(.2); return D
ev = []; matched = 0
for n, tk in enumerate(tks):
    nm = name(tk)
    if not nm: continue
    core = norm(nm); words = core.split()
    if not words or len(core) < 4: continue
    q = ' '.join(words[:2]); D = studies(q)
    if D.empty: continue
    D = D[D.sponsor.map(norm).str.startswith(core) | D.sponsor.map(norm).map(lambda s: len(core) >= 6 and s.startswith(core[:max(6, len(core) - 3)]))]
    D = D[D.pcd.notna()]
    if D.empty: continue
    matched += 1
    for r in D.itertuples():
        d = pd.Timestamp(r.pcd + '-15') if len(r.pcd) == 7 else pd.Timestamp(r.pcd); ev.append((tk, d, 'PHASE3' in r.phase))
    if n % 50 == 0: print('..', n, 'matched tickers', matched, flush=True)
E = pd.DataFrame(ev, columns=['tk', 'pcd', 'p3']); print('tickers matched with >=1 Phase2/3 trial with a date:', matched, 'of', len(tks), '| trials', len(E))
g = {t: x for t, x in E.groupby('tk')}
def win(lo, hi, p3=False):
    out = np.zeros(len(O), bool)
    for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
        x = g.get(tk)
        if x is None: continue
        v = x[x.p3] if p3 else x
        out[i] = ((v.pcd.values > d + np.timedelta64(lo, 'D')) & (v.pcd.values <= d + np.timedelta64(hi, 'D'))).any()
    return out
O['has'] = O.tk.isin(g); O['FWD35'] = win(0, 35); O['FWD35_P3'] = win(0, 35, True); O['BACK35'] = win(-35, 0); O['FAR'] = win(35, 70)
B = {'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = pd.DataFrame(index=D.index)
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
def pz(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 5 or n2 < 5: return np.nan
    p1, p2 = a.mean(), b.mean(); p = (a.sum() + b.sum()) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)); return (p1 - p2) / se if se > 0 else np.nan
for lab, cap in (('px<20', 20), ('px<10', 10), ('px<5', 5)):
    S = O[O.px < cap].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7)
    for gl, G in (('BASE GROUP top30%', S[S.s_ >= cut]), ('ALL health flips', S)):
        G = G[G.has]; print(f'\n######## {lab} {gl}: flips in matched-pipeline names {len(G)} (train {G.tr.sum()}, test {(~G.tr).sum()}) | REACH base {(G.PEAK >= .1).mean():.1%} REAL {(G.r >= .1).mean():.1%}')
        for fl in ('FWD35', 'FWD35_P3', 'BACK35', 'FAR'):
            a = G[G[fl]]; b = G[~G[fl]]; s = ''
            for e in ('2019-21', '2022', '2023-26'):
                x = a[a.era3 == e]; y = b[b.era3 == e]; s += f' {(x.PEAK >= .1).mean() * 100:3.0f}% v {(y.PEAK >= .1).mean() * 100:3.0f}% [{len(x)}] /' if len(x) else ' - /'
            print(f'  {fl:9s} n {len(a):4d} | REACH {(a.PEAK >= .1).mean():.0%} v {(b.PEAK >= .1).mean():.0%} (z {pz((a.PEAK >= .1).astype(float), (b.PEAK >= .1).astype(float)):+.1f}) | REAL {(a.r >= .1).mean():.0%} v {(b.r >= .1).mean():.0%} (z {pz((a.r >= .1).astype(float), (b.r >= .1).astype(float)):+.1f}) | median r {a.r.median() * 100:+.1f}% v {b.r.median() * 100:+.1f}% | era REACH:{s}')
