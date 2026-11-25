"""OPTION-FEATURE SIGNAL MODELS (Nov 24 2026; reports 457-459).  Read-only; never places an order.
Extra Discord signals beside the main screener, each a plain flag on the SAME flip (the main screener's entry): 
  NEG  negative gamma      prior-close net dealer gamma < 0 (live/neg_gamma.json; 15m track, options ratio >= 2: existing channel rule)
  PCV  put/call VOLUME     the name's put/call volume ratio (expiries <= 45d) in the top third of its own trailing 252-day history, prior close   [15m only]
  SKW  skew                OTM put IV (93-97% strike) minus OTM call IV (103-107%), 14-45 DTE, top third of own 252-day history                  [15m and 1h]
  CHM  charm               net dealer charm in the top third of own 252-day history                                                              [15m and 1h]
All need the options ratio >= 1.5 (the live filter). Backtests (23 names, 2019-2026, prior-close features): put/call volume 15m +66bp/trade t 5.3 (lift vs other flips +38bp t 3.1), skew 15m +74bp t 3.3 (lift t 2.6), charm 1h +113bp t 3.5 (lift t 2.2); negative gamma +46bp t 3.5 but no lift over the other flips (t 0.95). Context for the signal, not advice.
Daily row per name = computed from the 16:15 chain Webull log (live/options/<TK>/<date>.csv) -> live/opt_feat/<TK>.csv. History starts from the Theta-built seed (live_data/opt_feat_seed/); once 120 live rows exist the rank uses live rows only (Webull IV and volume are not byte-identical to Theta's).
  python3 option_models.py --update            compute and append today's row for every name that has a saved chain
  python3 option_models.py --tags NVDA         print the flags for a name"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); LIVE = os.getenv('LIVE_DIR', f'{ROOT}/live'); SEED = f'{ROOT}/live_data/opt_feat_seed'; OPT = f'{LIVE}/options'
NAMES = [x for x in os.getenv('OPT_MODEL_NAMES', ','.join(sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{SEED}/*.csv')))).split(',') if x]
def feat_path(tk): return f'{LIVE}/opt_feat/{tk}.csv'

def row_from_chain(df, now=None):
    """one day of features from a Webull chain table (live/options csv)."""
    import webull_greeks as WG
    now = now or pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    d = WG.chain(df); S = float(df['spot'].iloc[0]); vol = WG.pick(df, 'vol')
    g = WG.greeks(d, S, now); m = g.oi * 100; dte = (g.exp - now.normalize()).dt.days
    out = dict(gex=float((g.sgn * g.gamma * m * S * S * .01).sum()), charm=float((g.sgn * g.charm * m * S).sum()))
    if vol is not None:
        v = pd.to_numeric(vol, errors='coerce').reindex(df.index); kd = WG.pick(df, 'kind').astype(str).str.upper().str.startswith('C'); ex = pd.to_datetime(WG.pick(df, 'expiry'), errors='coerce'); dd = (ex - now.normalize()).dt.days; ok = (dd >= 0) & (dd <= 45)
        cv = float(v[ok & kd].sum()); pv = float(v[ok & ~kd].sum()); out['pc_vol'] = pv / max(cv, 1.0)
    mo = g.K / S; mid = (dte >= 14) & (dte <= 45)
    pu = g[(~g.call) & (mo > .93) & (mo < .97) & mid].iv; ca = g[g.call & (mo > 1.03) & (mo < 1.07) & mid].iv
    out['skew'] = float(pu.median() - ca.median()) if len(pu) and len(ca) else np.nan
    return out

def history(tk):
    seed = pd.read_csv(f'{SEED}/{tk}.csv', parse_dates=['date']) if os.path.exists(f'{SEED}/{tk}.csv') else pd.DataFrame(columns=['date', 'gex', 'pc_vol', 'skew', 'charm'])
    live = pd.read_csv(feat_path(tk), parse_dates=['date']) if os.path.exists(feat_path(tk)) else pd.DataFrame(columns=seed.columns)
    return (live if len(live) >= 120 else pd.concat([seed, live])).drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True), len(live)

def ranks(tk, upto=None):
    """rank of the LATEST row against the trailing 252 rows before it, per feature -> dict (None when there is not enough history)."""
    h, _ = history(tk)
    if len(h) < 61: return None
    last = h.iloc[-1]; prev = h.iloc[-253:-1]; r = {'date': str(pd.Timestamp(last.date).date())}
    for c in ('pc_vol', 'skew', 'charm'):
        a = prev[c].dropna().values; r[c] = float((a < last[c]).mean()) if len(a) > 30 and pd.notna(last[c]) else None
    a = prev['pc_vol'].dropna().values; r['pc_z'] = float((last['pc_vol'] - a.mean()) / a.std()) if len(a) > 30 and pd.notna(last['pc_vol']) and a.std() > 0 else None
    r['gex'] = float(last.gex) if pd.notna(last.gex) else None
    return r

def tags(tk, track='15m', max_age_days=5):
    """-> list of model flags for the name at this moment (prior-close features); [] if none, never raises."""
    try:
        if tk not in NAMES: return []
        r = ranks(tk)
        if not r or (pd.Timestamp.now() - pd.Timestamp(r['date'])).days > max_age_days: return []
        t = []
        if track == '15m' and r.get('pc_z') is not None and r['pc_z'] > 1: t.append('PCZ')      # stronger version (Nov 24 2026): z vs own trailing 252d > 1, replaces plain PCV
        elif track == '15m' and r['pc_vol'] is not None and r['pc_vol'] >= 2 / 3: t.append('PCV')
        if r['skew'] is not None and r['skew'] >= 2 / 3: t.append('SKW')
        if r['charm'] is not None and r['charm'] >= 2 / 3: t.append('CHM')
        return t
    except Exception as e:
        print(f'  ! option_models tags {tk}: {str(e)[:60]}'); return []

LABEL = dict(NEG='NEG γ', PCV='P/C VOL', PCZ='P/C VOL z>1', SKW='SKEW', CHM='CHARM')

def update():
    os.makedirs(f'{LIVE}/opt_feat', exist_ok=True); n = 0
    for tk in NAMES:
        fs = sorted(glob.glob(f'{OPT}/{tk}/*.csv'))
        if not fs: continue
        day = os.path.basename(fs[-1])[:10]; same = [f for f in fs if os.path.basename(f).startswith(day)]; f = next((x for x in same if x.endswith('_oi.csv')), same[-1])
        try:
            r = row_from_chain(pd.read_csv(f)); r['date'] = day; p = feat_path(tk)
            old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(); new = pd.concat([old, pd.DataFrame([r])]).drop_duplicates('date', keep='last'); new.to_csv(p, index=False); n += 1
        except Exception as e: print(f'  {tk}: skipped ({str(e)[:80]})')
    print(f'option_models: updated {n}/{len(NAMES)} names')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--update', action='store_true'); ap.add_argument('--tags', default=None); a = ap.parse_args()
    if a.update: update()
    if a.tags: print(a.tags, tags(a.tags, '15m'), ranks(a.tags))
