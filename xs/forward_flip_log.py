"""Forward flip log (Nov 22 2026).  Rey: log every flip live with its features and outcome so we get a clean out-of-sample test (the 2018 + 2022-26 test periods are research data now).
Usage: python3 xs/forward_flip_log.py [sessions=3]   -> pulls ~170 days of 15m RTH candles (LSE vault, key read from the scratchpad .lse_env at RUNTIME, never stored here) for the 54 research names, runs the vector (xs/vector_pine.run_state_machine), and APPENDS every bull flip of the last N sessions to data/forward_log.parquet (dedupe on tk + flip bar).  Also prints the current board: age in 15m bars, price vs the flip entry price, pulled-back flag, still-bull flag, vector stop level, ER20 / TOD-adjusted Z / jump stat J (reports 342-343 features).  NOT applied here: the live options-ratio >= 1.5 gate (needs the live options feed), so it is a log of raw flips on the research set.  Outcome columns (exit_time, exit_px, ret) are filled by `python3 xs/forward_flip_log.py score` once the vector bear-flips (exit = open of the bar after the bear-flip bar, 2+2 bp)."""
import os, sys, time, json, urllib.request, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; KEY = [l.split('=', 1)[1].strip() for l in open(SP + '.lse_env') if l.startswith('LSE_KEY')][0].strip('"\'')
LOG = 'data/forward_log.parquet'; TK = sorted(pd.read_parquet(SP + 'b302.parquet').tk.unique())
def get(tk, a, b, tries=6):
    for k in range(tries):
        try:
            r = urllib.request.Request(f'https://api.londonstrategicedge.com/vault/candles?symbol={tk}&timeframe=15m&start={a}&end={b}', headers={'x-api-key': KEY, 'User-Agent': 'curl/8.5.0', 'Accept': '*/*'}); return json.load(urllib.request.urlopen(r, timeout=60))
        except Exception as e: time.sleep(4 * (k + 1) if '429' in str(e) else 1.5 * (k + 1))
    return None
def bars(tk):
    end = pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=1); rows = []
    for a in (end - pd.Timedelta(days=170), end - pd.Timedelta(days=85)):
        d = get(tk, a.date(), min(a + pd.Timedelta(days=85), end).date())
        if d is None: return None
        rows += d; time.sleep(.2)
    H = pd.DataFrame(rows).drop_duplicates('ts'); H.index = pd.to_datetime(H.ts, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index(); mm = H.index.hour * 60 + H.index.minute; return H[(mm >= 570) & (mm < 960)]
def feats(H, i):  # i = flip bar index; everything known at that bar's close
    c = H.close.values; lc = np.log(c); r = np.r_[np.nan, np.diff(lc)]; a = np.abs(r); path = np.abs(np.diff(c[i - 20:i])).sum(); er = abs(c[i - 1] - c[i - 20]) / path if path > 0 and i >= 21 else np.nan
    bp = (np.pi / 2) * np.nanmean(a[i - 16:i] * a[i - 17:i - 1]) if i >= 18 else np.nan; J = r[i] / np.sqrt(bp) if bp and bp > 0 else np.nan
    e = lc - np.log(pd.Series(c).ewm(span=20, adjust=False).mean().values); bk = (H.index.hour * 60 + H.index.minute).values; prior = e[:i][bk[:i] == bk[i]][-60:]
    z = (e[i] - np.median(prior)) / (1.4826 * np.median(np.abs(prior - np.median(prior)))) if len(prior) >= 20 else np.nan
    return er, z, J
def scan(nsess):
    out = []
    for tk in TK:
        H = bars(tk)
        if H is None or len(H) < 400: print('skip', tk); continue
        regs, lv, atr, fl = run_state_machine(H); sess = sorted(set(H.index.normalize()))[-nsess:]; last_i = len(H) - 1
        for i, d in fl:
            if d != 1 or H.index[i].normalize() < sess[0] or i + 1 > last_i: continue
            er, z, J = feats(H, i); ent = H.open.values[i + 1]; px = H.close.values[last_i]; lo = H.low.values[i + 1:last_i + 1].min()
            out.append(dict(tk=tk, flip_bar=H.index[i], entry_time=H.index[i + 1], entry_px=ent, age_bars=last_i - i, last_time=H.index[last_i], last_px=px, pct_from_entry=px / ent - 1, pulled_back=bool(H.open.values[last_i] <= ent), min_low_since=lo, still_bull=bool(regs[last_i] == 1), vec_stop=lv[last_i], er20=er, z_aligned=z, J=J, exit_time=pd.NaT, exit_px=np.nan, ret=np.nan, logged_at=pd.Timestamp.utcnow().tz_localize(None)))
    return pd.DataFrame(out)
def score():
    L = pd.read_parquet(LOG); open_ = L[L.exit_time.isna()]
    for tk in open_.tk.unique():
        H = bars(tk)
        if H is None: continue
        regs, lv, atr, fl = run_state_machine(H)
        for j in open_[open_.tk == tk].index:
            i0 = H.index.get_loc(L.loc[j, 'flip_bar']) if L.loc[j, 'flip_bar'] in H.index else None
            if i0 is None: continue
            nxt = [i for i, d in fl if d == -1 and i > i0]
            if nxt and nxt[0] + 1 < len(H): ix = nxt[0] + 1; L.loc[j, ['exit_time', 'exit_px']] = [H.index[ix], H.open.values[ix]]; L.loc[j, 'ret'] = H.open.values[ix] / L.loc[j, 'entry_px'] - 1 - .0004
    L.to_parquet(LOG); print(L[L.exit_time.notna()][['tk', 'entry_time', 'exit_time', 'ret']].tail(20).to_string()); print(f'{L.exit_time.notna().sum()} closed of {len(L)}')
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'score': score(); sys.exit()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3; N = scan(n)
    if os.path.exists(LOG): L = pd.read_parquet(LOG); N = pd.concat([L, N]).drop_duplicates(['tk', 'flip_bar'], keep='first')
    N.to_parquet(LOG); cur = N[N.flip_bar >= N.flip_bar.max().normalize() - pd.Timedelta(days=n + 2)].sort_values('flip_bar', ascending=False)
    pd.set_option('display.width', 250); print(f'{len(N)} flips in log; last bar {N.last_time.max()}'); print(cur[['tk', 'flip_bar', 'entry_px', 'age_bars', 'pct_from_entry', 'pulled_back', 'still_bull', 'vec_stop', 'er20', 'z_aligned', 'J']].round(3).head(40).to_string(index=False))
