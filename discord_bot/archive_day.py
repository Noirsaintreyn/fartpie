"""Daily archive for THOD/TLOD research (called by the bot after the 16:05 post; can also be run by hand: python3 archive_day.py [YYYY-MM-DD]).
Writes discord_bot/archive/<date>/: issue.json (levels exactly as issued at 09:35), bars_<TK>.parquet (5m RTH + premarket for the day),
gex_<TK>.json (the GEX-by-strike profile the 09:30 chart used: walls, zero gamma, net gamma by strike near spot)."""
import os, sys, json, shutil
import pandas as pd
HERE = os.environ.get('DATA_DIR') or os.path.dirname(os.path.abspath(__file__))
def archive(date=None, gx_mod=None):
    iss = json.load(open(os.path.join(HERE, 'daily_issue.json'))); date = date or iss['date']
    if iss.get('date') != date: return f'no issue for {date}'
    out = os.path.join(HERE, 'archive', date); os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(HERE, 'daily_issue.json'), os.path.join(out, 'issue.json'))
    import yfinance as yf
    for tk in iss['issued']:
        try:
            b = yf.Ticker(tk).history(period='5d', interval='5m', prepost=True); b.index = b.index.tz_convert('America/New_York').tz_localize(None)
            b[b.index.normalize() == pd.Timestamp(date)].to_parquet(os.path.join(out, f'bars_{tk}.parquet'))
        except Exception as e: print('archive bars', tk, e)
        if gx_mod is not None:
            try:
                g = gx_mod.profile(tk); G = g['G']; s = g['spot']; G = G[(G.index > s * .85) & (G.index < s * 1.15)]; G = G[(G != 0).any(axis=1)]
                json.dump({k: (float(v) if v is not None else None) for k, v in g.items() if k in ('spot', 'call_wall', 'put_wall', 'zero_gamma', 'total')} |
                          {'asof': str(g['asof'].date()), 'G': {c: {str(float(k)): float(v) for k, v in G[c].items()} for c in G.columns}}, open(os.path.join(out, f'gex_{tk}.json'), 'w'))
            except Exception as e: print('archive gex', tk, e)
    return f'archived {date} -> {out}'
if __name__ == '__main__':
    sys.path.insert(0, HERE)
    try: import gexchart as GXm
    except Exception: GXm = None
    print(archive(sys.argv[1] if len(sys.argv) > 1 else None, GXm))
