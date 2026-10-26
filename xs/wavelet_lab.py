"""Wavelet-denoised price into the vector (Oct 25 2026). CAUSAL: at each bar the trailing WIN closes are wavelet-denoised and only the
last denoised value is kept, so no future data leaks in (denoising the whole series at once - the usual mistake - looks great and is
impossible to trade). High/low are shifted by the same amount as the close. Live engine otherwise (15m + 1h, sticky stop, LOCK).
 wav_db4  Daubechies-4, 3 levels, soft universal threshold, 128-bar window        wav_sym8  Symlet-8, 4 levels, 256-bar window
python3 xs/wavelet_lab.py BARS_DIR OUT_DIR WIDE_T"""
import sys, os, glob, numpy as np, pandas as pd, pywt
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'structure_lab.py')).read().split("W = pd.read_parquet(WT)")[0])
def denoise(c, wave, lev, win):
    x = np.log(c); out = np.full(len(x), np.nan)
    for i in range(win, len(x) + 1):
        w = x[i - win:i]; co = pywt.wavedec(w, wave, level=lev, mode='periodization')
        sig = np.median(np.abs(co[-1])) / .6745; th = sig * np.sqrt(2 * np.log(win))
        co = [co[0]] + [pywt.threshold(d, th, 'soft') for d in co[1:]]
        out[i - 1] = pywt.waverec(co, wave, mode='periodization')[-1]
    return np.exp(out)
def smooth_df(B, wave, lev, win):
    d = denoise(B.close.values, wave, lev, win); k = d / B.close.values
    return B.assign(open=B.open * k, high=B.high * k, low=B.low * k, close=d).dropna(subset=['close'])
def st(B, D):
    s = pd.Series(vec(D[['dt', 'open', 'high', 'low', 'close']].reset_index(drop=True)), index=D.index)
    return s.reindex(B.index).fillna(0).astype(int).values
W = pd.read_parquet(WT); CORE = sorted(set(W[W.sp & (W.ratio >= 1.5)].tk))
fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
for tk in (sys.argv[4:] or CORE):
    fo = f'{OUT}/{tk}.parquet'
    if os.path.exists(fo) or tk not in fs: continue
    H = load(fs[tk]); rows = []; hb = agg(H, '1h', '9h30min')
    for nm, wave, lev, win in (('wav_db4', 'db4', 3, 128), ('wav_sym8', 'sym8', 4, 256)):
        r15 = st(H, smooth_df(H, wave, lev, win)); r1 = known(H, hb, st(hb, smooth_df(hb, wave, lev, win)), '1h')
        for h1t in (False, True): rows.extend((tk, nm, h1t, 'trade') + t for t in run(H, r15, r1, h1t)[0])
    pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'kind', 'entry', 'exit', 'ret']).to_parquet(fo); print(tk, len(rows), flush=True)
