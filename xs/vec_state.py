"""Nov 2 2026: outside review #3, instrumented vector. An exact copy of signal_lab.vec() that also exports its state on every bar,
checked for parity (identical regimes on both tracks, identical live trade ledger) before any analysis.
Per bar: regime before/after, the active level used in the test, ATR, close, consecutive-break counters, the level-update reason
(1 init/flip, 2 ratchet), the bar the level last changed, the pivot behind it (occurred at bar i - L, confirmed at bar i).
DECLARED QUESTIONS (discovery only; any intervention is a separate, counted experiment):
 Q1 transition mechanism: at the first bar of the break run that produced each bullish flip, which of price / level / ATR changes
    were necessary for the trigger close > level + BREAK_ATR x ATR (counterfactual: freeze each at its previous-bar value)?
 Q2 boundary persistence: age of the level crossed, number and depth of earlier failed approaches, whether the pivot structure
    was making higher lows, time since the source pivot was confirmed.
 Q3 entry/exit attribution grid: vector entry + vector exit (live) | vector entry + random-duration exit | random entry + vector exit |
    random entry + random-duration exit.
 Reported per period: avg trade, big-winner ratio (share of top-5% trades / share of trades), bars held, worst drawdown inside the trade.
Output: /tmp/claude-0/ml/vs_flips.parquet (one row per live trade with its flip's mechanism + boundary features)."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
def vec_i(df, **kw):
    """signal_lab.vec, line for line, plus state export"""
    P = dict(BASE, **kw); L = P['PIVOT_LEN']
    high, low, close = df['high'].values, df['low'].values, df['close'].values; n = len(df)
    atr = compute_atr_series(high, low, close, P['ATR_LEN']); ef, es = ema(close, P['FAST_LEN']), ema(close, P['SLOW_LEN'])
    pl, ph = _piv(low, L, False), _piv(high, L, True); patr = np.r_[np.full(L, np.nan), atr[:-L]]
    LA, BA, MM, MB, CB = P['LEVEL_ATR'], P['BREAK_ATR'], P['MIN_MOVE_ATR'], P['MIN_BARS_RATCH'], P['CONFIRM_BARS']
    regime, level, bc, ac, lr = 0, np.nan, 0, 0, None; regs = np.zeros(n, int)
    S = {k: np.full(n, np.nan) for k in ('lvl', 'atr', 'ac', 'bc', 'upd', 'lr', 'rb', 'sb')}
    for i in range(n):
        a = atr[i]
        if np.isnan(a) or np.isnan(ef[i]) or np.isnan(es[i]) and regime == 0: regs[i] = regime; continue
        cs = pl[i] - patr[i] * LA if not np.isnan(pl[i]) and not np.isnan(patr[i]) else np.nan
        cr = ph[i] + patr[i] * LA if not np.isnan(ph[i]) and not np.isnan(patr[i]) else np.nan
        upd = 0
        if regime == 0: regime = 1 if ef[i] >= es[i] else -1; level = low[i] - a * LA if regime == 1 else high[i] + a * LA; lr = i; upd = 1
        S['lvl'][i] = level; S['atr'][i] = a                                     # the level and ATR the break test uses on this bar
        sb = regime == 1 and close[i] < level - a * BA; rb = regime == -1 and close[i] > level + a * BA
        if regime == 1: bc, ac = (bc + 1 if sb else 0), 0
        elif regime == -1: ac, bc = (ac + 1 if rb else 0), 0
        S['ac'][i], S['bc'][i], S['rb'][i], S['sb'][i] = ac, bc, rb, sb
        if regime == 1 and bc >= CB:
            regime, bc, ac = -1, 0, 0; nr = cr if (not np.isnan(cr) and cr > close[i]) else high[i] + a * LA; level = max(nr, close[i] + a * LA); lr = i; upd = 1
        elif regime == -1 and ac >= CB:
            regime, bc, ac = 1, 0, 0; ns = cs if (not np.isnan(cs) and cs < close[i]) else low[i] - a * LA; level = min(ns, close[i] - a * LA); lr = i; upd = 1
        elif regime == 1 and not np.isnan(cs) and cs < close[i] and cs > level + a * MM and (lr is None or i - lr >= MB): level, lr, upd = cs, i, 2
        elif regime == -1 and not np.isnan(cr) and cr > close[i] and cr < level - a * MM and (lr is None or i - lr >= MB): level, lr, upd = cr, i, 2
        S['upd'][i] = upd; S['lr'][i] = lr if lr is not None else np.nan; regs[i] = regime
    S = pd.DataFrame(S); S['reg'] = regs; S['close'] = close; S['pl'] = pl; S['ph'] = ph; S['CB'] = CB; S['BA'] = BA; return S
if __name__ == '__main__':
    X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet'); C = BASE['CONFIRM_BARS']; BA_ = BASE['BREAK_ATR']
    print(f"settings withheld; CONFIRM_BARS={C} (needed to locate the break run)", flush=True)
    rows = []; bad = 0
    for tk in sorted(X.tk.unique()):
        f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet'); H = load(f[0])
        reg0, s10 = states(H, vec); S15 = vec_i(H[['dt', 'open', 'high', 'low', 'close']]); r = agg(H, '1h', '9h30min'); S1h = vec_i(r)
        s1i = known(H, r, S1h.reg.values, '1h')
        par = (S15.reg.values == reg0).all() and (s1i == s10).all()
        T = [(tk, False) + t for t in run(H, S15.reg.values, s1i, False)[0]] + [(tk, True) + t for t in run(H, S15.reg.values, s1i, True)[0]]
        mine = set((a, b, pd.Timestamp(c)) for a, b, c, *_ in T); live = X[X.tk == tk]
        miss = sum((tk, h, e) not in mine for h, e in zip(live.h1, live.entry))
        if not par or miss: bad += 1
        print(f'  {tk:<5} regimes identical: {par}   live trades {len(live)}, not reproduced {miss}', flush=True)
        # flip features, both tracks
        kn1 = (r.dt + pd.Timedelta('1h')).values
        for _, x in live.iterrows():
            e = np.datetime64(x.entry)
            if not x.h1:
                j = np.searchsorted(H.dt.values, e); k = j - 1
                while k > j - 120 and k > 0 and not (S15.reg.values[k] == 1 and S15.reg.values[k - 1] == -1): k -= 1
                fb = k; S = S15; tt = H.dt.values
            else:
                k = np.searchsorted(kn1, e + np.timedelta64(15, 'm') - np.timedelta64(15, 'm'), side='right') - 1
                while k > 0 and not (S1h.reg.values[k] == 1 and S1h.reg.values[k - 1] == -1): k -= 1
                fb = k; S = S1h; tt = r.dt.values
            if fb < C + 2 or not (S.reg.values[fb] == 1 and S.reg.values[fb - 1] == -1): rows.append(dict(i=x.name, ok=False)); continue
            f_ = fb - C + 1; p = f_ - 1                                             # first bar of the break run, and the bar before it
            Cf, Cp, Lf, Lp, Af, Ap = S.close[f_], S.close[p], S.lvl[f_], S.lvl[p], S.atr[f_], S.atr[p]
            need_px = not (Cp > Lf + BA_ * Af); need_lv = not (Cf > Lp + BA_ * Af); need_at = not (Cf > Lf + BA_ * Ap)
            lr_ = int(S.lr[p]) if pd.notna(S.lr[p]) else p; age = f_ - lr_
            seg = S.iloc[lr_ + 1:f_]; z = (seg.close - seg.lvl - BA_ * seg.atr) / seg.atr
            appr = int(((z > -.5) & (z <= 0)).sum()); depth = float(z[z <= 0].max()) if (z <= 0).any() else np.nan
            pls = S.pl.values[:f_]; pls = pls[~np.isnan(pls)]; phs = S.ph.values[:f_]; phs = phs[~np.isnan(phs)]
            hl = float(len(pls) >= 2 and pls[-1] > pls[-2]); lh = float(len(phs) >= 2 and phs[-1] < phs[-2])
            reason = S.upd.values[lr_] if lr_ < len(S) else np.nan
            rows.append(dict(i=x.name, ok=True, need_px=need_px, need_lv=need_lv, need_at=need_at, dC=(Cf - Cp) / Af, dL=-(Lf - Lp) / Af, dA=-BA_ * (Af - Ap) / Af,
                             age=age, appr=appr, depth=depth, hl=hl, lh=lh, reason=reason, run=C))
    F = pd.DataFrame(rows).set_index('i'); Y = X.join(F); Y.to_parquet('/tmp/claude-0/ml/vs_flips.parquet')
    print(f'\nPARITY: {bad} ticker(s) with a mismatch; flips located for {Y.ok.fillna(False).mean():.0%} of trades', flush=True)
