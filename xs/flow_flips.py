"""Nov 9 2026: VECTOR FLIPS WITH EXTENDED DELTA, AND CVD FLIPS. DECLARED BEFORE RESULTS.
Order flow: Rey's MNQ TBBO ticks -> 1-minute aggressor buy / sell volume (mnq_orderflow_1min.parquet, 18 Dec 2024 - 18 Nov 2025 only) -> 5m delta on the NQ RTH bars.
SIGN FIX: build_mnq_orderflow_1min.py maps TBBO side A as buy - in Databento A is the SELL aggressor; signed_vol had corr -0.70 with the minute price change, so it is negated here.
Small sample: rules are read on the FIRST half (18 Dec 2024 - 31 May 2025) and checked on the SECOND (1 Jun - 18 Nov 2025). Points per trade at 1 contract, 1 pt cost.
A) EXTENDED DELTA on flips: the 15m flip bar's delta, signed by the flip side, / the stdev of the last 100 15m deltas = dz. Buckets: against (dz < 0), normal (0-1), extended (>= 1).
   Trades: every 15m vector flip (exit on the opposite flip, stop 0.15 ADR - xs/dte_season2.py run_sig2), the gamma-gated subset (V15a), and Rule A (xs/dte_greeks_prop.py).
B) CVD AGREES: session CVD (reset 09:30) above / below its 20-bar EMA on 5m, at the bar before entry, same side as the trade vs not.
C) CVD FLIPS as their own entries: CVD crosses its 20-bar EMA on a 5m close 09:45-15:30 -> enter next open on that side, exit on the opposite cross, stop 0.15 ADR, <= 4 a day;
   also only when the 15m vector agrees."""
_s2 = open('/home/claude/fartpie/xs/dte_season2.py').read(); exec(_s2[:_s2.rindex("if __name__ == '__main__':")])
OF = pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet'); OF['dt'] = pd.to_datetime(OF.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
OF['signed_vol'] = -OF['signed_vol']   # FIX: the file's buy/sell are swapped (Databento side A = SELL aggressor); corr with price -0.70 before the fix
OF['b5'] = OF.dt.dt.floor('5min'); D5 = OF.groupby('b5').signed_vol.sum()
H['delta'] = H.dt.map(D5); win = (H.dt >= '2024-12-18') & (H.dt < '2025-11-19'); H.loc[~win, 'delta'] = np.nan; dl = H.delta.values
cvd = H.groupby('day').delta.cumsum().where(win); H['cvd'] = cvd; H['cvd_ema'] = H.groupby('day').cvd.transform(lambda s_: s_.ewm(span=20, adjust=False).mean())
cst = np.sign(H.cvd - H.cvd_ema).values
d15 = pd.Series(dl, index=H.dt.values).groupby(pd.Series(H.dt.dt.floor('15min').values, index=H.dt.values)).sum(min_count=3)
z15 = d15 / d15.rolling(100, min_periods=50).std()
def feats(T_):
    out = []
    for r_ in T_.itertuples():
        d = pd.Timestamp(r_.day); ix = gi[d]; w = np.where(m[ix] == r_.m)[0]
        if not len(w) or w[0] == 0: out.append((np.nan, np.nan)); continue
        i = ix[w[0]]; t0 = pd.Timestamp(H.dt.values[i]) - pd.Timedelta(minutes=15); dz = z15.get(t0, np.nan) * r_.dir; ca = cst[i - 1] * r_.dir if np.isfinite(cst[i - 1]) else np.nan
        out.append((dz, ca))
    return out
def cvd_flips(need_vec=False):
    rows = []; vreg = dict(zip(B.end.values, B.reg.values)) if need_vec else None
    for d in days:
        if not (pd.Timestamp('2024-12-18') <= d < pd.Timestamp('2025-11-19')) or not ok(d): continue
        ix = gi[d]; busy = -1; cnt = 0
        for p in range(2, len(ix) - 1):
            i = ix[p]; t = m[i] + 5
            if t < 585 or t > 930 or cnt >= 4 or p + 1 <= busy: continue
            if not (np.isfinite(cst[i]) and np.isfinite(cst[i - 1])) or cst[i] == cst[i - 1] or cst[i] == 0: continue
            dr = int(cst[i])
            if need_vec:
                k_ = np.searchsorted(B.end.values, np.datetime64(H.dt.values[i] + np.timedelta64(5, 'm')), 'right') - 1
                if k_ < 0 or B.reg.values[k_] != dr: continue
            j = p + 1; e = o[ix[j]]; S = e - dr * .15 * DY.adr[d]; res = None
            for q in range(j, len(ix)):
                ii = ix[q]
                if m[ii] >= 955: res = ((c[ix[q - 1]] - e) * dr, q); break
                if (l[ii] <= S) if dr > 0 else (h[ii] >= S): res = ((S - e) * dr, q); break
                if q > j and cst[ii] == -dr and q + 1 < len(ix): res = ((o[ix[q + 1]] - e) * dr, q + 1); break
            if res is None: res = ((c[ix[-1]] - e) * dr, len(ix) - 1)
            rows.append((d, dr, res[0] - 1.0)); busy = res[1]; cnt += 1
    return pd.DataFrame(rows, columns=['day', 'dir', 'pts'])
if __name__ == '__main__':
    half = lambda T_: np.where(T_.day < pd.Timestamp('2025-06-01'), '1st', '2nd')
    V = run_sig2('s_vector'); Vg = V[~hig(V)]; RA_ = ra_full2(True); RA_ = RA_[RA_.sz > 0]
    print(f'order-flow days: {H[win].day.nunique()}, 5m bars with delta {np.isfinite(dl).sum():,}', flush=True)
    print('points per trade (trades) | 1st half / 2nd half', flush=True)
    for nm, T_ in (('15m vector, every flip', V), ('V15a (gamma-gated)', Vg), ('Rule A', RA_)):
        T_ = T_[(T_.day >= '2024-12-18') & (T_.day < '2025-11-19')].copy()
        if not len(T_): continue
        F_ = feats(T_); T_['dz'] = [f_[0] for f_ in F_]; T_['ca'] = [f_[1] for f_ in F_]; T_['h'] = half(T_)
        print(f'\n  {nm}: {len(T_)} trades, all {T_[T_.h == "1st"].pts.mean():+.2f} ({(T_.h == "1st").sum()}) / {T_[T_.h == "2nd"].pts.mean():+.2f} ({(T_.h == "2nd").sum()})', flush=True)
        for lab, msk in (('delta AGAINST the flip (dz < 0)', T_.dz < 0), ('delta normal (0-1)', (T_.dz >= 0) & (T_.dz < 1)), ('delta EXTENDED (dz >= 1)', T_.dz >= 1),
                         ('CVD agrees', T_.ca > 0), ('CVD disagrees', T_.ca < 0)):
            a, b = T_[msk & (T_.h == '1st')].pts, T_[msk & (T_.h == '2nd')].pts
            print(f'    {lab:<34} {a.mean():+6.2f} ({len(a):3d}) / {b.mean():+6.2f} ({len(b):3d})', flush=True)
    for lab, nv in (('CVD flips (own entries)', False), ('CVD flips, 15m vector agrees', True)):
        C_ = cvd_flips(nv); C_['h'] = half(C_); a, b = C_[C_.h == '1st'].pts, C_[C_.h == '2nd'].pts
        print(f'\n  {lab}: {a.mean():+6.2f} ({len(a)}) / {b.mean():+6.2f} ({len(b)}) | win {np.mean(C_.pts > 0):.0%}, total {C_.pts.sum():+.0f} pts', flush=True)
