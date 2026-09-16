"""
1:1 Python port of the "Regressive Support/Resistance - Pivot Ratchet" Pine
script (deterministic state machine: pivot highs/lows -> ATR-buffered close
confirmation -> regime flip; one-sided ATR-gated ratchet with a bar cooldown
within a regime). Same defaults as the script: pivotLen=8, atrLen=14,
breakATR=0.35, levelATR=0.15, minMoveATR=0.80, minBarsRatch=8, confirmBars=1,
EMA(21/55) trend filter for the initial seed regime only.

Two things get backtested, both against a random-timing control (same event
count, same direction labels, bar positions drawn uniformly from the same
eligible range) so any "edge" is measured against chance, not just eyeballed:

1. Regime-flip directional accuracy: does forward return over N bars actually
   move in the flip's implied direction (flip-to-support -> up, flip-to-
   resistance -> down)?
2. SNR: race each flip's forward path to +/-K*ATR. "Signal" = price reaches
   K*ATR in the flip's favor before K*ATR against it (real continuation).
   "Noise" = adverse hit first, or neither hit within the horizon (chop/
   false flag). SNR = n_signal / n_noise.

Also reports the "regime strip" agreement: across ALL bars (not just flip
moments), does current regime sign match forward N-bar return sign more
than the instrument's unconditional up-rate would predict.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_levels_coherent_rr import load_source

PIVOT_LEN = 8
ATR_LEN = 14
BREAK_ATR = 0.35
LEVEL_ATR = 0.15
MIN_MOVE_ATR = 0.80
MIN_BARS_RATCH = 8
CONFIRM_BARS = 1
FAST_LEN = 21
SLOW_LEN = 55
USE_FILTER = True

HORIZONS = [10, 20, 50]
SNR_THRESHOLDS = [0.5, 1.0, 1.5]
SNR_MAX_HORIZON = 60
N_RANDOM_DRAWS = 2000
SEED = 42


def wilder_rma(src, length):
    out = np.full(len(src), np.nan)
    if len(src) < length:
        return out
    seed = np.mean(src[:length])
    out[length - 1] = seed
    for i in range(length, len(src)):
        out[i] = (out[i - 1] * (length - 1) + src[i]) / length
    return out


def ema(src, length):
    out = np.full(len(src), np.nan)
    if len(src) < length:
        return out
    alpha = 2.0 / (length + 1)
    seed = np.mean(src[:length])
    out[length - 1] = seed
    for i in range(length, len(src)):
        out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


def compute_atr_series(high, low, close, length):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return wilder_rma(tr, length)


def find_pivots(values, left, right, is_high):
    # Vectorized: rolling window extremum (center-aligned) instead of a
    # per-bar Python loop - needed since 1m NQ data alone is ~4.2M rows.
    # Drops the exact-tie uniqueness check the naive loop had (two bars in
    # the same window sharing the identical float extremum) - a
    # negligible edge case for continuous float prices, not a behavior
    # change that matters here.
    n = len(values)
    s = pd.Series(values)
    window = left + right + 1
    roll = s.rolling(window, center=True, min_periods=window).max() if is_high \
        else s.rolling(window, center=True, min_periods=window).min()
    is_piv = (s.values == roll.values)
    piv_price = np.where(is_piv, values, np.nan)
    out = np.full(n, np.nan)
    if right < n:
        out[right:] = piv_price[:n - right]
    return out


def run_state_machine(df):
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    n = len(df)

    atr = compute_atr_series(high, low, close, ATR_LEN)
    ema_fast = ema(close, FAST_LEN)
    ema_slow = ema(close, SLOW_LEN)

    piv_low = find_pivots(low, PIVOT_LEN, PIVOT_LEN, is_high=False)
    piv_high = find_pivots(high, PIVOT_LEN, PIVOT_LEN, is_high=True)

    regime = 0
    level = np.nan
    below_count = 0
    above_count = 0
    last_ratchet_bar = None

    regimes = np.zeros(n, dtype=int)
    levels = np.full(n, np.nan)
    flips = []  # (bar_index, direction) direction: +1 = flip to support, -1 = flip to resistance

    for i in range(n):
        a = atr[i]
        if np.isnan(a):
            regimes[i] = regime
            levels[i] = level
            continue

        new_pivot_low = not np.isnan(piv_low[i])
        new_pivot_high = not np.isnan(piv_high[i])
        candidate_support = (piv_low[i] - a * LEVEL_ATR) if new_pivot_low else np.nan
        candidate_resistance = (piv_high[i] + a * LEVEL_ATR) if new_pivot_high else np.nan

        if regime == 0:
            trend_bull = ema_fast[i] >= ema_slow[i] if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]) else True
            regime = 1 if (USE_FILTER and trend_bull) or not USE_FILTER else -1
            level = (low[i] - a * LEVEL_ATR) if regime == 1 else (high[i] + a * LEVEL_ATR)

        support_break = regime == 1 and not np.isnan(level) and close[i] < level - a * BREAK_ATR
        resist_break = regime == -1 and not np.isnan(level) and close[i] > level + a * BREAK_ATR

        below_count = (below_count + 1) if (regime == 1 and support_break) else 0
        above_count = (above_count + 1) if (regime == -1 and resist_break) else 0

        flip_to_resistance = regime == 1 and below_count >= CONFIRM_BARS
        flip_to_support = regime == -1 and above_count >= CONFIRM_BARS

        if flip_to_resistance:
            regime = -1
            below_count = 0
            above_count = 0
            if new_pivot_high and candidate_resistance > close[i]:
                flip_resistance = candidate_resistance
            else:
                flip_resistance = high[i] + a * LEVEL_ATR
            level = max(flip_resistance, close[i] + a * LEVEL_ATR)
            last_ratchet_bar = i
            flips.append((i, -1))

        elif flip_to_support:
            regime = 1
            below_count = 0
            above_count = 0
            if new_pivot_low and candidate_support < close[i]:
                flip_support = candidate_support
            else:
                flip_support = low[i] - a * LEVEL_ATR
            level = min(flip_support, close[i] - a * LEVEL_ATR)
            last_ratchet_bar = i
            flips.append((i, 1))

        elif regime == 1:
            cooldown_ok = last_ratchet_bar is None or (i - last_ratchet_bar) >= MIN_BARS_RATCH
            if (new_pivot_low and candidate_support > level + a * MIN_MOVE_ATR
                    and candidate_support < close[i] and cooldown_ok):
                level = candidate_support
                last_ratchet_bar = i

        elif regime == -1:
            cooldown_ok = last_ratchet_bar is None or (i - last_ratchet_bar) >= MIN_BARS_RATCH
            if (new_pivot_high and candidate_resistance < level - a * MIN_MOVE_ATR
                    and candidate_resistance > close[i] and cooldown_ok):
                level = candidate_resistance
                last_ratchet_bar = i

        regimes[i] = regime
        levels[i] = level

    return regimes, levels, atr, flips


def z_test(p1, n1, p2, n2):
    x1, x2 = p1 * n1, p2 * n2
    pp = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    zz = (p1 - p2) / se
    return zz, 2 * (1 - norm.cdf(abs(zz)))


def n_draws_for(n_events, budget=5_000_000, lo=200, hi=N_RANDOM_DRAWS):
    # Instruments with far more flip events (e.g. NQ 1m has orders of
    # magnitude more flips than NQ 1h) would otherwise blow up total
    # replicate x event work - cap the total pairs evaluated instead of
    # holding N_RANDOM_DRAWS fixed regardless of n_events. For n_events
    # small enough that hi draws fit the budget, this is a no-op (same
    # draw count, same RNG consumption as before).
    if n_events <= 0:
        return lo
    return int(max(lo, min(hi, budget // n_events)))


def flip_accuracy(close, flips, horizons, eligible_lo, eligible_hi, rng):
    # Vectorized: one rng.integers() call of shape (n_draws, n_flips) per
    # horizon instead of N_RANDOM_DRAWS separate Python-loop draws - the
    # per-draw loop version was fine for ~1000-flip swing-timeframe runs
    # but never finishes for NQ 1m (tens of thousands of flips).
    n = len(close)
    real_bars = np.array([b for b, _ in flips])
    real_dirs = np.array([d for _, d in flips])
    n_flips = len(flips)
    n_draws = n_draws_for(n_flips)
    rand_bars = rng.integers(eligible_lo, eligible_hi, size=(n_draws, n_flips))

    rows = []
    for H in horizons:
        valid_real = real_bars + H < n
        fwd_real = close[real_bars[valid_real] + H] - close[real_bars[valid_real]]
        correct_real = np.where(real_dirs[valid_real] == 1, fwd_real > 0, fwd_real < 0)
        real_wr = correct_real.mean()
        real_n = valid_real.sum()

        valid2d = (rand_bars + H) < n
        idx_fwd = np.clip(rand_bars + H, 0, n - 1)
        idx_now = np.clip(rand_bars, 0, n - 1)
        fwd2d = close[idx_fwd] - close[idx_now]
        dirs2d = np.broadcast_to(real_dirs, rand_bars.shape)
        correct2d = np.where(dirs2d == 1, fwd2d > 0, fwd2d < 0).astype(float)
        correct2d[~valid2d] = np.nan
        rand_rates = np.nanmean(correct2d, axis=1)
        rand_rates = rand_rates[~np.isnan(rand_rates)]

        rand_wr = rand_rates.mean()
        pctile = (rand_rates < real_wr).mean() * 100
        zz, p = z_test(real_wr, real_n, rand_wr, len(rand_rates))
        rows.append({
            'horizon': H, 'n_flips': int(real_n), 'real_hit_rate': real_wr,
            'random_hit_rate': rand_wr, 'random_hit_std': rand_rates.std(),
            'diff': real_wr - rand_wr, 'random_percentile': pctile, 'z': zz, 'p': p,
        })
    return pd.DataFrame(rows)


def race_outcome_batch(close, atr, bars, dirs, k, max_h):
    # Vectorized race-to-+/-K*ATR: one pass of `max_h` steps over the WHOLE
    # event array at once (numpy), instead of a Python loop per event per
    # step. This is what made NQ 1m (tens of thousands of flips x
    # thousands of random replicates) intractable before - same race
    # logic, same "unresolved within horizon counts as noise" semantics.
    n = len(close)
    bars = np.asarray(bars, dtype=np.int64)
    dirs = np.asarray(dirs, dtype=np.float64)
    a = atr[np.clip(bars, 0, n - 1)]
    valid = np.isfinite(a) & (a > 0) & (bars >= 0) & (bars < n)
    entry = close[np.clip(bars, 0, n - 1)]
    target = entry + k * a * dirs
    stop = entry - k * a * dirs

    resolved = np.zeros(len(bars), dtype=bool)
    is_signal = np.zeros(len(bars), dtype=bool)
    is_noise = np.zeros(len(bars), dtype=bool)
    for step in range(1, max_h + 1):
        idx = bars + step
        in_range = idx < n
        active = valid & ~resolved & in_range
        if not active.any():
            continue
        idx_c = np.clip(idx, 0, n - 1)
        cur = close[idx_c]
        target_hit = np.where(dirs == 1, cur >= target, cur <= target)
        stop_hit = np.where(dirs == 1, cur <= stop, cur >= stop)
        sig = active & target_hit
        noi = active & stop_hit & ~sig
        is_signal |= sig
        is_noise |= noi
        resolved |= sig | noi
    unresolved_valid = valid & ~resolved
    is_noise |= unresolved_valid  # chop / no follow-through within horizon = noise
    return valid, is_signal, is_noise


def snr_test(close, atr, flips, thresholds, eligible_lo, eligible_hi, rng):
    real_bars = np.array([b for b, _ in flips])
    real_dirs = np.array([d for _, d in flips])
    n_flips = len(flips)
    n_draws = n_draws_for(n_flips)
    rand_bars = rng.integers(eligible_lo, eligible_hi, size=(n_draws, n_flips))
    rand_dirs = np.broadcast_to(real_dirs, rand_bars.shape)

    rows = []
    for k in thresholds:
        valid, sig, noi = race_outcome_batch(close, atr, real_bars, real_dirs, k, SNR_MAX_HORIZON)
        real_n = valid.sum()
        real_sig, real_noise = sig.sum(), noi.sum()
        real_rate = real_sig / real_n if real_n else np.nan
        real_snr = real_sig / real_noise if real_noise else np.inf

        valid_f, sig_f, noi_f = race_outcome_batch(
            close, atr, rand_bars.ravel(), rand_dirs.ravel(), k, SNR_MAX_HORIZON)
        valid2d = valid_f.reshape(rand_bars.shape)
        sig2d = sig_f.reshape(rand_bars.shape)
        noi2d = noi_f.reshape(rand_bars.shape)
        n_per_draw = valid2d.sum(axis=1)
        sig_per_draw = sig2d.sum(axis=1)
        noise_per_draw = noi2d.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            rate_per_draw = sig_per_draw / n_per_draw
            snr_per_draw = sig_per_draw / noise_per_draw
        rate_per_draw = rate_per_draw[n_per_draw > 0]
        snr_per_draw = snr_per_draw[np.isfinite(snr_per_draw) & (noise_per_draw > 0)]

        zz, p = z_test(real_rate, real_n, rate_per_draw.mean(), len(rate_per_draw))
        pctile = (rate_per_draw < real_rate).mean() * 100
        rows.append({
            'k_atr': k, 'n_flips': int(real_n),
            'real_signal_rate': real_rate, 'real_snr': real_snr,
            'random_signal_rate': rate_per_draw.mean(), 'random_snr_mean': snr_per_draw.mean(),
            'diff': real_rate - rate_per_draw.mean(), 'random_percentile': pctile, 'z': zz, 'p': p,
        })
    return pd.DataFrame(rows)


def regime_strip_agreement(close, regimes, horizons, warmup):
    rows = []
    n = len(close)
    base_up_rate = None
    for H in horizons:
        valid = np.arange(warmup, n - H)
        fwd_ret = close[valid + H] - close[valid]
        reg = regimes[valid]
        agree = ((reg == 1) & (fwd_ret > 0)) | ((reg == -1) & (fwd_ret < 0))
        agree_rate = agree.mean()
        up_rate = (fwd_ret > 0).mean()
        if base_up_rate is None:
            base_up_rate = up_rate
        support_frac = (reg == 1).mean()
        expected_by_chance = support_frac * up_rate + (1 - support_frac) * (1 - up_rate)
        n_valid = len(valid)
        zz, p = z_test(agree_rate, n_valid, expected_by_chance, n_valid)
        rows.append({
            'horizon': H, 'n_bars': n_valid, 'regime_agree_rate': agree_rate,
            'expected_by_chance': expected_by_chance, 'unconditional_up_rate': up_rate,
            'pct_bars_in_support': support_frac, 'diff': agree_rate - expected_by_chance,
            'z': zz, 'p': p,
        })
    return pd.DataFrame(rows)


def run_for(instrument, timeframe):
    df = load_source(instrument, timeframe)
    if df is None or len(df) < 300:
        print(f"  no/insufficient data for {instrument} {timeframe}, skipping")
        return
    print(f"\n{'='*90}\n{instrument} {timeframe}  ({df['datetime'].min()} -> {df['datetime'].max()}, {len(df)} bars)\n{'='*90}")

    regimes, levels, atr, flips = run_state_machine(df)
    close = df['close'].values
    n = len(df)
    warmup = max(ATR_LEN, SLOW_LEN, PIVOT_LEN * 2) + 5
    n_bull_flips = sum(1 for _, d in flips if d == 1)
    n_bear_flips = sum(1 for _, d in flips if d == -1)
    print(f"  {len(flips)} confirmed regime flips ({n_bull_flips} to support/bullish, {n_bear_flips} to resistance/bearish)")
    if flips:
        gaps = np.diff([b for b, _ in flips])
        print(f"  median bars between flips: {np.median(gaps):.0f}" if len(gaps) else "")

    eligible_lo, eligible_hi = warmup, n - max(HORIZONS + [SNR_MAX_HORIZON]) - 1
    if eligible_hi <= eligible_lo or len(flips) < 5:
        print("  not enough eligible flips/history, skipping stats")
        return
    rng = np.random.default_rng(SEED)

    print("\n-- Flip directional accuracy vs random-timing control --")
    acc = flip_accuracy(close, flips, HORIZONS, eligible_lo, eligible_hi, rng)
    print(acc.round(4).to_string(index=False))

    print("\n-- SNR (signal:noise, race to +/-K*ATR) vs random-timing control --")
    snr = snr_test(close, atr, flips, SNR_THRESHOLDS, eligible_lo, eligible_hi, rng)
    print(snr.round(4).to_string(index=False))

    print("\n-- Regime-strip agreement (ALL bars, not just flips) vs chance --")
    strip = regime_strip_agreement(close, regimes, HORIZONS, warmup)
    print(strip.round(4).to_string(index=False))

    return {'accuracy': acc, 'snr': snr, 'strip': strip, 'n_flips': len(flips)}


def main():
    import sys
    import traceback
    targets = [
        ('NQ', '15m'), ('NQ', '5m'), ('NQ', '1m'),
        ('AAPL', '15m'), ('AAPL', '5m'), ('AAPL', '1m'),
        ('AMD', '15m'), ('AMD', '5m'), ('AMD', '1m'),
    ]
    for instrument, timeframe in targets:
        try:
            run_for(instrument, timeframe)
        except Exception:
            print(f"  {instrument} {timeframe} FAILED:")
            traceback.print_exc()
        sys.stdout.flush()


if __name__ == '__main__':
    main()
