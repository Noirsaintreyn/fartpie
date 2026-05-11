"""
kalman.py — Kalman filter for smoothing NHP intensity and detecting slope changes
================================================================================
Smooths raw λ(t) series and computes slope (first derivative) with
standardized z-scores for signal generation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class KalmanOutput:
    """Output from Kalman filter applied to intensity series."""
    smoothed:   np.ndarray   # smoothed λ(t)
    slope:      np.ndarray   # slope (first derivative) of smoothed λ
    slope_z:    np.ndarray   # z-scored slope
    variance:   np.ndarray   # filter variance at each step
    raw:        np.ndarray   # original raw λ


def kalman_pipeline(
    prices:     np.ndarray,
    raw_lambda: np.ndarray,
    process_noise:    float = 0.01,
    measurement_noise: float = 0.1,
    slope_window:     int   = 5,
) -> KalmanOutput:
    """
    Apply 1D Kalman filter to smooth raw NHP intensity, then compute slope.

    The filter tracks the intensity level (state = [level, velocity]):
        x_t = A @ x_{t-1} + process_noise
        z_t = H @ x_t + measurement_noise

    After smoothing, computes the slope as the rate of change over
    `slope_window` steps, then z-scores the slope for threshold detection.

    Args:
        prices:            price array (used for scale reference)
        raw_lambda:        raw NHP intensity values
        process_noise:     Q diagonal value (higher = more responsive)
        measurement_noise: R value (higher = more smoothing)
        slope_window:      window for computing slope

    Returns:
        KalmanOutput with smoothed series, slope, and slope_z
    """
    n = len(raw_lambda)
    if n < 3:
        zeros = np.zeros(n)
        return KalmanOutput(
            smoothed=raw_lambda.copy(),
            slope=zeros,
            slope_z=zeros,
            variance=np.ones(n),
            raw=raw_lambda.copy(),
        )

    # State: [level, velocity]
    # Transition: level_t = level_{t-1} + velocity_{t-1}
    #             velocity_t = velocity_{t-1}
    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])  # observe level only
    Q = np.eye(2) * process_noise
    Q[1, 1] *= 0.1  # velocity changes slower
    R = np.array([[measurement_noise]])

    # Initialize
    x = np.array([raw_lambda[0], 0.0])  # [level, velocity]
    P = np.eye(2) * 1.0

    smoothed = np.zeros(n)
    velocities = np.zeros(n)
    variances = np.zeros(n)

    for t in range(n):
        # Predict
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q

        # Update
        z = np.array([raw_lambda[t]])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + (K @ (z - H @ x_pred)).flatten()
        P = (np.eye(2) - K @ H) @ P_pred

        smoothed[t] = x[0]
        velocities[t] = x[1]
        variances[t] = P[0, 0]

    # Compute slope over window
    slope = np.zeros(n)
    for t in range(slope_window, n):
        slope[t] = (smoothed[t] - smoothed[t - slope_window]) / slope_window

    # Z-score the slope
    slope_mean = np.mean(slope[slope_window:]) if n > slope_window else 0.0
    slope_std = np.std(slope[slope_window:]) if n > slope_window else 1.0
    slope_std = max(slope_std, 1e-8)
    slope_z = (slope - slope_mean) / slope_std

    return KalmanOutput(
        smoothed=smoothed,
        slope=slope,
        slope_z=slope_z,
        variance=variances,
        raw=raw_lambda.copy(),
    )
