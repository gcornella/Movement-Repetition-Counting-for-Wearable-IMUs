"""
 Method 4:  Threshold crossing
 Based on:  Accelerometer-Based Automated Counting of Ten Exercises without Exercise-Specific Training or Tuning
 Method 4a: - Record acceleration on 3 axes
            - Get the magnitude of acceleration.
            - Apply sign multiplication to avoid a half-wave rectification. If the signal was in the direction of gravity (using a long-timescale moving average), the signal was positive and if opposite, it was negative.
            - Tuning the threshold line position at two-third of the range between the minimum and maximum
            - When the accelerometer data cross the line in a positive direction, one count would be added.
            - A refractory period of 0.1 seconds has occurred to prevent counting another repetition needlessly.
 Method 4b: - Same as 4a, but adds a Butterworth filter to minimize the impact of high-frequency noise.
            - Low-pass filter using a moving average with the formula Ai+1 = 0.9 Ai+0.1 Xi.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import lowpass_filter_moving_average


def count_positive_crossings_adaptive_window(
    data: np.ndarray, fs: float, window_s: float = 3.0, refractory_s: float = 0.1, frac: float = 2/3
):
    """
    Method 4 thresholding (adaptive variant to avoid global outlier bias):
      The paper uses a threshold at: min + (2/3)*(max - min).
      To avoid a single outlier setting the threshold too high for the whole signal,
      we recompute min/max within successive windows of length `window_s` seconds.
      We then count positive-direction crossings with a global refractory.

    Steps per window [k, k+W):
      - Compute wmin, wmax from the window’s finite values.
      - Define thr = wmin + frac*(wmax - wmin), with frac=2/3 as in the paper.
      - Count one event when x[i-1] <= thr and x[i] > thr.
      - Enforce global refractory of `refractory_s` seconds between events.

    Args:
        data: 1D signal to be counted (e.g., signed magnitude)
        fs: Sampling rate [Hz]
        window_s: Threshold update window size in seconds (paper uses a fixed threshold; this variant adapts every window)
        refractory_s: Refractory period (paper uses 0.1 s)
        frac: Fraction between min and max for the threshold (paper uses 2/3)

    Returns:
        count: Total number of counts
        idxs:  Sample indices where counts fired (for plotting x)
        ths:   Threshold value at each fired index (for plotting y)
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n < 2:
        return 0, [], []

    # Convert seconds to samples
    W = max(1, int(round(window_s * fs)))
    rp = max(1, int(round(refractory_s * fs)))

    count = 0
    last_fire = -10**9  # last counted index (global refractory memory)
    idxs = []           # sample positions where we counted
    ths = []            # threshold values used at those positions

    start = 0
    while start < n:
        end = min(n, start + W)
        win = x[start:end]

        # Build threshold robustly from finite values only
        finite = np.isfinite(win)
        if not np.any(finite):
            start = end
            continue

        wmin = np.min(win[finite])
        wmax = np.max(win[finite])
        thr = wmin + frac * (wmax - wmin)

        # Scan inside this window, but honor the global refractory across windows
        i0 = max(start + 1, 1)  # ensure x[i-1] exists
        for i in range(i0, end):
            if i - last_fire < rp:
                continue
            if not (np.isfinite(x[i]) and np.isfinite(x[i-1])):
                continue

            # Positive-direction threshold crossing
            if x[i] > thr and x[i - 1] <= thr:
                count += 1
                last_fire = i
                idxs.append(i)
                ths.append(float(thr))

        start = end

    return count, idxs, ths


def gravity_unit_vector(x, y, z, fs, gravity_win_s=3.0, eps=1e-9):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    n = len(x)
    if not (n == len(y) == len(z)):
        raise ValueError(f"length mismatch x={len(x)}, y={len(y)}, z={len(z)}")

    # Desired window from seconds
    W = int(round(fs * gravity_win_s))

    # Clamp W to signal length, minimum 1
    W = max(1, min(W, n))

    # Make W odd (optional but nice for centering)
    if W % 2 == 0 and W > 1:
        W -= 1

    k = np.ones(W, dtype=float) / W

    gx = np.convolve(x, k, mode="same")
    gy = np.convolve(y, k, mode="same")
    gz = np.convolve(z, k, mode="same")

    # Now gx,gy,gz will be length n (because W <= n)
    gnorm = np.sqrt(gx*gx + gy*gy + gz*gz) + eps
    return gx/gnorm, gy/gnorm, gz/gnorm


def signed_magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float, gravity_win_s: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    if not (len(x) == len(y) == len(z)):
        raise ValueError(f"signed_magnitude: length mismatch x={len(x)}, y={len(y)}, z={len(z)}")

    ghatx, ghaty, ghatz = gravity_unit_vector(x, y, z, fs, gravity_win_s=gravity_win_s)

    # Extra assertion to catch exactly what you're seeing
    if not (len(ghatx) == len(x) == len(ghaty) == len(ghatz)):
        raise ValueError(f"ghat length mismatch: x={len(x)}, ghatx={len(ghatx)}, ghaty={len(ghaty)}, ghatz={len(ghatz)}")

    mag = np.sqrt(x*x + y*y + z*z)
    dot = x*ghatx + y*ghaty + z*ghatz

    sgn = np.sign(dot)
    sgn[sgn == 0] = 1.0
    return mag * sgn


# ---------- Method 4a: threshold crossing with sign multiplication ----------
def method4a_fcn(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float, plot: bool, info: []) -> int:
    """
    Method 4a — Positive-direction threshold crossing on signed magnitude:
      1) Record acceleration on 3 axes (x,y,z).
      2) Get magnitude and apply sign multiplication via gravity direction (avoid half-wave rectification).
      3) Define threshold at 2/3 between min and max — here adapted per-window to avoid global outliers.
      4) Count a rep when the signal crosses the threshold upward.
      5) Enforce a 0.1 s refractory to prevent double-counting.

    Returns:
        count: total repetition count for Method 4a
    """
    MIN_LEN = 30  # small safety margin, skip if data is too short

    if len(x) < MIN_LEN or len(y) < MIN_LEN or len(z) < MIN_LEN:
        return 0

    # Step 2: build the paper’s signed magnitude (long-timescale gravity window = 3 s)
    signed_mag = signed_magnitude(x, y, z, fs, gravity_win_s=3)

    # Step 3–5: adaptive (3–5 s) per-window thresholding + positive-direction crossing + 0.1 s refractory
    # (You set 5.0 s here; change to 3.0 if you want tighter adaptation.)
    count, idxs, ths = count_positive_crossings_adaptive_window(
        signed_mag, fs=fs, window_s=5.0, refractory_s=0.1, frac=2/3
    )

    # Visualize results
    if plot:
        plt.figure();
        plt.title(f"Method 4a; UserID: {info[0]}; Activity: {info[1]}")
        plt.plot(signed_mag, label='signed_mag')
        plt.scatter(idxs, ths, color='red', label='counts')
        plt.xlabel('Sample'); plt.ylabel('Value'); plt.grid(True);
        plt.legend(); plt.show()

    return count

# ---------- Method 4b: threshold crossing with sign multiplication and low-pass ----------
def method4b_fcn(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float, plot: bool, info: []) -> int:
    """
    Method 4b (paper) — Same as 4a, plus low-pass smoothing before thresholding:
      1) Build signed magnitude as in Method 4a.
      2) Apply the paper’s smoothing A[i] = 0.9*A[i-1] + 0.1*X[i] (EMA).
      3) Use the same thresholding rule (2/3 of min–max) with adaptive windows.
      4) Count positive-direction crossings with 0.1 s refractory.

    Returns:
        count: total repetition count for Method 4b
    """
    MIN_LEN = 30  # small safety margin, skip if data is too short

    if len(x) < MIN_LEN or len(y) < MIN_LEN or len(z) < MIN_LEN:
        return 0

    # Step 1: paper’s signed magnitude
    signed_mag = signed_magnitude(x, y, z, fs, gravity_win_s=3)

    # Step 2: paper’s low-pass smoothing
    signed_mag_lp = lowpass_filter_moving_average(signed_mag)

    # Step 3–4: adaptive threshold + positive-direction crossing + refractory
    count, idxs, ths = count_positive_crossings_adaptive_window(
        signed_mag_lp, fs=fs, window_s=5.0, refractory_s=0.1, frac=2/3
    )

    # Visualize results
    if plot:
        plt.figure();
        plt.title(f"Method 4b; UserID: {info[0]}; Activity: {info[1]}")
        plt.plot(signed_mag_lp, label='signed_mag (LP)')
        plt.scatter(idxs, ths, color='red', label='counts @ threshold')
        plt.xlabel('Sample'); plt.ylabel('Value'); plt.grid(True);
        plt.legend(); plt.show()

    return count
