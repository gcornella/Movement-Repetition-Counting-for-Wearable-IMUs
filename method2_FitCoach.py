"""
 Method 2:  FitCoach
 Based on:  FitCoach: Virtual fitness coach empowered by wearable mobile devices (Guo et al., IEEE INFOCOM 2017)
 Method:    - Signal Computation
                - Compute Magnitude of Linear Acceleration (MLA): ||raw|| - gravity (gravity via 0.5Hz LPF).
                - [Enhancement] Filter MLA (3Hz Low-Pass) to remove vibration noise before energy calculation.
                - Compute Short Time Energy (STE): Sliding window sum of MLA^2.
                  (Note: "Energy" in DSP implies squaring the signal. We square MLA to quantify motion intensity).
                - Smooth STE (1.5Hz LPF) to create a cohesive energy envelope.
            - Peak Detection & Segmentation
                - Detect peaks in STE using adaptive prominence (25% of dynamic range) and height thresholds.
                - Enforce minimum peak distance (0.65s) to prevent double-counting concentric/eccentric phases.
                - Identify repetition boundaries by locating local minima (valleys) between accepted peaks.

 Notes:
   • The original paper uses STE on MLA to ensure orientation independence.
   • We extended the STE window from the implied ~0.2s to 0.5s. This "bridges" the isometric pause between lifting and lowering, preventing the "double-counting" artifact common in controlled weight training.
   • A 3Hz pre-filter was added to the MLA to stabilize the energy calculation against sensor jitter.
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from typing import List, Tuple


def _clean_signal(data: np.ndarray) -> np.ndarray:
    nans = np.isnan(data)
    if not np.any(nans): return data
    if np.all(nans): return np.zeros_like(data)
    x_idxs = np.arange(len(data))
    data[nans] = np.interp(x_idxs[nans], x_idxs[~nans], data[~nans])
    return data


def _low_pass_filter(data: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    try:
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        if normal_cutoff >= 1.0: normal_cutoff = 0.99
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, data)
    except:
        return data


def _calculate_mla(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    mla_raw = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # Gravity estimation (0.5Hz)
    gravity = _low_pass_filter(mla_raw, fs, cutoff=0.5)
    # Linear Acceleration Magnitude
    mla = np.abs(mla_raw - gravity)
    return mla_raw, mla


def _calculate_ste(mla: np.ndarray, fs: float, win_sec: float) -> np.ndarray:
    win_len = int(round(win_sec * fs))
    if win_len < 1: win_len = 1
    energy_signal = mla ** 2
    window = np.ones(win_len)
    ste = np.convolve(energy_signal, window, mode='same')
    return ste


def _detect_boundaries(ste: np.ndarray, peaks: np.ndarray) -> List[int]:
    boundaries = []
    sig_len = len(ste)
    if len(peaks) == 0: return []

    if peaks[0] > 0:
        boundaries.append(np.argmin(ste[0:peaks[0]]))
    else:
        boundaries.append(0)

    for i in range(len(peaks) - 1):
        p_curr, p_next = peaks[i], peaks[i + 1]
        if p_next > p_curr:
            local_min = np.argmin(ste[p_curr:p_next]) + p_curr
            boundaries.append(local_min)

    if peaks[-1] < sig_len:
        local_min = np.argmin(ste[peaks[-1]:]) + peaks[-1]
        boundaries.append(local_min)
    else:
        boundaries.append(sig_len - 1)

    return sorted(list(set(boundaries)))


def method2_fcn(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float, plot: bool, info: list) -> int:
    """
    FitCoach Pipeline (Method 2) with tunable thresholds defined at the top.
    """

    # =========================================================================
    #                         ALGORITHM THRESHOLDS
    # =========================================================================

    # 1. MLA Pre-Filter Cutoff (Hz)
    #    Purpose: Removes high-frequency vibration/jitter from the accelerometer
    #    before calculating energy. Higher values allow more noise; lower values
    #    might smooth out quick, explosive movements.
    #    Default: 3.0 Hz
    MLA_LPF_CUTOFF = 3.0

    # 2. STE Window Size (Seconds)
    #    Purpose: Determines the "memory" of the energy calculation.
    #    - Too Small (<0.3s): "Double Counting" risk. The energy drops to zero
    #      during the pause between lift/lower, causing 1 rep to look like 2.
    #    - Too Large (>0.8s): Merges rapid reps together, missing counts.
    #    - Optimal: ~0.5s for controlled weight training (bridges the pause).
    STE_WINDOW_SEC = 0.5

    # 3. STE Smoothing Cutoff (Hz)
    #    Purpose: Smooths the resulting energy curve to make peak detection robust.
    #    It creates a clean "hill" for each rep.
    #    Default: 1.5 Hz
    STE_SMOOTH_CUTOFF = 1.5

    # 4. Minimum Peak Distance (Seconds)
    #    Purpose: The hard physiological limit for a full repetition cycle.
    #    Forces the algorithm to ignore the secondary "eccentric" peak if it
    #    occurs too close to the first one.
    #    Default: 0.65s (implies max ~92 reps/minute, safe for strength training).
    MIN_PEAK_DIST_SEC = 0.65

    # 5. Prominence Factor (0.0 to 1.0)
    #    Purpose: How much must a peak stand out from the surrounding signal?
    #    Calculated as (Factor * Dynamic Range).
    #    - Higher (0.5): Detects only very distinct, explosive reps. Misses weak ones.
    #    - Lower (0.1): Detects everything, including noise.
    #    Default: 0.5 (Strict, avoids false positives).
    PROMINENCE_FACTOR = 0.2


    # 6. Height Threshold Factor (0.0 to 1.0)
    #    Purpose: Minimum absolute energy required.
    #    Calculated as (5th Percentile + Factor * Dynamic Range).
    #    Default: 0.2 (Must be at least 20% up from the noise floor).
    HEIGHT_FACTOR = 0.5

    # =========================================================================

    # --- 1. Robust Input ---
    MIN_LEN = 30  # small safety margin, skip if data is too short

    if len(x) < MIN_LEN or len(y) < MIN_LEN or len(z) < MIN_LEN:
        return 0
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    if len(x) < fs * 0.5: return 0
    x, y, z = _clean_signal(x), _clean_signal(y), _clean_signal(z)

    # --- 2. Compute MLA ---
    mla_raw, mla = _calculate_mla(x, y, z, fs)

    # Pre-filter MLA to remove jitter
    mla_filtered = _low_pass_filter(mla, fs, cutoff=MLA_LPF_CUTOFF)

    # --- 3. Compute STE (Using Tuned Window) ---
    ste = _calculate_ste(mla_filtered, fs, win_sec=STE_WINDOW_SEC)
    ste_smooth = _low_pass_filter(ste, fs, cutoff=STE_SMOOTH_CUTOFF)

    # --- 4. Peak Detection ---
    min_dist = int(MIN_PEAK_DIST_SEC * fs)

    p95 = np.percentile(ste_smooth, 95)
    p05 = np.percentile(ste_smooth, 5)
    dynamic_range = p95 - p05

    if dynamic_range < 1e-3:
        rep_count = 0
        peaks = []
        boundaries = []
    else:
        prominence = PROMINENCE_FACTOR * dynamic_range
        height_thresh = p05 + HEIGHT_FACTOR * dynamic_range

        peaks, _ = signal.find_peaks(
            ste_smooth,
            distance=min_dist,
            height=height_thresh,
            prominence=prominence
        )
        boundaries = _detect_boundaries(ste_smooth, peaks)
        rep_count = len(peaks)

    # --- 5. Diagnostic Plotting ---
    if plot:
        try:
            t = np.arange(len(x)) / fs
            # Create 3 subplots: Raw, Window Analysis, Final Result
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

            title = f"FitCoach Analysis (Method 2) | Reps: {rep_count}"
            if info: title += f" | {info}"
            ax1.set_title(title)

            # Plot 1: Raw Signals
            ax1.plot(t, mla_raw, 'k--', alpha=0.3, label='MLA Raw')
            ax1.plot(t, mla_filtered, 'b-', linewidth=1.5, label=f'MLA (Filtered {MLA_LPF_CUTOFF}Hz)')
            ax1.set_ylabel('Accel (g)')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Window Size Sensitivity Analysis
            ax2.set_title("Effect of STE Window Size (Diagnostic)")
            # Compare current setting vs outliers
            test_windows = [0.2, 0.4, STE_WINDOW_SEC, 0.8]
            # Ensure unique sorted list
            test_windows = sorted(list(set(test_windows)))

            colors = plt.cm.viridis(np.linspace(0, 1, len(test_windows)))

            for i, win in enumerate(test_windows):
                # Calculate STE for this specific window
                ste_test = _calculate_ste(mla_filtered, fs, win_sec=win)
                ste_test = _low_pass_filter(ste_test, fs, cutoff=STE_SMOOTH_CUTOFF)

                # Plot
                lbl = f"Win {win}s"
                if win == STE_WINDOW_SEC: lbl += " (Current)"

                alpha = 0.9 if win == STE_WINDOW_SEC else 0.5
                linewidth = 2.0 if win == STE_WINDOW_SEC else 1.0

                ax2.plot(t, ste_test, color=colors[i], alpha=alpha, linewidth=linewidth, label=lbl)

            ax2.set_ylabel('Energy')
            ax2.legend(loc='upper right', fontsize='small')
            ax2.grid(True, alpha=0.3)

            # Plot 3: Final Algorithm Result
            ax3.set_title(f"Final Detection (Window {STE_WINDOW_SEC}s)")
            ax3.plot(t, ste_smooth, 'k-', linewidth=2, label='STE (Used)')
            ax3.plot(t[peaks], ste_smooth[peaks], 'rx', markersize=12, label='Peaks')

            if len(boundaries) > 0:
                ax3.plot(t[boundaries], ste_smooth[boundaries], 'bo', markersize=6, label='Boundaries')
                for b in boundaries:
                    ax3.axvline(t[b], color='b', linestyle=':', alpha=0.3)

            ax3.set_ylabel('Energy')
            ax3.set_xlabel('Time (s)')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plot error: {e}")

    return int(rep_count)