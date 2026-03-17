"""
 Method 1: RecoFit
 Based on:  RecoFit: Using a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises
 Method:    - Signal Computation
                - Remove high-low freq components using elliptical bandpass filter (0.15 Hz – 11 Hz)
                - Subtract the mean from the data, apply Principal Component Analysis (PCA), and project the data onto its first PC.
            - Peak detection
                - Compute a set of candidate peaks (local maxima).
                - Calculate [minPeriod, maxPeriod]
                - Sort peaks based on amplitude, accepting a candidate peak if it is at least minPeriod away from the closest already-accepted peak.
                - For each candidate, we compute the autocorrelation in a window centered on the peak.
                - The lag corresponding to this value is our estimate of the exercise period P for this candidate.
                - Repeat the process of sorting and filtering peaks, this time rejecting peaks that are closer to neighbors than 0.75*P.
                - Sort all candidate peaks based on amplitude and find the peak at the 40th percentile; we reject all peaks smaller than half the amplitude of this peak,

 Notes:
   • The original paper tunes min/max repetition periods per exercise. Here we scale the period bounds by fs for generality.
   • We use zero-phase filtering (filtfilt) to avoid peak shifts.
   • Peak pruning follows the RecoFit spirit: distance gating via estimated period from autocorrelation + amplitude screening via a 40th-percentile reference peak.
"""

import numpy as np
from scipy.signal import ellip, filtfilt, find_peaks, correlate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  # (Optional) enable if you want to visualize


def estimate_period(signal: np.ndarray, peak_idx: int, min_period: int, max_period: int) -> int:
    """
    Local repetition period estimate around a given peak using autocorrelation.
    ----------
    Input
    ----------
    signal : 1D np.ndarray
        The PCA-projected, band-passed signal.
    peak_idx : int
        Index of the candidate peak within `signal`.
    min_period : int
        Minimum plausible period (in samples).
    max_period : int
        Maximum plausible period (in samples).
    ----------
    Output
    -------
    int
        Estimated period (lag in samples) within [min_period, max_period].
        Falls back to max_period if estimation is not possible.
    """
    # Give autocorr enough context: window = ± 2*max_period
    half_window = 2 * max_period
    start = max(0, peak_idx - half_window)
    end   = min(len(signal), peak_idx + half_window)
    w = signal[start:end]

    # Need at least max_period samples to make a meaningful lag search
    if w.size < max_period + 1:
        return max_period

    # Biased autocorrelation; keep non-negative lags
    ac = correlate(w, w, mode='full')
    ac = ac[ac.size // 2:]

    # Search only within [min_period, max_period]
    if max_period <= min_period or ac.size <= max_period:
        return max_period

    search = ac[min_period:max_period]
    if search.size == 0 or not np.isfinite(search).all():
        return max_period

    lag = int(np.argmax(search)) + min_period
    return lag


def method1_fcn(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float, plot: bool, info: []) -> int:
    """
    Repetition counting using a RecoFit-style pipeline.
    ----------
    Steps
    ----------
    1) Preprocess & reduce to 1D:
       • Elliptical band-pass (0.15–11 Hz) on each axis (zero-phase).
       • Mean-center each axis.
       • PCA -> take the first principal component as a 1D waveform for counting.
    2) Candidate detection & pruning:
       • Find local maxima as candidate peaks.
       • Sort by amplitude descending.
       • Distance gate: accept a peak if ≥ minPeriod away from already accepted peaks.
       • Local period estimate: use autocorr around each accepted peak to get P.
       • Enforce spacing: keep peak only if it is > 0.75 * P from already-final peaks.
       • Amplitude screen: sort finals by amplitude; take the 40th-percentile peak
         (by rank) and discard peaks < 0.5 × that amplitude.
    ----------
    Output
    ----------
    int
        Estimated repetition count.
    """
    # ---------- 1) Preprocess & 1D reduction ----------
    MIN_LEN = 30  # small safety margin, skip if data is too short

    if len(x) < MIN_LEN or len(y) < MIN_LEN or len(z) < MIN_LEN:
        return 0

    # Band-pass per paper
    lowcut, highcut = 0.15, 11.0
    order = 4
    rp, rs = 0.5, 50  # passband ripple (dB), stopband attenuation (dB)

    # Elliptic band-pass (digital, with sampling freq)
    b, a = ellip(order, rp, rs, [lowcut, highcut], btype='band', analog=False, fs=fs)

    # Zero-phase filter each axis to prevent temporal shifts of peaks
    x_f = filtfilt(b, a, x)
    y_f = filtfilt(b, a, y)
    z_f = filtfilt(b, a, z)


    # Mean-center (PCA assumes zero-mean features for covariance structure)
    x0 = x_f - np.mean(x_f)
    y0 = y_f - np.mean(y_f)
    z0 = z_f - np.mean(z_f)

    # Stack to (N, 3) and take the first PC as the 1D signal
    data = np.vstack([x0, y0, z0]).T
    pca = PCA(n_components=3, svd_solver="full")
    pca.fit(data)
    signal1d = pca.transform(data)[:, 0]

    # ---------- 2) Candidate detection & pruning ----------
    # Initial candidates: local maxima without constraints (prominence can be added if needed)
    candidate_peaks, _ = find_peaks(signal1d)

    if candidate_peaks.size == 0:
        return 0  # Nothing to count

    # Sort by amplitude (desc)
    sorted_peaks = sorted(candidate_peaks, key=lambda i: signal1d[i], reverse=True)

    # Period bounds tied to sampling rate (paper uses class-specific values)
    # e.g., min 0.5 s, max 4.0 s by default → tune per exercise if you have labels
    minPeriod = int(0.75 * fs) # todo this should be adpated to each exercise
    maxPeriod = int(4.0 * fs)

    # Coarse distance gating (fast pass to avoid clustered duplicates)
    accepted_peaks = []
    for pk in sorted_peaks:
        if all(abs(pk - ap) >= minPeriod for ap in accepted_peaks):
            accepted_peaks.append(pk)

    # Local autocorr-based period + 0.75·P spacing rule (paper-inspired)
    final_peaks = []
    for pk in accepted_peaks:
        P = estimate_period(signal1d, pk, minPeriod, maxPeriod)
        # Keep if far enough from already-final peaks relative to local period
        if all(abs(pk - fp) > 0.75 * P for fp in final_peaks):
            final_peaks.append(pk)

    if len(final_peaks) == 0:
        return 0

    # Amplitude screening via 40th-percentile-by-rank reference
    sorted_final = sorted(final_peaks, key=lambda i: signal1d[i], reverse=True)
    # Guard for small lists: index ∈ [0, len-1]
    pidx = int(0.4 * (len(sorted_final) - 1)) if len(sorted_final) > 1 else 0
    ref_amp = signal1d[sorted_final[pidx]]

    # Reject peaks < 0.5 × reference amplitude (filters sub-repetition wiggles)
    filtered_peaks = [i for i in sorted_final if signal1d[i] >= 0.5 * ref_amp]

    # Visualize results
    if plot:
        plt.figure(figsize=(10, 4))
        plt.title(f"Method 1 (RecoFit); UserID: {info[0]}; Activity: {info[1]}")
        plt.plot(signal1d, label='1D (PC1)', color='gray', zorder=1)
        plt.scatter(candidate_peaks, signal1d[candidate_peaks], s=10, label='candidates', zorder=2)
        plt.scatter(accepted_peaks, signal1d[accepted_peaks], s=14, label='accepted', zorder=3)
        plt.scatter(final_peaks, signal1d[final_peaks], s=18, label='final (period-gated)',zorder=4)
        plt.scatter(filtered_peaks, signal1d[filtered_peaks], s=24, label='filtered (amp)', zorder=5)
        plt.legend(); plt.tight_layout(); plt.show()

    return int(len(filtered_peaks))
