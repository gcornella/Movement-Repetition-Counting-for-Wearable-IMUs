"""
 Method 3:  MiLift
 Based on:  MiLift: Efficient Smartwatch-Based Workout Tracking Using Automatic Segmentation

 Method 3a (Autocorr-based; paper’s “autocorrelation-based algorithm”):
  - Using gravity acceleration. If not provided, low-pass each accelerometer axis to extract just the gravity components.
  - Compute ACF per axis
  - Score ACF quality (periodicity/cleanliness), including period consistency
  - Count reps by naive peak detection on the selected axis’ ACF

 Method 3b (Revisit-based; “lightweight” in the paper):
  - Low-pass to gravity-ish signals
  - Select dominant axis by range (max–min)
  - Find naive peaks/valleys via derivative zero-crossings (no prominence)
  - Choose peaks vs valleys by local spikiness (mean |2nd-deriv| in ±Δ window)
  - Prune boundary artifacts using the 3-axis gravity vector

 Notes on “not_so_naive_peak_detection”:
    • The paper does not spell out the exact peak finder. We interpret their “naive” detection
        as local-extrema without heavy constraints.
    • Here we use first-derivative zero-crossings + second-derivative sign (curvature) plus a refractory min-distance.
    • This is a little smarter than the bare minimum (hence “not so naive”) but still simple—no prominence,
        no adaptive thresholds. Treat this as a faithful practical interpretation, not a verbatim spec.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, lfilter
from utils import lowpass_filter, pick_signal_with_highest_amplitude

# ------------------------------
# ACF helper (normalized)
# ------------------------------
def autocorrelation_homemade_fcn(signal):
    """
    Normalized ACF for non-negative lags:
      - mean-center
      - full correlate
      - keep non-negative lags and normalize by lag-0
    """
    signal_ = signal - np.mean(signal)
    acf = np.correlate(signal_, signal_, mode='full')
    acf = acf[acf.size // 2 :]
    if acf[0] != 0:
        acf = acf / acf[0]
    return acf

# -------------------------------------------------------
# ACF quality evaluation (periodicity / cleanliness)
# -------------------------------------------------------
def evaluate_autocorrelation(acf, fs, min_rep_hz=0.3, max_rep_hz=1.5):
    """
    Metrics from ACF (exclude lag 0), tuned to plausible rep rates:
      - avg secondary-peak amplitude (larger => more periodic energy)
      - decay slope of peak amplitudes (less negative => more sustained periodicity)
      - avg inter-peak distance (samples)
      - period consistency = 1 / (1 + CV) of inter-peak distance (higher => steadier rhythm)
      - area under |ACF| (overall periodic structure)
      - number of secondary peaks detected
    """
    acf1 = acf[1:]
    if acf1.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # Distance in samples from a max rep frequency bound
    # (prevents double-counting very close peaks)
    min_period = int(np.floor(fs / max_rep_hz)) if max_rep_hz > 0 else 1
    distance = max(5, min_period)

    peaks, _ = find_peaks(acf1, distance=distance, prominence=0.05)
    amps = acf1[peaks] if peaks.size else np.array([])

    avg_peak_amplitude = float(np.mean(amps)) if amps.size else 0.0

    # Decay slope across peak index (not time)—coarse envelope trend
    if amps.size >= 3:
        x = np.arange(len(amps))
        decay_slope = float(np.polyfit(x, amps, 1)[0])  # typically <= 0
    else:
        decay_slope = 0.0

    # Inter-peak distances (samples)
    if peaks.size >= 2:
        pk_dists = np.diff(peaks).astype(float)
        avg_peak_distance = float(pk_dists.mean())
        cv = float(pk_dists.std(ddof=1) / (pk_dists.mean() + 1e-9)) if pk_dists.size >= 2 else 0.0
        period_consistency = 1.0 / (1.0 + max(cv, 0.0))  # 0..1, higher=steadier
    else:
        avg_peak_distance = 0.0
        period_consistency = 0.0

    area = float(np.nansum(np.abs(acf1)))
    num_peaks = int(len(peaks))

    return (avg_peak_amplitude, decay_slope, avg_peak_distance,
            period_consistency, area, num_peaks)

# -------------------------------------------------------
# Naive extrema (derivative-based) without prominence
# -------------------------------------------------------
def not_so_naive_peak_detection(signal, fs, max_reps_hz=3.0):
    """
    INTERPRETATION of MiLift’s “naive peak detection” on the raw/gravity signal:
      - Extrema from first-derivative zero-crossings
      - Classify with second-derivative sign (curvature)
      - Enforce a refractory min-distance ≈ 1 / max_reps_hz
    Rationale:
      This is slightly more robust than strict neighbor comparisons, yet still simple.
      The original paper does not specify these exact steps.
    """
    # First/second derivatives
    d1 = np.gradient(signal)
    d2 = np.gradient(d1)

    # Zero-crossings of d1 are candidate extrema
    zc = np.where(np.diff(np.sign(d1)))[0]
    peaks   = [i for i in zc if d2[i] < 0]
    valleys = [i for i in zc if d2[i] > 0]

    # Refractory min-distance (prevents double counts in w-shapes)
    min_dist = int(max(1, round(fs / max_reps_hz)))

    def enforce_min_distance(idxs, score_fn, reverse=True):
        # Sort by amplitude proxy: peaks high first; valleys deep first (by -signal)
        sorted_ids = sorted(idxs, key=score_fn, reverse=reverse)
        accepted = []
        for k in sorted_ids:
            if all(abs(k - a) >= min_dist for a in accepted):
                accepted.append(k)
        return np.array(accepted, dtype=int)

    peaks   = enforce_min_distance(peaks,   lambda i: signal[i],  reverse=True)
    valleys = enforce_min_distance(valleys, lambda i: -signal[i], reverse=True)

    return peaks, valleys, d2

# -------------------------------------------------------
# Spikiness (mean |2nd-deriv| in ±Δ) and boundary prune
# -------------------------------------------------------
def choose_by_vertical_displacement(signal, peaks, valleys, fs, delta_sec, second_derivative):
    """
    Pick whether reps are better represented by peaks or valleys:
      - For each candidate, compute mean |second derivative| in a local ±Δ window
      - Compare average spikiness for peaks vs valleys; choose the larger
    """
    delta = max(1, int(round(fs * delta_sec)))
    n = len(signal)

    def mean_abs_s2(indices):
        vals = []
        for i in indices:
            if i - delta < 0 or i + delta >= n:
                continue
            s2w = np.abs(second_derivative[i - delta : i + delta + 1])
            if s2w.size:
                vals.append(float(s2w.mean()))
        return np.array(vals, float)

    P = mean_abs_s2(peaks)
    V = mean_abs_s2(valleys)
    Pmean = float(P.mean()) if P.size else 0.0
    Vmean = float(V.mean()) if V.size else 0.0

    if Pmean >= Vmean:
        return len(P), "peaks", peaks if P.size else np.array([], int)
    else:
        return len(V), "valleys", valleys if V.size else np.array([], int)

def boundary_outlier_prune(gx, gy, gz, idxs, trim_sec, fs, z_thresh=2.5):
    """
    Drop first/last candidate if its 3D gravity vector is an outlier vs. the core reps.
    Only checks events within 'trim_sec' of the recording boundaries, mirroring the paper’s boundary caution.
    """
    if idxs is None or len(idxs) < 3:
        return idxs

    trim = int(round(trim_sec * fs))
    n = len(gx)
    G = np.vstack([gx, gy, gz]).T
    core = idxs[1:-1]
    if len(core) < 3:
        return idxs

    means = G[core].mean(axis=0)
    stds  = G[core].std(axis=0, ddof=1) + 1e-8

    def max_axis_z(vec):
        z = np.abs((vec - means) / stds)
        return float(np.max(z))

    keep = set(idxs.tolist())
    for c in [idxs[0], idxs[-1]]:
        # Only consider candidates very near the start/end
        if c < trim or c > (n - 1 - trim):
            if max_axis_z(G[c]) > z_thresh:
                keep.discard(c)

    return np.array(sorted(keep), dtype=int)

# ===========================
#        METHOD 3a
# ===========================
def method3a_fcn(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float, plot: bool, info: []) -> int:
    """
    Autocorrelation-based rep counting (MiLift-3a style)
      1) LPF each axis to approximate gravity (slow-varying component)
      2) ACF per axis; compute a quality score per ACF
         - Includes 'period consistency' = 1/(1+CV) of inter-peak spacing
      3) Select best axis by score
      4) Naive peak count on that axis’ ACF (secondary peaks ~ reps)
    """
    MIN_LEN = 30  # small safety margin, skip if data is too short

    if len(x) < MIN_LEN or len(y) < MIN_LEN or len(z) < MIN_LEN:
        return 0

    # 1) Gravity-ish LPF
    x_lp = lowpass_filter(x, fs, cutoff_hz=1.0, order=4, zero_phase=True)
    y_lp = lowpass_filter(y, fs, cutoff_hz=1.0, order=4, zero_phase=True)
    z_lp = lowpass_filter(z, fs, cutoff_hz=1.0, order=4, zero_phase=True)

    # 2) ACF + metrics per axis
    axes = [x_lp, y_lp, z_lp]
    acfs = [autocorrelation_homemade_fcn(sig) for sig in axes]
    rows = []
    for i, acf in enumerate(acfs):
        avg_amp, decay_slope, avg_dist, period_consistency, area, n_peaks = evaluate_autocorrelation(acf, fs)
        rows.append(dict(
            Axis=i, AvgPeakAmp=avg_amp, DecaySlope=decay_slope, AvgPeakDist=avg_dist,
            PeriodConsistency=period_consistency, Area=area, NumPeaks=n_peaks
        ))
    df = pd.DataFrame(rows)

    # 3) Normalize & composite score (higher is better)
    def norm(v):
        v = pd.to_numeric(v, errors='coerce').fillna(0.0)
        den = (v.max() - v.min())
        return (v - v.min()) / den if den > 0 else 0.0

    df['S_amp']   = norm(df['AvgPeakAmp'])
    df['S_decay'] = norm(-df['DecaySlope'])        # less negative slope => better
    df['S_dist']  = norm(df['AvgPeakDist'])        # optional; helps prefer clearer period
    df['S_pc']    = norm(df['PeriodConsistency'])  # NEW: steadier rhythm => higher
    df['S_area']  = norm(df['Area'])
    df['S_npk']   = norm(df['NumPeaks'])

    # Weights can be tuned; we favor amplitude and consistency
    df['Score'] = (0.30*df['S_amp'] +
                   0.20*df['S_pc'] +
                   0.15*df['S_decay'] +
                   0.15*df['S_dist'] +
                   0.10*df['S_area'] +
                   0.10*df['S_npk'])

    best_axis = int(df['Score'].idxmax())

    # 4) Count secondary peaks on best axis' ACF
    acf_sel = acfs[best_axis][1:]  # ignore lag 0
    min_period = int(max(5, np.floor(fs / 1.5)))  # fastest ~1.5 Hz reps
    peaks, _ = find_peaks(acf_sel, distance=min_period, prominence=0.05)

    # Visualize results
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(6, 9))
        fig.suptitle(f"Method 3a; UserID: {info[0]}; Activity: {info[1]}")
        axs[0].plot(x_lp, label='x')
        axs[0].plot(y_lp, label='y')
        axs[0].plot(z_lp, label='z');
        axs[0].set_title("Gravity accelerometer data")
        axs[0].legend()

        for k in range(3): axs[1].plot(acfs[k], label=f'ACF axis {k}')
        axs[1].scatter(peaks, acf_sel[peaks], c='r', s=12, label='reps (ACF)')
        axs[1].set_title(f'Best ACF axis: {best_axis} and peaks')
        axs[1].legend()
        plt.tight_layout(); plt.show()

    return int(len(peaks))

# ===========================
#        METHOD 3b
# ===========================
def method3b_fcn(x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: float, plot: bool, info: [], delta_sec=0.08, boundary_trim_sec=0.5) -> int:
    """
    Revisit-based rep counting (MiLift-3b style; lightweight)
      1) LPF to gravity-ish
      2) Dominant axis (highest amplitude)
      3) Candidate extrema via derivative zero-crossings + curvature
         + refractory min-distance (no prominence)
      4) Choose peaks vs valleys by local spikiness (mean |2nd-deriv| in ±Δ)
      5) Boundary prune using 3-axis outlier check
    """
    MIN_LEN = 30  # small safety margin, skip if data is too short

    if len(x) < MIN_LEN or len(y) < MIN_LEN or len(z) < MIN_LEN:
        return 0
    # 1) Gravity-ish LPF
    xg = lowpass_filter(x, fs, cutoff_hz=1.0, order=4, zero_phase=True)
    yg = lowpass_filter(y, fs, cutoff_hz=1.0, order=4, zero_phase=True)
    zg = lowpass_filter(z, fs, cutoff_hz=1.0, order=4, zero_phase=True)
    gravityVector = {'X': xg, 'Y': yg, 'Z': zg}

    # 2) Dominant axis (largest dynamic range within the session)
    dominantSignalKey, signal = pick_signal_with_highest_amplitude(gravityVector, p_low=10.0, p_high=90.0)

    # 3) Extrema candidates on dominant axis (INTERPRETATION)
    peaks, valleys, d2 = not_so_naive_peak_detection(signal, fs, max_reps_hz=3.0)

    # 4) Pick mode by spikiness in a small Δ window
    counts, mode, rep_idxs = choose_by_vertical_displacement(signal, peaks, valleys, fs, delta_sec, d2)

    # 5) Boundary de-ambiguation using all three axes
    rep_idxs = boundary_outlier_prune(xg, yg, zg, rep_idxs, boundary_trim_sec, fs)
    counts = int(len(rep_idxs))

    # Visualize results
    if plot:
        plt.figure(figsize=(7,3))
        plt.plot(xg, label='x')
        plt.plot(yg, label='y')
        plt.plot(zg, label='z')
        plt.plot(signal, lw=1, label=dominantSignalKey)
        plt.scatter(peaks,   signal[peaks],   c='r', s=10, label='peaks')
        plt.scatter(valleys, signal[valleys], c='orange', s=10, label='valleys')
        plt.scatter(rep_idxs, signal[rep_idxs], c='b', s=12, label='reps')
        plt.title(f"Method 3b; UserID: {info[0]}; Activity: {info[1]}; Dominant axis: {dominantSignalKey} | Mode: {mode} | Reps: {counts}")
        plt.legend(); plt.tight_layout(); plt.show()

    return counts
