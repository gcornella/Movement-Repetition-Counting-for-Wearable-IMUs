import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, cheb2ord, cheby2


def method5_fcn(
    x,
    y,
    z,
    fs: float,
    plot: bool = False,
    info: list | None = None,
    remove_unwanted_segments: bool = True,
    connect_adjacent_segments: bool = True,
) -> int:
    """
    Method 5 (Džaja et al.), rep counting from an already segmented bout.

    Summary of what the function does
    1) Demean x, y, z and compute AVM = sqrt(x^2 + y^2 + z^2)
    2) Band-pass filter AVM using the paper fixed cutoffs (0.25 to 1.2 Hz)
    3) Estimate repetition frequency as the dominant spectral peak within that band
    4) Low-pass filter AVM using a Chebyshev Type II filter where:
         passband edge = dominant repetition frequency
         stopband edge = 2x dominant repetition frequency
       Filter order is computed from design specs since the paper does not publish ripple or attenuation.
    5) Repetition boundaries are extrema of the low-passed AVM.
       The paper selects minima vs maxima using exercise labels.
       This implementation is class agnostic: it evaluates both and picks the one that is
       most period-consistent and prominent, while explicitly avoiding half-period (2x) picks.
    6) Optional: remove unwanted start and end segments (paper rule)
    7) Optional: connect adjacent segments when two short segments together form one repetition

    Parameters
    ----------
    x, y, z : array-like
        Tri-axial accelerometer samples for one bout.
    fs : float
        Sampling rate in Hz.
    plot : bool
        If True, draw diagnostic plots.
    info : list | None
        Optional extra text for figure title.
    remove_unwanted_segments : bool
        If True, apply the paper artifact removal rule (start and end partial segments).
        Default True.
    connect_adjacent_segments : bool
        If True, apply an automatic pause split merge that merges two short adjacent
        segments when their summed duration matches one typical repetition.
        Default True.

    Returns
    -------
    int
        Estimated repetition count.
    """

    # ============================================================
    # Paper fixed parameters
    # ============================================================
    BP_LOW_HZ = 0.25
    BP_HIGH_HZ = 1.2
    STOPBAND_FACTOR = 2.0  # stopband edge = 2x passband edge (paper relationship)

    # Paper artifact removal thresholds
    ART_DUR_FACTOR = 0.5  # shorter than 0.5 * average rep time
    ART_AMP_FACTOR = 0.5  # and weaker than 0.5 * max amplitude in set

    # ============================================================
    # Practical choices not numerically fixed in the paper
    # ============================================================
    # Padding helps filtfilt reduce edge transients. Not described in paper.
    PAD_SECONDS = 1.0

    # Band-pass filter family and order are not specified by the paper.
    # Butterworth order 2 is a low ringing, stable default.
    BP_ORDER = 2

    # Chebyshev Type II order is not fixed by the paper.
    # Paper says it depends on pass and stop band.
    # Ripple and attenuation values are not given, so these are tunable.
    CHEB_PASS_RIPPLE_DB = 1.0
    CHEB_STOP_ATTEN_DB = 45.0  # slightly strong to help suppress within rep double bumps

    # Extrema detection gates are not specified by the paper.
    # Set to reduce noise extrema and prevent two extrema per repetition.
    MIN_DIST_FACTOR = 0.90
    PROMINENCE_STD_FACTOR = 0.35

    # Scoring weights for choosing minima vs maxima in a class agnostic way
    # Includes a penalty for half-period spacing to avoid 2x counting.
    SCORE_W_REGULARITY = 1.0
    SCORE_W_SHARPNESS = 0.25
    SCORE_W_MATCH_T = 1.0
    SCORE_W_PENALIZE_HALF = 1.0

    # Additional pruning, keep at most one boundary per repetition
    PRUNE_MIN_SEP_FACTOR = 0.80

    # Automatic adjacency merge settings
    # Trigger only when two short segments sum to about one repetition period.
    MERGE_SHORT_FRAC = 0.65
    MERGE_SUM_TOL = 0.20

    MIN_SIGNAL_SEC = 2.0

    # ============================================================
    # Step 1: Prepare signals and compute AVM
    # ============================================================
    MIN_LEN = 30  # small safety margin, skip if data is too short

    if len(x) < MIN_LEN or len(y) < MIN_LEN or len(z) < MIN_LEN:
        return 0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    n = len(x)
    if n == 0 or n != len(y) or n != len(z):
        return 0
    if n < int(fs * MIN_SIGNAL_SEC):
        return 0

    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    z = np.nan_to_num(z)

    # Demean each axis (paper step)
    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)

    # AVM (paper step)
    avm = np.sqrt(x * x + y * y + z * z)

    pad_samples = int(round(PAD_SECONDS * fs))
    avm_padded = np.pad(avm, pad_samples, mode="reflect") if pad_samples > 0 else avm.copy()
    nyq = 0.5 * fs

    # ============================================================
    # Step 2: Band-pass filtering for improved method (paper cutoffs)
    # ============================================================
    lo = max(BP_LOW_HZ / nyq, 1e-6)
    hi = min(BP_HIGH_HZ / nyq, 0.999999)
    if hi <= lo:
        return 0

    b_bp, a_bp = butter(BP_ORDER, [lo, hi], btype="bandpass")
    avm_bp = filtfilt(b_bp, a_bp, avm_padded)
    avm_bp_trim = avm_bp[pad_samples: pad_samples + n] if pad_samples > 0 else avm_bp

    # ============================================================
    # Step 3: Dominant repetition frequency from spectrum
    # ============================================================
    freqs = np.fft.rfftfreq(len(avm_bp), d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(avm_bp))

    valid = (freqs >= BP_LOW_HZ) & (freqs <= BP_HIGH_HZ)
    if np.any(valid):
        local_freqs = freqs[valid]
        local_mag = fft_mag[valid]
        local_peak_i = int(np.argmax(local_mag))
        f_rep = float(local_freqs[local_peak_i])
        fft_peak_mag = float(local_mag[local_peak_i])
    else:
        f_rep = 0.5 * (BP_LOW_HZ + BP_HIGH_HZ)
        fft_peak_mag = float(fft_mag[np.argmin(np.abs(freqs - f_rep))])

    if not np.isfinite(f_rep) or f_rep <= 0:
        f_rep = 0.5 * (BP_LOW_HZ + BP_HIGH_HZ)

    rep_period = 1.0 / f_rep

    # ============================================================
    # Step 4: Chebyshev Type II low-pass for segmentation (paper structure)
    # ============================================================
    # passband edge = f_rep, stopband edge = 2*f_rep
    wp = min(max(f_rep / nyq, 1e-6), 0.999999)
    ws = min(max((STOPBAND_FACTOR * f_rep) / nyq, wp + 1e-6), 0.999999)

    try:
        n_cheb, wn = cheb2ord(wp=wp, ws=ws, gpass=CHEB_PASS_RIPPLE_DB, gstop=CHEB_STOP_ATTEN_DB)
    except Exception:
        n_cheb, wn = 6, ws

    b_lp, a_lp = cheby2(N=n_cheb, rs=CHEB_STOP_ATTEN_DB, Wn=wn, btype="low")
    avm_lp = filtfilt(b_lp, a_lp, avm_padded)
    avm_lp = avm_lp[pad_samples: pad_samples + n] if pad_samples > 0 else avm_lp

    # ============================================================
    # Step 5: Candidate extrema sets (minima and maxima)
    # ============================================================
    min_dist_samples = max(1, int(round(MIN_DIST_FACTOR * rep_period * fs)))
    prominence_thr = PROMINENCE_STD_FACTOR * np.std(avm_lp)

    max_idx, max_props = find_peaks(avm_lp, distance=min_dist_samples, prominence=prominence_thr)
    min_idx, min_props = find_peaks(-avm_lp, distance=min_dist_samples, prominence=prominence_thr)

    # ============================================================
    # Step 6: Choose boundary type (class agnostic scoring)
    # ============================================================
    def score_boundaries(b_idx: np.ndarray, props: dict) -> float:
        if b_idx is None or len(b_idx) < 3:
            return -np.inf

        intervals = np.diff(b_idx) / fs
        med_i = float(np.median(intervals))
        mean_i = float(np.mean(intervals))
        std_i = float(np.std(intervals))
        if mean_i <= 0:
            return -np.inf

        # Regularity: prefer low CV
        cv = std_i / mean_i
        regularity = 1.0 / (1.0 + cv)

        # Sharpness: prefer larger prominence
        prom = props.get("prominences", None)
        mean_prom = float(np.mean(prom)) if prom is not None and len(prom) > 0 else 0.0
        sharpness = mean_prom / (np.std(avm_lp) + 1e-12)

        # Match to expected period
        T = rep_period
        match_T = 1.0 - min(abs(med_i - T) / (T + 1e-12), 1.0)

        # Penalize half-period spacing because it typically leads to 2x counting
        close_half = 1.0 - min(abs(med_i - 0.5 * T) / (0.5 * T + 1e-12), 1.0)

        return (
            SCORE_W_REGULARITY * regularity
            + SCORE_W_SHARPNESS * sharpness
            + SCORE_W_MATCH_T * match_T
            - SCORE_W_PENALIZE_HALF * close_half
        )

    score_max = score_boundaries(max_idx, max_props)
    score_min = score_boundaries(min_idx, min_props)

    if score_max >= score_min:
        boundaries = max_idx.astype(int)
        boundary_mode = "maxima"
    else:
        boundaries = min_idx.astype(int)
        boundary_mode = "minima"

    # ============================================================
    # Step 7: Period-based pruning to prevent double counting
    # ============================================================
    min_sep_sec = PRUNE_MIN_SEP_FACTOR * rep_period

    # Get prominences for chosen polarity so we can drop the weaker boundary if two are too close
    if boundary_mode == "maxima":
        all_idx, all_props = find_peaks(avm_lp, prominence=0)
    else:
        all_idx, all_props = find_peaks(-avm_lp, prominence=0)

    prom_map = {int(i): float(p) for i, p in zip(all_idx, all_props["prominences"])}

    pruned = []
    for b in boundaries:
        b = int(b)
        if not pruned:
            pruned.append(b)
            continue

        if (b - pruned[-1]) / fs >= min_sep_sec:
            pruned.append(b)
        else:
            if prom_map.get(b, 0.0) > prom_map.get(pruned[-1], 0.0):
                pruned[-1] = b

    boundaries = np.array(pruned, dtype=int)

    # ============================================================
    # Step 8: Convert boundaries to segments
    # ============================================================
    segments = []
    for i in range(len(boundaries) - 1):
        s = int(boundaries[i])
        e = int(boundaries[i + 1])
        if e > s:
            segments.append((s, e))

    # ============================================================
    # Step 9: Optional start and end segment removal (paper rule)
    # ============================================================
    if remove_unwanted_segments and segments:
        set_max_amp = float(np.max(avm_lp))
        filtered = []

        for (s, e) in segments:
            dur = (e - s) / fs
            seg_max = float(np.max(avm_lp[s:e]))

            too_short = dur < (ART_DUR_FACTOR * rep_period)
            too_weak = seg_max < (ART_AMP_FACTOR * set_max_amp)

            # Paper style coupled discard
            if too_short and too_weak:
                continue

            filtered.append((s, e))

        segments = filtered

    # ============================================================
    # Step 10: Optional connection of adjacent segments
    # ============================================================
    # Paper uses exercise specific rules for when to connect.
    # Here we use an automatic pause split detector:
    # merge two adjacent short segments if their combined duration is about one rep period.
    if connect_adjacent_segments and len(segments) >= 3:
        durs = np.array([(e - s) / fs for s, e in segments], dtype=float)
        T = rep_period

        short = durs < (MERGE_SHORT_FRAC * T)

        merged = []
        i = 0
        while i < len(segments):
            if i < len(segments) - 1 and short[i] and short[i + 1]:
                sum_d = durs[i] + durs[i + 1]
                if abs(sum_d - T) <= (MERGE_SUM_TOL * T):
                    merged.append((segments[i][0], segments[i + 1][1]))
                    i += 2
                    continue
            merged.append(segments[i])
            i += 1

        segments = merged

    rep_count = len(segments)

    # ============================================================
    # Plotting
    # ============================================================
    if plot:
        t = np.arange(n) / fs

        fig = plt.figure(figsize=(13, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3])

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, x, alpha=0.25, linewidth=0.8, label="x (demeaned)")
        ax1.plot(t, y, alpha=0.25, linewidth=0.8, label="y (demeaned)")
        ax1.plot(t, z, alpha=0.25, linewidth=0.8, label="z (demeaned)")
        ax1.plot(t, avm, linewidth=1.6, label="AVM")
        ax1.set_title("Step 1: Demean axes and compute AVM")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        ax2 = fig.add_subplot(gs[1, 0])
        show = (freqs >= 0) & (freqs <= 4.0)
        ax2.plot(freqs[show], fft_mag[show], linewidth=1.2)
        ax2.axvspan(BP_LOW_HZ, BP_HIGH_HZ, alpha=0.15, label="Search band")
        # Dominant frequency red dot
        ax2.scatter([f_rep], [fft_peak_mag], s=45, marker="o", color="red", label="Dominant peak")
        ax2.set_title("Step 3: Spectrum and dominant repetition frequency")
        ax2.set_xlabel("Hz")
        ax2.set_ylabel("Magnitude")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(t, avm_bp_trim, linewidth=1.2)
        ax3.set_title("Step 2: Band-pass filtered AVM (0.25 to 1.2 Hz)")
        ax3.set_xlabel("Time (s)")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[2, :])
        # Line must be blue
        ax4.plot(t, avm_lp, linewidth=1.8, color="blue", label="Low-pass AVM")

        # Scatter maxima candidates in green and minima candidates in red
        ax4.scatter(t[max_idx], avm_lp[max_idx], s=28, marker="o", color="green", label="Maxima candidates")
        ax4.scatter(t[min_idx], avm_lp[min_idx], s=28, marker="o", color="red", label="Minima candidates")

        # Chosen boundaries
        for b in boundaries:
            ax4.axvline(t[int(b)], linestyle=":", linewidth=1.0, alpha=0.8)

        # Highlight final segments
        for i, (s, e) in enumerate(segments):
            ax4.axvspan(t[s], t[e], alpha=0.18)
            mid = (s + e) // 2
            ax4.text(t[mid], np.max(avm_lp) * 0.9, str(i + 1), ha="center", fontsize=9)

        ax4.set_title(f"Steps 5 to 10: Boundaries ({boundary_mode}) and rep count = {rep_count}")
        ax4.set_xlabel("Time (s)")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="upper right")

        if info:
            plt.suptitle(" ".join(str(s) for s in info), fontsize=14)

        plt.tight_layout()
        plt.show()

    return int(rep_count)
