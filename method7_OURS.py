"""
============================================================
Method 7 Family: Hybrid WR + PCA + Period-Constrained Peak Counting
============================================================

Concept:
These methods combine ideas from uLift (period estimation),
RecoFit (peak pruning logic), and dimensionality reduction
(PCA) to create a class-agnostic repetition counter.

The core idea is:
- reduce 3D motion to a robust 1D repetition waveform,
- estimate the typical repetition period from signal periodicity,
- use that period to constrain peak detection and avoid double counting.

------------------------------------------------------------
Common pipeline (all variants)
------------------------------------------------------------

1) Read tri-axial accelerometer signals (x, y, z)

2) Low-pass filter each axis
   - Removes high-frequency noise and sub-movement artifacts
   - Keeps dominant repetition-scale motion

3) Form 3D signal and compute PCA
   - PCA finds dominant motion direction automatically
   - PC1 captures largest variance → often aligned with main exercise axis
   - Avoids manually selecting axes and improves robustness to orientation

4) Estimate repetition period using uLift-style weighted autocorrelation
   - Compute autocorrelation per axis
   - Combine axes via weighted sum (uLift concept)
   - Extract dominant lag → typical repetition duration (WR)
   - Convert to period in samples: P = WR * fs
   NOTE:
   - Autocorrelation is NOT used for counting peaks directly.
   - It only provides a physics-based spacing constraint.

5) Build 1D repetition waveform = PCA PC1 signal
   - Peak counting operates on this waveform.

------------------------------------------------------------
Method 7a: Adaptive peaks + RecoFit-inspired pruning
------------------------------------------------------------

Goal:
Adaptive detection when amplitude varies across repetitions.

Steps:
6) Estimate local signal scale using rolling MAD
   - Creates adaptive prominence threshold
   - Allows peak detection to adapt to changing intensity

7) Detect candidate peaks using adaptive prominence

8) Apply RecoFit-inspired pruning:
   - Greedy amplitude-ranked selection
   - Enforce minimum spacing (~0.65–0.75 * P)
   WHY "RecoFit-inspired":
   - Similar strategy: sort peaks by amplitude,
     iteratively remove nearby peaks to avoid duplicates.

9) Apply amplitude percentile rule
   - Remove small peaks relative to reference peak amplitude

10) Repetition count = number of remaining peaks


------------------------------------------------------------
Method 7b: uLift spacing constraint on PCA peaks
------------------------------------------------------------

Goal:
Simpler approach emphasizing temporal spacing rather than amplitude filtering.

Steps:
6) Detect peaks on PC1 using adaptive prominence

7) Apply spacing rule derived from uLift:
   - Keep peaks separated by >= 0.7 * P
   - WR provides expected cycle duration

8) Repetition count = accepted peaks

Key difference vs 7a:
- No RecoFit amplitude ranking or percentile filtering.
- Uses only temporal consistency constraint.


------------------------------------------------------------
Method 7c: Minimal hybrid (RecoFit-style with WR period)
------------------------------------------------------------

Goal:
Simple baseline using strong spacing logic without adaptive thresholds.

Steps:
6) Detect raw peaks on PC1 (basic detection)

7) Apply RecoFit-style distance gating (~0.75 * P)

8) Apply amplitude threshold (percentile-based)

9) Repetition count = filtered peaks

Key difference vs 7a:
- No local adaptive prominence (fixed peak detection).
- Relies purely on period-based pruning.


------------------------------------------------------------
Summary of design choices
------------------------------------------------------------

Why PCA:
- Wrist orientation changes across users/exercises.
- PCA automatically extracts dominant motion direction.
- Produces cleaner rep waveform than single-axis selection.

Why uLift WR:
- Provides robust period estimate from periodicity.
- Avoids manual tuning of rep duration ranges.

Why RecoFit-inspired pruning:
- Prevents double counting caused by sub-peaks.
- Uses amplitude-ranked greedy selection with spacing constraints.

"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from scipy.signal import find_peaks, correlate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils import lowpass_filter


# ============================================================
# Helpers
# ============================================================

def _stack_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    return np.stack([x, y, z], axis=1)


def pca_first_pc(acc_3d: np.ndarray) -> np.ndarray:
    """
    Mean-center axes then PCA to PC1.
    acc_3d: (N, 3)
    Returns: (N,) PC1 signal.
    """
    acc_3d = np.asarray(acc_3d, dtype=float)
    acc_3d = acc_3d - np.mean(acc_3d, axis=0, keepdims=True)

    pca = PCA(n_components=3, svd_solver="full")
    pca.fit(acc_3d)
    return pca.transform(acc_3d)[:, 0]


def ulift_compute_acf_1d(x: np.ndarray) -> np.ndarray:
    """
    uLift-style normalized autocorrelation (non-negative lags).
    """
    x = np.asarray(x, dtype=float)
    x_centered = x - np.mean(x)
    ac = correlate(x_centered, x_centered, mode="full")
    ac = ac[ac.size // 2:]
    if ac.size > 0 and ac[0] != 0:
        ac = ac / ac[0]
    return ac


def ulift_weighted_acf(acc_3d: np.ndarray) -> np.ndarray:
    """
    uLift weighted autocorrelation S(X) across 3 axes.
    acc_3d shape: (N, 3)
    """
    acc_3d = np.asarray(acc_3d, dtype=float)

    acfs = []
    weights = []
    for i in range(3):
        acf = ulift_compute_acf_1d(acc_3d[:, i])
        acfs.append(acf)
        weights.append(np.exp(np.linalg.norm(acf)))

    weights = np.asarray(weights, dtype=float)
    sum_w = float(np.sum(weights)) if float(np.sum(weights)) > 0 else 1.0

    s_x = np.zeros_like(acfs[0])
    for i in range(3):
        s_x += acfs[i] * (weights[i] / sum_w)

    return s_x


def ulift_get_tau_seconds_from_sx(s_x: np.ndarray, fs: float) -> float:
    """
    uLift-like period pick: first valid ACF peak after a start index.
    Returns tau in seconds, or 0.0 if none.
    """
    s_x = np.asarray(s_x, dtype=float)

    neg_idx = np.where(s_x < 0)[0]
    if len(neg_idx) > 0:
        start_search_idx = int(neg_idx[0])
    else:
        start_search_idx = int(0.5 * fs)

    min_phys_idx = int(0.4 * fs)
    start_search_idx = max(start_search_idx, min_phys_idx)

    peaks, _ = find_peaks(s_x, height=0.1, prominence=0.05)
    valid = peaks[peaks > start_search_idx]
    if len(valid) == 0:
        return 0.0

    return float(valid[0]) / float(fs)


def ulift_wr_seconds(
    acc_3d: np.ndarray,
    fs: float,
    window_size_sec: float,
    step_sec: float = 1.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Approx uLift WR:
      - sliding windows
      - each window: weighted ACF -> tau (seconds)
      - sp_vector = list of taus
      - WR = (p45 + p95) / 2
    """
    acc_3d = np.asarray(acc_3d, dtype=float)
    n = len(acc_3d)

    win = int(round(window_size_sec * fs))
    step = int(round(step_sec * fs))
    if step <= 0:
        step = int(fs)

    sp_vals: List[float] = []
    s_x_rep = None
    tau_rep = 0.0
    rep_time_sec = 0.0

    if win <= 0:
        return 0.0, {"sp_vector": np.array([]), "s_x_rep": None, "tau_rep": 0.0, "rep_time_sec": 0.0}

    if n <= win:
        s_x = ulift_weighted_acf(acc_3d)
        tau = ulift_get_tau_seconds_from_sx(s_x, fs)
        if tau > 0:
            sp_vals.append(tau)
        s_x_rep = s_x
        tau_rep = tau
    else:
        mid = n // 2
        for i in range(0, n - win + 1, step):
            w = acc_3d[i:i + win]
            s_x = ulift_weighted_acf(w)
            tau = ulift_get_tau_seconds_from_sx(s_x, fs)
            if tau > 0:
                sp_vals.append(tau)

            if i <= mid < (i + step):
                s_x_rep = s_x
                tau_rep = tau
                rep_time_sec = float(i) / float(fs)

        if s_x_rep is None:
            s_x_rep = ulift_weighted_acf(acc_3d[:win])
            tau_rep = ulift_get_tau_seconds_from_sx(s_x_rep, fs)

    sp_vector = np.asarray(sp_vals, dtype=float)
    if sp_vector.size == 0:
        return 0.0, {"sp_vector": sp_vector, "s_x_rep": s_x_rep, "tau_rep": tau_rep, "rep_time_sec": rep_time_sec}

    eta_45 = float(np.percentile(sp_vector, 45))
    eta_95 = float(np.percentile(sp_vector, 95))
    wr = 0.5 * (eta_45 + eta_95)

    dbg = {
        "sp_vector": sp_vector,
        "eta_45": eta_45,
        "eta_95": eta_95,
        "wr": wr,
        "s_x_rep": s_x_rep,
        "tau_rep": tau_rep,
        "rep_time_sec": rep_time_sec
    }
    return wr, dbg


def recofit_style_peak_prune(
    signal_1d: np.ndarray,
    period_samples: int,
    min_dist_scale: float = 0.75,
    final_dist_scale: float = 0.75,
    amp_ref_rank_frac: float = 0.40,
    amp_keep_ratio: float = 0.50
) -> Dict[str, Any]:
    """
    RecoFit-inspired pruning driven by a provided period_samples.
    """
    s = np.asarray(signal_1d, dtype=float)

    cand, _ = find_peaks(s)
    if cand.size == 0:
        return {
            "candidates": np.array([], dtype=int),
            "accepted": np.array([], dtype=int),
            "final": np.array([], dtype=int),
            "filtered": np.array([], dtype=int),
            "period_samples": int(max(1, period_samples))
        }

    sorted_cand = sorted(cand, key=lambda i: s[i], reverse=True)

    P = int(max(1, period_samples))
    min_dist = int(max(1, round(min_dist_scale * P)))
    final_dist = float(max(1, final_dist_scale * P))

    accepted = []
    for pk in sorted_cand:
        if all(abs(pk - ap) >= min_dist for ap in accepted):
            accepted.append(pk)

    final = []
    for pk in accepted:
        if all(abs(pk - fp) > final_dist for fp in final):
            final.append(pk)

    if len(final) == 0:
        return {
            "candidates": cand.astype(int),
            "accepted": np.asarray(accepted, dtype=int),
            "final": np.array([], dtype=int),
            "filtered": np.array([], dtype=int),
            "period_samples": P
        }

    final_sorted = sorted(final, key=lambda i: s[i], reverse=True)
    if len(final_sorted) == 1:
        ref_idx = 0
    else:
        ref_idx = int(round(amp_ref_rank_frac * (len(final_sorted) - 1)))
        ref_idx = int(np.clip(ref_idx, 0, len(final_sorted) - 1))

    ref_amp = float(s[final_sorted[ref_idx]])
    thr = amp_keep_ratio * ref_amp
    filtered = [i for i in final_sorted if s[i] >= thr]

    return {
        "candidates": cand.astype(int),
        "accepted": np.asarray(accepted, dtype=int),
        "final": np.asarray(final, dtype=int),
        "filtered": np.asarray(filtered, dtype=int),
        "period_samples": P,
        "amp_ref": ref_amp,
        "amp_thr": thr
    }


def _rolling_mad(x: np.ndarray, win: int) -> np.ndarray:
    """
    Robust rolling scale estimate using MAD.
    Returns an array length N with local MAD. Uses symmetric window via slicing.
    O(N*win) but win is small (period scale), acceptable for offline.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    win = int(max(5, win))
    half = win // 2
    out = np.zeros(n, dtype=float)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        w = x[a:b]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        out[i] = 1.4826 * mad
    return out


def _local_prominence_thresholds(signal: np.ndarray, fs: float, P: int) -> np.ndarray:
    """
    Build a local prominence threshold using rolling MAD and a period-scaled window.
    """
    s = np.asarray(signal, dtype=float)
    win = int(max(9, round(1.5 * P)))
    local_scale = _rolling_mad(s, win=win)
    floor = 1e-9 + 0.05 * np.std(s) if np.std(s) > 0 else 1e-9
    local_scale = np.maximum(local_scale, floor)
    return local_scale


def _plot_method7_1d(title: str, fs: float, signal_main: np.ndarray, peaks_dict: Dict[str, Any]) -> None:
    s = np.asarray(signal_main, dtype=float)
    t = np.arange(len(s)) / float(fs)

    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.plot(t, s, linewidth=1.2, label="signal")

    cand = peaks_dict.get("candidates", np.array([], dtype=int))
    acc = peaks_dict.get("accepted", np.array([], dtype=int))
    fin = peaks_dict.get("final", np.array([], dtype=int))
    filt = peaks_dict.get("filtered", np.array([], dtype=int))

    if cand.size > 0:
        plt.scatter(cand / fs, s[cand], s=12, label="candidates", alpha=0.6)
    if acc.size > 0:
        plt.scatter(acc / fs, s[acc], s=18, label="accepted", alpha=0.8)
    if fin.size > 0:
        plt.scatter(fin / fs, s[fin], s=22, label="final", alpha=0.9)
    if filt.size > 0:
        plt.scatter(filt / fs, s[filt], s=34, label="filtered", alpha=1.0)

    P = int(peaks_dict.get("period_samples", 0))
    if P > 0:
        plt.xlabel(f"time (s)   |   P≈{P / fs:.2f}s")
    else:
        plt.xlabel("time (s)")

    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.show()


def _plot_ulift_panels(
    title: str,
    fs: float,
    sp_vector: np.ndarray,
    eta_45: float,
    eta_95: float,
    wr: float,
    s_x_rep: Optional[np.ndarray],
    tau_rep: float
) -> None:
    nrows = 2 if s_x_rep is not None else 1
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 3.2 * nrows), sharex=False)
    if nrows == 1:
        axs = [axs]

    fig.suptitle(title, y=0.98)

    ax = axs[0]
    sp = np.asarray(sp_vector, dtype=float)
    ax.plot(np.arange(len(sp)), sp, "o-")
    ax.axhline(eta_45, linestyle="--", label="p45")
    ax.axhline(eta_95, linestyle="--", label="p95")
    ax.axhline(wr, linewidth=2, label="WR")
    ax.set_title("Workout Rate (uLift-style)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    ax.set_xlabel("window index (step=1s)")

    if s_x_rep is not None:
        S = np.asarray(s_x_rep, dtype=float)
        lags = np.arange(len(S)) / float(fs)
        ax2 = axs[1]
        ax2.plot(lags, S)
        if tau_rep and tau_rep > 0:
            ax2.axvline(tau_rep, linestyle="--")
        ax2.set_title("Representative weighted autocorrelation S(X)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("lag (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================
# method7a improved
# ============================================================

def method7a_fcn(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fs: float,
    plot: bool = True,
    info: Optional[list] = None,
    lowpass_hz: float = 2.0
) -> int:
    """
    method7a (improved):
    - lowpass x,y,z
    - PCA -> PC1
    - uLift WR -> period P
    - candidate peaks from PC1 using local (rolling MAD) prominence threshold
    - RecoFit-style distance gating using P
    - light amplitude screen (RecoFit percentile rule)

    Why this helps vs old 7a:
    - width-only filters often fail when rep shapes vary
    - local prominence adapts to changing amplitude across the set
    - distance gating uses a physics-derived period from WR
    """
    MIN_LEN = 60
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    if len(x) < MIN_LEN or len(y) < MIN_LEN or len(z) < MIN_LEN:
        return 0

    acc = _stack_xyz(x, y, z)
    acc_lp = lowpass_filter(acc, fs=fs, cutoff_hz=lowpass_hz, order=4, zero_phase=True)

    sig = pca_first_pc(acc_lp)

    total_time_sec = len(acc_lp) / float(fs)
    window_size_sec = float(np.clip(0.2 * total_time_sec, 8.0, total_time_sec))
    wr, dbg = ulift_wr_seconds(acc_lp, fs=fs, window_size_sec=window_size_sec, step_sec=1.0)
    if wr < 0.1:
        return 0

    P = int(max(1, round(wr * fs)))

    local_scale = _local_prominence_thresholds(sig, fs=fs, P=P)
    prom = float(np.median(local_scale)) * 1.2

    candidates, _ = find_peaks(sig, prominence=prom)
    if candidates.size == 0:
        return 0

    peaks_dict = recofit_style_peak_prune(
        signal_1d=sig,
        period_samples=P,
        min_dist_scale=0.65,
        final_dist_scale=0.75,
        amp_ref_rank_frac=0.40,
        amp_keep_ratio=0.45
    )

    if plot:
        meta = f" | {info[0]} {info[1]}" if info and isinstance(info, list) and len(info) >= 2 else ""
        _plot_ulift_panels(
            title=f"Method7a WR panels (LP {lowpass_hz}Hz){meta}",
            fs=fs,
            sp_vector=dbg.get("sp_vector", np.array([])),
            eta_45=float(dbg.get("eta_45", 0.0)),
            eta_95=float(dbg.get("eta_95", 0.0)),
            wr=float(dbg.get("wr", wr)),
            s_x_rep=dbg.get("s_x_rep", None),
            tau_rep=float(dbg.get("tau_rep", 0.0))
        )
        _plot_method7_1d(
            title=f"Method7a improved (LP {lowpass_hz}Hz + PCA + local-prom + RecoFit prune) | P≈{P/fs:.2f}s{meta}",
            fs=fs,
            signal_main=sig,
            peaks_dict={**peaks_dict, "candidates": candidates.astype(int)}
        )

    filtered = peaks_dict.get("filtered", np.array([], dtype=int))
    return int(len(filtered))


# ============================================================
# method7b improved
# ============================================================

def method7b_fcn(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fs: float,
    plot: bool = True,
    info: Optional[list] = None,
    lowpass_hz: float = 2.0
) -> int:
    """
    method7b (improved):
    - lowpass x,y,z
    - uLift WR -> period P
    - build PC1 signal (RecoFit strength)
    - use uLift rule for spacing (0.7*WR) but applied on PC1 peaks (not sp_vector peaks)
    - peak candidates use a local, adaptive prominence threshold (rolling MAD)

    """
    MIN_LEN = 60
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    if len(x) < MIN_LEN or len(y) < MIN_LEN or len(z) < MIN_LEN:
        return 0

    acc = _stack_xyz(x, y, z)
    acc_lp = lowpass_filter(acc, fs=fs, cutoff_hz=lowpass_hz, order=4, zero_phase=True)

    total_time_sec = len(acc_lp) / float(fs)
    window_size_sec = float(np.clip(0.2 * total_time_sec, 8.0, total_time_sec))
    wr, dbg = ulift_wr_seconds(acc_lp, fs=fs, window_size_sec=window_size_sec, step_sec=1.0)
    if wr < 0.1:
        return 0

    P = int(max(1, round(wr * fs)))
    min_dist = int(max(1, round(0.7 * wr * fs)))  # uLift rule in samples

    sig = pca_first_pc(acc_lp)

    local_scale = _local_prominence_thresholds(sig, fs=fs, P=P)
    prom = float(np.median(local_scale)) * 1.0

    candidates, _ = find_peaks(sig, prominence=prom, distance=max(1, int(0.45 * P)))
    if candidates.size == 0:
        return 0

    sorted_cand = sorted(candidates, key=lambda i: sig[i], reverse=True)
    accepted = []
    for pk in sorted_cand:
        if all(abs(pk - ap) >= min_dist for ap in accepted):
            accepted.append(pk)

    accepted = np.asarray(sorted(accepted), dtype=int)

    peaks_dict = {
        "candidates": candidates.astype(int),
        "accepted": accepted,
        "final": accepted,
        "filtered": accepted,
        "period_samples": P
    }

    if plot:
        meta = f" | {info[0]} {info[1]}" if info and isinstance(info, list) and len(info) >= 2 else ""
        _plot_ulift_panels(
            title=f"Method7b WR panels (LP {lowpass_hz}Hz){meta}",
            fs=fs,
            sp_vector=dbg.get("sp_vector", np.array([])),
            eta_45=float(dbg.get("eta_45", 0.0)),
            eta_95=float(dbg.get("eta_95", 0.0)),
            wr=float(dbg.get("wr", wr)),
            s_x_rep=dbg.get("s_x_rep", None),
            tau_rep=float(dbg.get("tau_rep", 0.0))
        )
        _plot_method7_1d(
            title=f"Method7b improved (LP {lowpass_hz}Hz + PCA peaks + 0.7*WR spacing) | P≈{P/fs:.2f}s{meta}",
            fs=fs,
            signal_main=sig,
            peaks_dict=peaks_dict
        )

    return int(len(accepted))


# ============================================================
# method7c (unchanged)
# ============================================================

def method7c_fcn(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fs: float,
    plot: bool = True,
    info: Optional[list] = None,
    lowpass_hz: float = 1.5
) -> int:
    """
    method7c:
    - lowpass filter x,y,z at lowpass_hz using utils.lowpass_filter
    - PCA -> PC1 as 1D signal
    - compute WR from uLift on lowpassed 3D, use P = WR*fs
    - prune peaks RecoFit-style using P (no minPeriod/maxPeriod tuning)
    """
    MIN_LEN = 60
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    if len(x) < MIN_LEN or len(y) < MIN_LEN or len(z) < MIN_LEN:
        return 0

    acc = _stack_xyz(x, y, z)
    acc_lp = lowpass_filter(acc, fs=fs, cutoff_hz=lowpass_hz, order=4, zero_phase=True)

    sig = pca_first_pc(acc_lp)

    total_time_sec = len(acc_lp) / float(fs)
    window_size_sec = float(np.clip(0.2 * total_time_sec, 8.0, total_time_sec))

    wr, dbg = ulift_wr_seconds(acc_lp, fs=fs, window_size_sec=window_size_sec, step_sec=1.0)
    if wr < 0.1:
        return 0

    P = int(max(1, round(wr * fs)))

    peaks_dict = recofit_style_peak_prune(
        signal_1d=sig,
        period_samples=P,
        min_dist_scale=0.75,
        final_dist_scale=0.75,
        amp_ref_rank_frac=0.40,
        amp_keep_ratio=0.50
    )

    if plot:
        meta = f" | {info[0]} {info[1]}" if info and isinstance(info, list) and len(info) >= 2 else ""
        _plot_ulift_panels(
            title=f"Method7c WR panels (LP {lowpass_hz}Hz){meta}",
            fs=fs,
            sp_vector=dbg.get("sp_vector", np.array([])),
            eta_45=float(dbg.get("eta_45", 0.0)),
            eta_95=float(dbg.get("eta_95", 0.0)),
            wr=float(dbg.get("wr", wr)),
            s_x_rep=dbg.get("s_x_rep", None),
            tau_rep=float(dbg.get("tau_rep", 0.0))
        )
        _plot_method7_1d(
            title=f"Method7c (LP {lowpass_hz}Hz + PCA + RecoFit prune with uLift P) | P≈{P/fs:.2f}s{meta}",
            fs=fs,
            signal_main=sig,
            peaks_dict=peaks_dict
        )

    filtered = peaks_dict.get("filtered", np.array([], dtype=int))
    return int(len(filtered))
