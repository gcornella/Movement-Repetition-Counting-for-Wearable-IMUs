"""
uLift Repetition Counter - Fixed & Verified
Reference: Lim, J., et al. (2024). uLift... IEEE Access.
"""

import numpy as np
from scipy.signal import correlate, find_peaks
import matplotlib.pyplot as plt
from typing import Tuple, List, Any

# Map activity IDs to readable names
WORKOUT_CLASSES = {
    "00": "SQUAT", "01": "PUSH_UP", "02": "LUNGE", "03": "JUMPING_JACK",
    "04": "BENCH_PRESS", "05": "GOOD_MORNING", "06": "DEAD_LIFT",
    "07": "PUSH_PRESS", "08": "BACK_SQUAT", "09": "ARM_CURL",
    "10": "BB_MILITARY_PRESS", "11": "BB_BENT_OVER_ROW",
    "12": "BURPEE", "13": "LEG_RAISED_CRUNCH", "14": "LATERAL_RAISE"
}


class ULiftRepCounter:
    def __init__(self, sampling_rate: int = 60):
        self.fs = sampling_rate

    @staticmethod
    def _smooth_data_ema(data: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Replicates 'PreProcess.smooth_data' from the uLift authors' repo.
        Formula: y[i] = alpha * y[i-1] + (1-alpha) * x[i]
        """
        smoothed = np.asarray(data, dtype=float).copy()
        prev = smoothed[0]
        for i in range(1, len(smoothed)):
            current = (prev * alpha) + (smoothed[i] * (1.0 - alpha))
            smoothed[i] = current
            prev = current
        return smoothed

    def _compute_acf(self, x: np.ndarray) -> np.ndarray:
        x_centered = x - np.mean(x)
        corr = correlate(x_centered, x_centered, mode='full')
        corr = corr[len(corr) // 2:]
        if corr[0] != 0:
            corr = corr / corr[0]
        return corr

    def _weighted_autocorrelation(self, segment: np.ndarray) -> np.ndarray:
        acfs = []
        weights = []
        for i in range(3):
            acf = self._compute_acf(segment[:, i])
            acfs.append(acf)
            norm_val = np.linalg.norm(acf)
            weights.append(np.exp(norm_val))

        weights = np.array(weights)
        sum_weights = np.sum(weights) if np.sum(weights) > 0 else 1.0

        s_x = np.zeros_like(acfs[0])
        for i in range(3):
            weight_normalized = weights[i] / sum_weights
            s_x += acfs[i] * weight_normalized
        return s_x

    def _get_signal_period(self, s_x: np.ndarray) -> float:
        negative_indices = np.where(s_x < 0)[0]
        if len(negative_indices) > 0:
            start_search_idx = negative_indices[0]
        else:
            start_search_idx = int(0.5 * self.fs)

        min_phys_idx = int(0.4 * self.fs)
        start_search_idx = max(start_search_idx, min_phys_idx)

        peaks, properties = find_peaks(s_x, height=0.1, prominence=0.05)
        valid_peaks = peaks[peaks > start_search_idx]

        if len(valid_peaks) == 0:
            return 0.0
        return valid_peaks[0] / self.fs

    def _calculate_sp_vector(self, data: np.ndarray, window_size_sec: float) -> Tuple[
        np.ndarray, np.ndarray, float, int, float]:
        n_samples = len(data)
        window_samples = int(window_size_sec * self.fs)
        step = int(self.fs * 1.0)

        sp_values = []
        s_x_representative = None
        peak_lag_representative = 0.0
        rep_val_idx = -1
        rep_time_sec = 0.0
        middle_index = n_samples // 2

        if n_samples <= window_samples:
            s_x = self._weighted_autocorrelation(data)
            tau = self._get_signal_period(s_x)
            if tau > 0: sp_values.append(tau)
            return np.array(sp_values), s_x, tau * self.fs, 0, 0.0

        for i in range(0, n_samples - window_samples + 1, step):
            window = data[i: i + window_samples]
            s_x = self._weighted_autocorrelation(window)
            tau = self._get_signal_period(s_x)
            if tau > 0: sp_values.append(tau)

            if i <= middle_index and (i + step) > middle_index:
                s_x_representative = s_x
                peak_lag_representative = tau * self.fs
                if tau > 0:
                    rep_val_idx = len(sp_values) - 1
                rep_time_sec = i / self.fs

        if s_x_representative is None and len(data) >= window_samples:
            s_x_representative = self._weighted_autocorrelation(data[:window_samples])

        return np.array(sp_values), s_x_representative, peak_lag_representative, rep_val_idx, rep_time_sec

    def count_reps(self, acc_data: np.ndarray, plot: bool = False, metadata: str = "") -> int:
        """Main Repetition Counting Pipeline [Section III-C]"""

        # --- 0. PREPROCESSING (FIX: Update acc_data with smoothed values) ---
        x_raw, y_raw, z_raw = acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]

        x = self._smooth_data_ema(x_raw, alpha=0.4)
        y = self._smooth_data_ema(y_raw, alpha=0.4)
        z = self._smooth_data_ema(z_raw, alpha=0.4)

        # IMPORTANT: Re-stack so subsequent steps use the smoothed data
        acc_data = np.stack([x, y, z], axis=1)
        total_time_sec = len(acc_data) / self.fs

        # 1. Dynamic Window Sizing
        window_size_ms = round(total_time_sec * 0.2 * 1000 / 100) * 100
        window_size = window_size_ms / 1000.0
        if window_size < 8.0: window_size = 8.0
        if window_size > total_time_sec: window_size = total_time_sec

        # 2. Workout Rate (WR) Calculation
        sp_vector, s_x_viz, peak_lag_viz, rep_idx, rep_time = self._calculate_sp_vector(acc_data,
                                                                                        window_size_sec=window_size)

        wr = 0.0
        eta_45, eta_95 = 0.0, 0.0
        if len(sp_vector) > 0:
            eta_45 = np.percentile(sp_vector, 45)
            eta_95 = np.percentile(sp_vector, 95)
            wr = (eta_45 + eta_95) / 2.0

        if wr < 0.1:
            if plot: print("No periodic signal found.")
            return 0

        # 3. Axis Selection (Robust Amplitude)
        p95 = np.percentile(acc_data, 95, axis=0)
        p05 = np.percentile(acc_data, 5, axis=0)
        robust_amplitudes = p95 - p05

        dominant_axis_idx = np.argmax(robust_amplitudes)
        axis_names = ['X', 'Y', 'Z']
        axis_used = axis_names[dominant_axis_idx]
        chosen_data = acc_data[:, dominant_axis_idx]

        # 4. Peak Detection with Hysteresis
        upper_threshold = 80
        lower_threshold = 65
        upper_line = np.percentile(chosen_data, upper_threshold)
        lower_line = np.percentile(chosen_data, lower_threshold)

        all_peaks, _ = find_peaks(chosen_data)
        candidate_peaks = all_peaks[chosen_data[all_peaks] >= upper_line]

        rep_idxs = []
        peak_ptr = 0

        # "Eliminated if within 0.7 * WR" (Repo Logic)
        forward_n_samples = int((0.7 * wr) * self.fs)

        while peak_ptr < candidate_peaks.shape[0]:
            first_peak_point = candidate_peaks[peak_ptr]

            # 1. Hysteresis
            future_signal_below = chosen_data[first_peak_point:] < lower_line
            if not np.any(future_signal_below):
                edge_peak_point = len(chosen_data)
            else:
                edge_peak_point = np.argmax(future_signal_below) + first_peak_point

            if edge_peak_point <= first_peak_point:
                edge_peak_point = first_peak_point + 1

            # 2. Refine Peak
            segment_slice = chosen_data[first_peak_point:edge_peak_point]
            true_peak_idx = first_peak_point
            if len(segment_slice) > 0:
                max_peak_offset = np.argmax(segment_slice)
                true_peak_idx = max_peak_offset + first_peak_point
                rep_idxs.append(true_peak_idx)
            else:
                rep_idxs.append(first_peak_point)

            # 3. Distance Constraint
            min_next_peak_idx = true_peak_idx + forward_n_samples

            # --- DEBUGGING REJECTIONS ---
            if plot:
                skipped_mask = (candidate_peaks > true_peak_idx) & (candidate_peaks <= min_next_peak_idx)


            # 4. Advance Pointer
            threshold_idx = max(edge_peak_point, min_next_peak_idx)
            future_peaks_mask = candidate_peaks > threshold_idx
            if not np.any(future_peaks_mask):
                break
            peak_ptr = np.argmax(future_peaks_mask)

        accepted_peaks = np.array(rep_idxs)

        if plot:
            self._plot_ulift_debug(
                x=x, y=y, z=z,
                sp_vector=sp_vector, eta_45=eta_45, eta_95=eta_95,
                S=s_x_viz, peak_lag=peak_lag_viz, rep_idx=rep_idx, rep_time=rep_time,
                chosen_sig=chosen_data,
                candidate_peaks=candidate_peaks,
                accepted_peaks=accepted_peaks,
                axis_used=axis_used,
                upper_line=upper_line,
                lower_line=lower_line,
                info={"WR_seconds": float(wr), "metadata": metadata, "window_size": window_size}
            )

        return int(len(accepted_peaks))

    def _plot_ulift_debug(self, x, y, z, sp_vector, eta_45, eta_95, S, peak_lag, rep_idx, rep_time, chosen_sig,
                          candidate_peaks, accepted_peaks, axis_used, upper_line, lower_line, info):
        t = np.arange(len(x)) / self.fs
        lags_s = np.arange(len(S)) / self.fs if S is not None else np.array([])

        # Force window position
        fig, axs = plt.subplots(4, 1, figsize=(9, 13), sharex=True)
        try:
            mngr = plt.get_current_fig_manager()
            if hasattr(mngr, 'window') and hasattr(mngr.window, 'wm_geometry'):
                mngr.window.wm_geometry("+50+50")
            elif hasattr(mngr, 'window') and hasattr(mngr.window, 'move'):
                mngr.window.move(50, 50)
        except Exception:
            pass

        wr = info.get('WR_seconds', 0)
        meta_txt = f" | {info.get('metadata')}"
        fig.suptitle(f"uLift Final (Smoothed) — WR={wr:.2f}s{meta_txt}", fontsize=14, y=0.97)

        # 1. Raw Data
        axs[0].plot(t, x, label='X', alpha=0.7)
        axs[0].plot(t, y, label='Y', alpha=0.7)
        axs[0].plot(t, z, label='Z', alpha=0.7)
        axs[0].set_title("1. Smoothed Input Data (EMA 0.4)", loc='left')
        axs[0].legend(loc='upper right', fontsize='small')

        # 2. SP Vector
        sp_x_axis = np.arange(len(sp_vector))
        axs[1].plot(sp_x_axis, sp_vector, 'o-', color='teal', label='SP(W)')
        axs[1].axhline(eta_45, color='blue', linestyle='--', label=f'45th %')
        axs[1].axhline(eta_95, color='orange', linestyle='--', label=f'95th %')
        axs[1].axhline(wr, color='red', linewidth=2, label=f'WR')
        axs[1].set_title("2. Workout Rate", loc='left')
        axs[1].legend(loc='upper right', fontsize='small')
        axs[1].grid(True, alpha=0.3)

        # 3. Weighted ACF
        if lags_s.size > 0:
            axs[2].plot(lags_s, S, color='purple', label='S(X)')
            if peak_lag is not None and peak_lag > 0:
                peak_time_s = peak_lag / self.fs
                axs[2].axvline(peak_time_s, linestyle='--', color='k', alpha=0.8)
                axs[2].text(peak_time_s + 0.1, np.nanmax(S) * 0.9, f"Tau ≈ {peak_time_s:.2f}s", ha='left')
            axs[2].set_title(f"3. Weighted ACF (Window @ {rep_time:.2f}s)", loc='left')

        # 4. Counting
        axs[3].plot(t, chosen_sig, label=f'Dominant Axis ({axis_used})', color='black', linewidth=1.5)
        axs[3].axhline(upper_line, linestyle='--', color='green', label='Upper 80%')
        axs[3].axhline(lower_line, linestyle=':', color='red', label='Lower 65%')

        if candidate_peaks.size > 0:
            axs[3].plot(candidate_peaks / self.fs, chosen_sig[candidate_peaks],
                        'o', color='orange', markersize=6, alpha=0.5, label='Candidates')

        if accepted_peaks.size > 0:
            axs[3].plot(accepted_peaks / self.fs, chosen_sig[accepted_peaks],
                        'x', color='lime', markersize=12, markeredgewidth=3, label='Final Reps')

        axs[3].set_title(f"4. Counting (Filter: 0.7*WR = {0.7 * wr:.2f}s)", loc='left')
        axs[3].legend(loc='upper right', fontsize='small')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


def method6_fcn(x_lp, y_lp, z_lp, fs=60, plot=True, info=None):
    MIN_LEN = 30  # small safety margin, skip if data is too short

    if len(x_lp) < MIN_LEN or len(y_lp) < MIN_LEN or len(z_lp) < MIN_LEN:
        return 0

    x = np.array(x_lp)
    y = np.array(y_lp)
    z = np.array(z_lp)
    acc_data = np.stack([x, y, z], axis=1)

    metadata_str = ""
    if info and isinstance(info, list) and len(info) >= 2:
        user_id = info[0]
        task_id = info[1]
        task_str = str(task_id)
        task_name = task_str
        if task_str in WORKOUT_CLASSES:
            task_name = WORKOUT_CLASSES[task_str]
        elif task_str.zfill(2) in WORKOUT_CLASSES:
            task_name = WORKOUT_CLASSES[task_str.zfill(2)]
        metadata_str = f"User: {user_id}, Task: {task_name}"

    counter = ULiftRepCounter(sampling_rate=fs)
    reps = counter.count_reps(
        acc_data,
        plot=plot,
        metadata=metadata_str
    )
    return reps