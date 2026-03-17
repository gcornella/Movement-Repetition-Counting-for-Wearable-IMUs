# eval_repcount_methods_ODRA_single_overall.py
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
PREFERRED_METHODS = [
    "method1", "method2", "method3a", "method3b", "method4b",
    "method5", "method6", "method7a","method7b","method7c"
]

SKIP_SET = {""}
Z_THRESH = 2.0

SCRIPT_DIR = Path(r"C:\Users\gcorn\PycharmProjects\FitmiSamsungCorrelation")

ODRA_PATH = SCRIPT_DIR / "repsfinal" / "Datasets" / "ODRA.json"
ALL_METHODS_PATH = SCRIPT_DIR / "paperReadyResultsFinal" / "ODRA_estimated_reps.json"

OUT_DIR = SCRIPT_DIR / "paperReadyResultsFinal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_OUT = OUT_DIR / "ODRA_Overall_MetricsExcel.xlsx"
PLOT_OUT = OUT_DIR / "ODRA_Overall_MetricsPlot.png"

# Metrics
METRIC_ORDER = ["ME", "MAE", "RMSE", "MAPE", "EXACT", "ACC1", "ACC2", "ACC3"]

METRIC_LABELS = {
    "ME": "ME (reps)",
    "MAE": "MAE (reps)",
    "RMSE": "RMSE (reps)",
    "MAPE": "MAPE (%)",
    "EXACT": "Exact match (%)",
    "ACC1": "Acc ±1 (%)",
    "ACC2": "Acc ±2 (%)",
    "ACC3": "Acc ±3 (%)",
}

# -----------------------------
# Load ODRA + predictions
# -----------------------------
with ODRA_PATH.open("r", encoding="utf-8") as f:
    odra = json.load(f)

with ALL_METHODS_PATH.open("r", encoding="utf-8") as f:
    counts_all_methods = json.load(f)

print("Loaded ODRA + estimated reps JSONs.")

METHODS = [m for m in PREFERRED_METHODS if m in counts_all_methods]
DATASETS = sorted(odra.keys())

# -----------------------------
# Helpers
# -----------------------------
def _skip_activity(label: str) -> bool:
    if label in SKIP_SET:
        return True
    return len(label) > 0 and label[0].upper() == "H"

def _pairs_to_arrays(pairs):
    if not pairs:
        return np.array([]), np.array([])
    gt_vals = np.array([g for g, _ in pairs], dtype=float)
    pr_vals = np.array([p for _, p in pairs], dtype=float)
    return gt_vals, pr_vals

def _trim_by_ae(gt_vals, pr_vals, z_thresh=Z_THRESH):
    if gt_vals.size == 0:
        return gt_vals, pr_vals, 0, 0

    ae = np.abs(pr_vals - gt_vals)
    mean_ae = np.mean(ae)
    std_ae = np.std(ae)

    if std_ae == 0.0:
        return gt_vals, pr_vals, int(gt_vals.size), int(gt_vals.size)

    keep = np.abs(ae - mean_ae) <= z_thresh * std_ae
    kept = np.count_nonzero(keep)
    return gt_vals[keep], pr_vals[keep], int(gt_vals.size), int(kept)

def compute_metrics_from_pairs(pairs, z_thresh=Z_THRESH):
    gt_vals, pr_vals = _pairs_to_arrays(pairs)
    gt_vals, pr_vals, n_raw, n_kept = _trim_by_ae(gt_vals, pr_vals, z_thresh=z_thresh)

    if gt_vals.size == 0:
        out = {k: np.nan for k in METRIC_ORDER}
        out["N_raw"] = n_raw
        out["N_kept"] = n_kept
        return out

    err = pr_vals - gt_vals
    abs_err = np.abs(err)

    me = float(np.mean(err))
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    mask_nz = gt_vals != 0
    if np.any(mask_nz):
        mape = float(np.mean(abs_err[mask_nz] / np.abs(gt_vals[mask_nz])) * 100.0)
    else:
        mape = np.nan

    exact = float(np.mean(abs_err == 0) * 100.0)
    acc1 = float(np.mean(abs_err <= 1) * 100.0)
    acc2 = float(np.mean(abs_err <= 2) * 100.0)
    acc3 = float(np.mean(abs_err <= 3) * 100.0)

    return {
        "ME": me,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "EXACT": exact,
        "ACC1": acc1,
        "ACC2": acc2,
        "ACC3": acc3,
        "N_raw": n_raw,
        "N_kept": n_kept,
    }

def iterate_pairs_for_method_across_all_datasets(method_name: str):
    """
    Yield (gt, pred) across ALL datasets combined.
    Uses:
      odra[dataset]["reps"][user][task]
      counts_all_methods[method][dataset][user][task]
    """
    method_block = counts_all_methods.get(method_name, {})

    for ds in DATASETS:
        gt_block = odra.get(ds, {}).get("reps", {})
        pred_block = method_block.get(ds, {})

        if not isinstance(pred_block, dict):
            continue

        for user_id, tasks in pred_block.items():
            if not isinstance(tasks, dict):
                continue

            gt_user = gt_block.get(user_id, {})
            if not isinstance(gt_user, dict):
                continue

            for task, pred_val in tasks.items():
                if _skip_activity(task):
                    continue
                if task not in gt_user:
                    continue
                gt_val = gt_user[task]
                if gt_val is None:
                    continue
                yield float(gt_val), float(pred_val)

# -----------------------------
# Compute ODRA overall metrics per method (all datasets pooled)
# -----------------------------
overall_rows = []
for m in METHODS:
    pairs = list(iterate_pairs_for_method_across_all_datasets(m))
    metrics = compute_metrics_from_pairs(pairs, z_thresh=Z_THRESH)
    metrics["method"] = m
    overall_rows.append(metrics)

    print(
        f"{m}: N_raw={metrics['N_raw']} N_kept={metrics['N_kept']}  "
        f"ME={metrics['ME']:.3f}  MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  "
        f"MAPE={metrics['MAPE']:.2f}%  Exact={metrics['EXACT']:.1f}%  "
        f"Acc@1={metrics['ACC1']:.1f}%  Acc@2={metrics['ACC2']:.1f}%  Acc@3={metrics['ACC3']:.1f}%"
    )

overall_df = pd.DataFrame(overall_rows).set_index("method")
overall_df = overall_df[["N_raw", "N_kept"] + METRIC_ORDER]

# -----------------------------
# Save to Excel
# -----------------------------
with pd.ExcelWriter(EXCEL_OUT, mode="w") as writer:
    overall_df.to_excel(writer, sheet_name="ODRA_Overall_AllMetrics")

print(f"Saved ODRA overall metrics Excel -> {EXCEL_OUT}")

# -----------------------------
# Plot: single ODRA figure (8 subplots)
# -----------------------------
def _annotate_bars(ax, xs, vals, fmt, y_is_percent: bool):
    for xi, v in zip(xs, vals):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    if y_is_percent:
        ax.set_ylim(bottom=0)

methods_idx = overall_df.index.tolist()
x = np.arange(len(methods_idx))

fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
axes = axes.ravel()

for i, metric in enumerate(METRIC_ORDER):
    ax = axes[i]
    vals = overall_df[metric].values

    ax.bar(x, vals, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_idx, rotation=30, ha="right")
    ax.set_ylabel(METRIC_LABELS[metric])

    is_percent = metric in {"MAPE", "EXACT", "ACC1", "ACC2", "ACC3"}
    if metric == "ME":
        ax.axhline(0, color="k", linewidth=0.8)

    fmt = "{:.1f}%" if is_percent else "{:.2f}"
    _annotate_bars(ax, x, vals, fmt, y_is_percent=is_percent)

fig.suptitle("ODRA: Overall Metrics by Method (All datasets pooled)", fontsize=14)
fig.savefig(PLOT_OUT, dpi=200)
plt.show()

print(f"Saved ODRA overall plot -> {PLOT_OUT}")
print("Done.")
