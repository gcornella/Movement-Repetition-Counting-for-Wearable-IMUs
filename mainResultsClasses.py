# odra_class_heatmaps_overall_metrics.py
# Creates 8 mini heatmaps (ME, MAE, RMSE, MAPE, EXACT, ACC1, ACC2, ACC3)
# Rows = methods (paper names), Cols = classes (fitness, rehab, adl)
# Cell value = class-aggregate metric across all datasets in that class
# Coloring = robust min-max (with clipping) where green = best, red = worst
# Yellow box = SINGLE best method per class column (for each metric)
#
# Notes:
# - For ME, "best" means abs(ME) closest to 0 (unbiased), not lowest.
# - For MAE/RMSE/MAPE, lower is better.
# - For EXACT/ACC1/ACC2/ACC3, higher is better.
# - Aggregation across datasets within a class is equal-weighted by dataset by default (recommended).
#   You can switch to pair-weighted if you want.

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------
# Config
# -----------------------------
RESULTS_DIR = Path(r"C:\Users\gcorn\PycharmProjects\FitmiSamsungCorrelation\paperReadyResults")
SHEET_NAME = "Overall_AllMetrics"

# Datasets -> class
DATASET_CLASS = {
    "recofit": "fitness",
    "crossfit": "fitness",
    "mmfit": "fitness",
    "uLift": "fitness",
    "CaraCount": "fitness",

    "JUIMU_ROM_ND": "rehab",
    "JUIMU_ROM_Stroke": "rehab",
    "ODRA-Rehab": "rehab",

    "uzh_healthy": "adl",
    "uzh_stroke": "adl",
}

# Class order (columns)
CLASS_ORDER = ["fitness", "rehab", "adl"]

# Methods and paper-friendly labels (rows)
METHOD_LABELS = {
    "method1": "RecoFit",
    "method2": "FitCoach",
    "method3a": "MiLift Autocorr (a)",
    "method3b": "MiLift Revisit (b)",
    "method4a": "Threshold Crossing",
    "method4b": "Threshold Cross LPF",
    "method5": "ExerSense",
    "method6": "Shimmer3",
    "method7": "LEAN",
    "method8": "uLift",
    "method9": "OURS",
}

# Ignore switches (toggle later)
IGNORE_METHOD5 = True
IGNORE_METHOD7 = True

# Desired method order in the figure
METHOD_ORDER = [
    "method1", "method2", "method3a", "method3b",
    "method4a", "method4b",
    "method6",
    "method8",
    "method9",
    # "method5", "method7" (optional later)
]

METRICS = ["ME", "MAE", "RMSE", "MAPE", "EXACT", "ACC1", "ACC2", "ACC3"]

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

# Directionality for "best"
BEST_MODE = {
    "ME": "abs_low",   # best is |ME| closest to 0
    "MAE": "low",
    "RMSE": "low",
    "MAPE": "low",
    "EXACT": "high",
    "ACC1": "high",
    "ACC2": "high",
    "ACC3": "high",
}

# Robust clipping for colormap only (raw values in cells stay unchanged)
# Choose: "percentile" (recommended), "std", or None
CLIP_MODE = "percentile"  # "std" or None
CLIP_PCT_LO, CLIP_PCT_HI = 5, 95
CLIP_STD_K = 2.0

# Class aggregation:
# - "dataset_equal": average metrics across datasets within class (each dataset weight 1)
# - "pair_weighted": weighted by N_kept (or N_raw) per dataset (bigger dataset dominates)
AGG_MODE = "dataset_equal"   # recommended
WEIGHT_COL = "N_kept"        # used only if AGG_MODE == "pair_weighted"

# Output
OUT_PNG = RESULTS_DIR / "ODRA_class_heatmaps_overall_metrics.png"
OUT_XLS = RESULTS_DIR / "ODRA_class_summary_overall_metrics.xlsx"

# Highlight style
HIGHLIGHT_EDGE = "yellow"

# -----------------------------
# Helpers
# -----------------------------
def find_excel_path(dataset_name: str) -> Path | None:
    p = RESULTS_DIR / f"{dataset_name}_MetricsExcel.xlsx"
    return p if p.exists() else None

def fmt_cell(metric: str, v: float) -> str:
    if not np.isfinite(v):
        return ""
    if metric in ("EXACT", "ACC1", "ACC2", "ACC3", "MAPE"):
        return f"{v:.1f}"
    if metric == "ME":
        return f"{v:.2f}"
    return f"{v:.2f}"

def add_highlight_box(ax, row_i: int, col_j: int):
    rect = patches.Rectangle(
        (col_j - 0.5, row_i - 0.5),
        1.0, 1.0,
        fill=False,
        edgecolor=HIGHLIGHT_EDGE,
        linewidth=2.5,
        zorder=10,
    )
    ax.add_patch(rect)

def best_row_index_for_column(metric: str, col_vals_raw: np.ndarray) -> int | None:
    v = np.asarray(col_vals_raw, dtype=float)
    finite = np.isfinite(v)
    if not np.any(finite):
        return None

    mode = BEST_MODE.get(metric, "high")

    if mode == "abs_low":
        vv = np.abs(v)
        vv[~finite] = np.nan
        return int(np.nanargmin(vv))

    if mode == "low":
        vv = v.copy()
        vv[~finite] = np.nan
        return int(np.nanargmin(vv))

    # high
    vv = v.copy()
    vv[~finite] = np.nan
    return int(np.nanargmax(vv))

def normalize_for_colormap(metric: str, raw: np.ndarray) -> np.ndarray:
    """
    Returns matrix in [0,1] for coloring, where 1 is best (green) and 0 is worst (red).
    Uses robust clipping so outliers do not flatten color range.
    Raw values shown inside cells remain unmodified.
    """
    x = raw.astype(float).copy()
    mode = BEST_MODE.get(metric, "high")

    # Convert to a "minimize" magnitude when needed (ME uses abs)
    if mode == "abs_low":
        x = np.abs(x)

    finite = np.isfinite(x)
    if not np.any(finite):
        return np.full_like(x, np.nan, dtype=float)

    xf = x[finite]

    # Robust bounds for colormap scaling only
    if CLIP_MODE == "percentile":
        lo = np.nanpercentile(xf, CLIP_PCT_LO)
        hi = np.nanpercentile(xf, CLIP_PCT_HI)
    elif CLIP_MODE == "std":
        mu = np.nanmean(xf)
        sd = np.nanstd(xf)
        lo = mu - CLIP_STD_K * sd
        hi = mu + CLIP_STD_K * sd
    else:
        lo = np.nanmin(xf)
        hi = np.nanmax(xf)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(xf)
        hi = np.nanmax(xf)
        if hi <= lo:
            hi = lo + 1.0

    x_clip = np.clip(x, lo, hi)
    scaled = (x_clip - lo) / (hi - lo + 1e-12)

    # Convert to "higher is better"
    # - For low/abs_low: lower raw is better -> invert
    # - For high: keep
    if mode in ("low", "abs_low"):
        scaled = 1.0 - scaled

    return scaled

def decide_methods_to_include() -> list[str]:
    ignore = set()
    if IGNORE_METHOD5:
        ignore.add("method5")
    if IGNORE_METHOD7:
        ignore.add("method7")
    return [m for m in METHOD_ORDER if (m in METHOD_LABELS and m not in ignore)]

# -----------------------------
# Load per-dataset Overall tables into a single long dataframe
# -----------------------------
method_ids = decide_methods_to_include()
row_labels = [METHOD_LABELS[m] for m in method_ids]

datasets = list(DATASET_CLASS.keys())

records = []
missing_files = []

for ds in datasets:
    xls_path = find_excel_path(ds)
    if xls_path is None:
        missing_files.append(ds)
        continue

    df = pd.read_excel(xls_path, sheet_name=SHEET_NAME)
    if "method" not in df.columns:
        raise ValueError(f"{xls_path} sheet '{SHEET_NAME}' has no 'method' column")

    need_cols = ["method", "N_raw", "N_kept"] + METRICS
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"{xls_path} sheet '{SHEET_NAME}' missing columns: {miss}")

    df = df[need_cols].copy()
    df["method"] = df["method"].astype(str)

    for m in method_ids:
        r = df[df["method"] == m]
        if r.empty:
            rec = {"dataset": ds, "class": DATASET_CLASS[ds], "method": m, "N_raw": np.nan, "N_kept": np.nan}
            for met in METRICS:
                rec[met] = np.nan
            records.append(rec)
            continue

        r = r.iloc[0]
        rec = {
            "dataset": ds,
            "class": DATASET_CLASS[ds],
            "method": m,
            "N_raw": float(r["N_raw"]),
            "N_kept": float(r["N_kept"]),
        }
        for met in METRICS:
            rec[met] = float(r[met])
        records.append(rec)

long_df = pd.DataFrame(records)

if missing_files:
    print("Missing Excel files for datasets (these datasets will be skipped):")
    for ds in missing_files:
        print("  -", ds)

# -----------------------------
# Aggregate dataset -> class (per method, per metric)
# -----------------------------
# We want one value per (method, class, metric)
# Recommended: equal-weight by dataset (each dataset counts once)
# Alternative: weight by N_kept so large datasets dominate

agg = {}  # agg[metric] = matrix rows=methods, cols=classes

for metric in METRICS:
    mat = np.full((len(method_ids), len(CLASS_ORDER)), np.nan, dtype=float)

    for ci, cls in enumerate(CLASS_ORDER):
        sub = long_df[long_df["class"] == cls].copy()
        if sub.empty:
            continue

        for ri, m in enumerate(method_ids):
            mm = sub[sub["method"] == m].copy()
            if mm.empty:
                continue

            vals = mm[metric].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue

            if AGG_MODE == "dataset_equal":
                mat[ri, ci] = float(np.nanmean(vals[finite]))
            elif AGG_MODE == "pair_weighted":
                w = mm[WEIGHT_COL].to_numpy(dtype=float)
                wf = np.isfinite(w) & finite
                if not np.any(wf):
                    mat[ri, ci] = float(np.nanmean(vals[finite]))
                else:
                    ww = w[wf]
                    vv = vals[wf]
                    mat[ri, ci] = float(np.sum(ww * vv) / (np.sum(ww) + 1e-12))
            else:
                raise ValueError(f"Unknown AGG_MODE: {AGG_MODE}")

    agg[metric] = mat

# -----------------------------
# Save class-aggregated table to Excel (raw values)
# -----------------------------
with pd.ExcelWriter(OUT_XLS, engine="openpyxl") as writer:
    for metric in METRICS:
        df_out = pd.DataFrame(agg[metric], index=row_labels, columns=CLASS_ORDER)
        df_out.to_excel(writer, sheet_name=metric)

print("Saved class summary Excel ->", OUT_XLS)

# -----------------------------
# Plot: 8 heatmaps (4x2), cols = classes
# Yellow box: ONE best method per class column (per metric)
# -----------------------------
fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
axes = axes.ravel()

cmap = plt.get_cmap("RdYlGn")
last_im = None

for ax, metric in zip(axes, METRICS):
    raw = agg[metric]
    norm = normalize_for_colormap(metric, raw)

    im = ax.imshow(norm, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    last_im = im

    ax.set_title(METRIC_LABELS.get(metric, metric))

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_xticks(np.arange(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=0)

    # One highlight per column: best method for that class (for this metric)
    for cj in range(raw.shape[1]):
        best_i = best_row_index_for_column(metric, raw[:, cj])
        if best_i is not None:
            add_highlight_box(ax, best_i, cj)

    # Annotate raw values
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            txt = fmt_cell(metric, raw[i, j])
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)

# shared colorbar
cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.9, pad=0.01)
cbar.set_label("Normalized performance (robust-scaled, green = best, red = worst)")

clip_note = ""
if CLIP_MODE == "percentile":
    clip_note = f"Colormap clip: {CLIP_PCT_LO}-{CLIP_PCT_HI} percentile"
elif CLIP_MODE == "std":
    clip_note = f"Colormap clip: mean ± {CLIP_STD_K} std"
else:
    clip_note = "Colormap: min-max"

agg_note = "Class aggregation: dataset-equal mean" if AGG_MODE == "dataset_equal" else f"Class aggregation: weighted by {WEIGHT_COL}"

fig.suptitle(
    f"ODRA class summary (rows = methods, cols = classes). Yellow = best per class. {agg_note}. {clip_note}.",
    fontsize=12,
)

fig.savefig(OUT_PNG, dpi=300)
plt.show()
print("Saved figure ->", OUT_PNG)
