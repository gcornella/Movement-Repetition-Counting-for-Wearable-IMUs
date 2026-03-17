from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------
# Config
# -----------------------------
RESULTS_DIR = Path(r"C:\Users\gcorn\PycharmProjects\FitmiSamsungCorrelation\paperReadyResults")

# Order of datasets (prefix in filenames: {dataset}_MetricsExcel.xlsx)
DATASETS = [
    "recofit",
    "crossfit",
    "mmfit",
    "uzh_healthy",
    "uzh_stroke",
    "JUIMU_ROM_ND",
    "JUIMU_ROM_Stroke",
    "uLift",
    "CaraCount",
    "ODRA-Rehab",
]

SHEET_NAME = "Overall_AllMetrics"

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
# - ME: best is abs closest to 0
# - MAE/RMSE/MAPE: lower is better
# - EXACT/ACC1/ACC2/ACC3: higher is better
BEST_MODE = {
    "ME": "abs_low",
    "MAE": "low",
    "RMSE": "low",
    "MAPE": "low",
    "EXACT": "high",
    "ACC1": "high",
    "ACC2": "high",
    "ACC3": "high",
}

# Robust clipping for colormap only (raw values in cells are unchanged)
# Choose: "percentile" (recommended), "std", or None
CLIP_MODE = "percentile"  # or "std" or None
CLIP_PCT_LO, CLIP_PCT_HI = 5, 95
CLIP_STD_K = 2.0

# Output
OUT_PNG = RESULTS_DIR / "paperReady_heatmaps_overall_metrics.png"

# Colors
COLOR_HEALTHY = "green"
COLOR_STROKE = "blue"
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


def normalize_for_colormap(metric: str, raw: np.ndarray) -> np.ndarray:
    """
    Returns matrix in [0,1] for coloring, where 1 is best (green) and 0 is worst (red).
    Uses robust clipping so outliers do not flatten color range.
    Raw values shown inside cells remain unmodified.
    """
    x = raw.astype(float).copy()
    mode = BEST_MODE.get(metric, "raw")

    # Transform for meaningful magnitude
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

    # Guard
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(xf)
        hi = np.nanmax(xf)
        if hi <= lo:
            hi = lo + 1.0

    # Clip for coloring only
    x_clip = np.clip(x, lo, hi)

    # Min-max scale (larger x_clip -> larger scaled)
    scaled = (x_clip - lo) / (hi - lo + 1e-12)

    # Convert to "higher is better" for the colormap
    if mode in ("low", "abs_low"):
        scaled = 1.0 - scaled
    elif mode == "high":
        pass
    else:
        # raw: treat higher as better
        pass

    return scaled


def best_row_index(metric: str, col_vals_raw: np.ndarray) -> int | None:
    """
    Given raw values for a single dataset column (one metric),
    return the row index of the best method.
    """
    v = col_vals_raw.astype(float).copy()
    finite = np.isfinite(v)
    if not np.any(finite):
        return None

    mode = BEST_MODE.get(metric, "raw")

    if mode == "abs_low":
        v = np.abs(v)
        # best is minimum abs
        idx = int(np.nanargmin(np.where(finite, v, np.nan)))
        return idx

    if mode == "low":
        idx = int(np.nanargmin(np.where(finite, v, np.nan)))
        return idx

    if mode == "high":
        idx = int(np.nanargmax(np.where(finite, v, np.nan)))
        return idx

    # default: high is best
    idx = int(np.nanargmax(np.where(finite, v, np.nan)))
    return idx


def add_highlight_box(ax, row_i: int, col_j: int):
    """
    Draw a yellow rectangle around cell (row_i, col_j) in imshow grid coords.
    imshow cells are centered on integer coordinates; cell spans [j-0.5, j+0.5] etc.
    """
    rect = patches.Rectangle(
        (col_j - 0.5, row_i - 0.5),
        1.0,
        1.0,
        fill=False,
        edgecolor=HIGHLIGHT_EDGE,
        linewidth=2.5,
        zorder=10,
    )
    ax.add_patch(rect)


# -----------------------------
# Decide methods to include
# -----------------------------
ignore = set()
if IGNORE_METHOD5:
    ignore.add("method5")
if IGNORE_METHOD7:
    ignore.add("method7")

method_ids = [m for m in METHOD_ORDER if (m in METHOD_LABELS and m not in ignore)]
row_labels = [METHOD_LABELS[m] for m in method_ids]

# -----------------------------
# Load all dataset tables
# -----------------------------
data = {ds: {} for ds in DATASETS}
missing_files = []

for ds in DATASETS:
    xls_path = find_excel_path(ds)
    if xls_path is None:
        missing_files.append(ds)
        continue

    df = pd.read_excel(xls_path, sheet_name=SHEET_NAME)
    if "method" not in df.columns:
        raise ValueError(f"{xls_path} sheet '{SHEET_NAME}' has no 'method' column")

    df["method"] = df["method"].astype(str)

    cols_needed = ["method"] + METRICS
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{xls_path} sheet '{SHEET_NAME}' missing columns: {missing_cols}")

    df = df[cols_needed].copy()

    for m in method_ids:
        row = df[df["method"] == m]
        if row.empty:
            data[ds][m] = {k: np.nan for k in METRICS}
            continue
        row = row.iloc[0]
        data[ds][m] = {k: float(row[k]) for k in METRICS}

if missing_files:
    print("Missing Excel files for datasets (these columns will be blank):")
    for ds in missing_files:
        print("  -", ds)

# -----------------------------
# Build raw matrices per metric: rows=methods, cols=datasets
# -----------------------------
raw_mats = {}
for metric in METRICS:
    mat = np.full((len(method_ids), len(DATASETS)), np.nan, dtype=float)
    for j, ds in enumerate(DATASETS):
        for i, m in enumerate(method_ids):
            mat[i, j] = data.get(ds, {}).get(m, {}).get(metric, np.nan)
    raw_mats[metric] = mat

# -----------------------------
# Plot: 8 heatmaps (4x2)
# -----------------------------
fig, axes = plt.subplots(4, 2, figsize=(18, 14), constrained_layout=True)
axes = axes.ravel()

cmap = plt.get_cmap("RdYlGn")  # red->yellow->green
last_im = None

for ax, metric in zip(axes, METRICS):
    raw = raw_mats[metric]
    norm = normalize_for_colormap(metric, raw)

    im = ax.imshow(norm, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    last_im = im

    ax.set_title(METRIC_LABELS.get(metric, metric))

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels(DATASETS, rotation=30, ha="right")

    # Annotate with raw values + highlight best per dataset column
    for j in range(raw.shape[1]):
        best_i = best_row_index(metric, raw[:, j])
        if best_i is not None:
            add_highlight_box(ax, best_i, j)

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            txt = fmt_cell(metric, raw[i, j])
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)

# Shared colorbar
cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.85, pad=0.01)
cbar.set_label("Normalized performance (robust-scaled, green = best, red = worst)")

clip_note = ""
if CLIP_MODE == "percentile":
    clip_note = f"Colormap clip: {CLIP_PCT_LO}–{CLIP_PCT_HI} percentile"
elif CLIP_MODE == "std":
    clip_note = f"Colormap clip: mean ± {CLIP_STD_K} std"
else:
    clip_note = "Colormap: min–max"

fig.suptitle(
    f"Overall metrics across datasets (rows = methods, cols = datasets). Yellow box = best per dataset. {clip_note}.",
    fontsize=13,
)

fig.savefig(OUT_PNG, dpi=300)
plt.show()

print("Saved figure to:", OUT_PNG)
