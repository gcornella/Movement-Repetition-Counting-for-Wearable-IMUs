from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# Config
# ============================================================
RESULTS_DIR = Path(r"C:\Users\gcorn\PycharmProjects\FitmiSamsungCorrelation\paperReadyResults")
RESULTS_DIR = Path(r"C:\Users\gcorn\PycharmProjects\FitmiSamsungCorrelation\paperReadyResultsFinal")
SHEET_NAME = "Overall_AllMetrics"

DATASETS = [
    "recofit",
    "crossfit",
    "mmfit",
    "uzh_healthy",
    "uzh_stroke",
    "JUIMU_ROM_ND",
    "JUIMU_ROM_Stroke",
    "uLift",
    "DWC-Fitness",
    "DWC-Rehab",
    "DWC-ADL",
    "CaRa-Fitness",
    "CaRa-ADL",
    "RehabSimUCI",
]


SHEET_NAME = "Overall_Pooled_AllMetrics"
DATASETS = [
    "ODRA-Fitness",
    "ODRA-Rehab",
    "ODRA-ADL",
]


METHOD_LABELS = {
    "method1": "RecoFit",
    "method2": "FitCoach",
    "method3a": "MiLift A",
    "method3b": "MiLift B",
    "method4b": "Threshold Cross B",
    "method5": "Shimmer3",
    "method6": "uLift",
    "method7a": "FusionRep A",
    "method7b": "FusionRep B",
    "method7c": "FusionRep C",
}

METHOD_ORDER = [
    "method1", "method2", "method3a", "method3b",
    "method4b", "method5", "method6",
    "method7a", "method7b", "method7c",
]

METRICS = ["ME", "MAE", "RMSE", "MAPE", "EXACT", "ACC1", "ACC2", "ACC3"]

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

VALUE_FMT = {
    "ME": "{:+.2f}",
    "MAE": "{:.2f}",
    "RMSE": "{:.2f}",
    "MAPE": "{:.2f}",
    "EXACT": "{:.1f}",
    "ACC1": "{:.1f}",
    "ACC2": "{:.1f}",
    "ACC3": "{:.1f}",
}

METRIC_ROW_LABELS = {
    "ME": "ME (reps)",
    "MAE": "MAE (reps)",
    "RMSE": "RMSE (reps)",
    "MAPE": "MAPE (%)",
    "EXACT": "Exact match (%)",
    "ACC1": "Acc ±1 (%)",
    "ACC2": "Acc ±2 (%)",
    "ACC3": "Acc ±3 (%)",
}

# ============================================================
# Colors (one per algorithm)
# ============================================================
def build_method_colors():
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    return {mid: cycle[i % len(cycle)] for i, mid in enumerate(METHOD_ORDER)}

METHOD_COLORS = build_method_colors()

# ============================================================
# IO helpers
# ============================================================
def find_excel_path(dataset_name: str) -> Path | None:
    p = RESULTS_DIR / f"{dataset_name}_MetricsExcel.xlsx"
    return p if p.exists() else None

def load_dataset_df(dataset_name: str) -> pd.DataFrame | None:
    xls_path = find_excel_path(dataset_name)
    if xls_path is None:
        return None

    df = pd.read_excel(xls_path, sheet_name=SHEET_NAME)

    if "method" not in df.columns:
        raise ValueError(f"{xls_path} sheet '{SHEET_NAME}' has no 'method' column")

    cols_needed = ["method"] + METRICS
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{xls_path} sheet '{SHEET_NAME}' missing columns: {missing_cols}")

    df = df[cols_needed].copy()
    df["method"] = df["method"].astype(str)

    rank_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    df["_rank"] = df["method"].map(rank_map).fillna(10_000).astype(int)
    df = df.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)

    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    return df

# ============================================================
# Winner selection
# ============================================================
def pick_winner(df: pd.DataFrame, metric: str) -> tuple[str, str, float] | None:
    if df is None or df.empty:
        return None

    mode = BEST_MODE.get(metric, "low")
    vals = df[metric].to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return None

    if mode == "abs_low":
        score = np.abs(vals)
        best_idx = int(np.nanargmin(np.where(finite, score, np.nan)))
    elif mode == "low":
        best_idx = int(np.nanargmin(np.where(finite, vals, np.nan)))
    elif mode == "high":
        best_idx = int(np.nanargmax(np.where(finite, vals, np.nan)))
    else:
        best_idx = int(np.nanargmin(np.where(finite, vals, np.nan)))

    method_id = str(df.loc[best_idx, "method"])
    value = float(df.loc[best_idx, metric])
    label = METHOD_LABELS.get(method_id, method_id)
    return method_id, label, value

# ============================================================
# Build winner table + legend set
# ============================================================
def build_winner_table():
    # rows = metrics, cols = datasets
    text = pd.DataFrame(index=METRICS, columns=DATASETS, dtype=object)
    winner_ids = pd.DataFrame(index=METRICS, columns=DATASETS, dtype=object)

    missing = []
    used_methods = set()

    for ds in DATASETS:
        df = load_dataset_df(ds)
        if df is None:
            missing.append(ds)

        for metric in METRICS:
            if df is None:
                text.loc[metric, ds] = "NA"
                winner_ids.loc[metric, ds] = None
                continue

            w = pick_winner(df, metric)
            if w is None:
                text.loc[metric, ds] = "NA"
                winner_ids.loc[metric, ds] = None
                continue

            method_id, method_label, val = w
            used_methods.add(method_id)

            fmt = VALUE_FMT.get(metric, "{:.2f}")
            #text.loc[metric, ds] = f"{method_label}\n{fmt.format(val)}"
            text.loc[metric, ds] = f"{fmt.format(val)}"

            winner_ids.loc[metric, ds] = method_id

    if missing:
        print("Missing Excel files (filled with NA):")
        for ds in missing:
            print("  -", ds)

    return text, winner_ids, used_methods

# ============================================================
# Plot (FLIPPED: rows=datasets, cols=metrics) + bottom legend
# ============================================================
# ============================================================
# Plot (FLIPPED: rows=datasets, cols=metrics)
# legend directly below table
# units on second header line
# ============================================================
def plot_colored_winner_table_flipped(
    text_df: pd.DataFrame,
    winner_ids_df: pd.DataFrame,
    used_methods: set[str],
    title: str = "Best algorithm per dataset and metric (cell color = winner)",
    figsize=(16, 6.0),
    fontsize_body=16,
    fontsize_header=16,
    cell_pad=0.02,
    header_linewidth=0.35,
    body_linewidth=0.12,
    na_facecolor="0.92",
    na_textcolor="0.25",
    legend_ncol=10,
):

    text_flip = text_df.T.copy()
    winners_flip = winner_ids_df.T.copy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # ---------- split header into two rows ----------
    col_labels = []
    for m in text_flip.columns.tolist():
        full = METRIC_ROW_LABELS.get(m, m)
        if "(" in full:
            name = full.split("(")[0].strip()
            unit = "(" + full.split("(")[1]
            col_labels.append(f"{name}\n{unit}")
        else:
            col_labels.append(full)

    row_labels = text_flip.index.tolist()

    # ---------- colors ----------
    cell_colors = []
    for r in range(text_flip.shape[0]):
        row_colors = []
        for c in range(text_flip.shape[1]):
            mid = winners_flip.iloc[r, c]
            raw = str(text_flip.iloc[r, c]).strip()
            if mid is None or raw.upper() == "NA":
                row_colors.append(na_facecolor)
            else:
                row_colors.append(METHOD_COLORS.get(mid, "1.0"))
        cell_colors.append(row_colors)

    tbl = ax.table(
        cellText=text_flip.values,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)

    # ---------- styling ----------
    # ---------- styling ----------
    for (r, c), cell in tbl.get_celld().items():

        # Increase header row height
        if r == 0:
            cell.set_height(cell.get_height() * 1.6)

        # Slightly increase column width
        if c >= 0:
            cell.set_width(cell.get_width() * 1.05)

        try:
            cell.PAD = cell_pad
        except:
            pass

        if r == 0 or c == -1:
            cell.set_linewidth(header_linewidth)
            cell.set_text_props(weight="bold", fontsize=fontsize_header)
        else:
            cell.set_linewidth(body_linewidth)
            cell.set_text_props(fontsize=fontsize_body)

        # gray NA text
        if r > 0 and c >= 0:
            raw = str(text_flip.iloc[r - 1, c]).strip()
            if raw.upper() == "NA":
                cell.get_text().set_color(na_textcolor)


    tbl.scale(1.0, 1.2)

    ax.set_title(title, fontsize=fontsize_header + 3, pad=6)

    # ---------- legend JUST below table ----------
    # ---------- legend JUST below table (2 rows) ----------
    if used_methods:
        used_sorted = [m for m in METHOD_ORDER if m in used_methods]

        handles = [
            Patch(facecolor=METHOD_COLORS[m], edgecolor="k", linewidth=0.3)
            for m in used_sorted
        ]

        labels = [METHOD_LABELS.get(m, m) for m in used_sorted]

        # force 2 rows
        n_items = len(labels)
        ncol = int(np.ceil(n_items / 2))  # half per row

        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=ncol,
            frameon=True,
            fontsize=fontsize_body,
            title="Algorithms",
            title_fontsize=fontsize_header,
            columnspacing=1.2,
            handletextpad=0.6,
        )

    plt.tight_layout()
    plt.show()

# ============================================================
# Print summary statistics
# ============================================================

def print_winner_statistics(winner_ids_df: pd.DataFrame):

    print("\n==============================")
    print(" WINNER SUMMARY ACROSS METRICS")
    print("==============================\n")

    # total number of valid cells
    valid_cells = winner_ids_df.notna().sum().sum()

    # overall win counts
    flat = winner_ids_df.values.flatten()
    flat = [x for x in flat if x is not None]

    overall_counts = pd.Series(flat).value_counts()

    print("Overall winner ranking (all metrics combined):\n")

    for method_id, count in overall_counts.items():

        label = METHOD_LABELS.get(method_id, method_id)
        perc = 100 * count / valid_cells

        print(f"{label:<15} : {count:3d} wins  ({perc:5.1f}%)")

    print("\n--------------------------------")

    # winner per metric
    print("\nWinner ranking PER METRIC:\n")

    for metric in METRICS:

        col = winner_ids_df.loc[metric].dropna()

        if len(col) == 0:
            continue

        counts = col.value_counts()

        winner_id = counts.index[0]
        winner_label = METHOD_LABELS.get(winner_id, winner_id)

        print(f"{metric:<8} → {winner_label} wins most ({counts.iloc[0]} datasets)")

    print("\n==============================\n")

# ============================================================
# Run
# ============================================================
text_df, winner_ids_df, used_methods = build_winner_table()
print_winner_statistics(winner_ids_df)

plot_colored_winner_table_flipped(
    text_df=text_df,
    winner_ids_df=winner_ids_df,
    used_methods=used_methods,
    title="Best algorithm per dataset and metric (cell color = winner)",
    figsize=(16, 6.0),
    fontsize_body=16,
    fontsize_header=14,
    cell_pad=0.01,
    header_linewidth=0.35,
    body_linewidth=0.12,
    legend_ncol=10,
)
