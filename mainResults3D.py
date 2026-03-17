from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

# -----------------------------
# Config
# -----------------------------
RESULTS_DIR = Path(r"C:\Users\gcorn\PycharmProjects\FitmiSamsungCorrelation\paperReadyResults")

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

METHOD_LABELS = {
    "method1": "RecoFit",
    "method2": "FitCoach",
    "method3a": "MiLift Autocorr (a)",
    "method3b": "MiLift Revisit (b)",
    # "method4a": "Threshold Crossing",
    "method4b": "Threshold Cross LPF",
    "method5": "ExerSense",
    "method6": "Shimmer3",
    "method7": "LEAN",
    "method8": "uLift",
    "method9": "OURS",
}

IGNORE_METHOD5 = True
IGNORE_METHOD7 = True

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

# 3D styling
SMALL_LINE_W = 1.1      # dataset-connector thin line width
SMALL_LINE_A = 0.55     # dataset-connector alpha
FILL_ALPHA = 0.18       # algorithm fill alpha
BIG_S = 90              # winner dot size

# -----------------------------
# Helpers
# -----------------------------
def find_excel_path(dataset_name: str) -> Path | None:
    p = RESULTS_DIR / f"{dataset_name}_MetricsExcel.xlsx"
    return p if p.exists() else None


def best_row_index(metric: str, col_vals_raw: np.ndarray) -> int | None:
    v = col_vals_raw.astype(float).copy()
    finite = np.isfinite(v)
    if not np.any(finite):
        return None

    mode = BEST_MODE.get(metric, "raw")

    if mode == "abs_low":
        vv = np.abs(v)
        return int(np.nanargmin(np.where(finite, vv, np.nan)))
    if mode == "low":
        return int(np.nanargmin(np.where(finite, v, np.nan)))
    if mode == "high":
        return int(np.nanargmax(np.where(finite, v, np.nan)))

    return int(np.nanargmax(np.where(finite, v, np.nan)))


def build_method_list() -> tuple[list[str], list[str]]:
    ignore = set()
    if IGNORE_METHOD5:
        ignore.add("method5")
    if IGNORE_METHOD7:
        ignore.add("method7")

    method_ids = [m for m in METHOD_ORDER if (m in METHOD_LABELS and m not in ignore)]
    row_labels = [METHOD_LABELS[m] for m in method_ids]
    return method_ids, row_labels


def load_raw_mats(method_ids: list[str]) -> dict[str, np.ndarray]:
    """
    Returns dict: metric -> matrix (n_methods x n_datasets)
    """
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
        print("Missing Excel files for datasets (these will be NaN in plots):")
        for ds in missing_files:
            print("  -", ds)

    raw_mats = {}
    for metric in METRICS:
        mat = np.full((len(method_ids), len(DATASETS)), np.nan, dtype=float)
        for j, ds in enumerate(DATASETS):
            for i, m in enumerate(method_ids):
                mat[i, j] = data.get(ds, {}).get(m, {}).get(metric, np.nan)
        raw_mats[metric] = mat

    return raw_mats


def plot_3d_all_metrics(raw_mats: dict[str, np.ndarray], row_labels: list[str]) -> None:
    """
    For each metric:
      - algorithm curve: algorithm-colored line + fill to z=0
      - dataset "connector": thin line joining all points for same dataset across algorithms
      - big scatter: winner per dataset (dataset-colored)
      - no dataset legend
      - axis labels repositioned for readability
    """

    # Use two different colormaps so dataset + algorithm colors are distinct
    # Algorithms: Set2/Set3-ish, Datasets: tab10/tab20
    alg_cmap = plt.get_cmap("Set2")
    ds_cmap = plt.get_cmap("tab10" if len(DATASETS) <= 10 else "tab20")

    alg_colors = [alg_cmap(i % alg_cmap.N) for i in range(len(row_labels))]
    ds_colors = [ds_cmap(j % ds_cmap.N) for j in range(len(DATASETS))]

    for metric in METRICS:
        raw = raw_mats[metric]  # shape: (methods, datasets)
        n_methods, n_datasets = raw.shape
        x_all = np.arange(n_datasets)

        fig = plt.figure(figsize=(17, 10))
        ax = fig.add_subplot(111, projection="3d")

        # ---- Make panes transparent and less occluding ----
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("0.90")
        ax.yaxis.pane.set_edgecolor("0.90")
        ax.zaxis.pane.set_edgecolor("0.90")

        # ---- Tick/label padding so axes are not behind labels ----
        ax.tick_params(axis="x", pad=14)
        ax.tick_params(axis="y", pad=12)
        ax.tick_params(axis="z", pad=8)

        # Relocated axis labels
        ax.set_xlabel("Datasets", labelpad=26)
        ax.set_ylabel("Algorithms", labelpad=30)
        ax.set_zlabel(METRIC_LABELS.get(metric, metric), labelpad=16)

        # ---- Plot algorithm curves + fills ----
        alg_handles = []
        for i, alg_label in enumerate(row_labels):
            z_all = raw[i, :].astype(float)
            if not np.any(np.isfinite(z_all)):
                continue

            finite = np.isfinite(z_all)
            idx = np.where(finite)[0]
            if idx.size == 0:
                continue

            splits = np.where(np.diff(idx) > 1)[0] + 1
            chunks = np.split(idx, splits)

            alg_color = alg_colors[i]

            for c in chunks:
                x = x_all[c]
                z = z_all[c]
                y = np.full_like(x, i)

                # Line (algorithm colored)
                ax.plot(x, y, z, linewidth=2.0, color=alg_color, zorder=8)

                # Fill under curve to z=0 (algorithm colored)
                verts = list(zip(x, y, z)) + list(zip(x[::-1], y[::-1], np.zeros_like(z)[::-1]))
                poly = Poly3DCollection([verts], facecolor=alg_color, edgecolor="none", alpha=FILL_ALPHA)
                ax.add_collection3d(poly)

            alg_handles.append(Line2D([0], [0], color=alg_color, lw=2, label=alg_label))

        # ---- NEW: Dataset connector lines (thin), joining points across algorithms ----
        # For each dataset j: connect (x=j, y=algorithm_index, z=value) across i
        for j in range(n_datasets):
            z_col = raw[:, j].astype(float)  # length n_methods
            finite = np.isfinite(z_col)
            if not np.any(finite):
                continue

            ys = np.where(finite)[0]
            zs = z_col[finite]
            xs = np.full_like(ys, j, dtype=float)

            # sort by y (already increasing) but keep safe
            order = np.argsort(ys)
            xs, ys, zs = xs[order], ys[order].astype(float), zs[order]

            ax.plot(
                xs, ys, zs,
                color=ds_colors[j],
                linewidth=SMALL_LINE_W,
                alpha=SMALL_LINE_A,
                zorder=6
            )

        # ---- Big scatter: winner per dataset ----
        for j in range(n_datasets):
            best_i = best_row_index(metric, raw[:, j])
            if best_i is None:
                continue
            z_best = raw[best_i, j]
            if not np.isfinite(z_best):
                continue

            ax.scatter(
                j, best_i, z_best,
                s=BIG_S,
                color=ds_colors[j],
                edgecolor="k",
                linewidth=0.9,
                depthshade=False,
                zorder=20
            )

        # ---- Ticks / tick label coloring (dataset colors on x labels) ----
        ax.set_xticks(x_all)
        ax.set_xticklabels(DATASETS, rotation=30, ha="right")
        for tick, c in zip(ax.get_xticklabels(), ds_colors):
            tick.set_color(c)

        ax.set_yticks(np.arange(n_methods))
        ax.set_yticklabels(row_labels)

        # View
        ax.view_init(elev=28, azim=-55)

        ax.set_title(
            f"{METRIC_LABELS.get(metric, metric)}: 3D overlay (thin dataset connectors, big dot = best per dataset)",
            pad=22
        )

        # Algorithm legend only
        ax.legend(
            handles=alg_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            title="Algorithms (line + fill)",
            frameon=True
        )

        plt.tight_layout()
        plt.show()


# -----------------------------
# Run
# -----------------------------
method_ids, row_labels = build_method_list()
raw_mats = load_raw_mats(method_ids)
plot_3d_all_metrics(raw_mats, row_labels)
