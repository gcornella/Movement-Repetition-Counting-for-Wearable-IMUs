import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.patches as mpatches
import json
import numpy as np
import math

# -----------------------------
# Load data
# -----------------------------
with open("reps_datasets/uLift_estimated_reps.json", "r") as f:
    counts_allMethods = json.load(f)

with open("reps_datasets/GTreps.json", "r") as f:
    gt = json.load(f)
print("GTdata.json loaded")

# -----------------------------
# Config
# -----------------------------
methods = ["method1", "method2", "method6","method3a", "method3b", "method4b", "method4a", "method5", "method7"]

# Activities to skip
SKIP_SET = {"H1","H4","H7","H10","C10"}

# Colors
user_palette = [
    'red','blue','green','yellow','purple','orange','pink','brown','cyan','magenta',
    'lime','teal','lavender','maroon','navy','olive','coral','turquoise','gold','indigo','violet'
]
prefix_color = {'A': 'tab:blue', 'C': 'orange'}

# -----------------------------
# Prepare GT map once: activity -> {user_id: gt_value}
# -----------------------------
transformed_data_gt = {}
for user_id, activities in gt.items():
    for activity, value in activities.items():
        transformed_data_gt.setdefault(activity, {})[user_id] = value

# -----------------------------
# Helpers
# -----------------------------
def build_method_map(method_name):
    """activity -> {user_id: estimated_value} for one method"""
    out = {}
    mdata = counts_allMethods.get(method_name, {})
    for user_id, activities in mdata.items():
        for activity, value in activities.items():
            out.setdefault(activity, {})[user_id] = value
    return out

def mae10_for_pairs(valuesx, valuesy, factor):
    """
    MAE scaled to '×10 reps' over aligned GT/est pairs.
    valuesx: GT list
    valuesy: est list
    factor: 1 or 2 (double set)
    """
    errs = []
    for x, y in zip(valuesx, valuesy):
        if x is None or x == 0:
            continue
        errs.append(abs(x - y*factor) * 10.0 / x)
    return float(np.mean(errs)) if errs else float('nan')

def group_mean_for(method_map, grp_prefix):
    """
    Mean of per-activity MAE×10 across all activities starting with grp_prefix.
    """
    per_activity = []
    for act, gt_users in transformed_data_gt.items():
        if not act.startswith(grp_prefix):
            continue
        if act in SKIP_SET:
            continue
        if act not in method_map:
            continue
        uids = sorted(set(method_map[act].keys()) & set(gt_users.keys()))
        if not uids:
            continue
        factor = 2 if act in DOUBLE_SET else 1
        xs = [transformed_data_gt[act][u] for u in uids]
        ys = [method_map[act][u] for u in uids]
        val = mae10_for_pairs(xs, ys, factor)
        if not np.isnan(val):
            per_activity.append(val)
    return float(np.mean(per_activity)) if per_activity else np.nan

# -----------------------------
# Figures 1 & 2 — iterate across methods
# -----------------------------
for method in methods:
    print(f"\n=== Plotting for {method} ===")
    transformed_data = build_method_map(method)

    # Common activities (preserve your base ordering)
    all_activities = list(transformed_data.keys())
    gt_activities_set = set(transformed_data_gt.keys())
    common_activities = [a for a in all_activities if a in gt_activities_set]

    # Build user color map for this method
    all_user_ids = sorted(set().union(*[set(transformed_data[a].keys()) for a in transformed_data.keys()]))
    # If users > palette, cycle colors
    color_iter = cycle(user_palette)
    color_map = {uid: next(color_iter) for uid in all_user_ids}

    # --- Figure 1: 4x5 scatter grid (kept like your base) ---
    '''fig, axs = plt.subplots(4, 5, figsize=(12, 10))
    axs = axs.flatten()

    mean_actvs_percent = []      # values for bar plot
    plotted_activities = []      # track labels actually plotted (so lengths match)
    for idx, activity in enumerate(common_activities):
        # ignore specific activity (your base rule)
        if activity in SKIP_SET:
            continue

        # prevent index errors if more than 20 activities
        if idx >= len(axs):
            break

        factor = 2 if activity in DOUBLE_SET else 1
        user_ids = sorted(set(transformed_data[activity].keys()) & set(transformed_data_gt[activity].keys()))
        ax = axs[idx]

        valuesx, valuesy = [], []
        for user_id in user_ids:
            x_val = transformed_data_gt[activity][user_id]
            y_val = transformed_data[activity][user_id]
            valuesx.append(x_val)
            valuesy.append(y_val)
            ax.scatter(x_val, y_val, c=color_map.get(user_id, 'gray'), label=f'User {user_id}')

        # MAE×10 (correct metric you used)
        act_mae10 = mae10_for_pairs(valuesx, valuesy, factor)
        mean_actvs_percent.append(act_mae10)
        plotted_activities.append(activity)

        # Correlation + best-fit line (only if enough points)
        if len(valuesx) >= 2:
            corr = np.corrcoef(valuesx, valuesy)[0, 1]
            coeffs = np.polyfit(valuesx, valuesy, deg=1)
            poly = np.poly1d(coeffs)
            x_range = np.linspace(min(valuesx), max(valuesx), 100)
            ax.plot(x_range, poly(x_range), color='red', linestyle='--',
                    label=f'Correlation: {corr:.2f}')
            ax.plot(x_range, x_range/factor, color='black', linestyle='-', label='identity')
        elif len(valuesx) == 1:
            x_range = np.linspace(valuesx[0], valuesx[0], 2)
            ax.plot(x_range, x_range/factor, color='black', linestyle='-', label='identity')

        ax.set_title(f'{activity}, {0 if np.isnan(act_mae10) else act_mae10:.2f}/10')
        ax.set_xlabel('GT')
        ax.set_ylabel('est.')
        ax.grid(True)

    # Remove any unused subplots
    for idx in range(len(plotted_activities), len(axs)):
        fig.delaxes(axs[idx])

    # Add a common legend (from first axes that has handles)
    for ax in axs[:len(plotted_activities)]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
            break

    fig.suptitle(f"{method}: GT vs Estimated per Activity", y=0.995, fontsize=14)
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    plt.show()
    '''

    # --- Figure 2: per-activity bar plot (MAE×10) ---
    '''plt.figure(figsize=(12, 6))
    labels = plotted_activities
    values = [0 if np.isnan(v) else v for v in mean_actvs_percent]
    bar_colors = [prefix_color.get(lbl[0], 'gray') for lbl in labels]

    x_idx = np.arange(len(labels))
    plt.bar(x_idx, values, color=bar_colors)
    plt.title(f'{method} — Mean Error per Activity (×10 reps)')
    plt.xlabel('Activity')
    plt.ylabel('Mean error (×10 reps)')
    plt.xticks(x_idx, labels, rotation=45, ha='right')

    # value labels
    for i, v in enumerate(mean_actvs_percent):
        if np.isnan(v):
            plt.text(i, 0, "n/a", ha='center', va='bottom')
        else:
            plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

    pref_handles = [mpatches.Patch(color=prefix_color[p], label=p) for p in ['A','C','H']]
    plt.legend(handles=pref_handles, title="Group")
    plt.tight_layout()
    plt.show()'''

# === FIGURE 3: A, H, C groups (x-axis) with methods side-by-side; color BY METHOD ===
import matplotlib.patches as mpatches

METHODS_F3   = methods   # methods shown as different colors
GROUPS_F3    = ["A", "C","H"]                        # order on x-axis
METHOD_COLOR = {"method1": "tab:red",
                "method2": "tab:blue",
                "method3a": "tab:orange",
                "method3b": "tab:green",
                "method4a": "tab:purple",
                "method4b": "tab:red",
                "method5": "tab:olive",
                "method6": "tab:pink",
                "method7": "tab:green"}

# Domain rules (same logic you used above)
DOUBLE_SET = {} # "A6","A9","A10","H1","H4","H7","H10"
SKIP_SET   = {"C10"}

def _build_method_map(method_name):
    """counts_allMethods[method][user][activity] -> activity -> {user: est}"""
    out = {}
    mdata = counts_allMethods.get(method_name, {})
    for uid, acts in mdata.items():
        for act, val in acts.items():
            out.setdefault(act, {})[uid] = val
    return out

def _mae10_for_activity(act, meth_map):
    """MAE ×10 across overlapping users for one activity (applies doubling, skips GT==0)."""
    if act in SKIP_SET or act not in transformed_data_gt or act not in meth_map:
        return np.nan
    uids = set(transformed_data_gt[act].keys()) & set(meth_map[act].keys())
    if not uids:
        return np.nan
    factor = 2 if act in DOUBLE_SET else 1
    errs = []
    for uid in uids:
        x_gt = transformed_data_gt[act][uid]
        y_est = meth_map[act][uid]
        if x_gt is None or x_gt == 0:
            continue
        errs.append(abs(x_gt - y_est*factor) * 10.0 / x_gt)
    return float(np.mean(errs)) if errs else np.nan

def _group_mean_mae10(meth_map, grp_prefix):
    """Mean of per-activity MAE×10 over all activities starting with grp_prefix."""
    vals = []
    for act in transformed_data_gt.keys():
        if not act.startswith(grp_prefix):
            continue
        v = _mae10_for_activity(act, meth_map)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

# Compute group means (methods × groups)
method_maps = {m: _build_method_map(m) for m in METHODS_F3}
gm = {m: {g: _group_mean_mae10(method_maps[m], g) for g in GROUPS_F3} for m in METHODS_F3}

# Optional: print to sanity-check (e.g., C should be lower than H)
print("Figure 3 — Group means (×10 reps):")
for g in GROUPS_F3:
    row = ", ".join(f"{m}={('n/a' if not np.isfinite(gm[m][g]) else f'{gm[m][g]:.2f}')}" for m in METHODS_F3)
    print(f"{g}: {row}")

# Layout: three group ticks (A,H,C); inside each, bars for all methods (colored by method)
x = np.arange(len(GROUPS_F3))
n_methods = len(METHODS_F3)
width = 0.8 / n_methods                         # keep total cluster width ≈ 0.8
offsets = (np.arange(n_methods) - (n_methods-1)/2.0) * width

plt.figure(figsize=(10, 5))

for i, m in enumerate(METHODS_F3):
    vals = [gm[m][g] for g in GROUPS_F3]
    heights = [0.0 if (v is None or not np.isfinite(v)) else v for v in vals]
    bars = plt.bar(x + offsets[i], heights, width=width,
                   color=METHOD_COLOR[m], edgecolor='black', label=m)

    # annotate values on each bar
    for xi, v in zip(x + offsets[i], vals):
        if not np.isfinite(v):
            plt.text(xi, 0, "n/a", ha='center', va='bottom', fontsize=8)
        else:
            plt.text(xi, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

# x-axis & guides
plt.xticks(x, GROUPS_F3)
plt.ylabel("Group mean error (×10 reps)")
plt.title("Group means by A/H/C with methods side-by-side (color = method)")

# method legend (color-coded)
plt.legend(title="Method", ncol=len(METHODS_F3), loc="upper right")

# (optional) faint separators between groups
for gx in x:
    plt.axvline(gx + 0.5, color='0.9', linestyle=':', linewidth=1)

plt.tight_layout()
plt.show()
