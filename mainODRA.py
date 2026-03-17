import json
from pathlib import Path

# Import algorithms
from paperReadyMethods.method1_RecoFit import method1_fcn
from paperReadyMethods.method2_FitCoach import method2_fcn
from paperReadyMethods.method3_MiLift import method3a_fcn, method3b_fcn
from paperReadyMethods.method4_Threshold import method4a_fcn, method4b_fcn
from paperReadyMethods.method5_Shimmer3 import method5_fcn
from paperReadyMethods.method6_uLift import method6_fcn
from paperReadyMethods.method7_OURS import method7a_fcn, method7b_fcn, method7c_fcn

from utils import find_ndarrays

# -----------------------------
# Choose sensor modality
# -----------------------------
SENSOR_MODE = "accel"  # "accel" or "gyro"
SKIP_DATASET_IF_GYRO_MISSING = True  # only applied when SENSOR_MODE == "gyro"

# -----------------------------
# Fixed local project path
# -----------------------------
SCRIPT_DIR = Path(r"C:\Users\gcorn\PycharmProjects\FitmiSamsungCorrelation")

ODRA_IN = SCRIPT_DIR / "repsfinal" / "Datasets" / "ODRA.json"
OUT_DIR = SCRIPT_DIR / "paperReadyResultsFinal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EST = OUT_DIR / "ODRA_estimated_reps.json"
IGNORED_TASKS_PATH = SCRIPT_DIR / "reps_datasets" / "ignoredTasks.json"

# -----------------------------
# Load ignored tasks
# -----------------------------
IGNORED_TASKS = set()
if IGNORED_TASKS_PATH.exists():
    with IGNORED_TASKS_PATH.open("r", encoding="utf-8") as f:
        ignored_tasks = json.load(f)
    for task_list in ignored_tasks.values():
        IGNORED_TASKS.update(task_list)

# -----------------------------
# Load ODRA container
# -----------------------------
with ODRA_IN.open("r", encoding="utf-8") as f:
    odra = json.load(f)

# -----------------------------
# Output format:
# counts_allMethods[method][dataset][user][task] = predicted reps
# -----------------------------
counts_allMethods = {
    "method1": {},
    "method2": {},
    "method3a": {},
    "method3b": {},
    "method4b": {},
    "method5": {},
    "method6": {},
    "method7a": {},
    "method7b": {},
    "method7c": {},
}

def get_xyz(payload, sensor_mode: str):
    """
    Returns (x, y, z) based on selected sensor_mode.

    sensor_mode:
      - "accel": ax, ay, az
      - "gyro":  gx, gy, gz

    Raises KeyError if required keys are missing.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"get_xyz expected dict payload, got {type(payload)}")

    if sensor_mode == "accel":
        keys = ("ax", "ay", "az")
    elif sensor_mode == "gyro":
        keys = ("gx", "gy", "gz")
    else:
        raise ValueError(f"Unknown sensor_mode '{sensor_mode}'. Use 'accel' or 'gyro'.")

    missing = [k for k in keys if k not in payload]
    if missing:
        raise KeyError(f"Missing {sensor_mode} keys {missing}. Keys present: {list(payload.keys())}")

    return payload[keys[0]], payload[keys[1]], payload[keys[2]]

def dataset_has_gyro(block, ignored_tasks: set) -> bool:
    """
    Returns True if the dataset block contains at least one valid user-task payload
    (not ignored) that has gx,gy,gz.
    """
    data_block = block.get("data", {})
    reps_block = block.get("reps", {})

    for user_id, tasks in data_block.items():
        if not isinstance(tasks, dict):
            continue
        gt_user = reps_block.get(user_id, {})
        if not isinstance(gt_user, dict):
            continue

        for task, payload in tasks.items():
            if task in ignored_tasks:
                continue
            if task not in gt_user:
                continue
            if isinstance(payload, dict) and all(k in payload for k in ("gx", "gy", "gz")):
                return True
    return False

# -----------------------------
# Decide which datasets to process
# -----------------------------
datasets_to_process = []
skipped_datasets = []

for ds, block in odra.items():
    if SENSOR_MODE == "gyro" and SKIP_DATASET_IF_GYRO_MISSING:
        if not dataset_has_gyro(block, IGNORED_TASKS):
            skipped_datasets.append(ds)
            continue
    datasets_to_process.append(ds)

if skipped_datasets:
    print(f"Skipping {len(skipped_datasets)} dataset(s) because gyro is missing (mode=gyro): {skipped_datasets}")

# -----------------------------
# Progress pre-count (only for datasets we will process)
# -----------------------------
total_pairs = 0
for ds in datasets_to_process:
    block = odra[ds]
    data_block = block.get("data", {})
    reps_block = block.get("reps", {})

    for user_id, tasks in data_block.items():
        if not isinstance(tasks, dict):
            continue
        gt_user = reps_block.get(user_id, {})
        if not isinstance(gt_user, dict):
            continue

        for task, payload in tasks.items():
            if task in IGNORED_TASKS:
                continue
            if task not in gt_user:
                continue

            # If in gyro mode, only count pairs that actually have gyro
            if SENSOR_MODE == "gyro":
                if not (isinstance(payload, dict) and all(k in payload for k in ("gx", "gy", "gz"))):
                    continue
            else:
                if not (isinstance(payload, dict) and all(k in payload for k in ("ax", "ay", "az"))):
                    continue

            total_pairs += 1

print(f"Sensor mode: {SENSOR_MODE}")
print(f"Total ODRA user-task pairs to process: {total_pairs}")

processed = 0

# -----------------------------
# Main loop
# -----------------------------
for ds in datasets_to_process:
    block = odra[ds]

    fs = block.get("_meta", {}).get("fs", None)
    if fs is None:
        raise ValueError(f"Missing fs in ODRA meta for dataset '{ds}'")

    data_block = block.get("data", {})
    reps_block = block.get("reps", {})

    # init per dataset in each method
    for k in counts_allMethods.keys():
        counts_allMethods[k].setdefault(ds, {})

    for user_id, tasks in data_block.items():
        if not isinstance(tasks, dict):
            continue

        for k in counts_allMethods.keys():
            counts_allMethods[k][ds].setdefault(user_id, {})

        gt_user = reps_block.get(user_id, {})
        if not isinstance(gt_user, dict):
            continue

        for task, payload in tasks.items():
            if task in IGNORED_TASKS:
                continue
            if task not in gt_user:
                continue

            # Enforce sensor availability per task
            try:
                x, y, z = get_xyz(payload, SENSOR_MODE)
            except KeyError:
                # In gyro mode we ignore tasks without gyro.
                # In accel mode we ignore tasks without accel.
                continue

            processed += 1
            pct = 100.0 * processed / max(total_pairs, 1)
            if processed == 1 or int(pct) % 5 == 0:
                print(f"Progress: {pct:.1f}% completed, {100.0 - pct:.1f}% remaining")

            info = [user_id, task]

            # Compute predictions
            counts_allMethods["method1"][ds][user_id][task] = method1_fcn(x, y, z, fs, plot=False, info=info)
            counts_allMethods["method2"][ds][user_id][task] = method2_fcn(x, y, z, fs, plot=False, info=info)

            counts_allMethods["method3a"][ds][user_id][task] = method3a_fcn(x, y, z, fs, plot=False, info=info)
            counts_allMethods["method3b"][ds][user_id][task] = method3b_fcn(x, y, z, fs, plot=False, info=info)

            counts_allMethods["method4b"][ds][user_id][task] = method4b_fcn(x, y, z, fs, plot=False, info=info)

            counts_allMethods["method5"][ds][user_id][task] = method5_fcn(x, y, z, fs, plot=False, info=info)
            counts_allMethods["method6"][ds][user_id][task] = method6_fcn(x, y, z, fs=fs, plot=False, info=info)

            counts_allMethods["method7a"][ds][user_id][task] = method7a_fcn(x, y, z, fs, plot=False, info=info)
            counts_allMethods["method7b"][ds][user_id][task] = method7b_fcn(x, y, z, fs, plot=False, info=info)
            counts_allMethods["method7c"][ds][user_id][task] = method7c_fcn(x, y, z, fs, plot=False, info=info)

# -----------------------------
# Save
# -----------------------------
with OUT_EST.open("w", encoding="utf-8") as f:
    find_ndarrays(counts_allMethods)
    json.dump(counts_allMethods, f, indent=2)

print(f"\nSaved ODRA estimated reps -> {OUT_EST}")
print("Done.")
