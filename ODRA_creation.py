import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

# -----------------------------
# Fixed local project path
# -----------------------------
SCRIPT_DIR = Path(r"C:\Users\gcorn\PycharmProjects\FitmiSamsungCorrelation")
FILES_DIR = SCRIPT_DIR / "repsfinal" / "Datasets"
FILES_DIR.mkdir(parents=True, exist_ok=True)

IGNORED_TASKS_PATH = SCRIPT_DIR / "reps_datasets" / "ignoredTasks.json"

OUT_ODRA = FILES_DIR / "ODRA.json"          # single file containing data+reps grouped by dataset
OUT_DATA = FILES_DIR / "ODRA_data.json"     # optional convenience: ODRA-only data (still grouped by dataset)
OUT_REPS = FILES_DIR / "ODRA_reps.json"     # optional convenience: ODRA-only reps (still grouped by dataset)

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

# Sampling frequencies per dataset name
FS_MAP = {
    "recofit": 50,
    "crossfit": 100,
    "mmfit": 50,
    "uzh_healthy": 60,
    "uzh_stroke": 60,
    "JUIMU_ROM_ND": 80,
    "JUIMU_ROM_Stroke": 80,
    "uLift": 60,
    "DWC-Fitness":100,
    "DWC-Rehab":100,
    "DWC-ADL":100,
    "CaRa-Fitness":100,
    "CaRa-ADL":100,
    "RehabSimUCI": 50,
}

# If True: force unique user IDs across datasets by prefixing with dataset
ALWAYS_NAMESPACE_USERS = False
# If True: force unique task names across datasets by prefixing with dataset
ALWAYS_NAMESPACE_TASKS = False

# Optional strict validation (recommended True while debugging)
STRICT_VALIDATE_PAYLOAD = True


# -----------------------------
# Helpers
# -----------------------------
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def atomic_save_json(path: Path, obj: Dict[str, Any], indent: int = 2) -> None:
    """
    Write JSON atomically: write to temp file, fsync, then rename.
    Prevents corrupted partial JSON if the process stops mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        # allow_nan=False forces failure if NaN/Inf appear (prevents invalid JSON)
        json.dump(obj, f, indent=indent, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())

    tmp.replace(path)

def validate_json_file(path: Path) -> Tuple[bool, str]:
    """
    Validate that a JSON file can be fully loaded.
    Returns (ok, message).
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True, "OK"
    except json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def unique_user_id(user_id: str, dataset: str, used_users: set) -> str:
    if ALWAYS_NAMESPACE_USERS:
        base = f"{user_id}__{dataset}"
    else:
        base = user_id if user_id not in used_users else f"{user_id}__{dataset}"

    k = 2
    out = base
    while out in used_users:
        out = f"{base}_{k}"
        k += 1
    return out

def unique_task_name(task: str, dataset: str, existing_tasks: set) -> str:
    if ALWAYS_NAMESPACE_TASKS:
        base = f"{task}__{dataset}"
    else:
        base = task if task not in existing_tasks else f"{task}__{dataset}"

    k = 2
    out = base
    while out in existing_tasks:
        out = f"{base}_{k}"
        k += 1
    return out

def validate_payload(task_payload: Any) -> bool:
    """
    Minimal check that payload contains ax/ay/az arrays of equal length.
    Returns True if ok, False otherwise.
    """
    if not isinstance(task_payload, dict):
        return False
    for key in ("ax", "ay", "az"):
        if key not in task_payload:
            return False
        if not isinstance(task_payload[key], list):
            return False

    n1 = len(task_payload["ax"])
    n2 = len(task_payload["ay"])
    n3 = len(task_payload["az"])
    if n1 == 0 or (n1 != n2) or (n1 != n3):
        return False

    # Optional: ensure values are JSON-safe numbers
    if STRICT_VALIDATE_PAYLOAD:
        for key in ("ax", "ay", "az"):
            for v in task_payload[key]:
                if not isinstance(v, (int, float)):
                    return False
                # disallow NaN/Inf, which break strict JSON dump allow_nan=False
                if isinstance(v, float) and (not (v == v) or v in (float("inf"), float("-inf"))):
                    return False
    return True


# -----------------------------
# Load ignored tasks
# -----------------------------
IGNORED_TASKS = set()
if IGNORED_TASKS_PATH.exists():
    ignored_tasks = load_json(IGNORED_TASKS_PATH)
    for task_list in ignored_tasks.values():
        IGNORED_TASKS.update(task_list)
else:
    print(f"[WARN] ignoredTasks.json not found at {IGNORED_TASKS_PATH}. Proceeding without ignores.")


# -----------------------------
# Build ODRA grouped structure
# -----------------------------
odra: Dict[str, Any] = {}
missing = []
total_pairs = 0
kept_pairs = 0
skipped_bad_payload = 0

# Pre-count for progress
for ds in DATASETS:
    data_path = FILES_DIR / f"{ds}_data.json"
    reps_path = FILES_DIR / f"{ds}_reps.json"
    if not data_path.exists() or not reps_path.exists():
        continue

    data_json = load_json(data_path)
    reps_json = load_json(reps_path)

    for user_id, tasks in data_json.items():
        if not isinstance(tasks, dict):
            continue
        reps_user = reps_json.get(user_id, {})
        if not isinstance(reps_user, dict):
            reps_user = {}

        for task in tasks.keys():
            if task in IGNORED_TASKS:
                continue
            if task in reps_user:
                total_pairs += 1

print(f"Total candidate user-task pairs (after ignores, present in reps): {total_pairs}")

processed = 0

for ds in DATASETS:
    data_path = FILES_DIR / f"{ds}_data.json"
    reps_path = FILES_DIR / f"{ds}_reps.json"

    if not data_path.exists() or not reps_path.exists():
        missing.append(ds)
        continue

    fs = FS_MAP.get(ds)
    if fs is None:
        raise ValueError(f"No fs mapping for dataset '{ds}'. Add it to FS_MAP.")

    data_json = load_json(data_path)
    reps_json = load_json(reps_path)

    odra.setdefault(ds, {"_meta": {"fs": fs}, "data": {}, "reps": {}})

    used_users = set(odra[ds]["data"].keys())

    for user_id, tasks in data_json.items():
        if not isinstance(tasks, dict):
            continue

        new_user = unique_user_id(user_id, ds, used_users)
        used_users.add(new_user)

        odra[ds]["data"].setdefault(new_user, {})
        odra[ds]["reps"].setdefault(new_user, {})

        existing_tasks = set(odra[ds]["data"][new_user].keys())

        reps_user = reps_json.get(user_id, {})
        if not isinstance(reps_user, dict):
            reps_user = {}

        for task, accel_payload in tasks.items():
            if task in IGNORED_TASKS:
                continue
            if task not in reps_user:
                continue

            processed += 1
            pct = 100.0 * processed / max(total_pairs, 1)
            if processed == 1 or int(pct) % 5 == 0:
                print(f"Progress: {pct:.1f}% completed, {100.0 - pct:.1f}% remaining")

            if not validate_payload(accel_payload):
                skipped_bad_payload += 1
                continue

            new_task = unique_task_name(task, ds, existing_tasks)
            existing_tasks.add(new_task)

            odra[ds]["data"][new_user][new_task] = accel_payload
            odra[ds]["reps"][new_user][new_task] = reps_user[task]
            kept_pairs += 1

print("\n=== ODRA BUILD COMPLETE ===")
print(f"Kept user-task pairs: {kept_pairs}")
if skipped_bad_payload:
    print(f"[WARN] Skipped {skipped_bad_payload} tasks due to invalid accel payload (missing ax/ay/az, bad lengths, or NaN/Inf).")

if missing:
    print("[WARN] Missing dataset files:")
    for ds in missing:
        print("  -", ds)

# -----------------------------
# Save atomically + validate
# -----------------------------
print("\nSaving ODRA.json atomically...")
atomic_save_json(OUT_ODRA, odra)

ok, msg = validate_json_file(OUT_ODRA)
if not ok:
    raise RuntimeError(f"ODRA.json failed validation after save. {msg}")
print("ODRA.json validation passed.")

# Save convenience split files (still grouped by dataset)
odra_data_only = {ds: {"_meta": odra[ds]["_meta"], "data": odra[ds]["data"]} for ds in odra}
odra_reps_only = {ds: {"_meta": odra[ds]["_meta"], "reps": odra[ds]["reps"]} for ds in odra}

print("Saving ODRA_data.json atomically...")
atomic_save_json(OUT_DATA, odra_data_only)
ok, msg = validate_json_file(OUT_DATA)
if not ok:
    raise RuntimeError(f"ODRA_data.json failed validation after save. {msg}")
print("ODRA_data.json validation passed.")

print("Saving ODRA_reps.json atomically...")
atomic_save_json(OUT_REPS, odra_reps_only)
ok, msg = validate_json_file(OUT_REPS)
if not ok:
    raise RuntimeError(f"ODRA_reps.json failed validation after save. {msg}")
print("ODRA_reps.json validation passed.")

print(f"\nSaved ODRA container -> {OUT_ODRA}")
print(f"Saved ODRA data-only  -> {OUT_DATA}")
print(f"Saved ODRA reps-only  -> {OUT_REPS}")
print("Done.")
