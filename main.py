
import json
import os

# Import algorithms
from paperReadyMethods.method1_RecoFit import method1_fcn
from paperReadyMethods.method2_FitCoach import method2_fcn
from paperReadyMethods.method3_MiLift import method3a_fcn, method3b_fcn
from paperReadyMethods.method4_Threshold import method4a_fcn, method4b_fcn
from paperReadyMethods.method5_Shimmer3 import method5_fcn
from paperReadyMethods.method6_uLift import method6_fcn
from paperReadyMethods.method7_OURS import method7a_fcn, method7b_fcn, method7c_fcn
from utils import find_ndarrays

# Connect to the database
counts_allMethods = {}
counts_method1 = {}
counts_method2 = {}
counts_method3a = {}
counts_method3b = {}
counts_method4a = {}
counts_method4b = {}
counts_method5 = {}
counts_method6 = {}
counts_method7a = {}
counts_method7b = {}
counts_method7c = {}

fs = 50  # Sampling rate

# Report results
model = "RehabSimUCI"
if model in ("recofit", "mmfit", "RehabSimUCI"):
    fs = 50
elif model in ("uLift", "uzh_healthy", "uzh_stroke"):
    fs = 60
elif model in ("JUIMU_ROM_ND", "JUIMU_ROM_Stroke"):
    fs = 80
elif model in ("crossfit", "DWC-Fitness", "DWC-Rehab", "DWC-ADL", "CaRa-Fitness", "CaRa-ADL"):
    fs = 100

else:
    raise ValueError(f"Unknown model: {model}")

repsDB = model + '_reps.json'
rawDB = model + '_data.json'
saveDB = model + '_estimated_reps.json'

filesDirectory = r"repsfinal\Datasets"

with open(os.path.join(filesDirectory, repsDB), 'r') as json_file:
    gt = json.load(json_file)  # Parse the JSON data into a Python list

with open(os.path.join(filesDirectory, rawDB), 'r') as json_file:
    datasetcut = json.load(json_file)

# All user IDs
user_ids = list(datasetcut.keys())
print(user_ids)

# All unique tasks across users
tasks = sorted({task for _, tasks_dict in datasetcut.items() for task in tasks_dict})
print(tasks)

# Ignored tasks to avoid large outliers
with open('repsfinal/Datasets/ignoredTasks.json', 'r') as json_file:
    ignored_tasks = json.load(json_file)
IGNORED_TASKS = set()
for task_list in ignored_tasks.values():
    IGNORED_TASKS.update(task_list)

for user_id in user_ids:
    print(f'********************** User: {user_id} **********************')
    counts_method1[user_id] = {}
    counts_method2[user_id] = {}
    counts_method3a[user_id] = {}
    counts_method3b[user_id] = {}
    counts_method4a[user_id] = {}
    counts_method4b[user_id] = {}
    counts_method5[user_id] = {}
    counts_method6[user_id] = {}
    counts_method7a[user_id] = {}
    counts_method7b[user_id] = {}
    counts_method7c[user_id] = {}

    for t in range(0, len(tasks)):
        task = tasks[t]
        if task not in datasetcut[user_id] or task in IGNORED_TASKS:
            continue

        if task not in gt.get(user_id, {}):
            continue

        print(f'--- Task: {task} --- with GT: {gt[user_id][task]}', )

        x = datasetcut[user_id][task]["ax"]
        y = datasetcut[user_id][task]["ay"]
        z = datasetcut[user_id][task]["az"]

        # Add to dict
        method1_reps = method1_fcn(x, y, z, fs, plot=False, info=[user_id, task])
        counts_method1[user_id][task] = method1_reps

        method2_reps = method2_fcn(x, y, z, fs, plot=False, info=[user_id, task])
        counts_method2[user_id][task] = method2_reps

        method3a_reps = method3a_fcn(x, y, z, fs, plot=False, info=[user_id, task])
        counts_method3a[user_id][task] = method3a_reps

        method3b_reps = method3b_fcn(x, y, z, fs, plot=False, info=[user_id, task])
        counts_method3b[user_id][task] = method3b_reps

        method4a_reps = method4a_fcn(x, y, z, fs, plot=False, info=[user_id, task])
        counts_method4a[user_id][task] = method4a_reps

        method4b_reps = method4b_fcn(x, y, z, fs, plot=False, info=[user_id, task])
        counts_method4b[user_id][task] = method4b_reps

        method5_reps = method5_fcn(x, y, z, fs, plot=False, info=[user_id, task])
        counts_method5[user_id][task] = method5_reps

        method6_reps = method6_fcn(x, y, z, fs=fs, plot=False, info=[user_id, task])
        counts_method6[user_id][task] = method6_reps

        method7a_reps = method7a_fcn(x, y, z, fs, plot=False, info=[user_id, task])
        counts_method7a[user_id][task] = method7a_reps

        method7b_reps = method7b_fcn(x, y, z, fs, plot=False, info=[user_id, task])
        counts_method7b[user_id][task] = method7b_reps

        method7c_reps = method7c_fcn(x, y, z, fs, plot=False, info=[user_id, task])
        counts_method7c[user_id][task] = method7c_reps

        print(' and estimated: ', method7a_reps)


# Open the file in write mode and use json.dump() to write the dictionary to the file
counts_allMethods['method1'] = counts_method1
counts_allMethods['method2'] = counts_method2
counts_allMethods['method3a'] = counts_method3a
counts_allMethods['method3b'] = counts_method3b
counts_allMethods['method4a'] = counts_method4a
counts_allMethods['method4b'] = counts_method4b
counts_allMethods['method5'] = counts_method5
counts_allMethods['method6'] = counts_method6
counts_allMethods['method7a'] = counts_method7a
counts_allMethods['method7b'] = counts_method7b
counts_allMethods['method7c'] = counts_method7c

save = True
if save:
    try:
        with open("paperReadyResultsFinal/" + saveDB, 'w') as json_file:
            find_ndarrays(counts_allMethods)
            json.dump(counts_allMethods, json_file, indent=4)  # indent=4 for pretty printing
        print(f"Dictionary successfully saved")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

