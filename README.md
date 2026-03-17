# Movement-Repetition-Counting-for-Wearable-IMUs
 ````markdown
# Wearable Movement Repetition Counting Benchmark for Real-World IMU Data

This repository showcases a **wearable sensing and movement algorithm pipeline** for turning raw **accelerometer and gyroscope time-series data** into robust **repetition counting metrics** across diverse real-world activities, an approach directly aligned with the kind of **movement intelligence, sensor algorithm development, and product-oriented validation** needed in next-generation wearable platforms.

## Overview

This repository contains a benchmarking framework for **task-agnostic repetition counting** using wrist-worn IMU data. It brings together multiple repetition counting methods adapted from prior literature and compares them across several datasets, activities, and users, including a combined benchmark container named **ODRA**.

The project was built to study how well different signal processing pipelines generalize across exercises, rehabilitation tasks, daily activities, and heterogeneous wearable datasets. It includes:

- implementations of several repetition counting methods adapted from published papers
- a unified pipeline to run all methods on multiple datasets
- support for both **accelerometer** and **gyroscope** modalities
- task filtering to exclude extreme outlier activities
- result export to JSON
- plotting and ranking utilities for method comparison

This repo is especially relevant for teams interested in:

- **movement sensing**
- **wearable algorithm development**
- **time-series analysis from IMUs**
- **activity and exercise quantification**
- **benchmarking sensor algorithms across datasets**
- **building robust movement features for production wearables**

---

## Motivation

Accurate repetition counting from wearable sensors remains difficult when algorithms are applied across datasets with different:

- sampling frequencies
- exercise types
- sensor placements
- user populations
- noise levels
- movement patterns

Most published methods perform well only under narrow conditions or on a single dataset. This repository was created to evaluate whether **task-agnostic movement counting** is feasible, and to help identify which method is best depending on the dataset characteristics, sensing modality, and error metric of interest.

---

## Main Contributions

- Unified benchmark of **multiple repetition counting methods** on wearable IMU datasets
- Side-by-side evaluation of methods adapted from prior papers
- Benchmarking on **ODRA**, a combined multi-dataset repetition counting resource
- Support for both **accelerometer** and **gyroscope** inputs
- Structured outputs for downstream ranking, plotting, and analysis
- A framework for studying **generalization, robustness, and method selection** in wearable movement analytics

---

## Repository Structure

```text
.
├── main.py
├── main_correlations.py
├── main_correlationsODRA.py
├── mainODRA.py
├── mainResult2d.py
├── mainResults3D.py
├── mainResultsClasses.py
├── mainResultsDensity.py
├── method1_RecoFit.py
├── method2_FitCoach.py
├── method3_MiLift.py
├── method4_Threshold.py
├── method5_Shimmer3.py
├── method6_uLift.py
├── method7_OURS.py
├── ODRA_creation.py
├── rankingAlgorithms.py
├── rankingODRA.py
├── paperReadyMethods/
├── repsfinal/
├── reps_datasets/
├── paperReadyResultsFinal/
└── utils.py
````

### Core scripts

* `main.py`
  Runs all implemented methods on a selected dataset and saves estimated repetition counts.

* `mainODRA.py`
  Runs all methods on the **ODRA** benchmark container, with support for both accelerometer and gyroscope modes.

* `ODRA_creation.py`
  Creates or organizes the ODRA benchmark dataset structure.

* `main_correlations.py` and `main_correlationsODRA.py`
  Compute and visualize correspondence between estimated repetitions and ground truth.

* `mainResult2d.py`, `mainResults3D.py`, `mainResultsClasses.py`, `mainResultsDensity.py`
  Visualization and analysis scripts for different result views.

* `rankingAlgorithms.py` and `rankingODRA.py`
  Rank methods according to selected performance metrics.

---

## Implemented Methods

Each method is implemented as a paper-based adaptation for cross-dataset benchmarking rather than a literal reproduction of the original publication setup.

### Method 1: RecoFit

Adapted from: **RecoFit: Using a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises**

Pipeline:

* elliptic band-pass filtering
* mean centering
* PCA to reduce triaxial IMU data to a 1D signal
* candidate peak detection
* local period estimation using autocorrelation
* spacing-based pruning
* amplitude-based filtering

### Method 2: FitCoach

Adapted from the FitCoach repetition counting method.

### Method 3a and 3b: MiLift

Adapted from MiLift-based approaches, with multiple variants included.

### Method 4a and 4b: Threshold

Threshold-based repetition counting baselines.

### Method 5: Shimmer3

Adapted from a Shimmer3-inspired repetition counting approach.

### Method 6: uLift

Adapted from the uLift method.

### Method 7a, 7b, 7c: Ours

Three variants of our proposed repetition counting method, designed to improve robustness across heterogeneous activities and datasets.

---

## Datasets

This repository supports benchmarking on several datasets with different sampling rates and activity types.

Examples from the codebase include:

* `RehabSimUCI`
* `recofit`
* `mmfit`
* `uLift`
* `uzh_healthy`
* `uzh_stroke`
* `JUIMU_ROM_ND`
* `JUIMU_ROM_Stroke`
* `crossfit`
* `DWC-Fitness`
* `DWC-Rehab`
* `DWC-ADL`
* `CaRa-Fitness`
* `CaRa-ADL`

### ODRA

A key benchmark structure in this repository is **ODRA**, a combined container used to standardize evaluation across multiple repetition-counting datasets.

ODRA stores:

* dataset-level metadata
* sampling frequency
* raw sensor signals
* repetition ground truth

This makes it possible to evaluate algorithm robustness across very different data sources using a unified interface.

---

## Input Format

Each task typically contains triaxial IMU signals, such as:

### Accelerometer mode

* `ax`
* `ay`
* `az`

### Gyroscope mode

* `gx`
* `gy`
* `gz`

Ground-truth repetition counts are stored separately and matched by:

* dataset
* user
* task

---

## Output Format

Estimated repetitions are saved in nested JSON structures such as:

```json
counts_allMethods[method][dataset][user][task] = predicted_reps
```

or for single-dataset runs:

```json
counts_allMethods[method][user][task] = predicted_reps
```

This structure makes it easy to perform downstream:

* ranking
* plotting
* activity-level comparisons
* dataset-level comparisons
* user-level error analysis

---

## Sensor Modes

The ODRA pipeline supports two sensor configurations:

* `accel`
* `gyro`

Example:

```python
SENSOR_MODE = "accel"   # or "gyro"
```

When running in gyroscope mode, datasets or user-task pairs without valid gyroscope signals can be skipped automatically.

---

## Ignored Tasks

Some tasks are excluded from evaluation to prevent large outliers from dominating comparisons.

Ignored tasks are loaded from:

```text
reps_datasets/ignoredTasks.json
```

This helps keep comparisons more stable and interpretable across methods.

---

## Sampling Frequency Handling

Sampling frequency is dataset dependent. The scripts automatically set `fs` based on the selected dataset or load it from ODRA metadata.

Examples:

* 50 Hz
* 60 Hz
* 80 Hz
* 100 Hz

This is important because many repetition counting methods depend strongly on timing assumptions, filtering cutoffs, and minimum peak spacing in samples.

---

## Example Workflow

### 1. Run all methods on a single dataset

Use `main.py` to:

* load one dataset
* apply all methods
* compare predictions against ground truth
* save results to JSON

### 2. Run all methods on ODRA

Use `mainODRA.py` to:

* load the ODRA container
* choose accelerometer or gyroscope mode
* process all valid dataset-user-task pairs
* save estimated repetitions for all methods

### 3. Visualize and compare methods

Use the result scripts to:

* generate scatter plots
* compare GT vs estimated repetitions
* compute per-activity error
* compute class-level or density-based summaries
* compare methods across activity groups

### 4. Rank methods

Use ranking scripts to identify the best method depending on:

* dataset
* activity class
* sensor modality
* evaluation metric

---

## Example Figures and Analyses

The repository supports analyses such as:

* **GT vs estimated repetitions** per activity
* **mean absolute error scaled by repetition count**
* **method comparison by activity group**
* **cross-user scatter plots**
* **group averages across activity classes**
* **correlation analyses across methods and datasets**

These tools help answer practical questions such as:

* Which repetition counting method generalizes best?
* Which methods perform better on rehabilitation vs fitness tasks?
* Does accelerometer or gyroscope data lead to better counting?
* Which algorithm is most robust to dataset shift?

---

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

Install dependencies:

```bash
pip install numpy scipy matplotlib scikit-learn
```

Depending on your scripts and environment, you may also need:

```bash
pip install pandas
```

---

## Usage

### Run single-dataset benchmark

```bash
python main.py
```

### Run ODRA benchmark

```bash
python mainODRA.py
```

### Generate comparison plots

```bash
python main_correlations.py
python main_correlationsODRA.py
```

### Rank algorithms

```bash
python rankingAlgorithms.py
python rankingODRA.py
```

---

## Example Method Adaptation Note

This repository intentionally treats each algorithm as a **paper-based adaptation** rather than an exact replication of the original publication. In practice, this means:

* parameters may be scaled by sampling frequency
* pipelines may be adjusted for cross-dataset use
* method logic is preserved while enabling more general benchmarking
* implementations are designed to be practical, readable, and comparable

This is especially important when evaluating whether a method can generalize beyond the narrow conditions in which it was originally proposed.

---

## Why This Repo Matters for Wearable Product Teams

For teams building movement features in wearable devices, this repository highlights several core product challenges:

* converting noisy raw IMU streams into reliable movement metrics
* making algorithms robust across users and tasks
* benchmarking signal pipelines before deployment
* understanding tradeoffs between generalization and task-specific tuning
* selecting features and models that can support production wearable experiences

The project reflects hands-on work in:

* **sensor algorithm development**
* **wearable time-series processing**
* **movement analytics**
* **signal processing from multi-axis IMUs**
* **validation across heterogeneous real-world data**

---

## Limitations

* Some methods remain sensitive to dataset-specific characteristics
* Not all papers provide enough detail for exact reproduction
* Task-agnostic repetition counting is still challenging for highly variable motions
* Some scripts use local paths that may need to be adapted before reuse
* Ground-truth conventions may vary across source datasets

---

## Future Improvements

* add standardized evaluation metrics into one shared module
* package all methods into a unified API
* add unit tests
* add dataset loaders with cleaner configuration
* add support for additional wearable datasets
* add summary tables for publication-ready benchmarking
* add cloud-ready batch execution for large-scale experiments

---

## Author

**Guillem Cornella i Barba**
Department of Mechanical and Aerospace Engineering
University of California, Irvine

---

## Citation

If you use this repository in academic work, please cite the relevant associated papers and this repository if applicable.

A BibTeX entry can be added here once the repository is publicly released.

---

## License

Add your preferred license here, for example:

```text
MIT License
```

---

## Contact

For questions, collaboration, or research inquiries:

**Guillem Cornella i Barba**
[cornellg@uci.edu](mailto:cornellg@uci.edu)

```

I can also make this into a more polished **GitHub-ready version with badges, a results snapshot section, and a short recruiter-facing project summary at the top**.
```
