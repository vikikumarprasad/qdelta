# QDELTA: Quantum and Classical Machine Learning for Computational Chemistry

This repository contains a scientific computing pipeline that applies both Quantum Machine Learning (QML) and Classical Machine Learning (CML) to improve the accuracy of computational chemistry predictions. All experiments are designed to run on a High-Performance Computing (HPC) cluster using SLURM and Apptainer containers.

---

## Table of Contents

1. [The Science: Delta-Learning](#the-science-delta-learning)
2. [Repository Structure](#repository-structure)
3. [Data](#data)
4. [Software Containers](#software-containers)
5. [Quantum ML Pipeline](#quantum-ml-pipeline)
   - [Models](#qml-models)
   - [Encoding Circuits](#encoding-circuits)
   - [Kernels](#kernels)
   - [Hyperparameter Tuners](#hyperparameter-tuners)
   - [Running a QML Job](#running-a-qml-job)
   - [QML SLURM Array Script](#qml-slurm-array-script)
6. [Classical ML Pipeline](#classical-ml-pipeline)
   - [Models](#cml-models)
   - [Feature Variants](#feature-variants)
   - [Running a CML Job](#running-a-cml-job)
7. [Output Structure](#output-structure)
8. [Custom CPKernel Encoding](#custom-cpkernel-encoding)
9. [Key Configuration Notes](#key-configuration-notes)

---

## The Science: Delta-Learning

Many computational chemistry methods sit on a spectrum between speed and accuracy. Semi-empirical methods like **PM7** are fast and can be run on thousands of molecules, but introduce significant errors. High-accuracy methods like **Density Functional Theory (DFT)** produce reliable results but are computationally expensive and cannot be applied at scale.

**Delta-Learning (Δ-learning)** bridges this gap. Instead of training a model to predict a molecular property directly, it trains a model to predict the *error* of a cheap method relative to an expensive one:

$$\Delta = \text{DFT value} - \text{PM7 value}$$

Once trained, the model is applied as a correction:

$$\text{Corrected prediction} = \text{PM7 value} + \hat{\Delta}$$

This repository targets two molecular properties:

- **Atomization Energy (AE)** — the energy required to break a molecule into its constituent atoms
- **Enthalpy of Reaction (ΔH)** — the heat of reaction

Both properties are predicted in **kcal/mol**. The pipeline also supports a **direct** mode, where the model is trained to predict the DFT value outright without a PM7 correction, and a **both** mode that runs delta and direct in a single job.

---

## Repository Structure

```
QDELTA/
│
├── Classical Machine Learning Scripts/
│   ├── HPC_CML_DNN.py          # Deep Neural Network
│   ├── HPC_CML_GPR.py          # Gaussian Process Regression
│   ├── HPC_CML_KRR.py          # Kernel Ridge Regression
│   ├── HPC_CML_SVR.py          # Support Vector Regression
│   └── HPC_CML_XGB.py          # XGBoost
│
├── Quantum Machine Learning Scripts/
│   ├── qml_lib/                # Core QML library
│   │   ├── custom/             # Custom encoding circuits
│   │   │   ├── kernel.py       # CPKernel circuit definition
│   │   │   └── utility.py      # Meta-Fibonacci qubit mapping utilities
│   │   ├── components.py       # PQC and kernel factory functions
│   │   ├── config.py           # Central config: models, encodings, optimizers
│   │   ├── data.py             # Data loading, scaling, PCA, validation
│   │   ├── local_kernel.py     # CPKernelWrapper (sQUlearn interface) + registration
│   │   ├── models.py           # Model factory (QSVR, QKRR, QGPR)
│   │   ├── pipeline.py         # Full pipeline orchestration
│   │   ├── reporting.py        # CSV saving, summary logging, PDF plot generation
│   │   └── tuning.py           # Hyperparameter tuners (grid, optuna, skopt, raytune)
│   └── HPC_QML.py              # Main QML driver script
│
├── data/
│   ├── train_df_new.csv        # Training set (features + targets)
│   ├── test_df_new.csv         # Test set (features + targets)
│   ├── smiles_db.csv           # Source database with SMILES strings
│   └── wb97xd3.csv             # Additional DFT reference data
│
├── cml_cb.sh                   # SLURM single-run script for CML models
├── qml_new_array.sh            # SLURM array script for QML sweep
├── CML.def                     # Apptainer definition file for CML container
├── QML.def                     # Apptainer definition file for QML container
└── ReadMe.md
```

---

## Data

All data files live in the `data/` directory. The two primary files used during training and evaluation are:

**`train_df_new.csv`** and **`test_df_new.csv`**

Both files share the same column schema. The key columns are:

| Column | Description |
|---|---|
| `ae_delta` | Target: AE(DFT) − AE(PM7) — used in delta mode |
| `ae_dft` | Target: raw DFT atomization energy — used in direct mode |
| `AE_mopac` | PM7 baseline for atomization energy |
| `dh_delta` | Target: ΔH(DFT) − ΔH(PM7) — used in delta mode |
| `dh_dft` | Target: raw DFT reaction enthalpy — used in direct mode |
| `DH_Mopac` | PM7 baseline for reaction enthalpy |
| All others | Molecular descriptor features |

The data loading module (`data.py`) automatically detects the PM7 column using a fallback search if the expected column name is not found. All features are scaled to **[−1, 1]** using MinMaxScaler fitted on the training set only. An optional PCA reduction can be applied after scaling.

---

## Software Containers

This project uses two separate Apptainer containers — one for classical ML and one for quantum ML — to keep the dependency stacks cleanly separated.

### Building the containers

From the root of the repository on the HPC login node:

```bash
# Build the QML container (qiskit, squlearn, pennylane, optuna, skopt, ray)
apptainer build QML.sif QML.def

# Build the CML container (tensorflow, xgboost, sklearn, shap, optuna)
apptainer build CML.sif CML.def
```

Both builds take several minutes. The resulting `.sif` files should be placed in your `BASE_DIR` on the cluster, wherever your SLURM scripts expect them.

### What each container provides

**`QML.sif`** (built from `QML.def`):
- Python 3, NumPy, pandas, SciPy, scikit-learn, matplotlib
- `qiskit`, `squlearn`, `pennylane`, `pennylane-qiskit`
- `optuna`, `scikit-optimize`, `ray[tune]`
- `rdkit-pypi`, `shap`, `symengine`, `pyDOE2`

**`CML.sif`** (built from `CML.def`):
- Python 3, NumPy, pandas, scikit-learn, matplotlib, seaborn
- `tensorflow==2.13.0`
- `xgboost`, `shap`, `optuna`, `joblib`

> **Note:** `CML.def` currently contains a `%files` section that copies a local directory into the container at build time. Before building on a new system, remove or update that section so the build does not depend on paths specific to the original cluster.

---

## Quantum ML Pipeline

The QML pipeline is fully modular. All behaviour is controlled through command-line arguments passed to `HPC_QML.py`. The library code lives in `qml_lib/`.

### QML Models

Three kernel-based quantum regression models are supported, all from the `sQUlearn` library:

| Argument | Model | Description |
|---|---|---|
| `qsvr` | Quantum SVR | Support vector regression with a quantum kernel |
| `qkrr` | Quantum KRR | Kernel ridge regression with a quantum kernel |
| `qgpr` | Quantum GPR | Gaussian process regression with a quantum kernel |

### Encoding Circuits

The encoding circuit (Parameterised Quantum Circuit, or PQC) maps classical feature vectors into the quantum Hilbert space. The circuit is specified with `--encoding`:

| Argument | Circuit |
|---|---|
| `yz_cx` | YZ-CX encoding circuit |
| `highdim` | High-dimensional encoding circuit |
| `hubregtsen` | Hubregtsen encoding circuit |
| `chebyshev` | Chebyshev PQC |
| `multicontrol` | Multi-control encoding circuit |
| `paramz` | Parameterised Z feature map |
| `cpkernel` | Custom CPKernel (see [Custom CPKernel](#custom-cpkernel-encoding)) |

All standard circuits are provided by `sQUlearn`. The `cpkernel` encoding is a custom circuit defined in `qml_lib/custom/kernel.py` and must be loaded with the `--load_custom` flag.

### Kernels

Two quantum kernel types are supported, specified with `--kernel`:

- **`projected`** (default): Projected Quantum Kernel (PQK) with a Gaussian outer kernel. The bandwidth `gamma` can be set with `--kernel-gamma` or tuned automatically.
- **`fidelity`**: Fidelity-based quantum kernel.

Both kernels default to the **Qiskit Aer statevector simulator** backend and fall back to PennyLane if Aer is unavailable. The backend can be forced with `--pqk-backend [auto|qiskit|pennylane]`.

> **Important:** Projected kernels use `Executor` objects that are not picklable. The pipeline automatically forces `--n_jobs 1` for projected kernel models to prevent multiprocessing errors.

Kernel parameters can be trained by passing `--train_kernel`. This is currently supported only for QGPR. For QSVR and QKRR, a warning is printed and training is skipped. The kernel optimizer is controlled by `--kernel_optimizer` and `--kernel_optimizer_iter`.

### Hyperparameter Tuners

The tuner is selected with `--tuner`. All tuners use cross-validated MAE as the objective.

| Argument | Tuner | Notes |
|---|---|---|
| `none` | No tuning | Uses model defaults |
| `grid` | GridSearchCV | Expands search space specs into explicit value lists |
| `optuna` | Optuna (TPE) | Supports pruning; n_jobs=1 internally, parallelism handled per fold |
| `skopt` | BayesSearchCV | Tunes gamma separately for projected kernels |
| `raytune` | Ray Tune | Requires Ray to be installed and initialised |

Cross-validation is configured with `--cv_type [kfold|repeated]`, `--cv_folds`, and `--cv_repeats`. The number of tuning trials is set with `--n_trials`.

### Running a QML Job

The main script is `HPC_QML.py`. It is not run directly — it is submitted to SLURM via a container. A minimal example command (inside the container) looks like this:

```bash
python3 HPC_QML.py \
  --model qgpr \
  --encoding hubregtsen \
  --kernel projected \
  --qubits 5 \
  --layers 3 \
  --features ch_f Mul ZPE_TS_P Freq lap_eig_1 \
  --target ae \
  --mode delta \
  --tuner skopt \
  --n_trials 50 \
  --cv_type repeated \
  --cv_folds 3 \
  --cv_repeats 1 \
  --seed 42 \
  --param_init random \
  --output_dir /path/to/output \
  --data_dir /path/to/data \
  --n_jobs 1 \
  --load_custom
```

**Key arguments:**

| Argument | Description |
|---|---|
| `--model` | One of `qsvr`, `qkrr`, `qgpr` |
| `--encoding` | Encoding circuit name (see table above) |
| `--qubits` | Number of qubits. For sequential re-encoding, must equal the number of features |
| `--layers` | Maximum number of circuit layers |
| `--features` | List of feature column names from the CSV. For sequential encoding, provide exactly `--qubits` features |
| `--target` | `ae` for atomization energy, `dh` for reaction enthalpy |
| `--mode` | `delta`, `direct`, or `both` |
| `--reencoding_type` | `sequential` (default) or `parallel`. Sequential requires features == qubits. Parallel requires exactly one feature |
| `--param_init` | `random` or `zeros` for circuit parameter initialisation |
| `--load_custom` | Must be passed when using `--encoding cpkernel` |
| `--save_model` | If set, saves the final trained model as a `.joblib` file |
| `--pca_components` | If set, reduces features to this many PCA components before training |

### QML SLURM Array Script

`qml_new_array.sh` submits a sweep of 140 jobs (7 encodings × 20 qubit/layer configurations) as a SLURM job array. Before submitting, edit the following at the top of the script:

```bash
BASE_DIR="/path/to/your/project"   # set to your project root
```

Also update `--account`, `--output`, and `--error` in the SBATCH header to match your cluster account and log directory.

The feature list is defined as a master array and sliced based on qubit count so that the number of features always equals the number of qubits:

```bash
master_features=("feature_1" "feature_2" ... "feature_9")
features_arg=$(IFS=" "; echo "${master_features[*]:0:$qubits}")
```

Replace the placeholder feature names with your actual dataset column names. The qubit sweep is defined by `QUBITS_LIST` and `LAYERS_LIST`. The encoding sweep is defined by `ENCODING_LIST`. The total number of jobs is `N_CONFIGS × N_ENCODINGS` and must match `--array=0-N` in the SBATCH header.

Submit with:

```bash
sbatch qml_new_array.sh
```

---

## Classical ML Pipeline

Each CML model is a self-contained Python script. All five scripts share the same general structure: correlation pruning, Optuna tuning, final model fitting, and artifact saving.

### CML Models

| Script | Model |
|---|---|
| `HPC_CML_SVR.py` | Support Vector Regression (sklearn) |
| `HPC_CML_KRR.py` | Kernel Ridge Regression (sklearn) |
| `HPC_CML_GPR.py` | Gaussian Process Regression (sklearn) |
| `HPC_CML_XGB.py` | XGBoost with early stopping |
| `HPC_CML_DNN.py` | Dense Neural Network (TensorFlow/Keras) |

All models run **both delta-learning and direct DFT** prediction in a single job and output results for both.

### Feature Variants

Every CML script supports two feature variants that can be run in a single job using `--run_variants [all|q9|both]`:

- **ALL**: Uses all available molecular descriptor features, with train-only correlation pruning at |r| > 0.90.
- **Q9**: Uses a fixed set of 9 features (`exp_mopac`, `AE_mopac`, `Par_n_Pople`, `Mul`, `ch_f`, `DH_Mopac`, `ZPE_TS_R`, `Freq`, `ZPE_P_R`). Correlation pruning is still applied within this subset.

The correlation pruning step removes highly redundant features computed from the training set only to prevent data leakage.

### Running a CML Job

The CML SLURM script is `cml_cb.sh`. It is designed to be fully configurable via `sbatch --export` without modifying the script itself. Before submitting, update the path variables at the top:

```bash
BASE_DIR="/path/to/your/project"
```

Also update `--account`, `--output`, and `--error` in the SBATCH header.

**Submitting with default settings:**

```bash
sbatch cml_cb.sh
```

**Overriding settings at submission time:**

```bash
sbatch --export=ALL,MODEL=gpr,TARGET_COL=dh_delta,N_TRIALS=100,RUN_VARIANTS=both cml_cb.sh
```

**Configurable variables:**

| Variable | Default | Options |
|---|---|---|
| `MODEL` | `svr` | `svr`, `krr`, `xgb`, `gpr`, `dnn` |
| `TARGET_COL` | `ae_delta` | `ae_delta`, `dh_delta` |
| `TUNER` | `optuna` | `optuna`, `grid`, `manual` |
| `N_TRIALS` | `50` | any integer |
| `SEED` | `42` | any integer |
| `CV_FOLDS` | `5` | any integer |
| `CV_REPEATS` | `2` | any integer |
| `FEATURE_SELECTION` | `corr90` | `corr90`, `none` |
| `NUM_FEATURES` | `all` | `all` or an integer N (slices the master feature list) |
| `PCA_COMPONENTS` | `none` | `none` or an integer |
| `RUN_VARIANTS` | `both` | `all`, `q9`, `both` (GPR and SVR/KRR/DNN) |

The script automatically maps `MODEL` to the correct Python entrypoint, creates a timestamped output directory, mirrors stdout/stderr into that directory, writes a `run_manifest.json`, and updates a `status.txt` file on exit.

> **Note for XGB:** The XGBoost script does not use Optuna tuning. It uses a fixed set of hyperparameters stored in the `BEST_HP` dictionary at the top of `HPC_CML_XGB.py`. Replace these values with the output of your own Optuna study before running at scale.

> **Note for GPR:** GPR is O(n³) in the number of training samples. Keep `CV_FOLDS` and `CV_REPEATS` modest, and consider enabling PCA with `PCA_COMPONENTS` to reduce dimensionality and stabilise training.

---

## Output Structure

### QML Outputs

Results are saved under `--output_dir` in the following structure:

```
output_dir/
├── summary_log.csv                  # One row per run; appended across all jobs
├── master_predictions.csv           # All raw predictions appended across all jobs
└── <CONFIG>/<UNIQUE_ID>/
    ├── results_predictions_<mode>.csv
    └── results_<mode>_dft.pdf
```

The `<CONFIG>` folder name is assembled from the run settings, for example:
`QGPR_hubregtsen_5Q_3L_Tuner-skopt_DELTA`

The `<UNIQUE_ID>` is the SLURM job ID and array task ID if running on a cluster, or a timestamp for local runs.

`summary_log.csv` records all configuration details and final metrics (MAE, STD, R²) for every run, making it easy to compare results across a sweep.

### CML Outputs

Each CML job creates a timestamped directory under `OUTPUT_ROOT`:

```
outputs/CML_Models/<MODEL>_<TIMESTAMP>_<JOBID>/
├── stdout.log
├── stderr.log
├── run_manifest.json               # Full run configuration
├── status.txt                      # 'running', 'success', or 'failed'
├── exit_code.txt
└── <variant>/                      # one folder per feature variant (all, q9)
    ├── plots/
    │   ├── <model>_parity_delta_only.pdf
    │   ├── <model>_parity_dft_from_delta.pdf
    │   └── <model>_parity_direct_dft.pdf
    ├── <model>_predictions.csv
    ├── <model>_*_pipeline.joblib
    ├── <model>_*_optuna_trials.csv
    ├── <model>_*_best_params.json
    ├── dropped_corr90.txt
    ├── features_after_prune.txt
    ├── metrics.json
    └── REPORT.md
```

All parity plots follow a consistent publication style: blue scatter points, dashed red y=x line, and a beige stats box showing R², MAE, and SD.

---

## Custom CPKernel Encoding

The CPKernel is a custom encoding circuit developed for this project. It is not part of `sQUlearn`'s standard library and lives in `qml_lib/custom/`.

The circuit layout is determined by a **Meta-Fibonacci qubit mapping** (`utility.py`). It has **6 trainable kernel parameters** (alpha, beta, gamma, param1, param2, param3) defined as symbolic Qiskit `Parameter` objects, separate from the feature parameters, and can be trained via `--train_kernel`.

The `CPKernelWrapper` in `local_kernel.py` adapts the circuit to the `EncodingCircuitBase` interface that `sQUlearn` expects. It is registered into the encoding map at runtime by passing `--load_custom` to `HPC_QML.py`.

```bash
python3 HPC_QML.py \
  --model qgpr \
  --encoding cpkernel \
  --load_custom \
  ...
```
---

## Key Configuration Notes

**Sequential vs. Parallel re-encoding:**
- `--reencoding_type sequential` (default): Features are encoded across qubits. The number of features must equal `--qubits`.
- `--reencoding_type parallel`: A single feature is repeated across all qubits and layers. Only one feature may be provided.

**Feature count and qubit count must match for sequential encoding.** The data loading step will raise a `ValueError` if they do not.

**Thread control:** Both pipelines set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to the number of requested CPUs to prevent oversubscription on shared cluster nodes.

**Reproducibility:** All scripts accept a `--seed` argument. The seed is passed to NumPy, TensorFlow (DNN), and all tuners. SLURM job IDs are included in output folder names to ensure no two runs overwrite each other.

**Extending the pipeline:** New encoding circuits can be added by implementing `EncodingCircuitBase` from `sQUlearn` and registering the class in `ENCODING_MAP` in `config.py`. New classical models can be added to any of the CML scripts by following the existing structure in `run_variant`.