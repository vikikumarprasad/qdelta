# QDELTA

A pipeline for applying quantum and classical ML to predict corrections to semiempirical (PM7) reaction barrier heights, targeting DFT-level accuracy via delta (Δ)-learning.

---

## How it works

Rather than predicting barrier heights directly, models are trained to predict the error between PM7 and DFT calculations:

```
ΔE = E_DFT − E_PM7
Corrected prediction = E_PM7 + ΔE
```

The pipeline supports both **delta** mode (learn the correction) and **direct** mode (predict DFT values outright).

---

## Repository structure

```
QDELTA/
├── Classical Machine Learning Scripts/
│   ├── HPC_CML_DNN.py
│   ├── HPC_CML_GPR.py
│   ├── HPC_CML_KRR.py
│   ├── HPC_CML_SVR.py
│   └── HPC_CML_XGB.py
├── Quantum Machine Learning Scripts/
│   ├── qml_lib/
│   │   ├── custom/          # CPKernel circuit and Meta-Fibonacci qubit mapping
│   │   ├── components.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── local_kernel.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── reporting.py
│   │   └── tuning.py
│   └── HPC_QML.py
├── data/
│   ├── train_df_new.csv
│   ├── test_df_new.csv
│   └── ...
├── cml_cb.sh               # SLURM script for CML jobs
├── qml_new_array.sh        # SLURM array script for QML sweep
├── CML.def                 # Apptainer container definition
└── QML.def                 # Apptainer container definition
```

---

## Quantum ML

Three kernel-based models are supported: **QSVR**, **QKRR**, and **QGPR**, using either a fidelity quantum kernel (FQK) or projected quantum kernel (PQK).

Seven encoding circuits are available: `chebyshev`, `cpkernel`, `highdim`, `hubregtsen`, `multicontrol`, `paramz`, `yz_cx`.

The full benchmark sweep runs 140 configurations (7 encodings × 4 qubit counts × 5 layer depths) as a SLURM job array via `qml_new_array.sh`. Feature count must match qubit count for sequential re-encoding (the default).

Example run:
```bash
python3 HPC_QML.py \
  --model qgpr \
  --encoding yz_cx \
  --kernel projected \
  --qubits 9 \
  --layers 5 \
  --features ch_f Mul ZPE_TS_P Freq lap_eig_1 SMR_VSA9 Par_n_Pople Balaban_J Labute_ASA \
  --target dh \
  --mode delta \
  --tuner skopt \
  --n_trials 50 \
  --seed 42 \
  --output_dir /path/to/output \
  --data_dir /path/to/data
```

## Classical ML

Each script runs both delta and direct prediction in a single job. Two feature variants are supported: **all** features (with correlation pruning at |r| > 0.9) or a fixed set of **9** SHAP-selected features.

Jobs are submitted via `cml_cb.sh` and configured through `--export` flags at submission:
```bash
sbatch --export=ALL,MODEL=svr,N_TRIALS=50 cml_cb.sh
```

---

## Setup

The pipeline runs on a SLURM cluster using two separate Apptainer containers — one for QML, one for CML. Build them from the repo root:

```bash
apptainer build QML.sif QML.def
apptainer build CML.sif CML.def
```

Set `BASE_DIR` in both SLURM scripts to your project root before submitting.
