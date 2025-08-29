#!/bin/bash
#SBATCH --job-name=QML_SingleRun
#SBATCH --account=rrg-vikikrpd
#SBATCH --time=10:00:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/armaank/projects/def-vikikrpd/armaank/logs/QML/Single_Run/QML_Job_%j.out
#SBATCH --error=/home/armaank/projects/def-vikikrpd/armaank/logs/QML/Single_Run/QML_Job_%j.err

# Models (--model):
#   qkrr       : Quantum Kernel Ridge Regression
#   qsvr       : Quantum Support Vector Regression
#   qgpr       : Quantum Gaussian Process Regression (custom subclass)
#   qnn-cpmap  : QNN with CPMap encoding (uses your custom CPMap wrapper)
#   qnn-iqp    : QNN with IQP encoding (requires qubits == features)

# Tuners (--tuner):
#   none       : No tuning, use defaults
#   grid       : GridSearchCV (uses threading backend)
#   optuna     : TPE sampler (seeded by --seed)
#   skopt      : Bayesian (gp_minimize)
#   raytune    : Ray Tune (loguniform/int spaces)

# Encodings (--encoding) for kernel/standard QNN/QRC models:
#   hubregtsen, chebyshev, yz_cx, highdim, , paramz, cpkernel, multicontrol
#   NOTE: --encoding is IGNORED for qnn-cpmap and qnn-iqp (they pick their own).

# Kernels (--kernel) for kernel models:
#   fidelity   : Standard FidelityKernel
#   projected  : ProjectedQuantumKernel (adds Tikhonov regularization internally)

# Re-encoding (--reencoding_type):
#   sequential : Repeat the encoding in depth (uses --layers)
#   parallel   : Repeat a single feature across qubits (forces layers=1)

set -euo pipefail

# Avoid CPU oversubscription from BLAS backends while joblib uses processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ----------------- Experiment knobs -----------------
MODEL="qsvr"
ENCODING="multicontrol"
QUBITS=3
LAYERS=2

# Tuner
TUNER="none"
TRIALS=100
# n_points is auto-chosen in code: min(max(2, n_jobs), 4).
# With 20 CPUs, you'll get n_points=4 automatically.

# Kernel model flags
TRAIN_KERNEL="false"
PARAM_INIT="random"
KERNEL="projected"

# CV
CV_TYPE="repeated"
CV_FOLDS=2
CV_REPEATS=1
SEED=42

# Data/paths
BASE_DIR="/home/armaank/projects/def-vikikrpd/armaank"
DATA_DIR="$BASE_DIR/data"
CODE_PATH="$BASE_DIR/Code_Files/QML_Models/HPC_QML.py"
SIF_PATH="$BASE_DIR/QML.sif"
OUTPUT_DIR="$BASE_DIR/Output_Files/QML_Models"

# Modes and target
MODE="both"     # run delta + direct in one shot
TARGET="ae"     # use 'dh' if you want ΔH instead

# Features (9 to match 9 qubits for sequential encoding)
master_features=("exp_mopac" "AE_mopac" "Par_n_Pople" "ch_f" "SMR_VSA9" "Mul" "ZPE_TS_P" "lap_eig_1" "Freq")

# ----------------------------------------------------

features_arg=$(IFS=" "; echo "${master_features[*]:0:$QUBITS}")
TRAIN_KERNEL_ARG=""
if [ "$TRAIN_KERNEL" == "true" ]; then
    TRAIN_KERNEL_ARG="--train_kernel"
fi

module load apptainer

echo "Starting Single QML Job"
echo "---"
echo "Model: $MODEL | Encoding: $ENCODING | Qubits: $QUBITS | Layers: $LAYERS"
echo "Tuner: $TUNER | Trials: $TRIALS | n_jobs: ${SLURM_CPUS_PER_TASK:-1} (BayesSearchCV will use n_points<=4)"
echo "Kernel: $KERNEL | Train-kernel: $TRAIN_KERNEL"
echo "CV: $CV_TYPE ${CV_FOLDS}x${CV_REPEATS} | Seed: $SEED"
echo "Mode: $MODE | Target: $TARGET"
echo "Features: ${features_arg}"
echo "---"

apptainer exec "$SIF_PATH" python3 "$CODE_PATH" \
    --model "$MODEL" \
    --encoding "$ENCODING" \
    --layers "$LAYERS" \
    --qubits "$QUBITS" \
    --tuner "$TUNER" \
    --n_trials "$TRIALS" \
    --features $features_arg \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --n_jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --seed "$SEED" \
    --kernel "$KERNEL" \
    --cv_type "$CV_TYPE" \
    --cv_folds "$CV_FOLDS" \
    --cv_repeats "$CV_REPEATS" \
    --reencoding_type sequential \
    --mode "$MODE" \
    --target "$TARGET" \
    $TRAIN_KERNEL_ARG \
    --param_init "$PARAM_INIT" \
    --load_custom
