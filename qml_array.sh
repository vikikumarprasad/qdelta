#!/bin/bash
#SBATCH --job-name=QML_Array
#SBATCH --account=rrg-vikikrpd
#SBATCH --array=1-20
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/armaank/projects/def-vikikrpd/armaank/logs/QML/New_Array/QKRR/Direct_%A_%a.out
#SBATCH --error=/home/armaank/projects/def-vikikrpd/armaank/logs/QML/New_Array/QKRR/Direct_%A_%a.err

set -euo pipefail

# ----- knobs -----
MODEL="qkrr"                 # qkrr | qsvr | qgpr | qnn-cpmap | qnn-iqp
ENCODING="cpkernel"        # hubregtsen | chebyshev | yz_cx | highdim | paramz | cpkernel | multicontrol
TUNER="skopt"                # grid | skopt | optuna | raytune
N_TRIALS=50

TRAIN_KERNEL="false"
KERNEL="projected"           # fidelity | projected
PARAM_INIT="random"

TARGET="ae"                  # ae | dh
MODE="both"                  # delta | direct | both
REENCODING="sequential"      # sequential | parallel

CV_TYPE="repeated"
CV_FOLDS=3
CV_REPEATS=2
SEED=42
VERBOSE=10                   # see CV progress

# sweep
QUBITS_LIST=(3 3 3 3 3  5 5 5 5 5  7 7 7 7 7  9 9 9 9 9)
LAYERS_LIST=(1 2 3 4 5  1 2 3 4 5  1 2 3 4 5  1 2 3 4 5)

# paths
BASE_DIR="/home/armaank/projects/def-vikikrpd/armaank"
CODE_PATH="$BASE_DIR/Code_Files/QML_Models/HPC_QML.py"
SIF_PATH="$BASE_DIR/QML.sif"
OUTPUT_DIR="$BASE_DIR/Output_Files/QML_Models/New_Array/QKRR"
LOG_DIR="$BASE_DIR/logs/QML/New_Array/QKRR"
DATA_DIR="$BASE_DIR/data"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# select this task
IDX=$((SLURM_ARRAY_TASK_ID - 1))
qubits=${QUBITS_LIST[$IDX]}
layers=${LAYERS_LIST[$IDX]}

# features for N=qubits (order matters!)
master_features=("Par_n_Pople" "AE_mopac" "ZPE_TS_P" "ch_f" "exp_mopac" "Mul" "SMR_VSA9" "lap_eig_1" "Freq")
features_arg=$(IFS=" "; echo "${master_features[*]:0:$qubits}")

# flags
TRAIN_KERNEL_ARG=""
KOPT_ARGS=()
if [ "$TRAIN_KERNEL" = "true" ]; then
  TRAIN_KERNEL_ARG="--train_kernel"
  KOPT_ARGS+=( --kernel_optimizer "lbfgsb" --kernel_optimizer_iter "100" )
fi

echo "Array task $SLURM_ARRAY_TASK_ID running: $MODEL, ${qubits}Q, ${layers}L | tuner=$TUNER trials=$N_TRIALS"

module load apptainer
# Avoid BLAS oversubscription while joblib handles CV parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

# Use srun to bind resources cleanly
srun -n 1 apptainer exec "$SIF_PATH" python3 "$CODE_PATH" \
  --model "$MODEL" \
  --encoding "$ENCODING" \
  --kernel "$KERNEL" \
  --mode "$MODE" \
  --target "$TARGET" \
  --reencoding_type "$REENCODING" \
  --layers "$layers" \
  --qubits "$qubits" \
  --features $features_arg \
  --output_dir "$OUTPUT_DIR" \
  --data_dir "$DATA_DIR" \
  --tuner "$TUNER" \
  --n_trials "$N_TRIALS" \
  --cv_type "$CV_TYPE" \
  --cv_folds "$CV_FOLDS" \
  --cv_repeats "$CV_REPEATS" \
  --seed "$SEED" \
  --param_init "$PARAM_INIT" \
  $TRAIN_KERNEL_ARG "${KOPT_ARGS[@]}" \
  --n_jobs "${SLURM_CPUS_PER_TASK}" \
  --verbose "$VERBOSE" \
  --load_custom
