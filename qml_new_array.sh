#!/bin/bash
#SBATCH --job-name=QGPR_SWEEP
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --array=0-139
#SBATCH --time=168:00:00
#SBATCH --mem=18G
#SBATCH --cpus-per-task=5
#SBATCH --output=YOUR_LOG_DIR/QGPR_SWEEP_%A_%a.out
#SBATCH --error=YOUR_LOG_DIR/QGPR_SWEEP_%A_%a.err

set -euo pipefail

# model and tuner settings
MODEL="qgpr"
TUNER="skopt"
N_TRIALS=50

TRAIN_KERNEL="false"
KERNEL="projected"
PARAM_INIT="random"

TARGET="ae"
MODE="delta"
REENCODING="sequential"

CV_TYPE="repeated"
CV_FOLDS=3
CV_REPEATS=1
SEED=42
VERBOSE=10

# 20 (qubits, layers) configurations to sweep
QUBITS_LIST=(3 3 3 3 3   5 5 5 5 5   7 7 7 7 7   9 9 9 9 9)
LAYERS_LIST=(1 2 3 4 5   1 2 3 4 5   1 2 3 4 5   1 2 3 4 5)

# encoding circuits to sweep
ENCODING_LIST=(hubregtsen chebyshev yz_cx highdim paramz cpkernel multicontrol)

N_CONFIGS=${#QUBITS_LIST[@]}
N_ENCODINGS=${#ENCODING_LIST[@]}
TOTAL_JOBS=$((N_CONFIGS * N_ENCODINGS))

if [ ${#LAYERS_LIST[@]} -ne "$N_CONFIGS" ]; then
  echo "ERROR: QUBITS_LIST and LAYERS_LIST must have same length" >&2
  exit 1
fi

# paths — edit these to match your cluster setup
BASE_DIR="/path/to/your/project"
CODE_PATH="$BASE_DIR/HPC_QML.py"
SIF_PATH="$BASE_DIR/QML.sif"
OUTPUT_DIR="$BASE_DIR/outputs/QGPR"
LOG_DIR="$BASE_DIR/logs/QGPR"
DATA_DIR="$BASE_DIR/data"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# decodes the SLURM array task index into encoding and config indices
TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "$TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "ERROR: TASK_ID $TASK_ID >= TOTAL_JOBS $TOTAL_JOBS" >&2
  exit 1
fi

enc_idx=$((TASK_ID / N_CONFIGS))
cfg_idx=$((TASK_ID % N_CONFIGS))

ENCODING=${ENCODING_LIST[$enc_idx]}
qubits=${QUBITS_LIST[$cfg_idx]}
layers=${LAYERS_LIST[$cfg_idx]}

# selects the first N features from the master list where N equals the qubit count
master_features=("feature_1" "feature_2" "feature_3" "feature_4" "feature_5" "feature_6" "feature_7" "feature_8" "feature_9")
features_arg=$(IFS=" "; echo "${master_features[*]:0:$qubits}")

# builds kernel training arguments only if train_kernel is enabled
TRAIN_KERNEL_ARG=""
KOPT_ARGS=()
if [ "$TRAIN_KERNEL" = "true" ]; then
  TRAIN_KERNEL_ARG="--train_kernel"
  KOPT_ARGS+=( --kernel_optimizer "lbfgsb" --kernel_optimizer_iter "100" )
fi

cpus="${SLURM_CPUS_PER_TASK:-5}"

echo "Task $TASK_ID: MODEL=$MODEL, ENC=$ENCODING, ${qubits}Q, ${layers}L, cpus=$cpus | tuner=$TUNER trials=$N_TRIALS"

module load apptainer

export OMP_NUM_THREADS=$cpus
export MKL_NUM_THREADS=$cpus
export OPENBLAS_NUM_THREADS=$cpus
export NUMEXPR_NUM_THREADS=$cpus
export MKL_DYNAMIC=FALSE

srun --ntasks=1 --cpus-per-task=$cpus \
  apptainer exec "$SIF_PATH" python3 "$CODE_PATH" \
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
  --n_jobs "$cpus" \
  --verbose "$VERBOSE" \
  --load_custom
  # --save_model
