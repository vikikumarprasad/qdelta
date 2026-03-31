#!/bin/bash
#SBATCH --job-name=CML_SingleRun
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=10:00:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=20
#SBATCH --output=YOUR_LOG_DIR/CML_Job_%j.out
#SBATCH --error=YOUR_LOG_DIR/CML_Job_%j.err

set -euo pipefail

# runtime settings — can be overridden by passing --export to sbatch
MODEL="${MODEL:-svr}"                       # svr | krr | xgb | gpr | dnn
TARGET_COL="${TARGET_COL:-ae_delta}"        # ae_delta | dh_delta
TUNER="${TUNER:-optuna}"                    # optuna | grid | manual
N_TRIALS="${N_TRIALS:-50}"
SEED="${SEED:-42}"
CV_FOLDS="${CV_FOLDS:-5}"
CV_REPEATS="${CV_REPEATS:-2}"
FEATURE_SELECTION="${FEATURE_SELECTION:-corr90}"
NUM_FEATURES="${NUM_FEATURES:-all}"         # all | N (slices the master feature list)
PCA_COMPONENTS="${PCA_COMPONENTS:-none}"    # none | int
RUN_VARIANTS="${RUN_VARIANTS:-both}"        # all | q9 | both (GPR only)

# paths — edit these to match cluster setup
BASE_DIR="/path/to//project"
DATA_DIR="$BASE_DIR/data"
OUTPUT_ROOT="$BASE_DIR/outputs/CML_Models"
SIF_PATH="$BASE_DIR/CML.sif"
LOG_DIR="$BASE_DIR/logs/CML/Single_Run"

# maps the model name to its Python entrypoint
case "$MODEL" in
  svr) CODE_PATH="$BASE_DIR/HPC_CML_SVR.py" ;;
  krr) CODE_PATH="$BASE_DIR/HPC_CML_KRR.py" ;;
  xgb) CODE_PATH="$BASE_DIR/HPC_CML_XGB.py" ;;
  gpr) CODE_PATH="$BASE_DIR/HPC_CML_GPR.py" ;;
  dnn) CODE_PATH="$BASE_DIR/HPC_CML_DNN.py" ;;
  *)   echo "Unknown MODEL='$MODEL'"; exit 2 ;;
esac

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

# creates a timestamped output directory for this run
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_ROOT/${MODEL^^}_${RUN_STAMP}_$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

# mirrors stdout and stderr into the run directory in addition to the SLURM log files
exec > >(tee -a "$RUN_DIR/stdout.log")
exec 2> >(tee -a "$RUN_DIR/stderr.log" >&2)

echo "========================================"
echo "Starting CML $MODEL job via Apptainer"
echo "Date: $(date)"
echo "JobID: $SLURM_JOB_ID  Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-NA}  Mem: ${SLURM_MEM_PER_NODE:-NA}"
echo "RUN_DIR: $RUN_DIR"
echo "----------------------------------------"

# sets thread counts for all linear algebra libraries
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$SEED"

# builds optional argument arrays to avoid passing empty or stray tokens
PCA_ARGS=()
FEATURES_ARGS=()
VARIANT_ARGS=()

if [ "$PCA_COMPONENTS" != "none" ]; then
  PCA_ARGS=(--pca_components "$PCA_COMPONENTS")
fi

if [ "$NUM_FEATURES" != "all" ]; then
  # edit this list to match dataset's feature columns
  master_features=("feature_1" "feature_2" "feature_3" "feature_4" "feature_5" "feature_6" "feature_7" "feature_8" "feature_9")
  FEATURES_ARGS=(--feature_set)
  FEATURES_ARGS+=("${master_features[@]:0:${NUM_FEATURES}}")
fi

# passes run_variants only to GPR since other models do not support it
if [ "$MODEL" = "gpr" ]; then
  VARIANT_ARGS=(--run_variants "$RUN_VARIANTS")
  if [ "$RUN_VARIANTS" = "both" ] && [ "$PCA_COMPONENTS" != "none" ]; then
    echo "NOTE: RUN_VARIANTS='both' and PCA_COMPONENTS='$PCA_COMPONENTS'."
    echo "      Both variants will use the same PCA setting. For Q9, PCA must be 'none'."
  fi
fi

module load apptainer

echo "----------------------------------------"
echo "Args:"
echo "  MODEL=$MODEL"
echo "  TARGET_COL=$TARGET_COL"
echo "  TUNER=$TUNER  N_TRIALS=$N_TRIALS"
echo "  FEATURE_SELECTION=$FEATURE_SELECTION"
echo "  NUM_FEATURES=$NUM_FEATURES"
echo "  PCA_COMPONENTS=$PCA_COMPONENTS"
[ "$MODEL" = "gpr" ] && echo "  RUN_VARIANTS=$RUN_VARIANTS"
echo "  SEED=$SEED  CV_FOLDS=$CV_FOLDS  CV_REPEATS=$CV_REPEATS"
echo "  CODE_PATH=$CODE_PATH"
echo "----------------------------------------"

# writes a JSON manifest of all run settings for reproducibility
cat > "$RUN_DIR/run_manifest.json" <<EOF
{
  "model": "$MODEL",
  "target_col": "$TARGET_COL",
  "tuner": "$TUNER",
  "n_trials": $N_TRIALS,
  "seed": $SEED,
  "cv_folds": $CV_FOLDS,
  "cv_repeats": $CV_REPEATS,
  "feature_selection": "$FEATURE_SELECTION",
  "num_features": "$NUM_FEATURES",
  "pca_components": "$PCA_COMPONENTS",
  "run_variants": "$RUN_VARIANTS",
  "code_path": "$CODE_PATH",
  "data_dir": "$DATA_DIR",
  "run_dir": "$RUN_DIR",
  "job_id": "$SLURM_JOB_ID"
}
EOF

# writes a status file that gets updated to 'success' or 'failed' on exit
echo "running" > "$RUN_DIR/status.txt"
trap 'ec=$?; echo $ec > "$RUN_DIR/exit_code.txt"; [ $ec -eq 0 ] && echo "success" > "$RUN_DIR/status.txt" || echo "failed" > "$RUN_DIR/status.txt"' EXIT

apptainer exec --bind "$BASE_DIR":"$BASE_DIR" "$SIF_PATH" \
  python3 -u "$CODE_PATH" \
    --model "$MODEL" \
    --target_col "$TARGET_COL" \
    --tuner "$TUNER" \
    --n_trials "$N_TRIALS" \
    --output_dir "$RUN_DIR" \
    --data_dir "$DATA_DIR" \
    --n_jobs "${SLURM_CPUS_PER_TASK:-6}" \
    --seed "$SEED" \
    --cv_folds "$CV_FOLDS" \
    --cv_repeats "$CV_REPEATS" \
    --feature_selection "$FEATURE_SELECTION" \
    "${PCA_ARGS[@]}" \
    "${FEATURES_ARGS[@]}" \
    "${VARIANT_ARGS[@]}"

echo "----------------------------------------"
echo "Done: $(date)"
echo "Artifacts in: $RUN_DIR"
