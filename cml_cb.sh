#!/bin/bash
#SBATCH --job-name=CML_SingleRun
#SBATCH --account=rrg-vikikrpd
#SBATCH --time=10:00:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=20
#SBATCH --output=/home/armaank/projects/def-vikikrpd/armaank/logs/CML/Single_Run/CML_Job_%j.out
#SBATCH --error=/home/armaank/projects/def-vikikrpd/armaank/logs/CML/Single_Run/CML_Job_%j.err

set -euo pipefail

# -------- Inputs via sbatch --export (defaults shown) --------
MODEL="${MODEL:-svr}"                  # svr | krr | xgb | gpr | dnn
TARGET_COL="${TARGET_COL:-ae_delta}"   # ae_delta | dh_delta
TUNER="${TUNER:-optuna}"               # optuna | grid | manual
N_TRIALS="${N_TRIALS:-50}"
SEED="${SEED:-17}"
CV_FOLDS="${CV_FOLDS:-5}"
CV_REPEATS="${CV_REPEATS:-2}"
FEATURE_SELECTION="${FEATURE_SELECTION:-corr90}"
NUM_FEATURES="${NUM_FEATURES:-all}"        # all | N (uses master list slicing)
PCA_COMPONENTS="${PCA_COMPONENTS:-none}"   # none | int
RUN_VARIANTS="${RUN_VARIANTS:-both}"        # all | q9 | both   (used only by GPR)

# -------- Paths --------
BASE_DIR="/home/armaank/projects/def-vikikrpd/armaank"
DATA_DIR="$BASE_DIR/data"
OUTPUT_ROOT="$BASE_DIR/Output_Files/CML_Models"
SIF_PATH="$BASE_DIR/CML.sif"
LOG_DIR="$BASE_DIR/logs/CML/Single_Run"

# Map MODEL -> Python entrypoint
case "$MODEL" in
  svr) CODE_PATH="$BASE_DIR/Code_Files/CML_Models/HPC_CML_SVR.py" ;;
  krr) CODE_PATH="$BASE_DIR/Code_Files/CML_Models/HPC_CML_KRR.py" ;;
  xgb) CODE_PATH="$BASE_DIR/Code_Files/CML_Models/HPC_CML_XGB.py" ;;
  gpr) CODE_PATH="$BASE_DIR/Code_Files/CML_Models/HPC_CML_GPR.py" ;;
  dnn) CODE_PATH="$BASE_DIR/Code_Files/CML_Models/HPC_CML_DNN.py" ;;
  *)   echo "Unknown MODEL='$MODEL'"; exit 2 ;;
esac

# Make sure key dirs exist
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

# -------- Per-run output directory --------
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_ROOT/${MODEL^^}_${RUN_STAMP}_$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

# Mirror stdout/stderr into RUN_DIR (keep SLURM .out/.err too)
exec > >(tee -a "$RUN_DIR/stdout.log")
exec 2> >(tee -a "$RUN_DIR/stderr.log" >&2)

echo "========================================"
echo "Starting CML $MODEL job via Apptainer"
echo "Date: $(date)"
echo "JobID: $SLURM_JOB_ID  Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-NA}  Mem: ${SLURM_MEM_PER_NODE:-NA}"
echo "RUN_DIR: $RUN_DIR"
echo "----------------------------------------"

# Threading / determinism
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$SEED"

# -------- Feature / PCA / Variant arg arrays (avoid stray tokens) --------
PCA_ARGS=()
FEATURES_ARGS=()
VARIANT_ARGS=()

if [ "$PCA_COMPONENTS" != "none" ]; then
  PCA_ARGS=(--pca_components "$PCA_COMPONENTS")
fi

if [ "$NUM_FEATURES" != "all" ]; then
  master_features=("exp_mopac" "AE_mopac" "Par_n_Pople" "Mul" "ch_f" "DH_Mopac" "ZPE_TS_R" "Freq" "ZPE_P_R")
  FEATURES_ARGS=(--feature_set)
  FEATURES_ARGS+=("${master_features[@]:0:${NUM_FEATURES}}")
fi

# Only GPR understands --run_variants; pass it conditionally
if [ "$MODEL" = "gpr" ]; then
  VARIANT_ARGS=(--run_variants "$RUN_VARIANTS")
  # Gentle heads-up if user requests 'both' with PCA != none
  if [ "$RUN_VARIANTS" = "both" ] && [ "$PCA_COMPONENTS" != "none" ]; then
    echo "NOTE: RUN_VARIANTS='both' and PCA_COMPONENTS='$PCA_COMPONENTS'."
    echo "      Both variants will use the same PCA setting. For Q9, PCA must be 'none'."
  fi
fi

# -------- Modules / container --------
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

# Write a small run manifest
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

# Clean exit status file
echo "running" > "$RUN_DIR/status.txt"
trap 'ec=$?; echo $ec > "$RUN_DIR/exit_code.txt"; [ $ec -eq 0 ] && echo "success" > "$RUN_DIR/status.txt" || echo "failed" > "$RUN_DIR/status.txt"' EXIT

# Bind BASE_DIR; run container
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
