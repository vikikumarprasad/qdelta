#!/bin/bash
# This script runs a "job array" in SLURM.
# A job array is a way to submit many similar jobs at once.
# This script will run 20 times, once for each task ID from 1 to 20.

# SBATCH settings for all jobs in the array.
# Note: Every single job in this array gets the same resources.
#SBATCH --job-name=QML_Sweep
#SBATCH --account=def-vikikrpd
#SBATCH --array=1-20
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/armaank/projects/def-vikikrpd/armaank/logs/QML/Array_Run/QML_Sweep_%A_%a.out
#SBATCH --error=/home/armaank/projects/def-vikikrpd/armaank/logs/QML/Array_Run/QML_Sweep_%A_%a.err

# Main settings for the experiment.
MODEL="qsvr"
ENCODING="hubregtsen"
TUNER="grid" # Options: skopt, optuna, raytune, grid
TRIALS=40    # How many trials for the tuner to run.

# Settings for training the quantum kernel.
TRAIN_KERNEL="true"     # Set to "true" to train kernel parameters.
KERNEL_OPTIMIZER="adam" # Optimizer for kernel training.
PARAM_INIT="random"     # How to initialize circuit parameters if not training.

# Other parameters for the model.
OPTIMIZER="adam"
KERNEL="projected"
CV_TYPE="repeated"
CV_FOLDS=5
CV_REPEATS=3
SEED=42

# File paths for your project.
BASE_DIR="/home/armaank/projects/def-vikikrpd/armaank"
CODE_PATH="$BASE_DIR/Code_Files/QML_Models/HPC_QML3.py"
SIF_PATH="$BASE_DIR/QML.sif" # Path to your apptainer/singularity container
OUTPUT_DIR="$BASE_DIR/Output_Files/QML_Models"

# A list of all possible features your model can use.
master_features=("exp_mopac" "AE_mopac" "Par_n_Pople" "ch_f" "DH_Mopac" "Mul" "ZPE_TS_R" "ZPE_P_R" "Freq")

# These lists map the job array's task ID to a specific experiment.
# For example, task ID 1 will use 3 qubits and 1 layer.
# Task ID 6 will use 5 qubits and 1 layer.
QUBITS_LIST=(3  3  3  3  3  5  5  5  5  5  7  7  7  7  7  9  9  9  9  9)
LAYERS_LIST=(1  2  3  4  5  1  2  3  4  5  1  2  3  4  5  1  2  3  4  5)

# SLURM's task ID starts at 1, but bash array indexes start at 0.
# So we subtract 1 to get the correct index.
INDEX=$(($SLURM_ARRAY_TASK_ID - 1))

# Get the number of qubits and layers for this specific job task.
qubits=${QUBITS_LIST[$INDEX]}
layers=${LAYERS_LIST[$INDEX]}

# A safety check to make sure we got valid parameters.
if [ -z "$qubits" ] || [ -z "$layers" ]; then
    echo "Error: Could not find parameters for Task ID $SLURM_ARRAY_TASK_ID."
    exit 1
fi

# Get the first 'n' features from the list, where n = number of qubits.
features_arg=$(IFS=" "; echo "${master_features[*]:0:$qubits}")

# Check if we need to add the --train_kernel flag.
TRAIN_KERNEL_ARG=""
if [ "$TRAIN_KERNEL" == "true" ]; then
    TRAIN_KERNEL_ARG="--train_kernel"
fi

# Load the apptainer module so we can run our container.
module load apptainer

# Print some info to the log file to help with debugging.
echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: Qubits=$qubits, Layers=$layers, Tuner=$TUNER"
echo "Kernel Training: $TRAIN_KERNEL"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"

# Run the python script inside the container.
apptainer exec "$SIF_PATH" python3 "$CODE_PATH" \
    --model "$MODEL" \
    --encoding "$ENCODING" \
    --layers "$layers" \
    --tuner "$TUNER" \
    --optimizer "$OPTIMIZER" \
    --qubits "$qubits" \
    --features $features_arg \
    --output_dir "$OUTPUT_DIR" \
    --cv_type "$CV_TYPE" \
    --cv_folds "$CV_FOLDS" \
    --cv_repeats "$CV_REPEATS" \
    --seed "$SEED" \
    --kernel "$KERNEL" \
    --n_trials "$TRIALS" \
    --n_jobs "$SLURM_CPUS_PER_TASK" \
    $TRAIN_KERNEL_ARG \
    --kernel_optimizer "$KERNEL_OPTIMIZER" \
    --param_init "$PARAM_INIT"

# Log that the task has finished.
echo "Task $SLURM_ARRAY_TASK_ID Finished."
