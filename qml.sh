#!/bin/bash
# This is a launcher script for submitting QML jobs to a SLURM cluster.

# SBATCH settings for the launcher script itself. It needs very few resources.
#SBATCH --job-name=QML_Launcher
#SBATCH --account=def-vikikrpd
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/armaank/projects/def-vikikrpd/armaank/logs/QML/launcher_%j.out
#SBATCH --error=/home/armaank/projects/def-vikikrpd/armaank/logs/QML/launcher_%j.err

#
# Main settings for the jobs you want to run.
#
MODEL="qsvr"
ENCODING="yz_cx"
TUNER="skopt" # Options: skopt, optuna, grid, raytune

#
# Settings for training the quantum kernel.
#
TRAIN_KERNEL="true"     # Set to "true" to train kernel parameters, "false" to skip.
KERNEL_OPTIMIZER="adam" # Optimizer for kernel training.
PARAM_INIT="random"     # How to initialize circuit parameters if not training.

#
# Other parameters for the model and cross-validation.
#
OPTIMIZER="adam"
KERNEL="projected"
CV_TYPE="repeated"
CV_FOLDS=5
CV_REPEATS=3
SEED=42
TRIALS=100

#
# File paths for your project.
#
BASE_DIR="/home/armaank/projects/def-vikikrpd/armaank"
CODE_PATH="$BASE_DIR/Code_Files/QML_Models/HPC_QML3.py"
SIF_PATH="$BASE_DIR/QML.sif" # Path to your apptainer/singularity container
OUTPUT_DIR="$BASE_DIR/Output_Files/QML_Models"
LOG_DIR="$BASE_DIR/logs/QML/${MODEL}"

# Make sure the directories for the output and logs exist.
mkdir -p "$OUTPUT_DIR/log"
mkdir -p "$LOG_DIR"

# A list of all possible features your model can use.
# The script will pick the first N features from this list, where N is the number of qubits.
master_features=("exp_mopac" "AE_mopac" "Par_n_Pople" "ch_f" "DH_Mopac" "Mul" "ZPE_TS_R" "ZPE_P_R" "Freq")


# This is the main part of the script.
# It loops through different numbers of qubits and layers and submits a separate job for each one.
echo "Starting job submission loop for MODEL=${MODEL}, TUNER=${TUNER}"

for qubits in 3 5 7 9; do
    # Get the first 'n' features from the list above, where n = number of qubits.
    features_arg=$(IFS=" "; echo "${master_features[*]:0:$qubits}")

    for layers in 1 2 3 4 5; do
        JOB_NAME="${MODEL}_${TUNER}_L${layers}_Q${qubits}"
        
        # This part sets different resources (time, memory) for jobs of different sizes.
        # This helps make sure small jobs don't ask for too much, and big jobs get enough.
        if [ "$qubits" -le 3 ]; then
            # Small jobs
            TIME="01:00:00"; MEM="8G"; CPUS=8
        elif [ "$qubits" -le 5 ]; then
            # Medium jobs
            TIME="08:00:00"; MEM="16G"; CPUS=8
        elif [ "$qubits" -le 7 ]; then
            # Large jobs
            TIME="24:00:00"; MEM="32G"; CPUS=8
        else
            # Extra-large jobs (9 qubits)
            TIME="72:00:00"; MEM="64G"; CPUS=8
        fi
        
        echo "Submitting job: ${JOB_NAME} with TIME=${TIME}, MEM=${MEM}, CPUS=${CPUS}"

        # This is the sbatch command that submits the actual job.
        # It uses a "here document" (<<EOF) to pass a script directly to sbatch.
        sbatch --job-name=${JOB_NAME} \
               --cpus-per-task=${CPUS} \
               --mem=${MEM} \
               --time=${TIME} \
               --output=${LOG_DIR}/${JOB_NAME}_%j.out \
               --error=${LOG_DIR}/${JOB_NAME}_%j.err \
               --account=def-vikikrpd <<EOF
#!/bin/bash

# Load the apptainer module so we can run our container.
module load apptainer

# Print some info to the log file to help with debugging.
echo "Job started at: \$(date)"
echo "Job Name: ${JOB_NAME}"
echo "Running on node: \$(hostname)"
echo "Allocated CPUs: \${SLURM_CPUS_PER_TASK}"

# Check if we need to add the --train_kernel flag.
TRAIN_KERNEL_ARG=""
if [ "$TRAIN_KERNEL" == "true" ]; then
    TRAIN_KERNEL_ARG="--train_kernel"
fi

# Run the python script inside the container.
# We pass all the settings from the top of this script as arguments to the python script.
apptainer exec "$SIF_PATH" python3 "$CODE_PATH" \\
    --model "$MODEL" \\
    --encoding "$ENCODING" \\
    --layers "$layers" \\
    --tuner "$TUNER" \\
    --optimizer "$OPTIMIZER" \\
    --qubits "$qubits" \\
    --features $features_arg \\
    --output_dir "$OUTPUT_DIR" \\
    --cv_type "$CV_TYPE" \\
    --cv_folds "$CV_FOLDS" \\
    --cv_repeats "$CV_REPEATS" \\
    --seed "$SEED" \\
    --kernel "$KERNEL" \\
    --n_trials "$TRIALS" \\
    --n_jobs \${SLURM_CPUS_PER_TASK} \\
    \$TRAIN_KERNEL_ARG \\
    --kernel_optimizer "$KERNEL_OPTIMIZER" \\
    --param_init "$PARAM_INIT"

echo "Job finished at: \$(date)"

EOF
        # A short pause to be nice to the SLURM scheduler.
        sleep 1
    done
done

echo "All jobs have been submitted."
