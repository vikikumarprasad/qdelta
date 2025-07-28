# HPC_QML.py
# Author: Armaan
# Last Updated: July 28, 2025
# Purpose: Main driver script for the QML analysis pipeline.

import argparse
import time
import subprocess
from qml_lib.pipeline import run_pipeline

def setup_arguments():
    """Defines all command-line arguments for the QML pipeline."""
    parser = argparse.ArgumentParser(description="Run a QML pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    run_args = parser.add_argument_group("Script and Run Arguments")
    run_args.add_argument("--data_dir", type=str, default="../../data", help="Directory with train_df.csv and test_df.csv.")
    run_args.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    run_args.add_argument("--seed", type=int, default=42, help="A number to make sure our results are reproducible.")
    run_args.add_argument("--n_jobs", type=int, default=1, help="How many CPUs to use for parallel tasks.")
    run_args.add_argument("--char_samples", type=int, default=5000, help="Samples for PQC characterization (default: 5000).")


    model_args = parser.add_argument_group("Model and PQC Arguments")
    model_args.add_argument("--model", type=str, required=True, choices=["qsvr", "qkrr", "qgpr", "qnn", "qrcr"], help="Which QML model to run.")
    model_args.add_argument("--encoding", type=str, required=True, choices=["hubregtsen", "chebyshev", "yz_cx", "highdim", "kyriienko", "paramz", "chebyshev_rx"], help="Which quantum circuit to use.")
    model_args.add_argument("--qubits", type=int, required=True, help="Number of qubits.")
    model_args.add_argument("--layers", type=int, required=True, help="Maximum number of layers in the circuit.")
    model_args.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgsb", "spsa", "slsqp"], help="Optimizer for QNN/QRCR models.")
    model_args.add_argument("--optimizer_iter", type=int, default=100, help="Number of iterations for the optimizer (kernel training or QNN/QRCR).")
    model_args.add_argument(
        "--reencoding_type", type=str, default="sequential", choices=["sequential", "parallel"],
        help="Re-encoding strategy. 'sequential' uses layers for repetition. 'parallel' uses qubits for repetition of a single feature."
    )
    
    kernel_args = parser.add_argument_group("Quantum Kernel Arguments")
    kernel_args.add_argument("--kernel", type=str, default="projected", choices=["projected", "fidelity"], help="Which type of kernel.")
    kernel_args.add_argument("--train_kernel", action="store_true", help="Set this flag to train the kernel's parameters.")
    kernel_args.add_argument("--param_init", type=str, default="random", choices=["random", "zeros"], help="How to initialize circuit parameters.")
    kernel_args.add_argument("--kernel_optimizer", type=str, default="adam", choices=["adam", "lbfgsb", "spsa", "slsqp"], help="Optimizer for trainable quantum kernels.")

    tuner_args = parser.add_argument_group("Hyperparameter Tuner Arguments")
    tuner_args.add_argument("--tuner", type=str, default="none", choices=["grid", "optuna", "skopt", "raytune", "none"], help="Which hyperparameter tuner to use.")
    tuner_args.add_argument("--n_trials", type=int, default=50, help="How many different settings the tuner should try.")
    tuner_args.add_argument("--cv_type", type=str, default="repeated", choices=["kfold", "repeated"], help="Which type of cross-validation to use.")
    tuner_args.add_argument("--cv_folds", type=int, default=5, help="Number of folds for cross-validation.")
    tuner_args.add_argument("--cv_repeats", type=int, default=3, help="Number of repeats for cross-validation.")

    data_args = parser.add_argument_group("Data and Feature Arguments")
    data_args.add_argument("--features", nargs='+', required=True, help="A list of the feature columns to use from the data files.")
    data_args.add_argument("--pca_components", type=int, default=None, help="If you want to use PCA, specify the number of components.")

    return parser.parse_args()

def main():
    """Main execution block."""
    start_time = time.time()
    args = setup_arguments()
    
    run_pipeline(args)
    
    end_time = time.time()
    print(f"\n--- Pipeline finished in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
