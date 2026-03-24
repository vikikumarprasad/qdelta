# HPC_QML.py
# Author: Armaan
# Main driver script for the QML analysis pipeline.

import argparse
import time
import inspect

import qml_lib.pipeline
print(f"Loading pipeline module from: {inspect.getfile(qml_lib.pipeline)}")

from qml_lib.pipeline import run_pipeline
from qml_lib.config import MODEL_CONFIG
import qiskit


def setup_arguments():
    """Defines and parses all command-line arguments for the QML pipeline."""
    parser = argparse.ArgumentParser(
        description="Run a QML pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    run_args = parser.add_argument_group("Script and Run Arguments")
    run_args.add_argument("--data_dir", type=str, default="../../data", help="Directory with train_df.csv and test_df.csv.")
    run_args.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    run_args.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    run_args.add_argument("--n_jobs", type=int, default=1, help="Number of CPUs to use for parallel tasks.")
    run_args.add_argument("--char_samples", type=int, default=5000, help="Samples for PQC characterization.")
    run_args.add_argument("--load_custom", action="store_true", help="Load custom components from 'local_kernel.py'.")
    run_args.add_argument("--verbose", type=int, default=0, help="Verbosity level for tuners and GridSearch (0..10).")
    run_args.add_argument("--save_model", action="store_true", help="Save the final trained model object to a file.")

    model_args = parser.add_argument_group("Model and PQC Arguments")
    model_args.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIG.keys()), help="Which QML model to run.")
    model_args.add_argument("--encoding", type=str, default=None, help="Which encoding circuit to use.")
    model_args.add_argument("--qubits", type=int, required=True, help="Number of qubits.")
    model_args.add_argument("--layers", type=int, required=True, help="Maximum number of circuit layers.")
    model_args.add_argument(
        "--reencoding_type",
        type=str,
        default="sequential",
        choices=["sequential", "parallel"],
        help="Re-encoding strategy: 'sequential' repeats across layers, 'parallel' repeats across qubits.",
    )
    model_args.add_argument(
        "--kernel-gamma",
        type=float,
        default=0.5,
        help="Gaussian outer-kernel bandwidth for ProjectedQuantumKernel.",
    )
    model_args.add_argument(
        "--pqk-backend",
        type=str,
        default="auto",
        choices=["auto", "qiskit", "pennylane"],
        help="Simulation backend for ProjectedQuantumKernel. 'auto' tries Qiskit Aer first.",
    )

    kernel_args = parser.add_argument_group("Quantum Kernel Arguments")
    kernel_args.add_argument("--kernel", type=str, default="projected", choices=["projected", "fidelity"], help="Type of quantum kernel to use.")
    kernel_args.add_argument("--train_kernel", action="store_true", help="Train the kernel's parameters.")
    kernel_args.add_argument("--param_init", type=str, default="random", choices=["random", "zeros"], help="Circuit parameter initialization strategy.")
    kernel_args.add_argument("--kernel_optimizer", type=str, default="adam", choices=["adam", "lbfgsb", "spsa", "slsqp"], help="Optimizer used when training the kernel.")
    kernel_args.add_argument("--kernel_optimizer_iter", type=int, default=100, help="Max iterations for the kernel optimizer.")

    tuner_args = parser.add_argument_group("Hyperparameter Tuner Arguments")
    tuner_args.add_argument("--tuner", type=str, default="none", choices=["grid", "optuna", "skopt", "raytune", "none"], help="Hyperparameter tuner to use.")
    tuner_args.add_argument("--n_trials", type=int, default=50, help="Number of hyperparameter configurations to evaluate.")
    tuner_args.add_argument("--cv_type", type=str, default="repeated", choices=["kfold", "repeated"], help="Cross-validation strategy.")
    tuner_args.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds.")
    tuner_args.add_argument("--cv_repeats", type=int, default=3, help="Number of CV repetitions.")

    data_args = parser.add_argument_group("Data and Feature Arguments")
    data_args.add_argument("--features", nargs="+", required=True, help="Feature columns to use from the data files.")
    data_args.add_argument("--pca_components", type=int, default=None, help="Number of PCA components to reduce features to.")
    data_args.add_argument(
        "--target",
        type=str,
        default="ae",
        choices=["ae", "dh"],
        help="Target property to model: 'ae' for atomization energy, 'dh' for enthalpy.",
    )
    data_args.add_argument(
        "--mode",
        type=str,
        default="delta",
        choices=["delta", "direct", "both"],
        help="Training mode: 'delta' corrects PM7, 'direct' trains on DFT, 'both' runs both.",
    )

    return parser.parse_args()


def main():
    """Entry point: pre-parses early flags, then runs the full pipeline."""
    # pre-parses flags needed before the full argument setup
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--load_custom", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    args, _ = parser.parse_known_args()

    print(f"Qiskit Version in Container: {qiskit.__version__} ---")

    if args.load_custom:
        from qml_lib.local_kernel import register_custom_components
        register_custom_components()

    start_time = time.time()
    args = setup_arguments()

    # forces single-threaded execution for projected kernels since Executor objects are not picklable
    if getattr(args, "kernel", "projected") == "projected":
        if args.n_jobs != 1:
            print("[PQK] Forcing n_jobs=1 for projected kernel to avoid pickling errors.")
        args.n_jobs = 1

    run_pipeline(args)

    end_time = time.time()
    print(f"\n--- Pipeline finished in {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()