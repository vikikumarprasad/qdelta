# HPC_QML.py
# Main driver script for the QML analysis pipeline.

import argparse
import time

from qml_lib.pipeline import run_pipeline
from qml_lib.config import MODEL_CONFIG
import qiskit


def setup_arguments():
    parser = argparse.ArgumentParser(
        description="Run a QML pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    run_args = parser.add_argument_group("Script and Run Arguments")
    run_args.add_argument("--data_dir",      type=str, default="../../data",  help="Directory with train_df.csv and test_df.csv.")
    run_args.add_argument("--output_dir",    type=str, required=True,         help="Directory to save results.")
    run_args.add_argument("--seed",          type=int, default=42,            help="Random seed for reproducibility.")
    run_args.add_argument("--n_jobs",        type=int, default=1,             help="Number of CPUs for parallel tasks.")
    run_args.add_argument("--char_samples",  type=int, default=5000,          help="Samples for PQC characterization.")
    run_args.add_argument("--load_custom",   action="store_true",             help="Load components from 'local_kernel.py'.")
    run_args.add_argument("--verbose",       type=int, default=0,             help="Verbosity for tuners/GridSearch/BayesSearch (0..10).")
    run_args.add_argument("--save_model",    action="store_true",             help="Save the final trained model to a file.")

    model_args = parser.add_argument_group("Model and PQC Arguments")
    model_args.add_argument("--model",           type=str, required=True, choices=list(MODEL_CONFIG.keys()), help="QML model to run.")
    model_args.add_argument("--encoding",        type=str, default=None,                                      help="Quantum encoding circuit (standard models only).")
    model_args.add_argument("--qubits",          type=int, required=True,                                     help="Number of qubits.")
    model_args.add_argument("--layers",          type=int, required=True,                                     help="Maximum number of circuit layers.")
    model_args.add_argument("--optimizer",       type=str, default="adam", choices=["adam", "lbfgsb", "spsa", "slsqp"], help="Optimizer for QNN models.")
    model_args.add_argument("--optimizer_iter",  type=int, default=100,                                       help="Iterations for the optimizer.")
    model_args.add_argument("--lr",              type=float, default=None,                                    help="Override QNN learning rate (tuner disabled only).")
    model_args.add_argument("--epochs",          type=int,   default=None,                                    help="Override QNN epochs (tuner disabled only).")
    model_args.add_argument("--batch_size",      type=int,   default=None,                                    help="Override QNN batch size (tuner disabled only).")
    model_args.add_argument("--variance",        type=float, default=None,                                    help="Override QNN variance (tuner disabled only).")
    model_args.add_argument("--reencoding_type", type=str, default="sequential", choices=["sequential", "parallel"], help="Re-encoding strategy.")
    model_args.add_argument("--kernel-gamma",    type=float, default=0.5,                                     help="RBF bandwidth for ProjectedQuantumKernel.")
    model_args.add_argument("--pqk-backend",     type=str, default="auto", choices=["auto", "qiskit", "pennylane"], help="Backend for ProjectedQuantumKernel.")

    kernel_args = parser.add_argument_group("Quantum Kernel Arguments")
    kernel_args.add_argument("--kernel",               type=str, default="projected", choices=["projected", "fidelity"], help="Kernel type.")
    kernel_args.add_argument("--train_kernel",         action="store_true",                                              help="Train the kernel's parameters.")
    kernel_args.add_argument("--param_init",           type=str, default="random",    choices=["random", "zeros"],       help="Circuit parameter initialisation.")
    kernel_args.add_argument("--kernel_optimizer",     type=str, default="adam",      choices=["adam", "lbfgsb", "spsa", "slsqp"], help="Optimizer for trainable kernels.")
    kernel_args.add_argument("--kernel_optimizer_iter", type=int, default=100,                                           help="Max iterations for trainable-kernel optimizer.")

    tuner_args = parser.add_argument_group("Hyperparameter Tuner Arguments")
    tuner_args.add_argument("--tuner",      type=str, default="none", choices=["grid", "optuna", "skopt", "raytune", "none"], help="Hyperparameter tuner.")
    tuner_args.add_argument("--n_trials",   type=int, default=50,                                                             help="Number of tuner trials.")
    tuner_args.add_argument("--cv_type",    type=str, default="repeated", choices=["kfold", "repeated"],                      help="Cross-validation strategy.")
    tuner_args.add_argument("--cv_folds",   type=int, default=5,                                                              help="Number of CV folds.")
    tuner_args.add_argument("--cv_repeats", type=int, default=3,                                                              help="Number of CV repeats.")

    data_args = parser.add_argument_group("Data and Feature Arguments")
    data_args.add_argument("--features",       nargs="+", required=True,                                     help="Feature columns to use from the data files.")
    data_args.add_argument("--pca_components", type=int,  default=None,                                      help="Number of PCA components (optional).")
    data_args.add_argument("--target",         type=str,  default="ae", choices=["ae", "dh"],                help="Target property to model.")
    data_args.add_argument("--mode",           type=str,  default="delta", choices=["delta", "direct", "both"], help="Training mode.")

    return parser.parse_args()


def main():
    # pre-parse the flags needed before full argument setup
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--load_custom", action="store_true")
    parser.add_argument("--verbose",     type=int, default=0)
    parser.add_argument("--n_jobs",      type=int, default=1)
    args, _ = parser.parse_known_args()

    print(f"Qiskit Version in Container: {qiskit.__version__} ---")

    if args.load_custom:
        from qml_lib.local_kernel import register_custom_components
        register_custom_components()

    start_time = time.time()
    args       = setup_arguments()

    # FastKernelRegressor (fidelity/projected) holds only QuantumCircuit + numpy arrays,
    # which are picklable, so n_jobs > 1 is safe with the 'threading' backend.
    # The threading backend also keeps the module-level simulation caches
    # (_PAULI_CACHE / _SV_CACHE) shared across workers rather than per-subprocess.
    # The legacy sQUlearn path is not picklable, so force n_jobs=1 as a safety net.
    is_fast_kernel = (
        args.model in ("qsvr", "qkrr", "qgpr")
        and getattr(args, "kernel", "projected") in ("projected", "fidelity")
    )
    if not is_fast_kernel and args.model in ("qsvr", "qkrr", "qgpr"):
        if args.n_jobs != 1:
            print("[Legacy kernel] Forcing n_jobs=1 to avoid pickling errors.")
        args.n_jobs = 1

    # fixed-feature-map QNN models do not use --encoding
    if args.model in ["qnn-cpmap", "qnn-iqp"] and args.encoding is not None:
        print(f"Warning: --encoding '{args.encoding}' is ignored for model '{args.model}'.")

    # qnn-iqp requires sequential re-encoding and exactly one feature per qubit
    if args.model == "qnn-iqp":
        if args.reencoding_type != "sequential":
            raise ValueError("qnn-iqp requires --reencoding_type sequential.")
        if len(args.features) != args.qubits:
            raise ValueError(
                f"qnn-iqp requires exactly {args.qubits} features (one per qubit); "
                f"got {len(args.features)}."
            )

    run_pipeline(args)

    end_time = time.time()
    print(f"\n--- Pipeline finished in {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
