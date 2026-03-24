# pipeline.py
# Orchestrates the full QML pipeline: data loading, tuning, training, evaluation, and saving results.

import os
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from .config import MODEL_CONFIG
from .data import load_data
from .models import create_model_from_params
from .tuning import tune_model
from .reporting import save_data_outputs, generate_results_plot


def run_pipeline(args):
    """
    Runs the full QML pipeline from data loading through to saving results.

    Supports 'delta', 'direct', and 'both' modes, running a separate pass for each.
    Applies target scaling for direct mode to stabilize training.
    """
    print(f"--- Starting QML Pipeline: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    print(
        f"Run Config: Model={args.model.upper()}, Tuner={args.tuner.upper()}, "
        f"PQC={args.encoding}, Qubits={args.qubits}, Max Layers={args.layers}, Seed={args.seed}"
    )
    print(
        f"Kernel Config: Train={args.train_kernel}, "
        f"Optimizer={args.kernel_optimizer if args.train_kernel else 'N/A'}"
    )
    print(f"Target: {getattr(args, 'target', 'ae').upper()} | Mode: {getattr(args, 'mode', 'delta')}")
    print(f"Features: {args.features}\n")

    # determines which modes to run
    mode_arg = getattr(args, "mode", "delta")
    if mode_arg not in ("delta", "direct", "both"):
        mode_arg = "delta"
    modes_to_run = ["delta", "direct"] if mode_arg == "both" else [mode_arg]

    # optional fairness toggle that drops the PM7 baseline feature in direct mode
    fair_direct = bool(getattr(args, "fair_direct", False))

    for mode in modes_to_run:
        print(f"\n=== [{mode.upper()}] Pass ===")

        features_local = list(args.features)
        original_qubits = args.qubits

        if mode == "direct" and fair_direct:
            base_map = {"ae": "AE_mopac", "dh": "DH_Mopac"}
            base_col = base_map.get(getattr(args, "target", "ae").lower())
            if base_col and base_col in features_local:
                features_local = [f for f in features_local if f != base_col]
                print(f"[direct] Dropped baseline feature '{base_col}' for fairness.")
                if args.reencoding_type == "sequential":
                    if len(features_local) < 1:
                        raise ValueError(
                            "All features were dropped for fairness in direct mode. "
                            "Provide additional features or disable fairness."
                        )
                    args.qubits = len(features_local)
                else:
                    if len(features_local) != 1:
                        raise ValueError(
                            f"'parallel' re-encoding requires exactly ONE feature; got {len(features_local)} "
                            "after fairness drop. Disable fairness or choose a single feature."
                        )

        # 1. loads and scales data
        label = "delta" if mode == "delta" else "dft"
        X_train, X_test, y_train, y_test, pm7_test, dft_true = load_data(
            args.data_dir,
            features_local,
            args.qubits,
            args.reencoding_type,
            getattr(args, "pca_components", None),
            args.seed,
            target=getattr(args, "target", "ae"),
            label=label,
        )
        args.input_dim = X_train.shape[1]
        print(f"[{mode}] Using features: {features_local} (qubits={args.qubits})")

        # 2. scales the target for direct mode to keep training stable
        if mode == "direct":
            y_scaler = StandardScaler()
            y_train_for_fitting = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            print(f"[{mode}] Applied StandardScaler to target Y for stable training.")
        else:
            y_scaler = None
            y_train_for_fitting = y_train

        # copies the search space and adds lr if kernel training is enabled
        _, search_space, *_ = MODEL_CONFIG[args.model]
        search_space = search_space.copy()
        if getattr(args, "train_kernel", False):
            search_space["lr"] = ("loguniform", 1e-4, 1e-1)

        # 3. tunes and trains
        best_params = tune_model(args, search_space, X_train, y_train_for_fitting)
        print("\n3. Training Final Model")
        print(f"Best settings found: {best_params}")

        layers_used = int(best_params.get("num_layers", args.layers))
        final_model = create_model_from_params(args, best_params)

        # checks if CPKernel used a different physical qubit count than requested
        if args.encoding == "cpkernel":
            try:
                pqc_obj = None
                if hasattr(final_model, "kernel_circuit"):
                    pqc_obj = final_model.kernel_circuit.encoding_circuit
                effective_qubits = getattr(pqc_obj, "num_qubits", args.qubits)
                if effective_qubits != original_qubits:
                    print(
                        f"[cpkernel] NOTE: CPKernel uses {effective_qubits} physical qubits "
                        f"for {args.input_dim} features. Updating args.qubits for logs/reports."
                    )
                    args.qubits = effective_qubits
            except Exception as e:
                print(
                    f"[cpkernel] Warning: failed to inspect effective CPKernel qubits "
                    f"({type(e).__name__}: {e}). Using requested qubits={args.qubits}."
                )

        final_model.fit(X_train, y_train_for_fitting)

        y_pred_test_scaled  = final_model.predict(X_test)
        y_pred_train_scaled = final_model.predict(X_train)

        # inverse transforms predictions back to the original target scale if scaled
        if y_scaler is not None:
            y_pred_test  = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
            y_pred_train = y_scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
            print(f"[{mode}] Inverse-transformed predictions to original DFT scale.")
        else:
            y_pred_test  = y_pred_test_scaled
            y_pred_train = y_pred_train_scaled

        # computes final DFT-space predictions
        dft_pred = y_pred_test + pm7_test if mode == "delta" else y_pred_test

        baseline_mae = mean_absolute_error(dft_true, pm7_test)
        final_mae    = mean_absolute_error(dft_true, dft_pred)
        errors       = dft_pred - dft_true
        error_std    = float(np.std(errors))

        try:
            r_value, _ = pearsonr(dft_true, dft_pred) if len(dft_true) > 1 else (np.nan, None)
            r_squared  = float(r_value ** 2) if r_value == r_value else float("nan")
        except Exception:
            r_squared = float("nan")

        print(f"  - Baseline MAE (PM7 vs DFT):              {baseline_mae:.4f} kcal/mol")
        print(f"  - Final MAE ({mode} vs DFT):              {final_mae:.4f} kcal/mol")
        print(f"  - Final Error STD ({mode} vs DFT):        {error_std:.4f} kcal/mol")
        print(f"  - Final R-squared ({mode} vs DFT):        {r_squared:.4f}")

        test_mse  = float(mean_squared_error(y_test, y_pred_test))
        train_mse = float(mean_squared_error(y_train, y_pred_train))

        # builds the output folder name from the run configuration
        config_parts = [args.model.upper()]
        if args.encoding:
            config_parts.append(args.encoding)
        config_parts.append(f"{args.qubits}Q")
        config_parts.append(f"{layers_used}L")
        if args.tuner != "none":
            config_parts.append(f"Tuner-{args.tuner}")
        config_parts.append(mode.upper())
        config_str = "_".join(config_parts)

        # uses SLURM IDs for unique folder names if available, otherwise falls back to a timestamp
        slurm_job_id  = os.environ.get("SLURM_JOB_ID")
        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

        if slurm_job_id and slurm_task_id:
            unique_id = f"{slurm_job_id}_{slurm_task_id}"
        elif slurm_job_id:
            unique_id = slurm_job_id
        else:
            unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_output_dir = os.path.join(args.output_dir, config_str, unique_id)
        os.makedirs(run_output_dir, exist_ok=True)

        fname_base    = os.path.join(run_output_dir, "results")
        final_metrics = {"mae": float(final_mae), "std": float(error_std), "r2": float(r_squared)}

        save_data_outputs(
            fname_base, args, y_test, y_pred_test, pm7_test, dft_true, best_params,
            baseline_mae, final_mae, error_std, best_cv_score=None,
            test_mse=test_mse, train_loss=train_mse, mode=mode, summary_log_dir=args.output_dir
        )

        generate_results_plot(
            fname_base, args, y_test, y_pred_test, pm7_test, dft_true, final_metrics, mode=mode
        )

        if getattr(args, "save_model", False):
            import joblib
            model_filename = f"{fname_base}_final_model.joblib"
            joblib.dump(final_model, model_filename)
            print(f"Final model object saved to {model_filename}")

        # restores the original qubit count before the next mode pass
        args.qubits = original_qubits