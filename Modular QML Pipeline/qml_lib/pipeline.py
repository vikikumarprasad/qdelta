# qml_lib/pipeline.py

import os
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from .config import MODEL_CONFIG
from .data import load_data
from .models import create_model_from_params
from .tuning import tune_model
from .reporting import save_data_outputs, generate_results_plot


def run_pipeline(args):
    """Runs the entire QML pipeline from data loading to saving results."""
    print(f"--- Starting QML Pipeline: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    # Display run config
    if args.model in ["qnn-cpmap", "qnn-iqp"]:
        print(
            f"Run Config: Model={args.model.upper()}, Qubits={args.qubits}, "
            f"Max Layers={args.layers}, Seed={args.seed}"
        )
    else:
        print(
            f"Run Config: Model={args.model.upper()}, Tuner={args.tuner.upper()}, "
            f"PQC={args.encoding}, Qubits={args.qubits}, Max Layers={args.layers}, Seed={args.seed}"
        )

    if args.model in ["qsvr", "qkrr", "qgpr"]:
        print(
            f"Kernel Config: Train={args.train_kernel}, "
            f"Optimizer={args.kernel_optimizer if args.train_kernel else 'N/A'}"
        )

    print(f"Target: {getattr(args, 'target', 'ae').upper()} | Mode: {getattr(args, 'mode', 'delta')}")
    print(f"Features: {args.features}\n")

    # Decide which modes to run
    mode_arg = getattr(args, "mode", "delta")
    if mode_arg not in ("delta", "direct", "both"):
        mode_arg = "delta"
    modes_to_run = ["delta", "direct"] if mode_arg == "both" else [mode_arg]

    fair_direct = bool(getattr(args, "fair_direct", False))

    for mode in modes_to_run:
        print(f"\n=== [{mode.upper()}] Pass ===")

        # --- Build per-pass feature list ---
        features_local = list(args.features)
        original_qubits = args.qubits  # we may temporarily adjust for this pass

        if mode == "direct" and fair_direct:
            # Drop baseline PM7/AE feature for further investigation
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
                    # Ensure features == qubits for sequential re-encoding
                    args.qubits = len(features_local)
                else:  # parallel encoding expects exactly one feature
                    if len(features_local) != 1:
                        raise ValueError(
                            f"'parallel' re-encoding requires exactly ONE feature; got {len(features_local)} "
                            "after fairness drop. Disable fairness or choose a single feature."
                        )

        # 1) Data load (label depends on mode)
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

        # 2) Search space (copy + minimal tweaks)
        _, search_space, *_ = MODEL_CONFIG[args.model]
        search_space = search_space.copy()
        # Only QNN wrappers have a meaningful "num_layers" to tune
        if args.model in ["qnn-cpmap", "qnn-iqp"]:
            search_space["num_layers"] = ("int", 1, args.layers)
        # Trainable-kernel LR (if enabled for kernel models)
        if getattr(args, "train_kernel", False) and args.model in ["qsvr", "qkrr", "qgpr"]:
            search_space["lr"] = ("loguniform", 1e-4, 1e-1)

        # 3) Tune + Train
        best_params = tune_model(args, search_space, X_train, y_train)
        print("\n3. Training Final Model")
        print(f"Best settings found: {best_params}")
        final_model = create_model_from_params(args, best_params)
        final_model.fit(X_train, y_train)

        # Predictions (model space)
        y_pred_test = final_model.predict(X_test)
        y_pred_train = final_model.predict(X_train)

        # 4) Evaluation (DFT space)
        dft_pred = y_pred_test + pm7_test if mode == "delta" else y_pred_test

        baseline_mae = mean_absolute_error(dft_true, pm7_test)
        print(f"  - Baseline MAE (PM7 vs DFT):             {baseline_mae:.4f} kcal/mol")

        final_mae = mean_absolute_error(dft_true, dft_pred)
        errors = dft_pred - dft_true
        error_std = float(np.std(errors))
        try:
            r_value, _ = pearsonr(dft_true, dft_pred) if len(dft_true) > 1 else (np.nan, None)
            r_squared = float(r_value ** 2) if r_value == r_value else float("nan")
        except Exception:
            r_squared = float("nan")

        print(f"  - Final MAE ({mode} vs DFT):             {final_mae:.4f} kcal/mol")
        print(f"  - Final Error STD ({mode} vs DFT):       {error_std:.4f} kcal/mol")
        print(f"  - Final R-squared ({mode} vs DFT):       {r_squared:.4f}")

        # Model-space MSEs (test & 'loss' ≈ train MSE)
        test_mse = float(mean_squared_error(y_test, y_pred_test))
        train_mse = float(mean_squared_error(y_train, y_pred_train))

        # 5) Persist artifacts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_str = f"{args.model.upper()}_q{args.qubits}_L{args.layers}_{args.tuner}_{mode.upper()}"
        fname_base = os.path.join(args.output_dir, f"{config_str}_{timestamp}")

        final_metrics = {"mae": float(final_mae), "std": float(error_std), "r2": float(r_squared)}

        # CSV + lean log (mode-aware)
        save_data_outputs(
            fname_base,
            args,
            y_test,
            y_pred_test,
            pm7_test,
            dft_true,
            best_params,
            baseline_mae,
            final_mae,
            error_std,
            best_cv_score=None,
            test_mse=test_mse,
            train_loss=train_mse,
            mode=mode,  # <<< IMPORTANT: pass mode through
        )

        # PDFs (DFT scatter + optional model-only) (mode-aware)
        generate_results_plot(
            fname_base,
            args,
            y_test,
            y_pred_test,
            pm7_test,
            dft_true,
            final_metrics,
            mode=mode,  # <<< IMPORTANT: pass mode through
        )

        if getattr(args, "save_model", False):
            import joblib
            model_filename = f"{fname_base}_final_model.joblib"
            joblib.dump(final_model, model_filename)
            print(f"Final model object saved to {model_filename}")

        # Restore original qubits for the next pass
        args.qubits = original_qubits
