# Make sure to have these imports at the top of your file
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from .config import MODEL_CONFIG
from .data import load_data
from .models import create_model_from_params
from .tuning import tune_model
from .reporting import save_data_outputs, generate_results_plot

def run_pipeline(args):
    """Runs the entire QML pipeline from data loading to saving results."""
    print(f"--- Starting QML Pipeline: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Run Config: Model={args.model.upper()}, Tuner={args.tuner.upper()}, PQC={args.encoding}, Qubits={args.qubits}, Max Layers={args.layers}, Seed={args.seed}")
    if args.model in ["qsvr", "qkrr", "qgpr"]:
        print(f"Kernel Config: Train={args.train_kernel}, Optimizer={args.kernel_optimizer if args.train_kernel else 'N/A'}")
    print(f"Features: {args.features}\n")

    X_train, X_test, y_train, y_test, pm7_test = load_data(
        args.data_dir, args.features, args.qubits, args.reencoding_type,
        args.pca_components, args.seed
    )

    _, search_space = MODEL_CONFIG[args.model]
    search_space = search_space.copy()
    search_space["num_layers"] = ("int", 1, args.layers)
    if args.train_kernel and args.model in ["qsvr", "qkrr", "qgpr"]:
        search_space["lr"] = ("loguniform", 1e-4, 1e-1)
    
    best_params = tune_model(args, search_space, X_train, y_train) 
    best_cv_score = None 

    print("\n3. Training Final Model")
    print(f"Best settings found: {best_params}")
    final_model = create_model_from_params(args, best_params)
    final_model.fit(X_train, y_train)
    predictions = final_model.predict(X_test)
    
    print("\n4. Final Evaluation & Saving Results")
    
    # Define true and predicted final energies
    dft_true = y_test + pm7_test
    dft_pred = predictions + pm7_test
    
    # Calculate the MAE of the baseline PM7 model (before correction)
    baseline_mae = mean_absolute_error(dft_true, pm7_test)
    print(f"  - Baseline MAE (PM7 vs DFT):             {baseline_mae:.4f} kcal/mol")

    # Calculate comprehensive metrics for the final QML-corrected model
    final_mae = mean_absolute_error(dft_true, dft_pred)
    errors = dft_pred - dft_true
    error_std = np.std(errors)
    r_value, _ = pearsonr(dft_true, dft_pred)
    r_squared = r_value**2
    
    print(f"  - Final Corrected MAE (QML vs DFT):      {final_mae:.4f} kcal/mol")
    print(f"  - Final Error STD (QML vs DFT):          {error_std:.4f} kcal/mol")
    print(f"  - Final R-squared (QML vs DFT):          {r_squared:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"{args.model.upper()}_{args.encoding}_q{args.qubits}_L{args.layers}_{args.tuner}"
    fname_base = os.path.join(args.output_dir, f"{config_str}_{timestamp}")
    
    # Pass all metrics to the reporting functions
    final_metrics = {"mae": final_mae, "std": error_std, "r2": r_squared}
    
    save_data_outputs(fname_base, args, y_test, predictions, pm7_test, best_params, 
                        baseline_mae, final_mae, error_std, best_cv_score)
                        
    generate_results_plot(fname_base, args, y_test, predictions, pm7_test, final_metrics)