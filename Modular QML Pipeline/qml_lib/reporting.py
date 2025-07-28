import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_results_plot(fname_base, args, y_test, y_pred, pm7_test, final_metrics):
    """Generates and saves a scatter plot of the final results with a metrics box."""
    dft_true = pm7_test + y_test
    dft_pred = pm7_test + y_pred

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define the limits for the plot
    lims = [min(dft_true.min(), dft_pred.min()) - 5, max(dft_true.max(), dft_pred.max()) + 5]
    
    # Scatter plot of the data
    ax.scatter(dft_true, dft_pred, alpha=0.6, edgecolor='k', s=25, label="Model Predictions")
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    metrics_text = (
        f'$R^2$ = {final_metrics["r2"]:.3f}'
        f'MAE = {final_metrics["mae"]:.2f} kcal/mol\n'
        f'SD = {final_metrics["std"]:.2f} kcal/mol\n'
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax.set_xlabel("Reference ΔH (kcal/mol) [DFT]", fontsize=12)
    ax.set_ylabel(f"Predicted ΔH (kcal/mol) [PM7 + {args.model.upper()}]", fontsize=12)
    ax.set_title(f"{args.model.upper()} | {args.encoding.title()} | {args.qubits}Q | Tuner: {args.tuner.title()}", fontsize=14, pad=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    plot_filename = f"{fname_base}_results_plot.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filename}")

def save_data_outputs(fname_base, args, y_test, y_pred, pm7_test, best_params,
                        baseline_mae, final_mae, error_std, best_cv_score):
    """Saves predictions to CSV and appends a fully comprehensive summary to the log file."""
    # Save predictions
    pred_df = pd.DataFrame({
        "AE_PM7": pm7_test, "True_Delta": y_test, "Predicted_Delta": y_pred,
        "True_DFT": pm7_test + y_test, "Predicted_DFT": pm7_test + y_pred
    })
    pred_filename = f"{fname_base}_predictions.csv"
    pred_df.to_csv(pred_filename, index=False)
    print(f"Predictions saved to {pred_filename}")

    log_path = os.path.join(args.output_dir, "summary_log.csv")
    
    # Check if model is a kernel-based method
    is_kernel_model = args.model in ["qsvr", "qkrr", "qgpr"]

    log_entry = {
        # Run and PQC Setup
        "Timestamp": os.path.basename(fname_base).split('_')[-1],
        "Model": args.model.upper(),
        "Encoding": args.encoding,
        "Qubits": args.qubits,
        "Max_Layers": args.layers,
        "Seed": args.seed,
        
        # Model Configuration Details
        "Kernel_Type": args.kernel if is_kernel_model else "N/A",
        "Train_Kernel": args.train_kernel,
        "Kernel_Optimizer": args.kernel_optimizer if args.train_kernel and is_kernel_model else "N/A",
        "Optimizer": args.optimizer if not is_kernel_model else "N/A",

        # Tuner Setup
        "Tuner": args.tuner,
        "CV_Folds": f"{args.cv_folds}x{args.cv_repeats}" if args.cv_type == "repeated" else args.cv_folds,

        # Data Setup
        "Features": ' '.join(args.features),
        "PCA_Components": args.pca_components if args.pca_components else "N/A",
        
        # Results
        "Baseline_MAE_kcal_mol": round(baseline_mae, 4),
        "CV_MAE_kcal_mol": round(best_cv_score, 4) if best_cv_score is not None else "N/A",
        "Test_MAE_kcal_mol": round(final_mae, 4),
        "Test_STD_kcal_mol": round(error_std, 4),
        
        # Hyperparameters
        "Best_Params": str(best_params) if best_params else "N/A"
    }
    header = not os.path.exists(log_path)
    pd.DataFrame([log_entry]).to_csv(log_path, mode="a", header=header, index=False)
    print(f"Results logged to {log_path}")