# reporting.py
# Handles saving predictions, appending to the summary log, and generating result plots.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from datetime import datetime


def _target_tag(args):
    """Returns a short label string for the target property used in plot axes."""
    try:
        return "AE" if getattr(args, "target", "ae") == "ae" else "ΔH"
    except Exception:
        return "AE"


def _title_from_args(args, prefix=None):
    """Builds a descriptive plot title from the run configuration."""
    enc = getattr(args, "encoding", "none")
    try:
        enc_disp = enc.title()
    except Exception:
        enc_disp = str(enc)
    tnr = getattr(args, "tuner", "none")
    try:
        tnr_disp = tnr.title()
    except Exception:
        tnr_disp = str(tnr)
    tgt  = getattr(args, "target", "ae").upper()
    base = f"{args.model.upper()} - {enc_disp} - {args.qubits}Q - {tgt} - Tuner: {tnr_disp}"
    return f"{prefix} - {base}" if prefix else base


def _scatter_to_pdf(path, title, x_vals, y_vals, metrics_box, x_label, y_label):
    """
    Renders a scatter plot with a y=x reference line and a metrics text box, then saves as PDF.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    # computes axis limits with a small padding margin
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    pad     = 5.0
    lim_min = min(x_min, y_min) - pad
    lim_max = max(x_max, y_max) + pad
    lims    = [lim_min, lim_max]

    ax.scatter(x_vals, y_vals, alpha=0.6, edgecolor='k', s=25, label="Model Predictions")
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)

    text = (
        f"$R^2$ = {metrics_box.get('r2', np.nan):.3f}\n"
        f"MAE = {metrics_box.get('mae', np.nan):.2f} kcal/mol\n"
        f"SD = {metrics_box.get('std', np.nan):.2f} kcal/mol\n"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Plot (PDF) saved to {path}")


def save_data_outputs(
    fname_base,
    args,
    y_test,
    y_pred,
    pm7_test,
    dft_true,
    best_params,
    baseline_mae,
    final_mae,
    error_std,
    best_cv_score,
    *,
    test_mse=None,
    train_loss=None,
    mode="delta",
    summary_log_dir=None
):
    """
    Saves per-run predictions to CSV and appends one row to the master summary log.
    """
    if mode == "delta":
        true_delta = y_test
        pred_delta = y_pred
        pred_dft   = pm7_test + y_pred
    elif mode == "direct":
        true_delta = dft_true - pm7_test
        pred_delta = y_pred - pm7_test
        pred_dft   = y_pred
    else:
        raise ValueError("mode must be 'delta' or 'direct'")

    # saves per-run predictions to a CSV file inside the run folder
    pred_filename = f"{fname_base}_predictions_{mode}.csv"
    pred_df = pd.DataFrame({
        "PM7":               pm7_test,
        "True_Delta":        true_delta,
        "Predicted_Delta":   pred_delta,
        "True_DFT":          dft_true,
        "Predicted_DFT":     pred_dft,
        "Mode":              mode,
        "Target":            getattr(args, "target", "ae")
    })
    pred_df.to_csv(pred_filename, index=False)
    print(f"Predictions saved to {pred_filename}")

    if summary_log_dir is None:
        summary_log_dir = args.output_dir

    # appends one row to the master summary log CSV
    log_path  = os.path.join(summary_log_dir, "summary_log.csv")
    log_entry = {
        "Timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Run_Folder":              os.path.dirname(fname_base),
        "Mode":                    mode,
        "Target":                  getattr(args, "target", "ae").upper(),
        "Model":                   args.model.upper(),
        "Encoding":                getattr(args, "encoding", "none"),
        "Qubits":                  args.qubits,
        "Max_Layers":              args.layers,
        "Seed":                    args.seed,
        "Kernel_Type":             args.kernel,
        "Train_Kernel":            args.train_kernel,
        "Kernel_Optimizer":        args.kernel_optimizer if getattr(args, "train_kernel", False) else "N/A",
        "PQK_Backend_Requested":   getattr(args, "pqk_backend", "N/A"),
        "PQK_Backend_Effective":   getattr(args, "_effective_pqk_backend", "N/A"),
        "N_Jobs":                  getattr(args, "n_jobs", "N/A"),
        "Tuner":                   args.tuner,
        "CV_Folds":                f"{args.cv_folds}x{args.cv_repeats}" if getattr(args, "cv_type", "kfold") == "repeated" else args.cv_folds,
        "Features":                ' '.join(getattr(args, "features", [])),
        "PCA_Components":          getattr(args, "pca_components", "N/A"),
        "Baseline_MAE_kcal_mol":   round(float(baseline_mae), 4),
        "CV_MAE_kcal_mol":         round(float(best_cv_score), 4) if best_cv_score is not None else "N/A",
        "Test_MAE_kcal_mol":       round(float(final_mae), 4),
        "Test_STD_kcal_mol":       round(float(error_std), 4),
        "Test_MSE_label":          round(float(test_mse), 6) if test_mse is not None else "N/A",
        "Train_Loss_or_TrainMSE":  round(float(train_loss), 6) if train_loss is not None else "N/A",
        "Best_Params":             str(best_params) if best_params else "N/A",
    }

    header = not os.path.exists(log_path)
    pd.DataFrame([log_entry]).to_csv(log_path, mode="a", header=header, index=False)
    print(f"Results logged to {log_path}")

    # appends detailed predictions to the master predictions file for error distribution analysis
    master_pred_path = os.path.join(summary_log_dir, "master_predictions.csv")

    pred_df["Error"]                 = pred_df["Predicted_DFT"] - pred_df["True_DFT"]
    pred_df["Model"]                 = args.model.upper()
    pred_df["Encoding"]              = getattr(args, "encoding", "none")
    pred_df["Qubits"]                = args.qubits
    pred_df["Max_Layers"]            = args.layers
    pred_df["Tuner"]                 = args.tuner
    pred_df["Target"]                = getattr(args, "target", "ae").upper()
    pred_df["Train_Kernel"]          = getattr(args, "train_kernel", False)
    pred_df["Kernel_Type"]           = getattr(args, "kernel", "N/A")
    pred_df["PQK_Backend_Requested"] = getattr(args, "pqk_backend", "N/A")
    pred_df["PQK_Backend_Effective"] = getattr(args, "_effective_pqk_backend", "N/A")
    pred_df["N_Jobs"]                = getattr(args, "n_jobs", "N/A")
    pred_df["Run_Folder"]            = os.path.dirname(fname_base)
    pred_df["Seed"]                  = args.seed

    header_pred = not os.path.exists(master_pred_path)
    pred_df.to_csv(master_pred_path, mode="a", header=header_pred, index=False)
    print(f"Raw predictions appended to {master_pred_path}")


def generate_results_pdfs(
    fname_base,
    args,
    y_test,
    y_pred,
    pm7_test,
    dft_true,
    final_metrics,
    *,
    save_model_only=False,
    mode="delta"
):

    tgt = _target_tag(args)

    if mode == "delta":
        dft_pred  = pm7_test + y_pred
        ylabel_A  = f"Predicted {tgt} (kcal/mol) [PM7 + {args.model.upper()}]"
        title_A   = _title_from_args(args)
    elif mode == "direct":
        dft_pred  = y_pred
        ylabel_A  = f"Predicted {tgt} (kcal/mol) [{args.model.upper()} (Direct)]"
        title_A   = _title_from_args(args)
    else:
        raise ValueError("mode must be 'delta' or 'direct'")

    path_A = f"{fname_base}_results_{mode}_dft.pdf"
    _scatter_to_pdf(
        path_A, title_A, dft_true, dft_pred, final_metrics,
        x_label=f"Reference {tgt} (kcal/mol) [DFT]",
        y_label=ylabel_A
    )

    result = {"dft_pdf": path_A, "model_only_pdf": None}

    if save_model_only:
        if mode == "delta":
            y_true_m = y_test
            y_pred_m = y_pred
        else:
            y_true_m = dft_true - pm7_test
            y_pred_m = y_pred - pm7_test

        try:
            r_val = float(np.corrcoef(y_true_m, y_pred_m)[0, 1]) if len(y_true_m) > 1 else np.nan
            r2    = r_val ** 2
        except Exception:
            r2 = np.nan
        mae_m    = float(mean_absolute_error(y_true_m, y_pred_m))
        sd_m     = float(np.std(y_pred_m - y_true_m))
        metrics2 = {"r2": r2, "mae": mae_m, "std": sd_m}

        title_B = _title_from_args(args, prefix="Model-Only (Δ)")
        path_B  = f"{fname_base}_results_{mode}_model_only.pdf"
        _scatter_to_pdf(
            path_B, title_B, y_true_m, y_pred_m, metrics2,
            x_label=f"Reference Δ{tgt} (kcal/mol)",
            y_label=f"Predicted Δ{tgt} (kcal/mol) [{args.model.upper()}]"
        )
        result["model_only_pdf"] = path_B

    return result


def generate_results_plot(
    fname_base,
    args,
    y_test,
    y_pred,
    pm7_test,
    dft_true,
    final_metrics,
    mode="delta"
):
    return generate_results_pdfs(
        fname_base, args, y_test, y_pred, pm7_test, dft_true, final_metrics,
        save_model_only=False, mode=mode
    )
