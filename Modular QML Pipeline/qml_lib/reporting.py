import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# ---------- small helpers ----------

def _target_tag(args):
    """Return short target tag for axis labels."""
    try:
        return "AE" if getattr(args, "target", "ae") == "ae" else "ΔH"
    except Exception:
        return "AE"

def _title_from_args(args, prefix=None):
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
    tgt = getattr(args, "target", "ae").upper()
    base = f"{args.model.upper()} - {enc_disp} - {args.qubits}Q - {tgt} - Tuner: {tnr_disp}"
    return f"{prefix} - {base}" if prefix else base

# ---------- plotting helpers ----------

def _scatter_to_pdf(path, title, x_vals, y_vals, metrics_box, x_label, y_label):
    """
    Render a scatter with MAE/R^2/SD box and y=x dashed line, and save as a PDF.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    # limits with small margin
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    pad = 5.0
    lim_min = min(x_min, y_min) - pad
    lim_max = max(x_max, y_max) + pad
    lims = [lim_min, lim_max]

    # scatter + diagonal
    ax.scatter(x_vals, y_vals, alpha=0.6, edgecolor='k', s=25, label="Model Predictions")
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)

    # metrics box
    text = (
        f"$R^2$ = {metrics_box.get('r2', np.nan):.3f}\n"
        f"MAE = {metrics_box.get('mae', np.nan):.2f} kcal/mol\n"
        f"SD = {metrics_box.get('std', np.nan):.2f} kcal/mol\n"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

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

# ---------- public API ----------

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
    mode="delta"       # 'delta' or 'direct'
):
    """
    Save predictions CSV and append a row to a CSV log (Excel-friendly).
    Mode-aware:
      - delta: Predicted_DFT = PM7 + y_pred ; True_Delta = y_test ; Predicted_Delta = y_pred
      - direct: Predicted_DFT = y_pred      ; True_Delta = DFT_true - PM7 ; Predicted_Delta = y_pred - PM7
    """
    # --- construct mode-aware columns ---
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

    # predictions CSV
    pred_df = pd.DataFrame({
        "PM7": pm7_test,
        "True_Delta": true_delta,
        "Predicted_Delta": pred_delta,
        "True_DFT": dft_true,
        "Predicted_DFT": pred_dft,
        "Mode": mode,
        "Target": getattr(args, "target", "ae")
    })
    pred_filename = f"{fname_base}_predictions_{mode}.csv"
    pred_df.to_csv(pred_filename, index=False)
    print(f"Predictions saved to {pred_filename}")

    # summary log (DFT-space metrics)
    log_path = os.path.join(args.output_dir, "summary_log.csv")
    is_kernel_model = args.model in ["qsvr", "qkrr", "qgpr"]

    log_entry = {
        # Run / setup
        "Timestamp": os.path.basename(fname_base).split('_')[-1],
        "Mode": mode,
        "Target": getattr(args, "target", "ae").upper(),
        "Model": args.model.upper(),
        "Encoding": getattr(args, "encoding", "none"),
        "Qubits": args.qubits,
        "Max_Layers": args.layers,
        "Seed": args.seed,

        # Model config
        "Kernel_Type": args.kernel if is_kernel_model else "N/A",
        "Train_Kernel": args.train_kernel if is_kernel_model else "N/A",
        "Kernel_Optimizer": args.kernel_optimizer if (getattr(args, "train_kernel", False) and is_kernel_model) else "N/A",
        "Optimizer": args.optimizer if not is_kernel_model else "N/A",

        # Tuner / CV
        "Tuner": args.tuner,
        "CV_Folds": f"{args.cv_folds}x{args.cv_repeats}" if getattr(args, "cv_type", "kfold") == "repeated" else args.cv_folds,

        # Data
        "Features": ' '.join(getattr(args, "features", [])),
        "PCA_Components": getattr(args, "pca_components", "N/A"),

        # Results (DFT space)
        "Baseline_MAE_kcal_mol": round(float(baseline_mae), 4),
        "CV_MAE_kcal_mol": round(float(best_cv_score), 4) if best_cv_score is not None else "N/A",
        "Test_MAE_kcal_mol": round(float(final_mae), 4),
        "Test_STD_kcal_mol": round(float(error_std), 4),

        # Aux metrics (label space)
        "Test_MSE_label": round(float(test_mse), 6) if test_mse is not None else "N/A",
        "Train_Loss_or_TrainMSE": round(float(train_loss), 6) if train_loss is not None else "N/A",

        # Hyperparams
        "Best_Params": str(best_params) if best_params else "N/A",
    }

    header = not os.path.exists(log_path)
    pd.DataFrame([log_entry]).to_csv(log_path, mode="a", header=header, index=False)
    print(f"Results logged to {log_path}")

def generate_results_pdfs(
    fname_base,
    args,
    y_test,
    y_pred,
    pm7_test,
    dft_true,
    final_metrics,
    *,
    save_model_only=False,   # DEFAULT OFF -> only DFT-space plots
    mode="delta"             # 'delta' or 'direct'
):
    """
    Create PDFs:
      (A) DFT-space scatter: x=DFT_true, y=DFT_pred (delta: PM7+pred ; direct: pred)
      (B) (optional) Model-only (Δ space): x=Δ_true, y=Δ_pred  [DISABLED by default]
    """
    tgt = _target_tag(args)

    # --- (A) DFT-space scatter ---
    if mode == "delta":
        dft_pred = pm7_test + y_pred
        ylabel_A = f"Predicted {tgt} (kcal/mol) [PM7 + {args.model.upper()}]"
        title_A  = _title_from_args(args)
    elif mode == "direct":
        dft_pred = y_pred
        ylabel_A = f"Predicted {tgt} (kcal/mol) [{args.model.upper()} (Direct)]"
        title_A  = _title_from_args(args)
    else:
        raise ValueError("mode must be 'delta' or 'direct'")

    path_A  = f"{fname_base}_results_{mode}_dft.pdf"
    _scatter_to_pdf(
        path_A, title_A, dft_true, dft_pred, final_metrics,
        x_label=f"Reference {tgt} (kcal/mol) [DFT]",
        y_label=ylabel_A
    )

    result = {"dft_pdf": path_A, "model_only_pdf": None}

    # --- (B) Optional Δ-space plot (kept available but OFF) ---
    if save_model_only:
        if mode == "delta":
            y_true_m = y_test
            y_pred_m = y_pred
        else:  # direct
            y_true_m = dft_true - pm7_test
            y_pred_m = y_pred - pm7_test

        try:
            r_val = float(np.corrcoef(y_true_m, y_pred_m)[0, 1]) if len(y_true_m) > 1 else np.nan
            r2 = r_val ** 2
        except Exception:
            r2 = np.nan
        mae_m = float(mean_absolute_error(y_true_m, y_pred_m))
        sd_m  = float(np.std(y_pred_m - y_true_m))
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

# ---- Backwards-compat wrapper ----
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
    """Legacy entry point. Generates ONLY the DFT-space PDF (no Δ-space plot)."""
    return generate_results_pdfs(
        fname_base, args, y_test, y_pred, pm7_test, dft_true, final_metrics,
        save_model_only=False, mode=mode
    )
