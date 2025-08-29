#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# HPC_CML_XGB.py
# ------------------------------------------------------------
# XGBoost for Δ-learning (DFT = PM7 + Δ) AND direct DFT prediction.
# - Reuses your saved Optuna best hyperparameters.
# - Exact Xabier-style correlation pruning (train-only, |r| > 0.90).
# - 80/20 train/validation split ON TRAIN ONLY for early stopping,
#   then refit on ALL training rows up to best_iteration.
# - Saves everything under --output_dir/<variant>/ organized folders:
#     models/, plots/, predictions/, meta/
# - Produces publication-style parity PDFs that match your example
#   (dashed red y=x, blue points, beige stats box).
# - Can run with all descriptors, the 9-feature “Q9” set, or both.
# - Ignores unknown CLI args so you can reuse your SLURM wrapper.
#
# Example:
#   python3 HPC_CML_XGB.py \
#     --data_dir /path/to/data \
#     --output_dir /path/to/out \
#     --run_variants both \
#     --feature_selection corr90
# ------------------------------------------------------------

from __future__ import annotations
import argparse, json, sys, platform
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xg
import shap

import matplotlib
matplotlib.use("Agg")  # always headless on HPC
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ========= CLI =========
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   required=True, type=str, help="Directory containing train_df_new.csv and test_df_new.csv")
parser.add_argument("--output_dir", required=True, type=str, help="Directory to write all artifacts")
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--feature_selection", default="corr90", choices=["none", "corr90"],
                    help="Xabier-style correlation pruning on TRAIN ONLY")
parser.add_argument("--run_variants", choices=["all", "q9", "both"], default="both",
                    help="Run with 'all' features, 9-feature 'q9', or both")
# Accept and ignore any extra args passed by your SLURM script to avoid errors
args, extra = parser.parse_known_args()
if extra:
    print(f"[WARN] Ignoring unused CLI args: {extra}", flush=True)

DATA = Path(args.data_dir)
OUT  = Path(args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(args.seed)

# ========= Matplotlib defaults (publication style) =========
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,       # embed TrueType
    "font.family": "DejaVu Serif",
    "font.size": 11, "axes.linewidth": 1.0, "axes.labelsize": 12,
    "axes.titlesize": 12, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4, "legend.frameon": False,
})

# ========= Hyperparameters (from your Optuna study) =========
BEST_HP = {
    "max_depth": 12,
    "learning_rate": 0.0244373296495741,
    "colsample_bytree": 0.5935165203818146,
    "subsample": 0.7934306405413262,
    "reg_alpha": 0.037448127616506344,
    "reg_lambda": 0.0028412844750713272,
    "gamma": 3.590765443858655e-05,
    "min_child_weight": 20.586964083896493,
    "objective": "reg:squarederror",
    "tree_method": "hist",    # switch to "gpu_hist" if you run on GPU
    "n_estimators": 20000,    # large cap; early stopping will trim
    "n_jobs": -1,
}

# ========= Utility: paper-like parity plot =========
def parity_plot_pdf(
    y_true, y_pred, out_pdf: Path,
    xlabel: str, ylabel: str,
    title: str | None = None
) -> dict:
    """Save a parity plot styled like your example and return metrics dict."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    sd  = float(np.std(y_pred - y_true, ddof=1))

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(6.0, 6.0))   # square like your sample
    # Blue points
    ax.scatter(y_true, y_pred, s=28, color="#2F5D86", edgecolors="none", alpha=0.85)
    # Dashed red y=x
    ax.plot([lo, hi], [lo, hi], "--", color="#D9534F", lw=1.6)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title, pad=8)

    # Beige stat box
    txt = rf"$R^2 = {r2:.3f}$" + f"\nMAE = {mae:.2f} kcal/mol\nSD = {sd:.2f} kcal/mol"
    box = dict(boxstyle="round,pad=0.5", fc="#EFE0C5", ec="0.6", lw=0.9)
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", bbox=box)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return {"r2": float(r2), "mae": float(mae), "sd": sd}

# ========= Load data =========
train_df = pd.read_csv(DATA / "train_df_new.csv", index_col=0)
test_df  = pd.read_csv(DATA / "test_df_new.csv",  index_col=0)
train_df.columns = train_df.columns.str.strip()
test_df.columns  = test_df.columns.str.strip()

# ========= Feature variants =========
Q9_DEFAULT = [
    "exp_mopac", "AE_mopac", "Par_n_Pople", "Mul", "ch_f",
    "lap_eig_1", "ZPE_TS_P", "Freq", "SMR_VSA9"
]


def run_variant(variant: str, q9_list: list[str] | None = None) -> dict:
    """
    Run both Δ-learning and direct DFT for a given feature variant ('all' or 'q9').
    Returns a metrics dict and writes lots of artifacts to disk.
    """
    # ----- Output folders (per variant) -----
    vroot = OUT / variant
    d_models = vroot / "models"
    d_plots  = vroot / "plots"
    d_preds  = vroot / "predictions"
    d_meta   = vroot / "meta"
    for d in (d_models, d_plots, d_preds, d_meta):
        d.mkdir(parents=True, exist_ok=True)

    # ----- Select features -----
    X_train_full = train_df.drop(columns=["dh_delta","ae_delta","dh_dft","ae_dft"])
    X_test_full  = test_df .drop(columns=["dh_delta","ae_delta","dh_dft","ae_dft"])

    if variant == "q9":
        feats = q9_list if q9_list is not None else Q9_DEFAULT
        feats = [c for c in feats if c in X_train_full.columns]
        if not feats:
            raise ValueError("Q9 feature list not found in DataFrame columns.")
        X_train_full = X_train_full[feats].copy()
        X_test_full  = X_test_full[feats].copy()
        (d_meta / "used_features.txt").write_text("\n".join(feats))

    # ----- Xabier-style correlation drop (train-only, |r| > 0.90) -----
    to_drop = []
    if args.feature_selection == "corr90":
        corr  = X_train_full.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] > 0.90).any()]
        if to_drop:
            (d_meta / "dropped_corr90.txt").write_text("\n".join(to_drop))
    Xtr = X_train_full.drop(columns=to_drop).copy()
    Xte = X_test_full.drop(columns=to_drop, errors="ignore").copy()

    (d_meta / "feature_counts.json").write_text(json.dumps({
        "variant": variant,
        "n_features_before": int(X_train_full.shape[1]),
        "n_features_after" : int(Xtr.shape[1]),
        "n_dropped_corr90": int(len(to_drop)),
    }, indent=2))

    print(f"[{variant}] features after pruning: {Xtr.shape[1]} (dropped {len(to_drop)})", flush=True)

    # ----- Robust PM7 column detection -----
    pm7_col = "AE_mopac"
    if pm7_col not in test_df.columns:
        cands = [c for c in test_df.columns if "mopac" in c.lower() or "pm7" in c.lower()]
        if not cands:
            raise KeyError("Could not find a PM7 column (e.g., 'AE_mopac') in test_df.")
        pm7_col = cands[0]

    # ---------------- A) Δ-learning ----------------
    y_train_delta = train_df.loc[Xtr.index, "ae_delta"].values.ravel()
    y_test_delta  = test_df .loc[Xte.index, "ae_delta"].values.ravel()

    # Early stopping on train-only split
    X_tr, X_val, y_tr, y_val = train_test_split(Xtr, y_train_delta, test_size=0.20, random_state=args.seed)

    xg_delta = xg.XGBRegressor(
        **BEST_HP, random_state=args.seed, eval_metric="mae",
        early_stopping_rounds=300, enable_categorical=False,
    )
    print(f"\n[{variant} | Δ] Training Δ-learning (ES on train/val)...", flush=True)
    xg_delta.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)
    best_n_delta = (xg_delta.best_iteration + 1) if xg_delta.best_iteration is not None else BEST_HP["n_estimators"]

    # Refit up to best_n on ALL training rows
    final_delta = xg.XGBRegressor(
        **{**BEST_HP, "n_estimators": best_n_delta},
        random_state=args.seed, eval_metric="mae", enable_categorical=False,
    )
    final_delta.fit(Xtr, y_train_delta)

    pred_delta          = final_delta.predict(Xte)
    pred_dft_from_delta = pred_delta + test_df.loc[Xte.index, pm7_col].values
    true_dft            = test_df.loc[Xte.index, "ae_dft"].values

    mae_delta = mean_absolute_error(y_test_delta, pred_delta)
    mae_dft_fd = mean_absolute_error(true_dft, pred_dft_from_delta)
    r2_dft_fd  = r2_score(true_dft, pred_dft_from_delta)

    # Save Δ-learning artifacts
    pd.DataFrame({
        "true_delta": y_test_delta,
        "pred_delta": pred_delta,
        "true_dft":   true_dft,
        "pred_dft":   pred_dft_from_delta,
    }, index=Xte.index).to_csv(d_preds / "XGB_predictions_test_delta.csv")

    final_delta.save_model(str(d_models / "xgb_model_delta.json"))
    (d_meta / "delta_best_estimators.json").write_text(json.dumps({"best_n_estimators": int(best_n_delta)}, indent=2))

    # SHAP (optional; guarded)
    try:
        expl = shap.TreeExplainer(final_delta)
        sv   = expl(Xtr.sample(min(2000, len(Xtr)), random_state=args.seed))  # speed tip: subsample
        shap.plots.beeswarm(sv, show=False)
        plt.tight_layout()
        plt.savefig(d_plots / "xgb_shap_delta.pdf", bbox_inches="tight", transparent=True)
        plt.close()
    except Exception as e:
        print(f"[{variant} | Δ] SHAP skipped: {e}", flush=True)

    # Δ -> DFT parity (publication-style)
    metrics_delta_dft = parity_plot_pdf(
        y_true=true_dft, y_pred=pred_dft_from_delta,
        out_pdf=d_plots / "xgb_parity_dft_from_delta.pdf",
        xlabel="Reference ΔH (kcal/mol) [DFT]",
        ylabel="Predicted ΔH (kcal/mol) [PM7 + Δ-XGB]",
        title=f"XGB | Δ-learning | {variant.upper()}"
    )
    # (Optional) pure Δ parity
    parity_plot_pdf(
        y_true=y_test_delta, y_pred=pred_delta,
        out_pdf=d_plots / "xgb_parity_delta_only.pdf",
        xlabel="Reference Δ (kcal/mol)",
        ylabel="Predicted Δ (kcal/mol)",
        title=f"XGB | Δ target | {variant.upper()}"
    )

    # ---------------- B) Direct DFT ----------------
    y_train_dft = train_df.loc[Xtr.index, "ae_dft"].values.ravel()
    y_test_dft  = test_df .loc[Xte.index, "ae_dft"].values.ravel()

    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(Xtr, y_train_dft, test_size=0.20, random_state=args.seed)

    xg_direct = xg.XGBRegressor(
        **BEST_HP, random_state=args.seed, eval_metric="mae",
        early_stopping_rounds=300, enable_categorical=False,
    )
    print(f"\n[{variant} | DFT] Training direct-DFT (ES on train/val)...", flush=True)
    xg_direct.fit(X_tr2, y_tr2, eval_set=[(X_val2, y_val2)], verbose=True)
    best_n_direct = (xg_direct.best_iteration + 1) if xg_direct.best_iteration is not None else BEST_HP["n_estimators"]

    final_direct = xg.XGBRegressor(
        **{**BEST_HP, "n_estimators": best_n_direct},
        random_state=args.seed, eval_metric="mae", enable_categorical=False,
    )
    final_direct.fit(Xtr, y_train_dft)

    pred_dft_direct = final_direct.predict(Xte)
    mae_dft_direct  = mean_absolute_error(y_test_dft, pred_dft_direct)
    r2_dft_direct   = r2_score(y_test_dft, pred_dft_direct)

    # Save direct artifacts
    pd.DataFrame({
        "true_dft": y_test_dft,
        "pred_dft": pred_dft_direct,
    }, index=Xte.index).to_csv(d_preds / "XGB_predictions_test_direct.csv")

    final_direct.save_model(str(d_models / "xgb_model_direct.json"))
    (d_meta / "direct_best_estimators.json").write_text(json.dumps({"best_n_estimators": int(best_n_direct)}, indent=2))

    # SHAP (optional)
    try:
        expl2 = shap.TreeExplainer(final_direct)
        sv2   = expl2(Xtr.sample(min(2000, len(Xtr)), random_state=args.seed))
        shap.plots.beeswarm(sv2, show=False)
        plt.tight_layout()
        plt.savefig(d_plots / "xgb_shap_direct.pdf", bbox_inches="tight", transparent=True)
        plt.close()
    except Exception as e:
        print(f"[{variant} | DFT] SHAP skipped: {e}", flush=True)

    # Direct DFT parity (publication-style)
    metrics_direct = parity_plot_pdf(
        y_true=y_test_dft, y_pred=pred_dft_direct,
        out_pdf=d_plots / "xgb_parity_direct_dft.pdf",
        xlabel="Reference ΔH (kcal/mol) [DFT]",
        ylabel="Predicted ΔH (kcal/mol) [XGB]",
        title=f"XGB | Direct DFT | {variant.upper()}"
    )

    # ----- Per-variant summary (also write a human-readable report) -----
    summary = {
        "variant": variant,
        "n_features_after_prune": int(Xtr.shape[1]),
        "delta": {
            "best_n_estimators": int(best_n_delta),
            "test_mae_delta":    float(mae_delta),
            "dft_from_delta": {
                "test_mae": float(metrics_delta_dft["mae"]),
                "test_r2":  float(metrics_delta_dft["r2"]),
                "test_sd":  float(metrics_delta_dft["sd"])
            }
        },
        "direct": {
            "best_n_estimators": int(best_n_direct),
            "test_mae_dft": float(metrics_direct["mae"]),
            "test_r2_dft":  float(metrics_direct["r2"]),
            "test_sd_dft":  float(metrics_direct["sd"])
        }
    }
    (d_meta / "XGB_metrics.json").write_text(json.dumps(summary, indent=2))

    # Markdown report
    md = []
    md.append(f"# XGB Results — {variant.upper()}\n")
    md.append(f"- Features after pruning: **{Xtr.shape[1]}** (dropped {len(to_drop)})\n")
    md.append("## Δ-learning (DFT = PM7 + Δ)\n")
    md.append(f"- Best trees: **{best_n_delta}**\n- Test MAE (Δ): **{mae_delta:.3f}** kcal/mol\n"
              f"- Test MAE (DFT from Δ): **{metrics_delta_dft['mae']:.3f}** kcal/mol\n"
              f"- Test R²  (DFT from Δ): **{metrics_delta_dft['r2']:.3f}**\n"
              f"- Test SD  (DFT from Δ residuals): **{metrics_delta_dft['sd']:.3f}** kcal/mol\n")
    md.append("\n## Direct DFT\n")
    md.append(f"- Best trees: **{best_n_direct}**\n- Test MAE (DFT): **{metrics_direct['mae']:.3f}** kcal/mol\n"
              f"- Test R²  (DFT): **{metrics_direct['r2']:.3f}**\n"
              f"- Test SD  (DFT residuals): **{metrics_direct['sd']:.3f}** kcal/mol\n")
    (d_meta / "REPORT.md").write_text("\n".join(md))

    print(f"[{variant}] Done. Artifacts: {vroot.resolve()}", flush=True)
    return summary


# ========= Run the requested variants =========
all_metrics = []
if args.run_variants in ("all", "both"):
    all_metrics.append(run_variant("all"))
if args.run_variants in ("q9", "both"):
    all_metrics.append(run_variant("q9", q9_list=None))

# ========= Write combined summary + environment info =========
combined = {
    "seed": args.seed,
    "feature_selection": args.feature_selection,
    "run_variants": args.run_variants,
    "metrics": all_metrics,
    "env": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xgboost": xg.__version__,
        "matplotlib": matplotlib.__version__,
        "shap": shap.__version__,
    }
}
(OUT / "XGB_metrics_combined.json").write_text(json.dumps(combined, indent=2))
print("\n[INFO] Combined metrics written to:", (OUT / "XGB_metrics_combined.json").resolve())
