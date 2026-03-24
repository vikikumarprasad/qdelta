# HPC_CML_SVR.py
# Support Vector Regression for delta-learning and direct DFT prediction.
# Supports ALL and Q9 feature variants with Optuna tuning and PDF parity plots.

import os, json, warnings, argparse, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import optuna as opt
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# matplotlib settings for paper-quality PDF output
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Serif",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "legend.frameon": False,
})

# fixed set of 9 physically meaningful features used in the Q9 variant
Q9_FEATURES = [
    "exp_mopac", "AE_mopac", "Par_n_Pople", "Mul", "ch_f",
    "DH_Mopac", "ZPE_TS_R", "Freq", "ZPE_P_R"
]


def log(msg: str):
    """Prints a timestamped message to stdout."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", flush=True)


def ensure_dir(p: Path) -> Path:
    """Creates a directory and all parents if they do not exist, then returns the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def beige_box_stats(ax, text: str):
    """Adds a beige stats text box to the given axes."""
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f3e9d2", edgecolor="#d3c6a3", alpha=0.9)
    )


def parity_plot_pdf(y_true, y_pred, out_pdf, xlabel, ylabel, title=None, mae=None, r2=None):
    """Saves a parity scatter plot with a dashed y=x line and a stats box to a PDF file."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    resid  = y_true - y_pred
    sd     = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    lo  = min(y_true.min(), y_pred.min())
    hi  = max(y_true.max(), y_pred.max())
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    ax.scatter(y_pred, y_true, s=14, alpha=0.8, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="crimson", lw=1.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=6)

    stats_lines = []
    if r2  is not None: stats_lines.append(f"R$^2$ = {r2:.3f}")
    if mae is not None: stats_lines.append(f"MAE = {mae:.2f}")
    stats_lines.append(f"SD = {sd:.2f}")
    beige_box_stats(ax, "\n".join(stats_lines))

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)


def build_svr_params(trial: opt.trial.Trial):
    """
    Returns an SVR kwargs dict for the given trial.

    Uses distinct parameter names for rbf and poly gamma to avoid Optuna distribution conflicts.
    """
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
    params = {
        "kernel":  kernel,
        "C":       trial.suggest_float("C", 1e-1, 1e3, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 0.3, log=True),
    }
    if kernel == "rbf":
        params["gamma"] = trial.suggest_float("gamma_rbf", 1e-4, 1.0, log=True)
    else:
        params["gamma"]  = trial.suggest_categorical("gamma_poly", ["scale", "auto"])
        params["degree"] = trial.suggest_int("degree", 2, 4)
        params["coef0"]  = trial.suggest_float("coef0", 0.0, 1.0)
    return params


def tune_svr(X, y, n_trials, folds, repeats, seed, out_csv: Path):
    """
    Runs Optuna hyperparameter search for SVR using cross-validated MAE.

    Returns the best CV MAE, the raw Optuna best params, and a cleaned SVR kwargs dict.
    """
    sampler = TPESampler(seed=seed, multivariate=True)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=3)
    study   = opt.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)

    def objective(trial: opt.trial.Trial) -> float:
        params  = build_svr_params(trial)
        Xv, yv  = X.values, y.astype(float).ravel()
        fold_mae = []
        for i, (tr_idx, va_idx) in enumerate(rkf.split(Xv), start=1):
            pipe = make_pipeline(StandardScaler(), SVR(**params))
            pipe.fit(Xv[tr_idx], yv[tr_idx])
            pred = pipe.predict(Xv[va_idx])
            fold_mae.append(mean_absolute_error(yv[va_idx], pred))
            trial.report(float(np.mean(fold_mae)), step=i)
            if trial.should_prune():
                raise opt.TrialPruned()
        return float(np.mean(fold_mae))

    log(f"Optuna tuning: trials={n_trials}, CV={folds}x{repeats}")
    study.optimize(objective, n_trials=n_trials)
    study.trials_dataframe().to_csv(out_csv, index=False)

    best = study.best_params

    # maps kernel-specific gamma keys back to the single 'gamma' SVR argument
    svr_params = {k: v for k, v in best.items() if k not in ("gamma_rbf", "gamma_poly")}
    svr_params["gamma"] = best["gamma_rbf"] if svr_params["kernel"] == "rbf" else best["gamma_poly"]

    return study.best_value, best, svr_params


def corr90_prune(X_train: pd.DataFrame, X_test: pd.DataFrame, out_txt: Path):
    """
    Removes features with pairwise correlation above 0.90 using train-only statistics.

    Returns the pruned train and test DataFrames and the list of dropped column names.
    """
    corr    = X_train.corr().abs()
    upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > 0.90).any()]
    with open(out_txt, "w") as f:
        for c in to_drop:
            f.write(c + "\n")
    return X_train.drop(columns=to_drop), X_test.drop(columns=to_drop, errors="ignore"), to_drop


def run_variant(variant_name: str,
                X_train_full: pd.DataFrame,
                X_test_full: pd.DataFrame,
                train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                args,
                root_out: Path,
                summary_rows: list):
    """
    Runs delta-learning and direct DFT SVR training for one feature variant.

    Applies correlation pruning, tunes with Optuna, fits final pipelines,
    saves predictions, parity PDFs, a metrics JSON, and a Markdown report.
    """
    vdir = ensure_dir(root_out / variant_name)
    pdir = ensure_dir(vdir / "plots")

    # selects the Q9 feature subset or uses all available features
    if variant_name == "q9":
        present = [c for c in Q9_FEATURES if c in X_train_full.columns]
        if len(present) < len(Q9_FEATURES):
            log(f"[{variant_name}] Missing Q9 columns: {set(Q9_FEATURES) - set(present)}")
        Xtr0 = X_train_full[present].copy()
        Xte0 = X_test_full[present].copy()
    else:
        Xtr0, Xte0 = X_train_full.copy(), X_test_full.copy()

    # removes highly correlated features using train-only statistics
    if args.feature_selection.lower() == "corr90":
        Xtr, Xte, dropped = corr90_prune(Xtr0, Xte0, out_txt=vdir / "dropped_corr90.txt")
        log(f"[{variant_name}] Corr>0.90 dropped: {len(dropped)}; features remain: {Xtr.shape[1]}")
    else:
        Xtr, Xte = Xtr0, Xte0
        log(f"[{variant_name}] Corr pruning disabled; features: {Xtr.shape[1]}")

    # resolves target and PM7 column names based on the chosen target
    tcol = args.target_col
    if tcol not in ("ae_delta", "dh_delta"):
        raise ValueError("target_col must be 'ae_delta' or 'dh_delta'.")

    if tcol == "ae_delta":
        dft_col, pm7_col       = "ae_dft", "AE_mopac"
        delta_label, dft_label = "ΔAE (kcal/mol)", "AE (kcal/mol)"
    else:
        dft_col, pm7_col       = "dh_dft", "DH_Mopac"
        delta_label, dft_label = "ΔH‡ (kcal/mol)", "H‡ (kcal/mol)"

    # falls back to searching for a PM7 column by name if the default is missing
    if pm7_col not in test_df.columns:
        candidates = [c for c in test_df.columns if "mopac" in c.lower() or "pm7" in c.lower()]
        if not candidates:
            raise KeyError(f"[{variant_name}] PM7 base column '{pm7_col}' not found in test_df.")
        log(f"[{variant_name}] PM7 '{pm7_col}' not found; using '{candidates[0]}'")
        pm7_col = candidates[0]

    # aligns target arrays to the pruned feature index
    y_tr_delta = train_df.loc[Xtr.index, tcol].values.ravel()
    y_te_delta = test_df.loc[Xte.index, tcol].values.ravel()
    y_tr_dft   = train_df.loc[Xtr.index, dft_col].values.ravel()
    y_te_dft   = test_df.loc[Xte.index, dft_col].values.ravel()
    pm7_base   = test_df.loc[Xte.index, pm7_col].values.ravel()

    # delta-learning: tune on delta, fit final pipeline, reconstruct DFT = PM7 + delta
    cv_mae_delta, raw_params_delta, svr_params_delta = tune_svr(
        X=Xtr, y=pd.Series(y_tr_delta), n_trials=args.n_trials,
        folds=args.cv_folds, repeats=args.cv_repeats, seed=args.seed,
        out_csv=vdir / "svr_delta_optuna_trials.csv"
    )
    with open(vdir / "svr_delta_best_params_raw.json", "w") as f:
        json.dump(raw_params_delta, f, indent=2)
    with open(vdir / "svr_delta_best_params_mapped.json", "w") as f:
        json.dump(svr_params_delta, f, indent=2)

    pipe_delta          = make_pipeline(StandardScaler(), SVR(**svr_params_delta))
    pipe_delta.fit(Xtr.values, y_tr_delta)
    pred_delta          = pipe_delta.predict(Xte.values)
    pred_dft_from_delta = pred_delta + pm7_base

    mae_delta          = mean_absolute_error(y_te_delta, pred_delta)
    mae_dft_from_delta = mean_absolute_error(y_te_dft, pred_dft_from_delta)
    r2_dft_from_delta  = r2_score(y_te_dft, pred_dft_from_delta)

    # direct DFT: tune on DFT values directly, fit final pipeline, predict DFT
    cv_mae_dft, raw_params_dft, svr_params_dft = tune_svr(
        X=Xtr, y=pd.Series(y_tr_dft), n_trials=args.n_trials,
        folds=args.cv_folds, repeats=args.cv_repeats, seed=args.seed,
        out_csv=vdir / "svr_direct_optuna_trials.csv"
    )
    with open(vdir / "svr_direct_best_params_raw.json", "w") as f:
        json.dump(raw_params_dft, f, indent=2)
    with open(vdir / "svr_direct_best_params_mapped.json", "w") as f:
        json.dump(svr_params_dft, f, indent=2)

    pipe_direct     = make_pipeline(StandardScaler(), SVR(**svr_params_dft))
    pipe_direct.fit(Xtr.values, y_tr_dft)
    pred_dft_direct = pipe_direct.predict(Xte.values)
    mae_dft_direct  = mean_absolute_error(y_te_dft, pred_dft_direct)
    r2_dft_direct   = r2_score(y_te_dft, pred_dft_direct)

    # saves fitted pipelines and combined predictions CSV
    import joblib
    joblib.dump(pipe_delta,  vdir / "svr_delta_pipeline.joblib")
    joblib.dump(pipe_direct, vdir / "svr_direct_dft_pipeline.joblib")

    pd.DataFrame({
        "true_delta":          y_te_delta,
        "pred_delta":          pred_delta,
        "true_dft":            y_te_dft,
        "pred_dft_from_delta": pred_dft_from_delta,
        "pred_dft_direct":     pred_dft_direct,
    }, index=Xte.index).to_csv(vdir / "svr_predictions.csv", index=True)

    with open(vdir / "features_after_prune.txt", "w") as f:
        for c in Xtr.columns:
            f.write(c + "\n")

    # generates parity PDFs for delta, DFT-from-delta, and direct DFT predictions
    parity_plot_pdf(
        y_true=y_te_delta, y_pred=pred_delta,
        out_pdf=pdir / "svr_parity_delta_only.pdf",
        xlabel=f"Predicted {delta_label}", ylabel=f"True {delta_label}",
        mae=mae_delta, r2=None
    )
    parity_plot_pdf(
        y_true=y_te_dft, y_pred=pred_dft_from_delta,
        out_pdf=pdir / "svr_parity_dft_from_delta.pdf",
        xlabel=f"Predicted {dft_label}", ylabel=f"True {dft_label}",
        mae=mae_dft_from_delta, r2=r2_dft_from_delta
    )
    parity_plot_pdf(
        y_true=y_te_dft, y_pred=pred_dft_direct,
        out_pdf=pdir / "svr_parity_direct_dft.pdf",
        xlabel=f"Predicted {dft_label}", ylabel=f"True {dft_label}",
        mae=mae_dft_direct, r2=r2_dft_direct
    )

    # collects all metrics for this variant into a single dict
    metrics = {
        "variant":                   variant_name,
        "target_col":                tcol,
        "feature_count_after_prune": int(Xtr.shape[1]),
        "cv_folds":                  args.cv_folds,
        "cv_repeats":                args.cv_repeats,
        "n_trials":                  args.n_trials,
        "delta_cv_mae":              float(cv_mae_delta),
        "delta_test_mae_delta":      float(mae_delta),
        "delta_test_mae_dft":        float(mae_dft_from_delta),
        "delta_test_r2_dft":         float(r2_dft_from_delta),
        "direct_cv_mae":             float(cv_mae_dft),
        "direct_test_mae_dft":       float(mae_dft_direct),
        "direct_test_r2_dft":        float(r2_dft_direct),
    }
    with open(vdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    summary_rows.append(metrics)

    # writes a minimal Markdown report summarising results and artifact paths
    report = textwrap.dedent(f"""
    # SVR Report — Variant: {variant_name}

    **Features after prune:** {Xtr.shape[1]}
    **Target:** {tcol}

    ## Δ-learning
    - CV MAE (Δ): {cv_mae_delta:.4f}
    - Test MAE (Δ): {mae_delta:.4f}
    - Test MAE (DFT = PM7 + Δ): {mae_dft_from_delta:.4f}
    - Test R²  (DFT = PM7 + Δ): {r2_dft_from_delta:.4f}

    ## Direct DFT
    - CV MAE (DFT): {cv_mae_dft:.4f}
    - Test MAE (DFT): {mae_dft_direct:.4f}
    - Test R²  (DFT): {r2_dft_direct:.4f}

    ## Artifacts
    - Plots: `plots/svr_parity_delta_only.pdf`, `plots/svr_parity_dft_from_delta.pdf`, `plots/svr_parity_direct_dft.pdf`
    - Pipelines: `svr_delta_pipeline.joblib`, `svr_direct_dft_pipeline.joblib`
    - Tuning logs: `svr_*_optuna_trials.csv`, best params JSONs
    """).strip()
    (vdir / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    """Parses arguments, loads data, and runs all requested feature variants."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",          required=True, type=str)
    parser.add_argument("--output_dir",        required=True, type=str)
    parser.add_argument("--model",             default="svr")
    parser.add_argument("--target_col",        default="ae_delta", choices=["ae_delta", "dh_delta"])
    parser.add_argument("--tuner",             default="optuna")
    parser.add_argument("--n_trials",          type=int, default=100)
    parser.add_argument("--n_jobs",            type=int, default=6)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--cv_folds",          type=int, default=5)
    parser.add_argument("--cv_repeats",        type=int, default=2)
    parser.add_argument("--feature_selection", default="corr90", choices=["corr90", "none"])
    parser.add_argument("--run_variants",      default="both", choices=["both", "all", "q9"])
    parser.add_argument("--pca_components",    default="none")
    parser.add_argument("--feature_set",       nargs="*", default=None)

    args, unknown = parser.parse_known_args()
    if unknown:
        log(f"Ignoring unknown args: {unknown}")

    os.makedirs(args.output_dir, exist_ok=True)

    # sets thread counts for all linear algebra libraries
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(args.n_jobs)
    np.random.seed(args.seed)

    # loads train and test CSVs and strips any accidental whitespace from column names
    train_df = pd.read_csv(Path(args.data_dir) / "train_df_new.csv", index_col=0)
    test_df  = pd.read_csv(Path(args.data_dir) / "test_df_new.csv",  index_col=0)
    train_df.columns = train_df.columns.str.strip()
    test_df.columns  = test_df.columns.str.strip()

    # drops target columns to isolate features
    X_train_full = train_df.drop(columns=["dh_delta", "ae_delta", "dh_dft", "ae_dft"]).copy()
    X_test_full  = test_df.drop(columns=["dh_delta", "ae_delta", "dh_dft", "ae_dft"]).copy()

    variants     = ["all", "q9"] if args.run_variants == "both" else [args.run_variants]
    summary_rows = []

    for v in variants:
        log(f"=== Running variant: {v} ===")
        run_variant(
            variant_name=v,
            X_train_full=X_train_full,
            X_test_full=X_test_full,
            train_df=train_df,
            test_df=test_df,
            args=args,
            root_out=Path(args.output_dir),
            summary_rows=summary_rows
        )

    # saves a top-level summary CSV across all variants
    pd.DataFrame(summary_rows).to_csv(Path(args.output_dir) / "SVR_summary_all_vs_q9.csv", index=False)
    log("Done. Artifacts saved under per-variant folders and top-level summary CSV created.")


if __name__ == "__main__":
    main()