# HPC_CML_GPR.py — Gaussian Process: Δ-learning + Direct DFT
# Runs BOTH feature variants in one job (ALL-pruned and Q9) if desired.
# - Train-only correlation pruning (|r|>0.90)
# - Optional PCA (applied inside a Pipeline with StandardScaler)
# - Optuna tuning with CV (RepeatedKFold)
# - Publication-style vector PDFs (blue dots, dashed red y=x, beige stats box)
# - Saves per-variant artifacts + a top-level summary CSV
#
# Example:
#   python HPC_CML_GPR.py --data_dir /path/to/data --output_dir /path/to/run \
#       --target_col ae_delta --n_trials 60 --cv_folds 2 --cv_repeats 1 \
#       --pca_components 96 --run_variants both
#
# Notes:
# - GPR is O(n^3) in samples; keep CV modest and consider PCA to stabilize & speed up.
# - CLI keeps compatibility with your SLURM wrapper and ignores unknown flags safely.

import os, json, warnings, argparse, time, textwrap
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
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------- Matplotlib "paper-like" defaults ----------
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Serif",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "legend.frameon": False,
})

# ---------- Q9 features (your nine) ----------
Q9_FEATURES = [
    "exp_mopac", "AE_mopac", "Par_n_Pople", "Mul", "ch_f",
    "DH_Mopac", "ZPE_TS_R", "Freq", "ZPE_P_R"
]

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", flush=True)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def beige_box_stats(ax, text: str):
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f3e9d2", edgecolor="#d3c6a3", alpha=0.9)
    )

def parity_plot_pdf(y_true, y_pred, out_pdf, xlabel, ylabel, title=None, mae=None, r2=None):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    resid  = y_true - y_pred
    sd     = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    ax.scatter(y_pred, y_true, s=14, alpha=0.8, edgecolors="none")               # blue default
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="crimson", lw=1.4)         # dashed red y=x
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", "box")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title, pad=6)

    stats_lines = []
    if r2 is not None:  stats_lines.append(f"R$^2$ = {r2:.3f}")
    if mae is not None: stats_lines.append(f"MAE = {mae:.2f}")
    stats_lines.append(f"SD = {sd:.2f}")
    beige_box_stats(ax, "\n".join(stats_lines))

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)

# ---------- Correlation drop ----------
def corr90_prune(X_train: pd.DataFrame, X_test: pd.DataFrame, out_txt: Path):
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > 0.90).any()]
    with open(out_txt, "w") as f:
        for c in to_drop: f.write(c + "\n")
    return X_train.drop(columns=to_drop), X_test.drop(columns=to_drop, errors="ignore"), to_drop

# ---------- PCA parsing ----------
def parse_pca_components(arg_val):
    if isinstance(arg_val, str):
        if arg_val.lower() == "none":
            return None
        try:
            return int(arg_val)
        except Exception:
            return None
    try:
        v = int(arg_val)
        return v if v > 0 else None
    except Exception:
        return None

# ---------- GPR builders ----------
def make_preproc(seed: int, pca_n: int | None):
    steps = [("scaler", StandardScaler())]
    if pca_n and pca_n > 0:
        steps.append(("pca", PCA(n_components=pca_n, random_state=seed)))
    return Pipeline(steps)

def make_gpr(params: dict, seed: int):
    # Kernel: Constant * RBF + White
    k = ConstantKernel(constant_value=params["c_const"], constant_value_bounds="fixed") \
        * RBF(length_scale=params["length_scale"]) \
        + WhiteKernel(noise_level=params["noise_level"])
    return GaussianProcessRegressor(
        kernel=k,
        alpha=params["alpha"],         # extra jitter to diagonal (stability)
        normalize_y=True,
        copy_X_train=False,
        random_state=seed,
        # n_restarts_optimizer=0  # set >0 if you can afford it
    )

# ---------- Optuna tuning ----------
def tune_gpr(X: pd.DataFrame, y: np.ndarray,
             n_trials: int, folds: int, repeats: int, seed: int,
             pca_n: int | None, out_csv: Path):
    sampler = TPESampler(seed=seed, multivariate=True)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=3)
    study   = opt.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)

    Xv = X.values.astype(float)
    yv = y.astype(float).ravel()

    def objective(trial: opt.trial.Trial) -> float:
        params = {
            "c_const":     trial.suggest_float("c_const",     1e-3, 1e3, log=True),
            "length_scale":trial.suggest_float("length_scale",1e-2, 1e2, log=True),
            "noise_level": trial.suggest_float("noise_level", 1e-8, 1e-1, log=True),
            "alpha":       trial.suggest_float("alpha",       1e-10,1e-2, log=True),
        }

        fold_mae = []
        for i, (tr_idx, va_idx) in enumerate(rkf.split(Xv), start=1):
            pre = make_preproc(seed, pca_n)
            gpr = make_gpr(params, seed)
            pipe = Pipeline(pre.steps + [("gpr", gpr)])
            pipe.fit(Xv[tr_idx], yv[tr_idx])
            pred = pipe.predict(Xv[va_idx])
            fold_mae.append(mean_absolute_error(yv[va_idx], pred))
            trial.report(float(np.mean(fold_mae)), step=i)
            if trial.should_prune():
                raise opt.TrialPruned()
        return float(np.mean(fold_mae))

    log(f"Optuna tuning: trials={n_trials}, CV={folds}x{repeats}, PCA={pca_n or 'none'}")
    study.optimize(objective, n_trials=n_trials)

    df_trials = study.trials_dataframe()
    df_trials.to_csv(out_csv, index=False)

    best = study.best_params
    return study.best_value, best

# ---------- Variant runner ----------
def run_variant(variant_name: str,
                X_train_full: pd.DataFrame,
                X_test_full: pd.DataFrame,
                train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                args,
                root_out: Path,
                summary_rows: list):

    vdir = ensure_dir(root_out / variant_name)
    pdir = ensure_dir(vdir / "plots")

    # 1) Feature subset (Q9 vs ALL)
    if variant_name == "q9":
        present = [c for c in Q9_FEATURES if c in X_train_full.columns]
        if len(present) < len(Q9_FEATURES):
            log(f"[{variant_name}] Missing Q9 columns: {set(Q9_FEATURES) - set(present)}")
        Xtr0 = X_train_full[present].copy()
        Xte0 = X_test_full[present].copy()
    else:
        Xtr0, Xte0 = X_train_full.copy(), X_test_full.copy()

    # 2) Train-only correlation pruning
    if args.feature_selection.lower() == "corr90":
        Xtr, Xte, dropped = corr90_prune(
            Xtr0, Xte0, out_txt=vdir / "dropped_corr90.txt"
        )
        log(f"[{variant_name}] Corr>0.90 dropped: {len(dropped)}; features remain: {Xtr.shape[1]}")
    else:
        Xtr, Xte = Xtr0, Xte0
        log(f"[{variant_name}] Corr pruning disabled; features: {Xtr.shape[1]}")

    # 3) Targets / columns
    tcol = args.target_col
    if tcol == "ae_delta":
        dft_col, pm7_col = "ae_dft", "AE_mopac"
        delta_label, dft_label = "ΔAE (kcal/mol)", "AE (kcal/mol)"
    elif tcol == "dh_delta":
        dft_col, pm7_col = "dh_dft", "DH_Mopac"
        delta_label, dft_label = "ΔH‡ (kcal/mol)", "H‡ (kcal/mol)"
    else:
        raise ValueError("target_col must be 'ae_delta' or 'dh_delta'.")

    if pm7_col not in test_df.columns:
        candidates = [c for c in test_df.columns if "mopac" in c.lower() or "pm7" in c.lower()]
        if not candidates:
            raise KeyError(f"[{variant_name}] PM7 base column '{pm7_col}' not found in test_df.")
        log(f"[{variant_name}] PM7 '{pm7_col}' not found; using '{candidates[0]}'")
        pm7_col = candidates[0]

    # Robust alignment to indices
    y_tr_delta = train_df.loc[Xtr.index, tcol].values.ravel()
    y_te_delta = test_df .loc[Xte.index, tcol].values.ravel()
    y_tr_dft   = train_df.loc[Xtr.index, dft_col].values.ravel()
    y_te_dft   = test_df .loc[Xte.index, dft_col].values.ravel()
    pm7_base   = test_df .loc[Xte.index, pm7_col].values.ravel()

    # 4A) Tune GPR on Δ, fit on all train, predict Δ, reconstruct DFT
    cv_mae_delta, best_params_delta = tune_gpr(
        X=Xtr, y=y_tr_delta, n_trials=args.n_trials,
        folds=args.cv_folds, repeats=args.cv_repeats, seed=args.seed,
        pca_n=args.pca_n, out_csv=vdir / "gpr_delta_optuna_trials.csv"
    )
    with open(vdir / "gpr_delta_best_params.json", "w") as f:
        json.dump(best_params_delta, f, indent=2)

    pre_delta = make_preproc(args.seed, args.pca_n)
    gpr_delta = make_gpr(best_params_delta, args.seed)
    pipe_delta = Pipeline(pre_delta.steps + [("gpr", gpr_delta)])

    t0 = time.time()
    pipe_delta.fit(Xtr.values, y_tr_delta)
    log(f"[{variant_name} | Δ] Final fit time: {time.time()-t0:.1f}s")
    try:
        log(f"[{variant_name} | Δ] Learned kernel: {pipe_delta.named_steps['gpr'].kernel_}")
    except Exception:
        pass

    pred_delta = pipe_delta.predict(Xte.values)
    pred_dft_from_delta = pred_delta + pm7_base

    mae_delta = mean_absolute_error(y_te_delta, pred_delta)
    mae_dft_from_delta = mean_absolute_error(y_te_dft, pred_dft_from_delta)
    r2_dft_from_delta  = r2_score(y_te_dft, pred_dft_from_delta)

    # 4B) Tune GPR on DFT (direct), fit, predict
    cv_mae_direct, best_params_direct = tune_gpr(
        X=Xtr, y=y_tr_dft, n_trials=args.n_trials,
        folds=args.cv_folds, repeats=args.cv_repeats, seed=args.seed,
        pca_n=args.pca_n, out_csv=vdir / "gpr_direct_optuna_trials.csv"
    )
    with open(vdir / "gpr_direct_best_params.json", "w") as f:
        json.dump(best_params_direct, f, indent=2)

    pre_direct = make_preproc(args.seed, args.pca_n)
    gpr_direct = make_gpr(best_params_direct, args.seed)
    pipe_direct = Pipeline(pre_direct.steps + [("gpr", gpr_direct)])

    t0 = time.time()
    pipe_direct.fit(Xtr.values, y_tr_dft)
    log(f"[{variant_name} | DFT] Final fit time: {time.time()-t0:.1f}s")
    try:
        log(f"[{variant_name} | DFT] Learned kernel: {pipe_direct.named_steps['gpr'].kernel_}")
    except Exception:
        pass

    pred_dft_direct = pipe_direct.predict(Xte.values)
    mae_dft_direct  = mean_absolute_error(y_te_dft, pred_dft_direct)
    r2_dft_direct   = r2_score(y_te_dft, pred_dft_direct)

    # 5) Save predictions and pipelines
    import joblib
    joblib.dump(pipe_delta,  vdir / "gpr_delta_pipeline.joblib")
    joblib.dump(pipe_direct, vdir / "gpr_direct_dft_pipeline.joblib")

    pd.DataFrame({
        "true_delta": y_te_delta,
        "pred_delta": pred_delta,
        "true_dft":   y_te_dft,
        "pred_dft_from_delta": pred_dft_from_delta,
        "pred_dft_direct":     pred_dft_direct,
    }, index=Xte.index).to_csv(vdir / "gpr_predictions.csv", index=True)

    with open(vdir / "features_after_prune.txt", "w") as f:
        for c in Xtr.columns: f.write(c + "\n")

    # 6) Publication-style PDFs
    parity_plot_pdf(
        y_true=y_te_delta, y_pred=pred_delta,
        out_pdf=pdir / "gpr_parity_delta_only.pdf",
        xlabel=f"Predicted {delta_label}", ylabel=f"True {delta_label}",
        mae=mae_delta, r2=None
    )
    parity_plot_pdf(
        y_true=y_te_dft, y_pred=pred_dft_from_delta,
        out_pdf=pdir / "gpr_parity_dft_from_delta.pdf",
        xlabel=f"Predicted {dft_label}", ylabel=f"True {dft_label}",
        mae=mae_dft_from_delta, r2=r2_dft_from_delta
    )
    parity_plot_pdf(
        y_true=y_te_dft, y_pred=pred_dft_direct,
        out_pdf=pdir / "gpr_parity_direct_dft.pdf",
        xlabel=f"Predicted {dft_label}", ylabel=f"True {dft_label}",
        mae=mae_dft_direct, r2=r2_dft_direct
    )

    # 7) Per-variant metrics JSON
    metrics = {
        "variant": variant_name,
        "target_col": tcol,
        "feature_count_after_prune": int(Xtr.shape[1]),
        "cv_folds": args.cv_folds, "cv_repeats": args.cv_repeats,
        "n_trials": args.n_trials,
        "pca_components": (args.pca_n if args.pca_n else "none"),

        "delta_cv_mae": float(cv_mae_delta),
        "delta_test_mae_delta": float(mae_delta),
        "delta_test_mae_dft": float(mae_dft_from_delta),
        "delta_test_r2_dft": float(r2_dft_from_delta),

        "direct_cv_mae": float(cv_mae_direct),
        "direct_test_mae_dft": float(mae_dft_direct),
        "direct_test_r2_dft": float(r2_dft_direct),
    }
    with open(vdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 8) Append to top-level summary
    summary_rows.append(metrics)

    # 9) Minimal Markdown report
    report = textwrap.dedent(f"""
    # GPR Report — Variant: {variant_name}

    **Features after prune:** {Xtr.shape[1]}
    **Target:** {tcol}
    **PCA components:** {(args.pca_n if args.pca_n else "none")}

    ## Δ-learning
    - CV MAE (Δ): {cv_mae_delta:.4f}
    - Test MAE (Δ): {mae_delta:.4f}
    - Test MAE (DFT = PM7 + Δ): {mae_dft_from_delta:.4f}
    - Test R²  (DFT = PM7 + Δ): {r2_dft_from_delta:.4f}

    ## Direct DFT
    - CV MAE (DFT): {cv_mae_direct:.4f}
    - Test MAE (DFT): {mae_dft_direct:.4f}
    - Test R²  (DFT): {r2_dft_direct:.4f}

    ## Artifacts
    - Plots: `plots/gpr_parity_delta_only.pdf`, `plots/gpr_parity_dft_from_delta.pdf`, `plots/gpr_parity_direct_dft.pdf`
    - Pipelines: `gpr_delta_pipeline.joblib`, `gpr_direct_dft_pipeline.joblib`
    - Tuning logs: `gpr_*_optuna_trials.csv`, best params JSONs
    """).strip()
    (vdir / "REPORT.md").write_text(report, encoding="utf-8")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    # Compatibility with your SLURM wrapper (kept but not strictly needed)
    parser.add_argument("--model", default="gpr")
    parser.add_argument("--target_col", default="ae_delta", choices=["ae_delta","dh_delta"])
    parser.add_argument("--tuner", default="optuna")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cv_folds", type=int, default=2)   # keep modest for GPR
    parser.add_argument("--cv_repeats", type=int, default=1)
    parser.add_argument("--feature_selection", default="corr90", choices=["corr90","none"])
    parser.add_argument("--run_variants", default="both", choices=["both","all","q9"])
    parser.add_argument("--pca_components", default="none")  # 'none' or int (applies to ALL variants)
    # ignored legacy flags (avoid errors if SLURM passes them)
    parser.add_argument("--feature_set", nargs="*", default=None)

    args, unknown = parser.parse_known_args()
    if unknown:
        log(f"Ignoring unknown args: {unknown}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Thread/BLAS clamps (GPR itself is single-threaded; BLAS may parallelize PCA/scale)
    for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(args.n_jobs)
    np.random.seed(args.seed)

    # Parse PCA setting
    args.pca_n = parse_pca_components(args.pca_components)

    # Load data
    train_df = pd.read_csv(Path(args.data_dir) / "train_df_new.csv", index_col=0)
    test_df  = pd.read_csv(Path(args.data_dir) / "test_df_new.csv",  index_col=0)
    train_df.columns = train_df.columns.str.strip()
    test_df.columns  = test_df.columns.str.strip()

    # Base features (drop targets)
    X_train_full = train_df.drop(columns=["dh_delta","ae_delta","dh_dft","ae_dft"]).copy()
    X_test_full  = test_df .drop(columns=["dh_delta","ae_delta","dh_dft","ae_dft"]).copy()

    # Run variants
    variants = ["all","q9"] if args.run_variants == "both" else [args.run_variants]
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

    # Top-level summary CSV
    pd.DataFrame(summary_rows).to_csv(Path(args.output_dir) / "GPR_summary_all_vs_q9.csv", index=False)
    log("Done. Artifacts saved under per-variant folders and top-level summary CSV created.")

if __name__ == "__main__":
    main()
